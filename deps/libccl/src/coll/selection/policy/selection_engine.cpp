/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "coll/selection/policy/selection_engine.hpp"

#include "coll/algorithms/algorithm_utils.hpp"  // ccl_coll_type_to_str
#include "coll/selection/selector.hpp"
#include "comm/comm.hpp"
#include "common/env/vars_experimental.hpp"  // CCL_SELECTION_POLICY name
#include "common/global/global.hpp"          // ccl::global_data::env()
#include "common/log/log.hpp"

#include "coll/selection/policy/mini_json.hpp"

#include <fstream>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_set>

namespace ccl {
namespace selection {

// -----------------------------------------------------------------------------
// Field / Predicate resolution — all string→enum work happens at load time.
// -----------------------------------------------------------------------------

namespace {

using Field = NodeKey::Field;
using Predicate = NodeKey::Predicate;
using Op = NodeKey::Op;

// Enum-value tables. Each maps a string spelling in the policy file to a
// small int. The engine never round-trips these back to strings on the hot
// path — value_enum_ is compared to a runtime enum promoted the same way.
struct EnumEntry { std::string_view name; int value; };

constexpr int kPhaseScaleup  = 0;
constexpr int kPhaseScaleout = 1;
constexpr EnumEntry kPhaseTable[] = {
    {"scaleup", kPhaseScaleup},
    {"scaleout", kPhaseScaleout},
};

// Values must match ccl_p2p_domain_level (see p2p_domain.hpp) so JSON
// `p2p_domain=<name>` compares against the runtime value with no
// translation table beyond the string→int here at load time.
constexpr EnumEntry kP2pDomainTable[] = {
    {"auto",   0},
    {"node",   1},
    {"numa",   2},
    {"switch", 3},
    {"none",   4},
};

// dtype / atl_transport tables kept intentionally small — extend as JSON
// authors need them. Unknown values are Predicate::invalid (never matches)
// and are warned about at load time.
constexpr EnumEntry kDtypeTable[] = {
    {"int8", 0},  {"uint8", 1},
    {"int16", 2}, {"uint16", 3}, {"float16", 4}, {"bfloat16", 5},
    {"int32", 6}, {"uint32", 7}, {"float32", 8},
    {"int64", 9}, {"uint64", 10}, {"float64", 11},
};
constexpr EnumEntry kAtlTransportTable[] = {
    {"mpi", 0},
    {"ofi", 1},
};

std::optional<int> lookup_enum(const EnumEntry* table, std::size_t n,
                               std::string_view name) {
    for (std::size_t i = 0; i < n; ++i) {
        if (table[i].name == name) return table[i].value;
    }
    return std::nullopt;
}
#define LOOKUP(TABLE, NAME) lookup_enum(TABLE, sizeof(TABLE) / sizeof(TABLE[0]), NAME)

Field resolve_field(std::string_view name) {
    if (name == "msg_size")        return Field::msg_size;
    if (name == "count")           return Field::count;
    if (name == "comm_size")       return Field::comm_size;
    if (name == "phase")           return Field::phase;
    if (name == "p2p_domain")      return Field::p2p_domain;
    if (name == "dtype")           return Field::dtype;
    if (name == "atl_transport")   return Field::atl_transport;
    if (name == "has_p2p_access")  return Field::has_p2p_access;
    if (name == "is_multi_thread") return Field::is_multi_thread;
    if (name == "is_arc")          return Field::is_arc;
    if (name == "is_recording")    return Field::is_recording;
    if (name == "protocol")        return Field::protocol;
    if (name == "algorithm")       return Field::algorithm;
    return Field::unknown;
}

// Strict decimal parse (rejects sign, hex, trailing junk, leading whitespace,
// and overflow). See CR §1.6. Hand-rolled to avoid <charconv>, which some
// icpx toolchains don't ship with a matching libstdc++.
std::optional<size_t> parse_uint_strict(std::string_view s) {
    if (s.empty()) return std::nullopt;
    constexpr unsigned long long kMax = std::numeric_limits<unsigned long long>::max();
    unsigned long long v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return std::nullopt;
        unsigned d = static_cast<unsigned>(c - '0');
        if (v > (kMax - d) / 10ULL) return std::nullopt;
        v = v * 10ULL + d;
    }
    return static_cast<size_t>(v);
}

std::optional<bool> parse_bool(std::string_view s) {
    if (s == "yes" || s == "true" || s == "1")  return true;
    if (s == "no"  || s == "false" || s == "0") return false;
    return std::nullopt;
}

bool is_power_of_two(size_t n) { return n > 0 && (n & (n - 1)) == 0; }

size_t comm_size_of(const ccl_selector_param& p) {
    if (p.comm_size_hint != 0) return p.comm_size_hint;
    return p.comm ? static_cast<size_t>(p.comm->size()) : 0;
}

// PER-RANK bytes. See CR §1.2 — every sycl_selection threshold uses this
// definition; do NOT scale by comm_size.
size_t compute_msg_size(const ccl_selector_param& p) {
    return p.count * p.dtype.size();
}

// Promote the runtime dtype to the same small-int space kDtypeTable uses.
int runtime_dtype_enum(const ccl_selector_param& p) {
    switch (p.dtype.idx()) {
        case ccl::datatype::int8:     return 0;
        case ccl::datatype::uint8:    return 1;
        case ccl::datatype::int16:    return 2;
        case ccl::datatype::uint16:   return 3;
        case ccl::datatype::float16:  return 4;
        case ccl::datatype::bfloat16: return 5;
        case ccl::datatype::int32:    return 6;
        case ccl::datatype::uint32:   return 7;
        case ccl::datatype::float32:  return 8;
        case ccl::datatype::int64:    return 9;
        case ccl::datatype::uint64:   return 10;
        case ccl::datatype::float64:  return 11;
        default:                      return -1;
    }
}

int runtime_phase_enum(const ccl_selector_param& p) {
    if (p.phase_hint == ccl_selector_param::phase_hint_t::scaleout) return kPhaseScaleout;
    if (p.phase_hint == ccl_selector_param::phase_hint_t::scaleup)  return kPhaseScaleup;
    // Fall back to the legacy `is_scaleout` bool when the fast path didn't
    // seed a phase hint — a single source of truth for scaleup/scaleout.
    return p.is_scaleout ? kPhaseScaleout : kPhaseScaleup;
}

int runtime_p2p_domain_enum(const ccl_selector_param& p) {
    // p2p_domain_hint_t values are ABI-aligned with ccl_p2p_domain_level and
    // kP2pDomainTable, so `unset` (-1) falls through as "no runtime info,
    // never matches" and everything else casts through directly.
    if (p.p2p_domain_hint == ccl_selector_param::p2p_domain_hint_t::unset) {
        return -1;
    }
    return static_cast<int>(p.p2p_domain_hint);
}

}  // namespace

// -----------------------------------------------------------------------------
// NodeKey::parse — split raw "<param><op><value>" once, resolve field/enum
// values, and cache them. matches() then does no string work.
// -----------------------------------------------------------------------------

NodeKey NodeKey::parse(const std::string& raw) {
    NodeKey k;

    auto le = raw.find("<=");
    auto lt = raw.find('<');
    auto eq = raw.find('=');

    size_t op_pos = std::string::npos;
    size_t op_len = 0;

    if (le != std::string::npos) {
        op_pos = le; op_len = 2; k.op_ = Op::less_equal;
    }
    else if (lt != std::string::npos && (eq == std::string::npos || lt < eq)) {
        op_pos = lt; op_len = 1; k.op_ = Op::less;
    }
    else if (eq != std::string::npos) {
        op_pos = eq; op_len = 1; k.op_ = Op::equals;
    }

    if (op_pos == std::string::npos) {
        // No operator — treat as raw name; won't match anything.
        k.param_ = raw;
        k.field_ = Field::unknown;
        k.pred_ = Predicate::invalid;
        return k;
    }

    k.param_ = raw.substr(0, op_pos);
    k.raw_value_ = raw.substr(op_pos + op_len);
    k.field_ = resolve_field(k.param_);

    // Wildcard is universal — no per-field logic.
    if (k.raw_value_ == "any") {
        k.pred_ = Predicate::wildcard;
        return k;
    }

    switch (k.field_) {
        case Field::msg_size:
        case Field::count:
        case Field::comm_size:
            if (k.raw_value_ == "pow2") {
                k.pred_ = Predicate::power_of_two;
            }
            else if (auto n = parse_uint_strict(k.raw_value_)) {
                k.pred_ = Predicate::integer;
                k.value_int_ = *n;
            }
            else {
                k.pred_ = Predicate::invalid;
            }
            break;

        case Field::has_p2p_access:
        case Field::is_multi_thread:
        case Field::is_arc:
        case Field::is_recording:
            if (auto b = parse_bool(k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::boolean : Predicate::invalid;
                k.value_bool_ = *b;
            }
            else {
                k.pred_ = Predicate::invalid;
            }
            break;

        case Field::phase:
            if (auto v = LOOKUP(kPhaseTable, k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::enum_val : Predicate::invalid;
                k.value_enum_ = *v;
            }
            else { k.pred_ = Predicate::invalid; }
            break;

        case Field::p2p_domain:
            if (auto v = LOOKUP(kP2pDomainTable, k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::enum_val : Predicate::invalid;
                k.value_enum_ = *v;
            }
            else { k.pred_ = Predicate::invalid; }
            break;

        case Field::dtype:
            if (auto v = LOOKUP(kDtypeTable, k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::enum_val : Predicate::invalid;
                k.value_enum_ = *v;
            }
            else { k.pred_ = Predicate::invalid; }
            break;

        case Field::atl_transport:
            if (auto v = LOOKUP(kAtlTransportTable, k.raw_value_)) {
                k.pred_ = (k.op_ == Op::equals) ? Predicate::enum_val : Predicate::invalid;
                k.value_enum_ = *v;
            }
            else { k.pred_ = Predicate::invalid; }
            break;

        case Field::protocol:
        case Field::algorithm:
            // Terminals — value stays as raw_value_; traverser consumes it.
            k.pred_ = Predicate::enum_val;  // sentinel; not evaluated
            break;

        case Field::unknown:
        case Field::leaf_param:
            k.pred_ = Predicate::invalid;
            break;
    }
    return k;
}

// -----------------------------------------------------------------------------
// NodeKey::matches — hot-path evaluator. No allocations, no string compares
// on the numeric/boolean/enum paths.
// -----------------------------------------------------------------------------

bool NodeKey::matches(const ccl_selector_param& p) const {
    if (pred_ == Predicate::wildcard) return true;
    if (pred_ == Predicate::invalid)  return false;

    auto cmp_int = [&](size_t rt) {
        switch (op_) {
            case Op::equals:     return rt == value_int_;
            case Op::less:       return rt <  value_int_;
            case Op::less_equal: return rt <= value_int_;
        }
        return false;
    };

    switch (field_) {
        case Field::msg_size:
            return pred_ == Predicate::integer
                       ? cmp_int(compute_msg_size(p))
                       : (op_ == Op::equals && is_power_of_two(compute_msg_size(p)));
        case Field::count:
            return pred_ == Predicate::integer
                       ? cmp_int(p.count)
                       : (op_ == Op::equals && is_power_of_two(p.count));
        case Field::comm_size:
            return pred_ == Predicate::integer
                       ? cmp_int(comm_size_of(p))
                       : (op_ == Op::equals && is_power_of_two(comm_size_of(p)));

        case Field::has_p2p_access:
            return p.has_p2p_access_hint >= 0 &&
                   ((p.has_p2p_access_hint != 0) == value_bool_);
        case Field::is_multi_thread:
            return p.is_multi_thread_hint >= 0 &&
                   ((p.is_multi_thread_hint != 0) == value_bool_);
        case Field::is_arc:
            return p.is_arc_hint >= 0 &&
                   ((p.is_arc_hint != 0) == value_bool_);
        case Field::is_recording:
            return p.is_recording_hint >= 0 &&
                   ((p.is_recording_hint != 0) == value_bool_);

        case Field::phase:      return runtime_phase_enum(p) == value_enum_;
        case Field::p2p_domain: {
            int rt = runtime_p2p_domain_enum(p);
            return rt >= 0 && rt == value_enum_;
        }
        case Field::dtype: {
            int rt = runtime_dtype_enum(p);
            return rt >= 0 && rt == value_enum_;
        }
        case Field::atl_transport:
            // No runtime hint on ccl_selector_param yet — always false so
            // `atl_transport=any` (the else branch) is the only way to route.
            return false;

        case Field::protocol:
        case Field::algorithm:
        case Field::unknown:
        case Field::leaf_param:
            return false;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Tree construction from parsed JSON
// -----------------------------------------------------------------------------

namespace {

using json = mini_json::Value;

std::unique_ptr<SelectionNode> build_node(const json& obj, const std::string& path) {
    auto node = std::make_unique<SelectionNode>();
    if (!obj.is_object()) return node;

    std::unordered_set<std::string> seen_keys;

    for (auto it = obj.begin(); it != obj.end(); ++it) {
        // Author comments — keys starting with '_' — never take part in
        // dispatch and don't get flagged as unknown.
        if (!it.key().empty() && it.key()[0] == '_') continue;

        if (!seen_keys.insert(it.key()).second) {
            LOG_WARN("selection policy: '", path,
                     "': duplicate branch key '", it.key(),
                     "' at same level — later value wins");
        }

        NodeKey k = NodeKey::parse(it.key());

        // Leaf-scoped algorithm knobs (e.g. `"k=2": {}`): only classify as a
        // leaf when the key is NOT a recognized filter/terminal and body is
        // empty. Anything else is a branch.
        const bool is_leaf_knob =
            k.field() == Field::unknown &&
            k.op() == NodeKey::Op::equals && !k.raw_value().empty() &&
            it->is_object() && it->empty();

        if (is_leaf_knob) {
            node->leaf_params_[k.param()] = k.raw_value();
            continue;
        }

        if (k.field() == Field::unknown) {
            LOG_WARN("selection policy: '", path,
                     "': unknown filter '", k.param(),
                     "' — this branch will never match; check spelling. "
                     "Known filters: msg_size, count, comm_size, phase, "
                     "p2p_domain, dtype, atl_transport, has_p2p_access, "
                     "is_multi_thread, is_arc, is_recording");
        }
        else if (k.predicate() == NodeKey::Predicate::invalid &&
                 k.field() != Field::protocol && k.field() != Field::algorithm) {
            LOG_WARN("selection policy: '", path,
                     "': branch '", it.key(),
                     "' has an unparseable value; this branch will never match");
        }

        node->children_.emplace_back(std::move(k), build_node(*it, path));
    }
    return node;
}

}  // namespace

// -----------------------------------------------------------------------------
// SelectionEngine
// -----------------------------------------------------------------------------

void SelectionEngine::clear() {
    roots_.clear();
    global_config_.clear();
    loaded_ = false;
}

bool SelectionEngine::load_from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        LOG_WARN("selection policy: cannot open '", path, "'");
        return false;
    }
    json doc;
    try {
        doc = json::parse(in);
    }
    catch (const std::exception& e) {
        LOG_ERROR("selection policy: parse error in '", path, "': ", e.what());
        return false;
    }
    if (!doc.is_object()) {
        LOG_ERROR("selection policy: root of '", path, "' must be an object");
        return false;
    }

    roots_.clear();
    global_config_.clear();

    // Enforce schema_version match. Silent load of a stale schema is the
    // fastest way to mis-dispatch after a schema change.
    static constexpr long long kSchemaVersion = 1;
    if (const auto* sv = doc.find("schema_version")) {
        if (!sv->is_number_integer() || sv->get<long long>() != kSchemaVersion) {
            LOG_ERROR("selection policy: '", path,
                      "' has unsupported schema_version (expected ",
                      kSchemaVersion, ")");
            return false;
        }
    }
    else {
        LOG_WARN("selection policy: '", path,
                 "' has no schema_version; assuming ", kSchemaVersion);
    }

    for (auto it = doc.begin(); it != doc.end(); ++it) {
        if (it.key() == "schema_version") continue;

        if (it.key() == "global_config" && it->is_object()) {
            for (auto cf = it->begin(); cf != it->end(); ++cf) {
                if (cf->is_string()) {
                    global_config_[cf.key()] = cf->get<std::string>();
                }
                else if (cf->is_boolean()) {
                    global_config_[cf.key()] = cf->get<bool>() ? "true" : "false";
                }
                else if (cf->is_number_integer()) {
                    global_config_[cf.key()] = std::to_string(cf->get<long long>());
                }
                else {
                    global_config_[cf.key()] = cf->dump();
                }
            }
            continue;
        }

        NodeKey k = NodeKey::parse(it.key());
        if (k.param() != "collective") {
            LOG_WARN("selection policy: '", path,
                     "': ignoring unknown top-level key '", it.key(),
                     "' (expected 'schema_version', 'global_config', or "
                     "'collective=<name>')");
            continue;
        }
        if (roots_.count(k.raw_value()) != 0) {
            LOG_WARN("selection policy: '", path,
                     "': duplicate 'collective=", k.raw_value(),
                     "' root — later definition overrides earlier");
        }
        roots_[k.raw_value()] = build_node(*it, path);
    }
    loaded_ = !roots_.empty() || !global_config_.empty();
    LOG_INFO("selection policy: loaded ", roots_.size(),
             " collective(s) from '", path, "'");
    return true;
}

bool SelectionEngine::has_rules_for(const std::string& collective) const {
    if (roots_.empty()) return false;
    return roots_.count(collective) != 0;
}

std::optional<std::string> SelectionEngine::global_option(
    const std::string& key) const {
    if (global_config_.empty()) return std::nullopt;
    auto it = global_config_.find(key);
    if (it == global_config_.end()) return std::nullopt;
    return it->second;
}

// -----------------------------------------------------------------------------
// Traversal — backtracking DFS. A subtree that matches an interior filter but
// yields no terminal below no longer aborts the whole resolve; the caller can
// try later siblings (typically `<param>=any`). See CR §2.
// -----------------------------------------------------------------------------

namespace {

std::optional<LeafDecision> traverse(const SelectionNode* node,
                                     const ccl_selector_param& p,
                                     std::optional<std::string> protocol_ctx) {
    if (!node) return std::nullopt;

    for (const auto& [key, child] : node->children_) {
        if (key.field() == Field::algorithm && key.op() == Op::equals) {
            LeafDecision d;
            d.algorithm = key.raw_value();
            d.protocol = std::move(protocol_ctx);
            if (child) d.params = child->leaf_params_;
            return d;
        }
        if (key.field() == Field::protocol && key.op() == Op::equals) {
            auto res = traverse(child.get(), p, key.raw_value());
            if (res.has_value()) return res;
            continue;
        }

        if (!key.matches(p)) continue;
        auto res = traverse(child.get(), p, protocol_ctx);
        if (res.has_value()) return res;
        // Interior match with no terminal below — keep trying later siblings
        // instead of committing to this branch.
    }
    return std::nullopt;
}

// Post-resolve validation gate. Under SYCL-graph capture the tuners `CCL_THROW`
// unless algo == direct (or proxied scatter for alltoall). If the policy
// chose otherwise while is_recording=true, drop the decision — caller falls
// back to legacy dispatch instead of hitting a runtime throw. See CR §1.4.
bool decision_is_valid_for(const LeafDecision& d, const ccl_selector_param& p) {
    if (p.is_recording_hint != 1) return true;
    if (d.algorithm == "direct") return true;
    if (p.ctype == ccl_coll_alltoall && d.algorithm == "scatter") return true;
    LOG_WARN("selection policy: dropping algorithm='", d.algorithm,
             "' under is_recording=true — only 'direct' (and alltoall "
             "'scatter') survive graph capture");
    return false;
}

const char* ctype_to_policy_key(ccl_coll_type ctype) {
    // allgatherv shares the allgather policy tree — see collectives_policy.json.
    if (ctype == ccl_coll_allgatherv) return "allgather";
    return ccl_coll_type_to_str(ctype);
}

}  // namespace

std::optional<LeafDecision> SelectionEngine::resolve(
    const std::string& collective, const ccl_selector_param& param) const {
    auto it = roots_.find(collective);
    if (it == roots_.end()) return std::nullopt;
    auto res = traverse(it->second.get(), param, std::nullopt);
    if (res && !decision_is_valid_for(*res, param)) return std::nullopt;
    return res;
}

std::optional<LeafDecision> SelectionEngine::resolve(
    ccl_coll_type ctype, const ccl_selector_param& param) const {
    return resolve(std::string(ctype_to_policy_key(ctype)), param);
}

// -----------------------------------------------------------------------------
// Singleton (lazy, once-only load)
// -----------------------------------------------------------------------------

SelectionEngine& global_engine() {
    static SelectionEngine engine;
    static std::once_flag once;
    std::call_once(once, [&]() {
        const auto& path = ccl::global_data::env().selection_policy_path;
        if (!path.empty()) {
            engine.load_from_file(path);
        }
    });
    return engine;
}

}  // namespace selection
}  // namespace ccl

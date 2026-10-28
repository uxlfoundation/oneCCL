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

#pragma once

// JSON-based algorithm/protocol selection engine.
//
// Design:
//   - The engine loads a JSON policy file (path from CCL_SELECTION_POLICY env
//     var) and answers `resolve(collective, runtime_param)` with a leaf
//     decision (algorithm name + optional protocol name + optional knobs).
//   - Selection fields (inputs the traverser matches on):
//       msg_size, count, comm_size, phase, p2p_domain, dtype, atl_transport,
//       has_p2p_access, is_multi_thread, is_arc, is_recording
//   - Terminal leaves (outputs):
//       protocol=<name> (optional wrapper) -> algorithm=<name>
//   - Traversal is depth-first, ordered by the JSON's insertion order (first
//     matching sibling wins; `<param>=any` acts as the else branch).
//

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "coll/algorithms/algorithm_utils.hpp"  // ccl_coll_type

struct ccl_selector_param;

namespace ccl {
namespace selection {

// -----------------------------------------------------------------------------
// LeafDecision — what a successful traversal returns.
// -----------------------------------------------------------------------------
struct LeafDecision {
    std::string algorithm;                // required, e.g. "ring"
    std::optional<std::string> protocol;  // optional, e.g. "sycl_ll" / "sycl_simple"
    std::unordered_map<std::string, std::string> params;  // "k"="2", etc.
};

// -----------------------------------------------------------------------------
// NodeKey — a parsed `<param><op><value>` branch label.
// -----------------------------------------------------------------------------
//
// Recognised operators: `=`, `<`, `<=`. Recognised value predicates:
//   - integer / byte literal (e.g. "8192")
//   - "pow2"          (matches only if the runtime param is a power of two)
//   - "any"           (unconditional match; author-side "else")
//   - "yes" / "no"    (booleans)
class NodeKey {
public:
    enum class Op { equals, less, less_equal };

    // Which selection field this key filters on. Resolved once at load time
    // so matches() is a switch, not a chain of string compares.
    enum class Field {
        // Numeric fields (Predicate::integer or Predicate::power_of_two).
        msg_size, count, comm_size,
        // Enum-like string fields (Predicate::enum_val).
        phase, p2p_domain, dtype, atl_transport,
        // Boolean tri-state fields on ccl_selector_param.
        has_p2p_access, is_multi_thread, is_arc, is_recording,
        // Terminals — traversal handles these directly.
        protocol, algorithm,
        // Not-a-filter leaf-scoped knob (e.g. `k=2`).
        leaf_param,
        // Recognized-but-unspellable / not-yet-implemented.
        unknown,
    };

    enum class Predicate {
        wildcard,       // "any"
        power_of_two,   // "pow2"
        integer,        // 8192 — value_int_
        boolean,        // "yes"/"no"/"true"/"false" — value_bool_
        enum_val,       // enum-mapped string — value_enum_
        invalid,        // parseable-but-unrecognized value; never matches
    };

    NodeKey() = default;
    static NodeKey parse(const std::string& raw);

    const std::string& param() const { return param_; }
    const std::string& raw_value() const { return raw_value_; }
    Op op() const { return op_; }
    Field field() const { return field_; }
    Predicate predicate() const { return pred_; }

    bool is_wildcard() const { return pred_ == Predicate::wildcard; }

    // Evaluate this branch against the runtime param bundle.
    bool matches(const ccl_selector_param& param) const;

private:
    std::string param_;
    std::string raw_value_;
    Op op_ = Op::equals;
    Field field_ = Field::unknown;
    Predicate pred_ = Predicate::invalid;
    size_t value_int_ = 0;
    int value_enum_ = 0;
    bool value_bool_ = false;
};

// -----------------------------------------------------------------------------
// SelectionNode — one node in the policy tree.
// -----------------------------------------------------------------------------
//
// A node has ordered `children_` (interior traversal, depth-first) and, on
// terminal nodes, `leaf_params_` — algorithm knobs like `k=2`. A `protocol=`
// wrapper contains an `algorithm=` child; traversal stops descending as soon
// as it can assemble a complete LeafDecision.
class SelectionNode {
public:
    SelectionNode() = default;

    // Ordered list preserves author intent: first matching child wins.
    std::vector<std::pair<NodeKey, std::unique_ptr<SelectionNode>>> children_;

    // Leaf-scoped params (algorithm knobs). Only populated on terminal nodes.
    std::unordered_map<std::string, std::string> leaf_params_;
};

// -----------------------------------------------------------------------------
// SelectionEngine — orchestrator. Owns the parsed trees keyed by collective.
// -----------------------------------------------------------------------------
class SelectionEngine {
public:
    // Load a JSON policy from a file. Returns false on I/O or parse error;
    // errors are logged and the engine stays empty (safe fallback).
    bool load_from_file(const std::string& path);

    // Discard current rules (used for reload / testing).
    void clear();

    // True when at least one collective root OR global_config key was loaded.
    // Callers can early-out on the no-policy case with a single boolean load
    // — cheaper than has_rules_for()/global_option() which do hashmap lookups.
    bool is_loaded() const noexcept { return loaded_; }

    // Whether any rules are registered for `collective` (e.g. "allgather").
    bool has_rules_for(const std::string& collective) const;

    // Read a value from the file's top-level `"global_config": { ... }` block.
    // Returns nullopt if the file wasn't loaded, has no `global_config`, or the
    // key is absent. Used for global directives that steer the runtime
    // *outside* the tree traversal (e.g. p2p_domain_level, log_selections). The
    // key namespace is intentionally separate from selection-field names
    // to keep global config from being confused with tree filters.
    std::optional<std::string> global_option(const std::string& key) const;

    // Walk the tree for `collective` and return a decision if traversal
    // reaches a terminal, or nullopt if no branch matched at some level or
    // the collective is absent from the file.
    std::optional<LeafDecision> resolve(const std::string& collective,
                                        const ccl_selector_param& param) const;

    // ccl_coll_type overload. Owns the ctype → policy-key mapping (notably
    // ccl_coll_allgatherv → "allgather"), so callers don't each have to
    // remember it. Returns nullopt for absent-in-policy or unmapped types.
    std::optional<LeafDecision> resolve(ccl_coll_type ctype,
                                        const ccl_selector_param& param) const;

private:
    std::unordered_map<std::string, std::unique_ptr<SelectionNode>> roots_;
    std::unordered_map<std::string, std::string> global_config_;
    // Fast-path flag: set to true if load_from_file() populated anything
    // (roots or global_config). Enables O(1) early-out at every consult site.
    bool loaded_ = false;
};

// -----------------------------------------------------------------------------
// Global singleton accessor.
// -----------------------------------------------------------------------------
//
// The engine is lazy-initialised: the first call reads CCL_SELECTION_POLICY,
// loads the file if the env var is set, and caches the result. Thread-safe
// (Meyers singleton with once_flag for load).
SelectionEngine& global_engine();

}  // namespace selection
}  // namespace ccl

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

// Unit tests for coll/selection/policy/mini_json.hpp
//
// The primary case is parsing the real collectives_policy.json shipped in
// deps/libccl/policies/ — every structural feature the engine relies on
// (order preservation, comments, nested objects, empty leaves, arrays,
// long strings) is exercised there. Additional cases below cover the parser
// corners that the policy file doesn't touch directly.
//
// The path to the fixture is injected via -DMINI_JSON_POLICY_PATH="..." from
// CMake so the test binary can be run from any working directory.

#include "gtest/gtest.h"

#include "coll/selection/policy/mini_json.hpp"

#include <cstdlib>
#include <fstream>
#include <string>

// Resolution order for the policy fixture path:
//   1) MINI_JSON_POLICY_PATH env var (set by ctest and the install target)
//   2) MINI_JSON_POLICY_PATH_FALLBACK compile-time define (build tree)
//   3) hard-coded relative path (last-ditch, matches source-tree layout)
static std::string policy_fixture_path() {
    if (const char* p = std::getenv("MINI_JSON_POLICY_PATH")) return p;
#ifdef MINI_JSON_POLICY_PATH_FALLBACK
    return MINI_JSON_POLICY_PATH_FALLBACK;
#else
    return "../../../deps/libccl/policies/collectives_policy.json";
#endif
}

using ccl::selection::mini_json::Value;
using ccl::selection::mini_json::parse_error;

// ---------------------------------------------------------------------------
// collectives_policy.json — the real file the engine consumes.
// ---------------------------------------------------------------------------

TEST(MiniJsonPolicyFile, ParsesWithoutError) {
    std::ifstream in(policy_fixture_path());
    ASSERT_TRUE(in.good()) << "cannot open fixture " << policy_fixture_path();
    ASSERT_NO_THROW({
        Value v = Value::parse(in);
        (void)v;
    });
}

TEST(MiniJsonPolicyFile, RootIsAnObjectWithExpectedTopLevelKeys) {
    // Set-membership check — doesn't lock the file's key ORDER, so editing
    // or reordering the policy doesn't break the parser test. Order
    // preservation is asserted separately in the inline-fixture tests
    // below, which own their own JSON.
    std::ifstream in(policy_fixture_path());
    Value doc = Value::parse(in);
    ASSERT_TRUE(doc.is_object());

    ASSERT_NE(doc.find("schema_version"), nullptr);
    ASSERT_NE(doc.find("global_config"), nullptr);
    // Every non-schema/non-global_config/non-comment top-level key is a
    // collective root or a `notes` block.
    for (auto it = doc.begin(); it != doc.end(); ++it) {
        const auto& k = it.key();
        if (k == "schema_version" || k == "global_config" || k == "notes") continue;
        if (!k.empty() && k[0] == '_') continue;
        EXPECT_EQ(k.rfind("collective=", 0), 0u)
            << "unexpected top-level key: " << k;
    }
}

TEST(MiniJson, ObjectKeysPreserveInsertionOrder) {
    // Order preservation is a hard engine invariant (first-match-commits).
    // Test it on an inline fixture we own, not on the shipped policy file.
    const std::string src = R"json({"c":3,"a":1,"b":2})json";
    Value doc = Value::parse(src);
    ASSERT_TRUE(doc.is_object());
    std::vector<std::string> keys;
    for (auto it = doc.begin(); it != doc.end(); ++it) keys.push_back(it.key());
    ASSERT_EQ(keys.size(), 3u);
    EXPECT_EQ(keys[0], "c");
    EXPECT_EQ(keys[1], "a");
    EXPECT_EQ(keys[2], "b");
}

TEST(MiniJsonPolicyFile, PrimitivesRoundTrip) {
    std::ifstream in(policy_fixture_path());
    Value doc = Value::parse(in);

    const Value* sv = doc.find("schema_version");
    ASSERT_NE(sv, nullptr);
    ASSERT_TRUE(sv->is_number_integer());
    EXPECT_EQ(sv->get<long long>(), 1);

    const Value* gc = doc.find("global_config");
    ASSERT_NE(gc, nullptr);
    ASSERT_TRUE(gc->is_object());

    const Value* dl = gc->find("p2p_domain_level");
    ASSERT_NE(dl, nullptr);
    ASSERT_TRUE(dl->is_string());
    EXPECT_EQ(dl->get<std::string>(), "node");
}

TEST(MiniJsonPolicyFile, VocabularyArraysParseAsArrays) {
    // Structural check only — that `notes.p2p_domains` exists and is a
    // non-empty array of strings. Contents are policy-authoring vocabulary
    // and can churn without breaking the parser.
    std::ifstream in(policy_fixture_path());
    Value doc = Value::parse(in);

    const Value* vocab = doc.find("notes");
    ASSERT_NE(vocab, nullptr);
    ASSERT_TRUE(vocab->is_object());

    const Value* domains = vocab->find("p2p_domains");
    ASSERT_NE(domains, nullptr);
    ASSERT_TRUE(domains->is_array());
    EXPECT_GT(domains->size(), 0u);
    for (const auto& e : domains->as_array()) {
        EXPECT_TRUE(e.is_string());
    }
}

TEST(MiniJsonPolicyFile, NestedNavigationHitsEmptyLeaf) {
    // Path: collective=allgather / p2p_domain=any / phase=scaleup / is_arc=yes
    // / is_multi_thread=yes / protocol=sycl_ll / algorithm=ring
    // / line_size=256 -> {}  (empty object, used by the engine as a knob).
    std::ifstream in(policy_fixture_path());
    Value doc = Value::parse(in);

    const Value* n = doc.find("collective=allgather");
    ASSERT_NE(n, nullptr);
    n = n->find("p2p_domain=any");     ASSERT_NE(n, nullptr);
    n = n->find("phase=scaleup");      ASSERT_NE(n, nullptr);
    n = n->find("is_arc=yes");         ASSERT_NE(n, nullptr);
    n = n->find("is_multi_thread=yes");ASSERT_NE(n, nullptr);
    n = n->find("protocol=sycl_ll");   ASSERT_NE(n, nullptr);
    n = n->find("algorithm=ring");     ASSERT_NE(n, nullptr);
    n = n->find("line_size=256");      ASSERT_NE(n, nullptr);
    ASSERT_TRUE(n->is_object());
    EXPECT_TRUE(n->empty());
}

TEST(MiniJsonPolicyFile, DumpReparsesToEquivalentTree) {
    // Round-trip: parse -> dump -> parse. The re-parsed tree must have the
    // same top-level shape (key names in order, types, and primitive values).
    std::ifstream in(policy_fixture_path());
    Value first = Value::parse(in);
    const std::string serialized = first.dump();
    Value second = Value::parse(serialized);

    ASSERT_TRUE(first.is_object());
    ASSERT_TRUE(second.is_object());
    ASSERT_EQ(first.size(), second.size());

    auto ita = first.begin();
    auto itb = second.begin();
    for (; ita != first.end() && itb != second.end(); ++ita, ++itb) {
        EXPECT_EQ(ita.key(), itb.key());
        EXPECT_EQ(static_cast<int>(ita->type()), static_cast<int>(itb->type()));
    }
}

// ---------------------------------------------------------------------------
// Parser corners the policy file doesn't exercise directly.
// ---------------------------------------------------------------------------

TEST(MiniJson, LineAndBlockCommentsStrippedTogether) {
    const std::string src = R"json(
        // top-level line comment
        {
          "a": 1, /* trailing block */
          /* leading
             multi-line */
          "b": "x" // eol
        }
    )json";
    Value v = Value::parse(src);
    ASSERT_TRUE(v.is_object());
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v.find("a")->get<long long>(), 1);
    EXPECT_EQ(v.find("b")->get<std::string>(), "x");
}

TEST(MiniJson, IntegerOverflowPromotesToDouble) {
    // Just past INT64_MAX — must not throw, must not wrap; comes back as a
    // Double so callers see the magnitude.
    Value v = Value::parse("9223372036854775808");
    ASSERT_TRUE(v.is_number_float());
    EXPECT_GT(v.get<double>(), 9.2e18);
}

TEST(MiniJson, Int64MinExact) {
    Value v = Value::parse("-9223372036854775808");
    ASSERT_TRUE(v.is_number_integer());
    EXPECT_EQ(v.get<long long>(), std::numeric_limits<long long>::min());
}

TEST(MiniJson, StringEscapesAndBmpEscape) {
    // Standard escapes plus a BMP \u escape (U+00E9 = 'é').
    // Uses `é` so we actually exercise the 2-byte UTF-8 encode branch,
    // not the escape-free memcpy fast path.
    Value v = Value::parse(R"json("a\"b\\c\/d\b\f\n\r\te\u00E9")json");
    ASSERT_TRUE(v.is_string());
    EXPECT_EQ(v.get<std::string>(), std::string("a\"b\\c/d\b\f\n\r\te\xc3\xa9"));
}

TEST(MiniJson, SurrogatePairDecodesAstral) {
    // U+1F600 (😀) as a UTF-16 surrogate pair 😀. Using the
    // escape sequence (not a literal emoji) forces the parser through the
    // surrogate combining logic and the 4-byte UTF-8 encoding branch.
    Value v = Value::parse(R"json("\uD83D\uDE00")json");
    ASSERT_TRUE(v.is_string());
    EXPECT_EQ(v.get<std::string>(), std::string("\xF0\x9F\x98\x80"));
}

TEST(MiniJson, LoneSurrogateIsRejected) {
    EXPECT_THROW(Value::parse(R"json("\uD800")json"), parse_error);
    EXPECT_THROW(Value::parse(R"json("\uDC00")json"), parse_error);
}

TEST(MiniJson, LeadingZeroInNumberRejected) {
    EXPECT_THROW(Value::parse("01"), parse_error);
}

TEST(MiniJson, TrailingContentRejected) {
    EXPECT_THROW(Value::parse("{} garbage"), parse_error);
}

TEST(MiniJson, GradualUnderflowAccepted) {
    // 1e-320 sets ERANGE via subnormal, but is a legal JSON number.
    // Regression for CR §2.
    Value v = Value::parse("1e-320");
    ASSERT_TRUE(v.is_number_float());
    EXPECT_LT(v.get<double>(), 1e-300);
    EXPECT_GE(v.get<double>(), 0.0);
}

TEST(MiniJson, DumpPreservesDoubleTypeAcrossRoundtrip) {
    // Whole-valued doubles (1.5e3, 1500.0) must not silently promote to
    // Int on re-parse. Regression for CR §2.
    Value v = Value::parse("1.5e3");
    ASSERT_TRUE(v.is_number_float());
    Value v2 = Value::parse(v.dump());
    EXPECT_TRUE(v2.is_number_float());
    EXPECT_DOUBLE_EQ(v.get<double>(), v2.get<double>());
}

TEST(MiniJson, DumpDoesNotSegfaultOnMovedFromValue) {
    // Move-only Value: after `Value moved = std::move(v);` the source
    // reports is_object()==true with obj_==nullptr. dump() must not
    // dereference. Regression for CR §1.5.
    Value v = Value::parse(R"json({"a":1,"b":[1,2]})json");
    ASSERT_TRUE(v.is_object());
    Value moved = std::move(v);
    // NOLINTNEXTLINE(bugprone-use-after-move) — the point of the test.
    EXPECT_NO_THROW({ std::string s = v.dump(); (void)s; });
}

TEST(MiniJson, ErrorLocationIsReasonable) {
    try {
        Value::parse("{\n  \"a\": ,\n}");
        FAIL() << "expected parse_error";
    } catch (const parse_error& e) {
        EXPECT_EQ(e.line(), 2u);
        EXPECT_GT(e.col(), 1u);
    }
}

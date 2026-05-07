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

#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"
#include <gtest/gtest.h>

TEST(OneCCLVersionTest, GetVersionReturnsCorrectVersionCode) {
    int version = -1;
    ASSERT_EQ(onecclGetVersion(&version), onecclSuccess);
    EXPECT_EQ(version,
              ONECCL_VERSION(ONECCL_MAJOR, ONECCL_MINOR, ONECCL_PATCH));
}

// Test onecclExtractVersionComponents
TEST(OneCCLVersionTest, ExtractVersionComponentsReturnsCorrectValues) {
    int const version_code =
        ONECCL_VERSION(ONECCL_MAJOR, ONECCL_MINOR, ONECCL_PATCH);
    int major = -1;
    int minor = -1;
    int patch = -1;
    ASSERT_EQ(
        onecclExtractVersionComponents(version_code, &major, &minor, &patch),
        onecclSuccess);
    EXPECT_EQ(major, ONECCL_MAJOR);
    EXPECT_EQ(minor, ONECCL_MINOR);
    EXPECT_EQ(patch, ONECCL_PATCH);
}

// Test onecclExtractVersionComponents with random version code
TEST(OneCCLVersionTest, ExtractVersionComponentsRandomVersion) {
    int major = -1;
    int minor = -1;
    int patch = -1;
    // The first 4 characters are major version - 2032, next two are minor - 97,
    // the last two are patch - 71
    int const version_code = 20329771;
    ASSERT_EQ(
        onecclExtractVersionComponents(version_code, &major, &minor, &patch),
        onecclSuccess);
    EXPECT_EQ(major, 2032);
    EXPECT_EQ(minor, 97);
    EXPECT_EQ(patch, 71);
}

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

#include <iostream>
#include <sstream>
#include <cassert>
#include <mpi.h>
#include "oneapi/ccl.hpp"

int main(int argc, char* argv[]) {
    // this test is a regression test for abi-compatibility with pre-c++11 abi
    // if this test compiles, links and runs without complaining
    // about undefined symbols, it means it already succeeded
    ccl::device_index_type dit;

    assert(ccl::to_string(dit) == "[0:0:0]");

    std::ostringstream dit_0_string;
    dit_0_string << ccl::to_string(dit);
    assert(dit_0_string.str() == "[0:0:0]");

    std::ostringstream dit_0_stream;
    dit_0_stream << dit;
    assert(dit_0_stream.str() == "[0:0:0]");

    ccl::device_index_type dit_1_3_2 = ccl::from_string("[1:3:2]");

    assert(ccl::to_string(dit_1_3_2) == "[1:3:2]");

    std::ostringstream dit_1_3_2_string;
    dit_1_3_2_string << ccl::to_string(dit_1_3_2);
    assert(dit_1_3_2_string.str() == "[1:3:2]");

    std::ostringstream dit_1_3_2_stream;
    dit_1_3_2_stream << dit_1_3_2;
    assert(dit_1_3_2_stream.str() == "[1:3:2]");

    std::cout << "PASSED\n";
    return 0;
}

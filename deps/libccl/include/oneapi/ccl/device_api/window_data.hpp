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

#include <cstddef>

namespace ccl {
namespace v1 {

/**
 * Device-accessible window data, in device memory and written once by an H2D copy
 * at registration — the host works off its own shadow instead (window_record in
 * src/ccl_api_functions.cpp). Internal: users see only the opaque window_t.
 */
struct window_data {
    void** ptrs;            /**< device array of peer base pointers, indexed by LSA (node) rank (size = lsa_size) */
    int nranks;             /**< number of entries in ptrs[] (equals lsa_size) */
    size_t size;            /**< size of each peer's window region in bytes */
    int lsa_rank;           /**< this rank's position in the LSA team (its node rank) */
    int world_rank;         /**< this rank's position in the world communicator */
    int lsa_size;           /**< number of ranks in the LSA team */
};

} // namespace v1
} // namespace ccl

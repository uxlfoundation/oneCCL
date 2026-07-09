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

#include <sycl/sycl.hpp>
#include <vector>

#include "oneapi/ccl.hpp"

// Stage data into `dst` and apply the custom pre-operation there, returning the resulting
// collective event. Handles both shapes the SYCL collectives need:
//   * Out-of-place: `src` is non-null and differs from `dst` -- the helper memcpys src->dst
//     before running the pre-op kernel on `dst`.
//   * In-place:    `src` is null or equal to `dst` -- the helper skips the copy and just chains
//     the pre-op on `dep_events`, preserving the caller's input dependencies.
//
// The returned event is the type-dispatched ccl::event created by invoke_collective,
// suitable for pushing onto a vector_class<event> dep chain.
ccl::event stage_and_apply_custom_preop(sycl::queue& q,
                                        void* dst,
                                        const void* src,
                                        size_t count,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        size_t tmp_buf_size,
                                        const std::vector<sycl::event>& dep_events);

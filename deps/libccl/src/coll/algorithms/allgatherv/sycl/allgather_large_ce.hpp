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
#include "oneapi/ccl.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"  // for sycl_ptrs_type

// Forward declarations
class ccl_comm;
class ccl_stream;

/*
 * Allgather for large payload using copy engine
 */
ccl::event allgather_large_copy_engine(sycl::queue& q,
                                       const void* send_buf,
                                       size_t send_count,
                                       void* recv_buf,
                                       const ccl::vector_class<size_t>& recv_counts,
                                       const ccl::vector_class<size_t>& offsets,
                                       ccl::datatype dtype,
                                       ccl_comm* comm,
                                       ccl_stream* global_stream,
                                       sycl_ptrs_type& sycl_ptrs,
                                       const ccl::vector_class<ccl::event>& deps);

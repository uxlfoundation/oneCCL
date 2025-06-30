/*
 Copyright 2016-2020 Intel Corporation
 
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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"

#define SYCL_ALLREDUCE_FUNCTIONS(MSGSIZE) \
    void init_allreduce_##MSGSIZE(ccl::datatype dtype, \
                                  sycl::queue& queue, \
                                  ccl_comm* comm, \
                                  ccl_stream* stream, \
                                  uint32_t rank_in, \
                                  uint32_t world_in); \
    ccl::event run_allreduce_##MSGSIZE(ccl::datatype dtype, \
                                       sycl::queue& q, \
                                       const void* in_buf, \
                                       void* out_buf, \
                                       size_t count, \
                                       ccl::reduction reduction, \
                                       const ccl::vector_class<ccl::event>& deps, \
                                       bool& done);

SYCL_ALLREDUCE_FUNCTIONS(small)
SYCL_ALLREDUCE_FUNCTIONS(medium)
SYCL_ALLREDUCE_FUNCTIONS(large)

namespace ccl {

namespace v1 {
event allreduce_sycl(sycl::queue& q,
                     const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction reduction,
                     ccl_comm* comm,
                     ccl_stream* op_stream,
                     const allreduce_attr& attr,
                     const vector_class<event>& deps,
                     bool& done);

} // namespace v1

} // namespace ccl

ccl::event allreduce_small(const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           ccl::reduction reduction,
                           ccl_comm* comm,
                           ccl_stream* global_stream,
                           const ccl::vector_class<ccl::event>& deps);

ccl::event allreduce_large(const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           ccl::reduction reduction,
                           ccl_comm* comm,
                           ccl_stream* global_stream,
                           const ccl::vector_class<ccl::event>& deps);

// ring with LL protocols
ccl::event allreduce_ll_ring(const void* src,
                             void* dst,
                             size_t count,
                             ccl::datatype dtype,
                             ccl::reduction reduction,
                             ccl_comm* comm,
                             ccl_stream* global_stream,
                             bool& done);

ccl::event allreduce_scaleout_sycl(sycl::queue& q,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   ccl::reduction reduction,
                                   ccl_comm* comm,
                                   const ccl::vector_class<ccl::event>& deps,
                                   bool original_deps,
                                   sycl_allreduce_tune_attr tune_attr,
                                   bool& done,
                                   bool is_cpu_buffers = false);

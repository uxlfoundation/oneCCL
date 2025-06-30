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

#include "coll/algorithms/utils/sycl_selection.hpp"

#define SYCL_ALLGATHERV_FUNCTIONS(MSGSIZE) \
    void init_allgatherv_##MSGSIZE(ccl::datatype dtype, \
                                   sycl::queue& queue, \
                                   ccl_comm* comm, \
                                   ccl_stream* stream, \
                                   uint32_t rank_in, \
                                   uint32_t world_in); \
    ccl::event run_allgatherv_##MSGSIZE(ccl::datatype dtype, \
                                        sycl::queue& q, \
                                        const void* send_buf, \
                                        size_t send_count, \
                                        void* rev_buf, \
                                        const ccl::vector_class<size_t>& recv_counts, \
                                        bool& done);

SYCL_ALLGATHERV_FUNCTIONS(small)
SYCL_ALLGATHERV_FUNCTIONS(medium)

namespace ccl {
namespace v1 {

ccl::event allgather_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      const ccl::vector_class<size_t>& offsets,
                                      ccl::datatype dtype,
                                      ccl_comm* comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done,
                                      sycl_coll_scaleup_attr coll_attr = {});

ccl::event allgather_sycl(sycl::queue& q,
                          const void* send_buf,
                          size_t send_count,
                          void* recv_buf,
                          const ccl::vector_class<size_t>& recv_counts,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          ccl_stream* op_stream,
                          const allgatherv_attr& attr,
                          const vector_class<event>& deps,
                          bool& done);
} // namespace v1
} // namespace ccl

ccl::event allgatherv_small(const void* send_buf,
                            size_t send_count,
                            void* recv_buf,
                            const ccl::vector_class<size_t>& recv_counts,
                            const ccl::vector_class<size_t>& offsets,
                            ccl::datatype dtype,
                            ccl_comm* comm,
                            ccl_stream* global_stream,
                            const ccl::vector_class<ccl::event>& deps);

ccl::event allgatherv_large(const void* send_buf,
                            size_t send_count,
                            void* recv_buf,
                            const ccl::vector_class<size_t>& recv_counts,
                            const ccl::vector_class<size_t>& offsets,
                            ccl::datatype dtype,
                            ccl_comm* comm,
                            ccl_stream* global_stream,
                            const ccl::vector_class<ccl::event>& deps,
                            sycl_coll_scaleup_attr coll_attr = {});

// ring with LL protocols
ccl::event allgatherv_ll_ring(const void* send_buf,
                              size_t send_count,
                              void* recv_buf,
                              const ccl::vector_class<size_t>& recv_counts,
                              const ccl::vector_class<size_t>& offsets,
                              ccl::datatype dtype,
                              ccl_comm* comm,
                              ccl_stream* global_stream,
                              const ccl::vector_class<ccl::event>& deps,
                              bool& done);

ccl::event allgatherv_scaleout_sycl(sycl::queue& q,
                                    const void* send_buf,
                                    size_t send_count,
                                    void* recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    ccl::datatype dtype,
                                    ccl_comm* comm,
                                    const ccl::vector_class<ccl::event>& deps,
                                    bool original_deps,
                                    bool& done,
                                    sycl_allgatherv_tune_attr tune_attr,
                                    bool is_cpu_buffers = false);

ccl::event allgatherv_scaleout_sycl_direct(sycl::queue& q,
                                           const void* send_buf,
                                           size_t send_count,
                                           void* recv_buf,
                                           const ccl::vector_class<size_t>& recv_counts,
                                           ccl::datatype dtype,
                                           ccl_comm* comm,
                                           const ccl::vector_class<ccl::event>& deps,
                                           bool& done,
                                           bool copy_to_host,
                                           bool is_cpu_buffers = false);

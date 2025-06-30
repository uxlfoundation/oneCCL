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

#define SYCL_REDUCE_SCATTER_FUNCTIONS(MSGSIZE) \
    void init_reduce_scatter_##MSGSIZE(ccl::datatype dtype, \
                                       sycl::queue& queue, \
                                       ccl_comm* comm, \
                                       ccl_stream* stream, \
                                       uint32_t rank_in, \
                                       uint32_t world_in); \
    ccl::event run_reduce_scatter_##MSGSIZE(ccl::datatype dtype, \
                                            sycl::queue& q, \
                                            const void* send_buf, \
                                            void* rev_buf, \
                                            size_t recv_count, \
                                            ccl::reduction reduction, \
                                            const ccl::vector_class<ccl::event>& deps, \
                                            bool& done);

SYCL_REDUCE_SCATTER_FUNCTIONS(small)
SYCL_REDUCE_SCATTER_FUNCTIONS(medium)
SYCL_REDUCE_SCATTER_FUNCTIONS(large)

namespace ccl {
namespace v1 {

ccl::event reduce_scatter_sycl_single_node(sycl::queue& q,
                                           const void* send_buf,
                                           void* recv_buf,
                                           size_t recv_count,
                                           datatype dtype,
                                           reduction reduction,
                                           ccl_comm* comm,
                                           ccl_stream* global_stream,
                                           const vector_class<event>& deps,
                                           bool& done,
                                           sycl_coll_scaleup_attr coll_attr = {});

ccl::event reduce_scatter_sycl(sycl::queue& q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t recv_count,
                               datatype dtype,
                               ccl::reduction reduction,
                               ccl_comm* comm,
                               ccl_stream* global_stream,
                               const reduce_scatter_attr& attr,
                               const vector_class<event>& deps,
                               bool& done);

} // namespace v1
} // namespace ccl

ccl::event reduce_scatter_scaleout_sycl(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t recv_count,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        ccl_comm* comm,
                                        const ccl::vector_class<ccl::event>& deps,
                                        bool original_deps,
                                        sycl_reduce_scatter_tune_attr& tune_attr,
                                        bool& done,
                                        bool is_cpu_buffers = false);

ccl::event reduce_scatter_small(const void* send_buf,
                                void* recv_buf,
                                size_t recv_count,
                                size_t rem_count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm* comm,
                                ccl_stream* global_stream,
                                const ccl::vector_class<ccl::event>& deps);

ccl::event reduce_scatter_large(const void* send_buf,
                                void* recv_buf,
                                size_t recv_count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm* comm,
                                ccl_stream* global_stream,
                                const ccl::vector_class<ccl::event>& deps,
                                sycl_coll_scaleup_attr coll_attr = {});

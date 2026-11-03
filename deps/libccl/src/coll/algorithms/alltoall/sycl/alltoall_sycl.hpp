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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"

namespace ccl {
namespace v1 {

ccl::event alltoall_scaleup_sycl(sycl::queue& q,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 ccl_comm* comm,
                                 const std::shared_ptr<ccl_comm> p2p_comm,
                                 bool is_numa_comm,
                                 int numa_split,
                                 ccl_stream* global_stream,
                                 const vector_class<event>& deps,
                                 bool& done);

ccl::event alltoall_sycl(sycl::queue q,
                         const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         ccl_comm* comm,
                         ccl_stream* op_stream,
                         const alltoall_attr& attr,
                         const vector_class<event>& deps,
                         bool& done);
} // namespace v1
} // namespace ccl

// ring with LL protocols
ccl::event alltoall_ll(const void* send_buf,
                       void* recv_buf,
                       size_t send_count,
                       const ccl::vector_class<size_t>& offsets,
                       ccl::datatype dtype,
                       ccl_comm* comm,
                       const std::shared_ptr<ccl_comm> p2p_comm,
                       ccl_stream* global_stream,
                       const ccl::vector_class<ccl::event>& deps,
                       bool& done);

ccl::event alltoall_large(const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          std::vector<size_t>& offsets,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          const std::shared_ptr<ccl_comm> p2p_comm,
                          ccl_stream* global_stream,
                          const ccl::vector_class<ccl::event>& deps);

ccl::event alltoall_scaleout_sycl(sycl::queue& q,
                                  const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  ccl_comm* comm,
                                  ccl_stream* global_stream,
                                  const ccl::vector_class<ccl::event>& deps,
                                  bool original_deps,
                                  sycl_scaleout_tune_attr tune_attr,
                                  bool& done);

// with use_proxy, the scale-out exchange runs on the proxy thread instead of in a
// host_task on the caller's queue
ccl::event alltoall_sycl_numa_pairwise_rdma_oneshot(sycl::queue& q,
                                                    const void* send_buf,
                                                    void* recv_buf,
                                                    size_t count,
                                                    ccl::datatype dtype,
                                                    ccl_comm* comm,
                                                    std::shared_ptr<ccl_comm> p2p_comm,
                                                    std::shared_ptr<ccl_comm> r2r_comm,
                                                    ccl_stream* global_stream,
                                                    const ccl::vector_class<ccl::event>& deps,
                                                    bool batch_mode,
                                                    bool use_proxy,
                                                    bool& done);

// Proxy-thread entry points declaration
struct ccl_proxy_op;
struct ccl_proxy_context;
void proxy_alltoall_pairwise(const ccl_proxy_op& op, ccl_proxy_context& ctx);

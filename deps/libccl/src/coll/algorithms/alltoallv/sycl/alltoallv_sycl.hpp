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

struct alltoallv_ctx {
    size_t max_count{}; // max send/recv count among all ranks
    size_t total_max_count{}; // max total send+recv count among all ranks
    size_t chunk_bytes{};
    bool copy_to_host{};
    bool is_cpu_buffers{};
    bool is_pipeline{};
};

namespace ccl {
namespace v1 {

ccl::event alltoallv_scaleup_sycl(sycl::queue& q,
                                  const void* send_buf,
                                  const ccl::vector_class<size_t>& send_counts,
                                  const ccl::vector_class<size_t>& send_displs,
                                  void* recv_buf,
                                  const ccl::vector_class<size_t>& recv_counts,
                                  const ccl::vector_class<size_t>& recv_displs,
                                  ccl::datatype dtype,
                                  ccl_comm* comm,
                                  std::shared_ptr<ccl_comm> p2p_comm,
                                  bool is_numa_comm,
                                  int numa_split,
                                  ccl_stream* global_stream,
                                  const vector_class<event>& deps,
                                  const alltoallv_ctx& options,
                                  bool& done);

ccl::event alltoallv_sycl(sycl::queue q,
                          const void* send_buf,
                          const ccl::vector_class<size_t>& send_counts,
                          void* recv_buf,
                          const ccl::vector_class<size_t>& recv_counts,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          ccl_stream* op_stream,
                          const alltoallv_attr& attr,
                          const vector_class<event>& deps,
                          bool& done);
} // namespace v1
} // namespace ccl

// ring with LL protocols
ccl::event alltoallv_ll(const void* send_buf,
                        const ccl::vector_class<size_t>& send_counts,
                        const ccl::vector_class<size_t>& send_offsets,
                        void* recv_buf,
                        const ccl::vector_class<size_t>& recv_counts,
                        const ccl::vector_class<size_t>& recv_offsets,
                        ccl::datatype dtype,
                        ccl_comm* comm,
                        const std::shared_ptr<ccl_comm> p2p_comm,
                        ccl_stream* global_stream,
                        const ccl::vector_class<ccl::event>& deps,
                        const size_t max_count,
                        bool& done);

ccl::event alltoallv_large(const void* send_buf,
                           const std::vector<size_t>& send_counts,
                           const std::vector<size_t>& send_offsets,
                           void* recv_buf,
                           const std::vector<size_t>& recv_counts,
                           const std::vector<size_t>& recv_offsets,
                           ccl::datatype dtype,
                           ccl_comm* comm,
                           ccl_stream* global_stream,
                           const ccl::vector_class<ccl::event>& deps,
                           const alltoallv_ctx& options);

ccl::event alltoallv_scaleout_sycl(sycl::queue& q,
                                   const void* send_buf,
                                   const ccl::vector_class<size_t>& send_counts,
                                   const ccl::vector_class<size_t>& send_displs,
                                   void* recv_buf,
                                   const ccl::vector_class<size_t>& recv_counts,
                                   const ccl::vector_class<size_t>& recv_displs,
                                   ccl::datatype dtype,
                                   ccl_comm* comm,
                                   ccl_stream* global_stream,
                                   const ccl::vector_class<ccl::event>& deps,
                                   bool original_deps,
                                   sycl_scaleout_tune_attr tune_attr,
                                   const alltoallv_ctx& options,
                                   bool& done);

sycl::event alltoallv_scaleout_sycl_simple_nonblocking(sycl::queue& q,
                                                       const void* send_buf,
                                                       const ccl::vector_class<size_t>& send_counts,
                                                       void* recv_buf,
                                                       const ccl::vector_class<size_t>& recv_counts,
                                                       ccl::datatype dtype,
                                                       ccl_comm* comm,
                                                       const ccl::vector_class<ccl::event>& deps,
                                                       bool original_deps,
                                                       const alltoallv_ctx& options,
                                                       bool& done);

sycl::event alltoallv_scaleout_sycl_simple_blocking(sycl::queue& q,
                                                    const void* send_buf,
                                                    const ccl::vector_class<size_t>& send_counts,
                                                    void* recv_buf,
                                                    const ccl::vector_class<size_t>& recv_counts,
                                                    ccl::datatype dtype,
                                                    ccl_comm* comm,
                                                    const ccl::vector_class<ccl::event>& deps,
                                                    bool original_deps,
                                                    const alltoallv_ctx& options,
                                                    bool& done);

inline sycl::event alltoallv_scaleout_sycl_direct(sycl::queue& q,
                                                  const void* send_buf,
                                                  const ccl::vector_class<size_t>& send_counts,
                                                  void* recv_buf,
                                                  const ccl::vector_class<size_t>& recv_counts,
                                                  ccl::datatype dtype,
                                                  ccl_comm* comm,
                                                  const ccl::vector_class<ccl::event>& deps,
                                                  bool original_deps,
                                                  const alltoallv_ctx& options,
                                                  bool& done) {
    if (ccl::global_data::env().enable_op_sync) {
        return alltoallv_scaleout_sycl_simple_blocking(q,
                                                       send_buf,
                                                       send_counts,
                                                       recv_buf,
                                                       recv_counts,
                                                       dtype,
                                                       comm,
                                                       deps,
                                                       original_deps,
                                                       options,
                                                       done);
    }
    else {
        return alltoallv_scaleout_sycl_simple_nonblocking(q,
                                                          send_buf,
                                                          send_counts,
                                                          recv_buf,
                                                          recv_counts,
                                                          dtype,
                                                          comm,
                                                          deps,
                                                          original_deps,
                                                          options,
                                                          done);
    };
}

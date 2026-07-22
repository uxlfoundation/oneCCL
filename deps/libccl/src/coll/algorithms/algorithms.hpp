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

#include "sched/sched.hpp"
#include "internal_types.hpp"

#include <map>
#include <type_traits>

#define CCL_UNDEFINED_ALGO_ID (-1)

// allgather(v)
ccl::status ccl_coll_build_direct_allgather(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm);
ccl::status ccl_coll_build_naive_allgather(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t count,
                                           const ccl_datatype& dtype,
                                           ccl_comm* comm);
ccl::status ccl_coll_build_direct_allgatherv(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             size_t send_count,
                                             ccl_buffer recv_buf,
                                             const size_t* recv_counts,
                                             const ccl_datatype& dtype,
                                             ccl_comm* comm);
ccl::status ccl_coll_build_naive_allgatherv(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            size_t send_count,
                                            ccl_buffer recv_buf,
                                            const size_t* recv_counts,
                                            const std::vector<ccl_buffer>& recv_device_bufs,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm,
                                            bool is_scaleout = false);
ccl::status ccl_coll_build_ring_allgatherv(ccl_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           ccl_buffer send_buf,
                                           size_t send_count,
                                           ccl_buffer recv_buf,
                                           const size_t* recv_counts,
                                           const std::vector<ccl_buffer>& recv_device_bufs,
                                           const ccl_datatype& dtype,
                                           ccl_comm* comm,
                                           bool is_scaleout = false);
ccl::status ccl_coll_build_flat_allgatherv(ccl_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param);
ccl::status ccl_coll_build_multi_bcast_allgatherv(ccl_sched* main_sched,
                                                  std::vector<ccl_sched*>& scheds,
                                                  const ccl_coll_param& coll_param,
                                                  size_t data_partition_count);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_allgatherv(ccl_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

// allreduce
ccl::status ccl_coll_build_direct_allreduce(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl::reduction reduction,
                                            ccl_comm* comm);
ccl::status ccl_coll_build_rabenseifner_allreduce(ccl_sched* sched,
                                                  ccl_buffer send_buf,
                                                  ccl_buffer recv_buf,
                                                  size_t count,
                                                  const ccl_datatype& dtype,
                                                  ccl::reduction reduction,
                                                  ccl_comm* comm);
ccl::status ccl_coll_build_nreduce_allreduce(ccl_sched* sched,
                                             ccl_buffer send_buf,
                                             ccl_buffer recv_buf,
                                             size_t count,
                                             const ccl_datatype& dtype,
                                             ccl::reduction reduction,
                                             ccl_comm* comm);
ccl::status ccl_coll_build_ring_allreduce(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const std::vector<ccl_buffer>& recv_device_bufs,
                                          const ccl_datatype& dtype,
                                          ccl::reduction reduction,
                                          ccl_comm* comm);
ccl::status ccl_coll_build_ring_rma_allreduce(ccl_sched* sched,
                                              ccl_buffer send_buf,
                                              ccl_buffer recv_buf,
                                              size_t count,
                                              const ccl_datatype& dtype,
                                              ccl::reduction reduction,
                                              ccl_comm* comm);
ccl::status ccl_coll_build_recursive_doubling_allreduce(ccl_sched* sched,
                                                        ccl_buffer send_buf,
                                                        ccl_buffer recv_buf,
                                                        size_t count,
                                                        const ccl_datatype& dtype,
                                                        ccl::reduction reduction,
                                                        ccl_comm* comm);
ccl::status ccl_coll_build_2d_allreduce(ccl_sched* sched,
                                        ccl_buffer send_buf,
                                        ccl_buffer recv_buf,
                                        size_t count,
                                        const ccl_datatype& dtype,
                                        ccl::reduction reduction,
                                        ccl_comm* comm);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_allreduce(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction reduction,
                                          ccl_comm* comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

// alltoall(v)
ccl::status ccl_coll_build_direct_alltoall(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t count,
                                           const ccl_datatype& dtype,
                                           ccl_comm* comm);

ccl::status ccl_coll_build_direct_alltoallv(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            const size_t* send_counts,
                                            ccl_buffer recv_buf,
                                            const size_t* recv_counts,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm);
ccl::status ccl_coll_build_naive_alltoallv(ccl_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param);
ccl::status ccl_coll_build_scatter_alltoallv(ccl_sched* main_sched,
                                             std::vector<ccl_sched*>& scheds,
                                             const ccl_coll_param& coll_param);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_alltoallv(ccl_sched* main_sched,
                                          std::vector<ccl_sched*>& scheds,
                                          const ccl_coll_param& coll_param);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

// barrier
ccl::status ccl_coll_build_direct_barrier(ccl_sched* sched, ccl_comm* comm);
ccl::status ccl_coll_build_dissemination_barrier(ccl_sched* sched, ccl_comm* comm);

// bcast
ccl::status ccl_coll_build_direct_bcast(ccl_sched* sched,
                                        ccl_buffer buf,
                                        size_t count,
                                        const ccl_datatype& dtype,
                                        int root,
                                        ccl_comm* comm);
ccl::status ccl_coll_build_scatter_ring_allgather_bcast(ccl_sched* sched,
                                                        ccl_buffer buf,
                                                        size_t count,
                                                        const ccl_datatype& dtype,
                                                        int root,
                                                        ccl_comm* comm);
ccl::status ccl_coll_build_naive_bcast(ccl_sched* sched,
                                       ccl_buffer buf,
                                       size_t count,
                                       const ccl_datatype& dtype,
                                       int root,
                                       ccl_comm* comm);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_bcast(ccl_sched* sched,
                                      ccl_buffer buf,
                                      size_t count,
                                      const ccl_datatype& dtype,
                                      int root,
                                      ccl_comm* comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

// broadcast
ccl::status ccl_coll_build_direct_broadcast(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            int root,
                                            ccl_comm* comm);
ccl::status ccl_coll_build_scatter_ring_allgather_broadcast(ccl_sched* sched,
                                                            ccl_buffer send_buf,
                                                            ccl_buffer recv_buf,
                                                            size_t count,
                                                            const ccl_datatype& dtype,
                                                            int root,
                                                            ccl_comm* comm);
ccl::status ccl_coll_build_naive_broadcast(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t count,
                                           const ccl_datatype& dtype,
                                           int root,
                                           ccl_comm* comm);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_broadcast(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          int root,
                                          ccl_comm* comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

// recv
ccl::status ccl_coll_build_direct_recv(ccl_sched* sched,
                                       ccl_buffer buf,
                                       size_t count,
                                       const ccl_datatype& dtype,
                                       int peer_rank,
                                       ccl_comm* comm);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_recv(ccl_sched* sched,
                                     ccl_buffer buf,
                                     size_t count,
                                     const ccl_datatype& dtype,
                                     int peer_rank,
                                     ccl_comm* comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

// reduce
ccl::status ccl_coll_build_direct_reduce(ccl_sched* sched,
                                         ccl_buffer send_buf,
                                         ccl_buffer recv_buf,
                                         size_t count,
                                         const ccl_datatype& dtype,
                                         ccl::reduction reduction,
                                         int root,
                                         ccl_comm* comm);
ccl::status ccl_coll_build_rabenseifner_reduce(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction reduction,
                                               int root,
                                               ccl_comm* comm);
ccl::status ccl_coll_build_ring_reduce(ccl_sched* sched,
                                       ccl_buffer send_buf,
                                       ccl_buffer recv_buf,
                                       size_t count,
                                       const ccl_datatype& dtype,
                                       ccl::reduction reduction,
                                       int root,
                                       ccl_comm* comm);
ccl::status ccl_coll_build_binomial_reduce(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t count,
                                           const ccl_datatype& dtype,
                                           ccl::reduction reduction,
                                           int root,
                                           ccl_comm* comm);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_reduce_fill(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl::reduction reduction,
                                            int root,
                                            ccl_comm* comm);
ccl::status ccl_coll_build_topo_reduce(ccl_sched* sched,
                                       ccl_buffer send_buf,
                                       ccl_buffer recv_buf,
                                       size_t count,
                                       const ccl_datatype& dtype,
                                       ccl::reduction reduction,
                                       int root,
                                       ccl_comm* comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

// reduce_scatter
ccl::status ccl_coll_build_direct_reduce_scatter(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 ccl_buffer recv_buf,
                                                 size_t send_count,
                                                 const ccl_datatype& dtype,
                                                 ccl::reduction reduction,
                                                 ccl_comm* comm);
ccl::status ccl_coll_build_naive_reduce_scatter(ccl_sched* sched,
                                                ccl_buffer send_buf,
                                                ccl_buffer recv_buf,
                                                size_t send_count,
                                                const ccl_datatype& dtype,
                                                ccl::reduction reduction,
                                                ccl_comm* comm);
ccl::status ccl_coll_build_reduce_scatter_block(ccl_sched* sched,
                                                ccl_buffer send_buf,
                                                ccl_buffer recv_buf,
                                                size_t send_count,
                                                const ccl_datatype& dtype,
                                                ccl::reduction reduction,
                                                ccl_comm* comm);
ccl::status ccl_coll_build_ring_reduce_scatter(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t recv_count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction reduction,
                                               ccl_comm* comm);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_reduce_scatter_fill(ccl_sched* sched,
                                                    ccl_buffer send_buf,
                                                    ccl_buffer recv_buf,
                                                    size_t send_count,
                                                    size_t chunk_count,
                                                    bool inplace,
                                                    const ccl_datatype& dtype,
                                                    ccl::reduction reduction,
                                                    ccl_comm* comm);
ccl::status ccl_coll_build_topo_reduce_scatter(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t send_count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction reduction,
                                               ccl_comm* comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

// scatter
ccl::status ccl_coll_build_naive_scatter(ccl_sched* sched,
                                         ccl_buffer send_buf,
                                         ccl_buffer recv_buf,
                                         size_t count,
                                         const ccl_datatype& dtype,
                                         int root,
                                         ccl_comm* comm);

// send
ccl::status ccl_coll_build_direct_send(ccl_sched* sched,
                                       ccl_buffer buf,
                                       size_t count,
                                       const ccl_datatype& dtype,
                                       int peer_rank,
                                       ccl_comm* comm);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_send(ccl_sched* sched,
                                     ccl_buffer buf,
                                     size_t count,
                                     const ccl_datatype& dtype,
                                     int peer_rank,
                                     ccl_comm* comm);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

class ccl_double_tree;
ccl::status ccl_coll_build_double_tree_op(ccl_sched* sched,
                                          ccl_coll_type coll_type,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction reduction,
                                          const ccl_double_tree& dtree,
                                          ccl_comm* comm);

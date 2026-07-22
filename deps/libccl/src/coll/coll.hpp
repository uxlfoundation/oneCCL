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

#include "coll/algorithms/algorithm_utils.hpp"
#include "coll/coll_param.hpp"
#include "coll/group/group.hpp"

#include "comm/comm.hpp"
#include "common/datatype/datatype.hpp"
#include "common/stream/stream.hpp"
#include "common/utils/buffer.hpp"

#include "coll/attr/ccl_common_op_attrs.hpp"

#include "internal_types.hpp"

class ccl_sched;
class ccl_request;

ccl::status ccl_coll_build_allgather(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     ccl_buffer recv_buf,
                                     size_t count,
                                     const ccl_datatype& dtype,
                                     ccl_comm* comm,
                                     bool is_scaleout);

ccl::status ccl_coll_build_allgatherv(ccl_sched* sched,
                                      ccl_buffer send_buf,
                                      size_t send_count,
                                      ccl_buffer recv_buf,
                                      const size_t* recv_counts,
                                      const std::vector<ccl_buffer>& recv_device_bufs,
                                      const ccl_datatype& dtype,
                                      ccl_comm* comm,
                                      bool is_scaleout,
                                      bool is_hmem_enabled = false);

// TODO: pack this arguments in ccl_coll_build parameters structure
ccl::status ccl_coll_build_allreduce(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     ccl_buffer recv_buf,
                                     size_t count,
                                     const std::vector<ccl_buffer>& recv_device_bufs,
                                     const ccl_datatype& dtype,
                                     ccl::reduction reduction,
                                     ccl_comm* comm,
                                     bool is_scaleout);

ccl::status ccl_coll_build_alltoall(ccl_sched* sched,
                                    ccl_buffer send_buf,
                                    ccl_buffer recv_buf,
                                    size_t count,
                                    const ccl_datatype& dtype,
                                    ccl_comm* comm,
                                    bool is_scaleout);

ccl::status ccl_coll_build_alltoallv(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     const size_t* send_counts,
                                     ccl_buffer recv_buf,
                                     const size_t* recv_counts,
                                     const ccl_datatype& dtype,
                                     ccl_comm* comm,
                                     bool is_scaleout);

ccl::status ccl_coll_build_barrier(ccl_sched* sched, ccl_comm* comm);

ccl::status ccl_coll_build_bcast(ccl_sched* sched,
                                 ccl_buffer buf,
                                 size_t count,
                                 const ccl_datatype& dtype,
                                 int root,
                                 ccl_comm* comm);
ccl::status ccl_coll_build_broadcast(ccl_sched* sched,
                                     ccl_buffer send_buf,
                                     ccl_buffer recv_buf,
                                     size_t count,
                                     const ccl_datatype& dtype,
                                     int root,
                                     ccl_comm* comm);

ccl::status ccl_coll_build_reduce(ccl_sched* sched,
                                  ccl_buffer send_buf,
                                  ccl_buffer recv_buf,
                                  size_t count,
                                  const ccl_datatype& dtype,
                                  ccl::reduction reduction,
                                  int root,
                                  ccl_comm* comm,
                                  bool is_scaleout);

ccl::status ccl_coll_build_reduce_scatter(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction reduction,
                                          ccl_comm* comm,
                                          bool is_scaleout,
                                          bool from_allreduce = false);

ccl::status ccl_coll_build_scatter(ccl_sched* sched,
                                   ccl_buffer send_buf,
                                   ccl_buffer recv_buf,
                                   size_t count,
                                   const ccl_datatype& dtype,
                                   int root,
                                   ccl_comm* comm,
                                   bool is_scaleout);

ccl::status ccl_coll_build_recv(ccl_sched* sched,
                                ccl_buffer buf,
                                size_t count,
                                const ccl_datatype& dtype,
                                int peer,
                                ccl_comm* comm);

ccl::status ccl_coll_build_send(ccl_sched* sched,
                                ccl_buffer buf,
                                size_t count,
                                const ccl_datatype& dtype,
                                int peer,
                                ccl_comm* comm);

ccl::event ccl_allgather(const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         const ccl_coll_attr& attr,
                         ccl_comm* comm,
                         const ccl_stream* stream,
                         const std::vector<ccl::event>& deps);

ccl_request* ccl_allgather_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps);

ccl::event ccl_alltoall(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        ccl::datatype dtype,
                        const ccl_coll_attr& attr,
                        ccl_comm* comm,
                        const ccl_stream* stream,
                        const std::vector<ccl::event>& deps);

ccl::event ccl_allgatherv(const void* send_buf,
                          size_t send_count,
                          void* recv_buf,
                          const ccl::vector_class<size_t>& recv_counts,
                          ccl::datatype dtype,
                          const ccl_coll_attr& attr,
                          ccl_comm* comm,
                          const ccl_stream* stream,
                          const std::vector<ccl::event>& deps);

ccl_request* ccl_allgatherv_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const size_t* recv_counts,
                                 ccl::datatype dtype,
                                 const ccl_coll_attr& attr,
                                 ccl_comm* comm,
                                 const ccl_stream* stream,
                                 const std::vector<ccl::event>& deps);

ccl::event ccl_allreduce(const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         ccl::reduction reduction,
                         const ccl_coll_attr& attr,
                         ccl_comm* comm,
                         const ccl_stream* stream,
                         const std::vector<ccl::event>& deps);

ccl_request* ccl_allreduce_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps);

ccl::event ccl_alltoall(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        ccl::datatype dtype,
                        const ccl_coll_attr& attr,
                        ccl_comm* comm,
                        const ccl_stream* stream,
                        const std::vector<ccl::event>& deps);

ccl_request* ccl_alltoall_impl(const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               ccl::datatype dtype,
                               const ccl_coll_attr& attr,
                               ccl_comm* comm,
                               const ccl_stream* stream,
                               const std::vector<ccl::event>& deps);

ccl::event ccl_alltoallv(const void* send_buf,
                         const size_t* send_counts,
                         void* recv_buf,
                         const size_t* recv_counts,
                         ccl::datatype dtype,
                         const ccl_coll_attr& attr,
                         ccl_comm* comm,
                         const ccl_stream* stream,
                         const std::vector<ccl::event>& deps);

ccl_request* ccl_alltoallv_impl(const void* send_buf,
                                const size_t* send_counts,
                                void* recv_buf,
                                const size_t* recv_counts,
                                ccl::datatype dtype,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps);

ccl::event ccl_barrier(ccl_comm* comm,
                       const ccl_stream* stream,
                       const std::vector<ccl::event>& deps);

ccl_request* ccl_barrier_impl(ccl_comm* comm,
                              const ccl_stream* stream,
                              const std::vector<ccl::event>& deps);

ccl::event ccl_broadcast(void* buf,
                         size_t count,
                         ccl::datatype dtype,
                         int root,
                         const ccl_coll_attr& attr,
                         ccl_comm* comm,
                         const ccl_stream* stream,
                         const std::vector<ccl::event>& deps);

ccl_request* ccl_broadcast_impl(void* buf,
                                size_t count,
                                ccl::datatype dtype,
                                int root,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps);

ccl::event ccl_broadcast(void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         int root,
                         const ccl_coll_attr& attr,
                         ccl_comm* comm,
                         const ccl_stream* stream,
                         const std::vector<ccl::event>& deps);

ccl_request* ccl_broadcast_impl(void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                int root,
                                const ccl_coll_attr& attr,
                                ccl_comm* comm,
                                const ccl_stream* stream,
                                const std::vector<ccl::event>& deps);

ccl::event ccl_reduce(const void* send_buf,
                      void* recv_buf,
                      size_t count,
                      ccl::datatype dtype,
                      ccl::reduction reduction,
                      int root,
                      const ccl_coll_attr& attr,
                      ccl_comm* comm,
                      const ccl_stream* stream,
                      const std::vector<ccl::event>& deps);

ccl_request* ccl_reduce_impl(const void* send_buf,
                             void* recv_buf,
                             size_t count,
                             ccl::datatype dtype,
                             ccl::reduction reduction,
                             int root,
                             const ccl_coll_attr& attr,
                             ccl_comm* comm,
                             const ccl_stream* stream,
                             const std::vector<ccl::event>& deps);

ccl::event ccl_reduce_scatter(const void* send_buf,
                              void* recv_buf,
                              size_t recv_count,
                              ccl::datatype dtype,
                              ccl::reduction reduction,
                              const ccl_coll_attr& attr,
                              ccl_comm* comm,
                              const ccl_stream* stream,
                              const std::vector<ccl::event>& deps);

ccl_request* ccl_reduce_scatter_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t recv_count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     const ccl_coll_attr& attr,
                                     ccl_comm* comm,
                                     const ccl_stream* stream,
                                     const std::vector<ccl::event>& deps);

ccl::event ccl_scatter(const void* send_buf,
                       void* recv_buf,
                       size_t count,
                       ccl::datatype dtype,
                       int root,
                       const ccl_coll_attr& attr,
                       ccl_comm* comm,
                       const ccl_stream* stream,
                       const std::vector<ccl::event>& deps);

ccl_request* ccl_scatter_impl(const void* send_buf,
                              void* recv_buf,
                              size_t count,
                              ccl::datatype dtype,
                              int root,
                              const ccl_coll_attr& attr,
                              ccl_comm* comm,
                              const ccl_stream* stream,
                              const std::vector<ccl::event>& deps);

// wrapper for ccl_recv_impl
ccl::event ccl_recv(void* recv_buf,
                    size_t count,
                    ccl::datatype dtype,
                    int peer,
                    const ccl_coll_attr& attr,
                    ccl_comm* comm,
                    const ccl_stream* stream,
                    const std::vector<ccl::event>& deps);

ccl_request* ccl_recv_impl(void* recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           int peer,
                           const ccl_coll_attr& attr,
                           ccl_comm* comm,
                           const ccl_stream* stream,
                           const std::vector<ccl::event>& deps);

// wrapper for ccl_send_impl
ccl::event ccl_send(const void* send_buf,
                    size_t send_count,
                    ccl::datatype dtype,
                    int peer_rank,
                    const ccl_coll_attr& attr,
                    ccl_comm* comm,
                    const ccl_stream* stream,
                    const std::vector<ccl::event>& deps);

ccl_request* ccl_send_impl(const void* send_buf,
                           size_t count,
                           ccl::datatype dtype,
                           int peer,
                           const ccl_coll_attr& attr,
                           ccl_comm* comm,
                           const ccl_stream* stream,
                           const std::vector<ccl::event>& deps);
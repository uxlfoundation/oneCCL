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

namespace ccl {

namespace v1 {

/******************** COMMUNICATOR ********************/

/**
 * Generating API types for collective operations
 * of the communicator class (communicator)
 */
#define API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(BufferType) \
\
    template event CCL_API allgather(const BufferType* send_buf, \
                                     BufferType* recv_buf, \
                                     size_t count, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const allgather_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API allgather(const BufferType* send_buf, \
                                     BufferType* recv_buf, \
                                     size_t count, \
                                     const communicator& comm, \
                                     const allgather_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API allgather(const BufferType* send_buf, \
                                     vector_class<BufferType*>& recv_buf, \
                                     size_t count, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const allgather_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API allgather(const BufferType* send_buf, \
                                     vector_class<BufferType*>& recv_buf, \
                                     size_t count, \
                                     const communicator& comm, \
                                     const allgather_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API allgatherv(const BufferType* send_buf, \
                                      size_t send_count, \
                                      BufferType* recv_buf, \
                                      const vector_class<size_t>& recv_counts, \
                                      const communicator& comm, \
                                      const stream& op_stream, \
                                      const allgatherv_attr& attr, \
                                      const vector_class<event>& deps); \
\
    template event CCL_API allgatherv(const BufferType* send_buf, \
                                      size_t send_count, \
                                      BufferType* recv_buf, \
                                      const vector_class<size_t>& recv_counts, \
                                      const communicator& comm, \
                                      const allgatherv_attr& attr, \
                                      const vector_class<event>& deps); \
\
    template event CCL_API allgatherv(const BufferType* send_buf, \
                                      size_t send_count, \
                                      vector_class<BufferType*>& recv_bufs, \
                                      const vector_class<size_t>& recv_counts, \
                                      const communicator& comm, \
                                      const stream& op_stream, \
                                      const allgatherv_attr& attr, \
                                      const vector_class<event>& deps); \
\
    template event CCL_API allgatherv(const BufferType* send_buf, \
                                      size_t send_count, \
                                      vector_class<BufferType*>& recv_bufs, \
                                      const vector_class<size_t>& recv_counts, \
                                      const communicator& comm, \
                                      const allgatherv_attr& attr, \
                                      const vector_class<event>& deps); \
\
    template event CCL_API allreduce(const BufferType* send_buf, \
                                     BufferType* recv_buf, \
                                     size_t count, \
                                     reduction reduction, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const allreduce_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API allreduce(const BufferType* send_buf, \
                                     BufferType* recv_buf, \
                                     size_t count, \
                                     reduction reduction, \
                                     const communicator& comm, \
                                     const allreduce_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API alltoall(const BufferType* send_buf, \
                                    BufferType* recv_buf, \
                                    size_t count, \
                                    const communicator& comm, \
                                    const stream& op_stream, \
                                    const alltoall_attr& attr, \
                                    const vector_class<event>& deps); \
\
    template event CCL_API alltoall(const BufferType* send_buf, \
                                    BufferType* recv_buf, \
                                    size_t count, \
                                    const communicator& comm, \
                                    const alltoall_attr& attr, \
                                    const vector_class<event>& deps); \
\
    template event CCL_API alltoall(const vector_class<BufferType*>& send_buf, \
                                    const vector_class<BufferType*>& recv_buf, \
                                    size_t count, \
                                    const communicator& comm, \
                                    const stream& op_stream, \
                                    const alltoall_attr& attr, \
                                    const vector_class<event>& deps); \
\
    template event CCL_API alltoall(const vector_class<BufferType*>& send_buf, \
                                    const vector_class<BufferType*>& recv_buf, \
                                    size_t count, \
                                    const communicator& comm, \
                                    const alltoall_attr& attr, \
                                    const vector_class<event>& deps); \
\
    template event CCL_API alltoallv(const BufferType* send_buf, \
                                     const vector_class<size_t>& send_counts, \
                                     BufferType* recv_buf, \
                                     const vector_class<size_t>& recv_counts, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const alltoallv_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API alltoallv(const BufferType* send_buf, \
                                     const vector_class<size_t>& send_counts, \
                                     BufferType* recv_buf, \
                                     const vector_class<size_t>& recv_counts, \
                                     const communicator& comm, \
                                     const alltoallv_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API alltoallv(const vector_class<BufferType*>& send_bufs, \
                                     const vector_class<size_t>& send_counts, \
                                     const vector_class<BufferType*>& recv_bufs, \
                                     const vector_class<size_t>& recv_counts, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const alltoallv_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API alltoallv(const vector_class<BufferType*>& send_bufs, \
                                     const vector_class<size_t>& send_counts, \
                                     const vector_class<BufferType*>& recv_bufs, \
                                     const vector_class<size_t>& recv_counts, \
                                     const communicator& comm, \
                                     const alltoallv_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API broadcast(BufferType* buf, \
                                     size_t count, \
                                     int root, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const broadcast_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API broadcast(BufferType* buf, \
                                     size_t count, \
                                     int root, \
                                     const communicator& comm, \
                                     const broadcast_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API broadcast(BufferType* send_buf, \
                                     BufferType* recv_buf, \
                                     size_t count, \
                                     int root, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const broadcast_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API broadcast(BufferType* send_buf, \
                                     BufferType* recv_buf, \
                                     size_t count, \
                                     int root, \
                                     const communicator& comm, \
                                     const broadcast_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API reduce(const BufferType* send_buf, \
                                  BufferType* recv_buf, \
                                  size_t count, \
                                  reduction reduction, \
                                  int root, \
                                  const communicator& comm, \
                                  const stream& op_stream, \
                                  const reduce_attr& attr, \
                                  const vector_class<event>& deps); \
\
    template event CCL_API reduce(const BufferType* send_buf, \
                                  BufferType* recv_buf, \
                                  size_t count, \
                                  reduction reduction, \
                                  int root, \
                                  const communicator& comm, \
                                  const reduce_attr& attr, \
                                  const vector_class<event>& deps); \
\
    template event CCL_API reduce_scatter(const BufferType* send_buf, \
                                          BufferType* recv_buf, \
                                          size_t recv_count, \
                                          reduction reduction, \
                                          const communicator& comm, \
                                          const stream& op_stream, \
                                          const reduce_scatter_attr& attr, \
                                          const vector_class<event>& deps); \
\
    template event CCL_API reduce_scatter(const BufferType* send_buf, \
                                          BufferType* recv_buf, \
                                          size_t recv_count, \
                                          reduction reduction, \
                                          const communicator& comm, \
                                          const reduce_scatter_attr& attr, \
                                          const vector_class<event>& deps); \
\
    template event CCL_API scatter(const BufferType* send_buf, \
                                   BufferType* recv_buf, \
                                   size_t count, \
                                   int root, \
                                   const communicator& comm, \
                                   const stream& op_stream, \
                                   const scatter_attr& attr, \
                                   const vector_class<event>& deps); \
\
    template event CCL_API scatter(const BufferType* send_buf, \
                                   BufferType* recv_buf, \
                                   size_t count, \
                                   int root, \
                                   const communicator& comm, \
                                   const scatter_attr& attr, \
                                   const vector_class<event>& deps); \
\
    template event CCL_API recv(BufferType* recv_buf, \
                                size_t recv_count, \
                                int peer, \
                                const communicator& comm, \
                                const stream& op_stream, \
                                const pt2pt_attr& attr, \
                                const vector_class<event>& deps); \
\
    template event CCL_API recv(BufferType* recv_buf, \
                                size_t recv_count, \
                                int peer, \
                                const communicator& comm, \
                                const pt2pt_attr& attr, \
                                const vector_class<event>& deps); \
\
    template event CCL_API send(BufferType* send_buf, \
                                size_t send_count, \
                                int peer, \
                                const communicator& comm, \
                                const stream& op_stream, \
                                const pt2pt_attr& attr, \
                                const vector_class<event>& deps); \
\
    template event CCL_API send(BufferType* send_buf, \
                                size_t send_count, \
                                int peer, \
                                const communicator& comm, \
                                const pt2pt_attr& attr, \
                                const vector_class<event>& deps);

#define API_COMM_OP_REF_EXPLICIT_INSTANTIATION(BufferObjectType) \
\
    template event CCL_API allgather(const BufferObjectType& send_buf, \
                                     BufferObjectType& recv_buf, \
                                     size_t count, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const allgather_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API allgather(const BufferObjectType& send_buf, \
                                     BufferObjectType& recv_buf, \
                                     size_t count, \
                                     const communicator& comm, \
                                     const allgather_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API allgather( \
        const BufferObjectType& send_buf, \
        vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf, \
        size_t count, \
        const communicator& comm, \
        const stream& op_stream, \
        const allgather_attr& attr, \
        const vector_class<event>& deps); \
\
    template event CCL_API allgather( \
        const BufferObjectType& send_buf, \
        vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf, \
        size_t count, \
        const communicator& comm, \
        const allgather_attr& attr, \
        const vector_class<event>& deps); \
\
    template event CCL_API allgatherv(const BufferObjectType& send_buf, \
                                      size_t send_count, \
                                      BufferObjectType& recv_buf, \
                                      const vector_class<size_t>& recv_counts, \
                                      const communicator& comm, \
                                      const stream& op_stream, \
                                      const allgatherv_attr& attr, \
                                      const vector_class<event>& deps); \
\
    template event CCL_API allgatherv(const BufferObjectType& send_buf, \
                                      size_t send_count, \
                                      BufferObjectType& recv_buf, \
                                      const vector_class<size_t>& recv_counts, \
                                      const communicator& comm, \
                                      const allgatherv_attr& attr, \
                                      const vector_class<event>& deps); \
\
    template event CCL_API allgatherv( \
        const BufferObjectType& send_buf, \
        size_t send_count, \
        vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs, \
        const vector_class<size_t>& recv_counts, \
        const communicator& comm, \
        const stream& op_stream, \
        const allgatherv_attr& attr, \
        const vector_class<event>& deps); \
\
    template event CCL_API allgatherv( \
        const BufferObjectType& send_buf, \
        size_t send_count, \
        vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs, \
        const vector_class<size_t>& recv_counts, \
        const communicator& comm, \
        const allgatherv_attr& attr, \
        const vector_class<event>& deps); \
\
    template event CCL_API allreduce(const BufferObjectType& send_buf, \
                                     BufferObjectType& recv_buf, \
                                     size_t count, \
                                     reduction reduction, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const allreduce_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API allreduce(const BufferObjectType& send_buf, \
                                     BufferObjectType& recv_buf, \
                                     size_t count, \
                                     reduction reduction, \
                                     const communicator& comm, \
                                     const allreduce_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API alltoall(const BufferObjectType& send_buf, \
                                    BufferObjectType& recv_buf, \
                                    size_t count, \
                                    const communicator& comm, \
                                    const stream& op_stream, \
                                    const alltoall_attr& attr, \
                                    const vector_class<event>& deps); \
\
    template event CCL_API alltoall(const BufferObjectType& send_buf, \
                                    BufferObjectType& recv_buf, \
                                    size_t count, \
                                    const communicator& comm, \
                                    const alltoall_attr& attr, \
                                    const vector_class<event>& deps); \
\
    template event CCL_API alltoall( \
        const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf, \
        const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf, \
        size_t count, \
        const communicator& comm, \
        const stream& op_stream, \
        const alltoall_attr& attr, \
        const vector_class<event>& deps); \
\
    template event CCL_API alltoall( \
        const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf, \
        const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf, \
        size_t count, \
        const communicator& comm, \
        const alltoall_attr& attr, \
        const vector_class<event>& deps); \
\
    template event CCL_API alltoallv(const BufferObjectType& send_buf, \
                                     const vector_class<size_t>& send_counts, \
                                     BufferObjectType& recv_buf, \
                                     const vector_class<size_t>& recv_counts, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const alltoallv_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API alltoallv(const BufferObjectType& send_buf, \
                                     const vector_class<size_t>& send_counts, \
                                     BufferObjectType& recv_buf, \
                                     const vector_class<size_t>& recv_counts, \
                                     const communicator& comm, \
                                     const alltoallv_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API alltoallv( \
        const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs, \
        const vector_class<size_t>& send_counts, \
        const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs, \
        const vector_class<size_t>& recv_counts, \
        const communicator& comm, \
        const stream& op_stream, \
        const alltoallv_attr& attr, \
        const vector_class<event>& deps); \
\
    template event CCL_API alltoallv( \
        const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs, \
        const vector_class<size_t>& send_counts, \
        const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs, \
        const vector_class<size_t>& recv_counts, \
        const communicator& comm, \
        const alltoallv_attr& attr, \
        const vector_class<event>& deps); \
\
    template event CCL_API broadcast(BufferObjectType& buf, \
                                     size_t count, \
                                     int root, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const broadcast_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API broadcast(BufferObjectType& buf, \
                                     size_t count, \
                                     int root, \
                                     const communicator& comm, \
                                     const broadcast_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API broadcast(BufferObjectType& send_buf, \
                                     BufferObjectType& recv_buf, \
                                     size_t count, \
                                     int root, \
                                     const communicator& comm, \
                                     const stream& op_stream, \
                                     const broadcast_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API broadcast(BufferObjectType& send_buf, \
                                     BufferObjectType& recv_buf, \
                                     size_t count, \
                                     int root, \
                                     const communicator& comm, \
                                     const broadcast_attr& attr, \
                                     const vector_class<event>& deps); \
\
    template event CCL_API reduce(const BufferObjectType& send_buf, \
                                  BufferObjectType& recv_buf, \
                                  size_t count, \
                                  reduction reduction, \
                                  int root, \
                                  const communicator& comm, \
                                  const stream& op_stream, \
                                  const reduce_attr& attr, \
                                  const vector_class<event>& deps); \
\
    template event CCL_API reduce(const BufferObjectType& send_buf, \
                                  BufferObjectType& recv_buf, \
                                  size_t count, \
                                  reduction reduction, \
                                  int root, \
                                  const communicator& comm, \
                                  const reduce_attr& attr, \
                                  const vector_class<event>& deps); \
\
    template event CCL_API reduce_scatter(const BufferObjectType& send_buf, \
                                          BufferObjectType& recv_buf, \
                                          size_t recv_count, \
                                          reduction reduction, \
                                          const communicator& comm, \
                                          const stream& op_stream, \
                                          const reduce_scatter_attr& attr, \
                                          const vector_class<event>& deps); \
\
    template event CCL_API reduce_scatter(const BufferObjectType& send_buf, \
                                          BufferObjectType& recv_buf, \
                                          size_t recv_count, \
                                          reduction reduction, \
                                          const communicator& comm, \
                                          const reduce_scatter_attr& attr, \
                                          const vector_class<event>& deps); \
\
    template event CCL_API scatter(const BufferObjectType& send_buf, \
                                   BufferObjectType& recv_buf, \
                                   size_t count, \
                                   int root, \
                                   const communicator& comm, \
                                   const stream& op_stream, \
                                   const scatter_attr& attr, \
                                   const vector_class<event>& deps); \
\
    template event CCL_API scatter(const BufferObjectType& send_buf, \
                                   BufferObjectType& recv_buf, \
                                   size_t count, \
                                   int root, \
                                   const communicator& comm, \
                                   const scatter_attr& attr, \
                                   const vector_class<event>& deps); \
\
    template event CCL_API recv(BufferObjectType& recv_buf, \
                                size_t recv_count, \
                                int peer, \
                                const communicator& comm, \
                                const stream& op_stream, \
                                const pt2pt_attr& attr, \
                                const vector_class<event>& deps); \
\
    template event CCL_API recv(BufferObjectType& recv_buf, \
                                size_t recv_count, \
                                int peer, \
                                const communicator& comm, \
                                const pt2pt_attr& attr, \
                                const vector_class<event>& deps); \
\
    template event CCL_API send(BufferObjectType& send_buf, \
                                size_t send_count, \
                                int peer, \
                                const communicator& comm, \
                                const stream& op_stream, \
                                const pt2pt_attr& attr, \
                                const vector_class<event>& deps); \
\
    template event CCL_API send(BufferObjectType& send_buf, \
                                size_t send_count, \
                                int peer, \
                                const communicator& comm, \
                                const pt2pt_attr& attr, \
                                const vector_class<event>& deps);

} // namespace v1

} // namespace ccl

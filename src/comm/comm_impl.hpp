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

#include "comm/comm.hpp"

#include "common/request/request.hpp"
#include "common/event/impls/host_event.hpp"

#include "coll/coll.hpp"
#include "coll/attr/ccl_common_op_attrs.hpp"

/* allgather */
template <class buffer_type>
ccl::event ccl_comm::allgather_impl(const buffer_type* send_buf,
                                    buffer_type* recv_buf,
                                    size_t count,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allgather_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    return ccl_allgather(reinterpret_cast<const void*>(send_buf),
                         reinterpret_cast<void*>(recv_buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

template <class buffer_type>
ccl::event ccl_comm::allgather_impl(const buffer_type* send_buf,
                                    ccl::vector_class<buffer_type*>& recv_buf,
                                    size_t count,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allgather_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    return ccl_allgather(reinterpret_cast<const void*>(send_buf),
                         (void*)(recv_buf.data()),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

template <class buffer_type>
ccl::event ccl_comm::allgather_impl(const buffer_type& send_buf,
                                    buffer_type& recv_buf,
                                    size_t count,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allgather_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_allgather(reinterpret_cast<const void*>(&send_buf),
                         reinterpret_cast<void*>(&recv_buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

template <class buffer_type>
ccl::event ccl_comm::allgather_impl(
    const buffer_type& send_buf,
    ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    size_t count,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgather_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_allgather(reinterpret_cast<const void*>(&send_buf),
                         (void*)(recv_buf.data()),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

/* allgatherv */
template <class buffer_type>
ccl::event ccl_comm::allgatherv_impl(const buffer_type* send_buf,
                                     size_t send_count,
                                     buffer_type* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    return ccl_allgatherv(reinterpret_cast<const void*>(send_buf),
                          send_count,
                          reinterpret_cast<void*>(recv_buf),
                          recv_counts,
                          ccl::native_type_info<buffer_type>::dtype,
                          attr,
                          this,
                          get_stream_ptr(stream),
                          deps);
}

template <class buffer_type>
ccl::event ccl_comm::allgatherv_impl(const buffer_type* send_buf,
                                     size_t send_count,
                                     ccl::vector_class<buffer_type*>& recv_bufs,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    return ccl_allgatherv(reinterpret_cast<const void*>(send_buf),
                          send_count,
                          (void*)(recv_bufs.data()),
                          recv_counts,
                          ccl::native_type_info<buffer_type>::dtype,
                          internal_attr,
                          this,
                          get_stream_ptr(stream),
                          deps);
}

template <class buffer_type>
ccl::event ccl_comm::allgatherv_impl(const buffer_type& send_buf,
                                     size_t send_count,
                                     buffer_type& recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgatherv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_allgatherv(reinterpret_cast<const void*>(&send_buf),
                          send_count,
                          reinterpret_cast<void*>(&recv_buf),
                          recv_counts,
                          ccl::native_type_info<buffer_type>::dtype,
                          internal_attr,
                          this,
                          get_stream_ptr(stream),
                          deps);
}

template <class buffer_type>
ccl::event ccl_comm::allgatherv_impl(
    const buffer_type& send_buf,
    size_t send_count,
    ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_bufs,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::allgatherv_attr& attr,
    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_allgatherv(reinterpret_cast<const void*>(&send_buf),
                          send_count,
                          (void*)(recv_bufs.data()),
                          recv_counts,
                          ccl::native_type_info<buffer_type>::dtype,
                          internal_attr,
                          this,
                          get_stream_ptr(stream),
                          deps);
}

/* allreduce */
template <class buffer_type>
ccl::event ccl_comm::allreduce_impl(const buffer_type* send_buf,
                                    buffer_type* recv_buf,
                                    size_t count,
                                    ccl::reduction reduction,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allreduce_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    return ccl_allreduce(reinterpret_cast<const void*>(send_buf),
                         reinterpret_cast<void*>(recv_buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         reduction,
                         attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

template <class buffer_type>
ccl::event ccl_comm::allreduce_impl(const buffer_type& send_buf,
                                    buffer_type& recv_buf,
                                    size_t count,
                                    ccl::reduction reduction,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::allreduce_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_allreduce(reinterpret_cast<const void*>(&send_buf),
                         reinterpret_cast<void*>(&recv_buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         reduction,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

/* alltoall */
template <class buffer_type>
ccl::event ccl_comm::alltoall_impl(const buffer_type* send_buf,
                                   buffer_type* recv_buf,
                                   size_t count,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    return ccl_alltoall(reinterpret_cast<const void*>(send_buf),
                        reinterpret_cast<void*>(recv_buf),
                        count,
                        ccl::native_type_info<buffer_type>::dtype,
                        attr,
                        this,
                        get_stream_ptr(stream),
                        deps);
}

template <class buffer_type>
ccl::event ccl_comm::alltoall_impl(const ccl::vector_class<buffer_type*>& send_buf,
                                   const ccl::vector_class<buffer_type*>& recv_buf,
                                   size_t count,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

template <class buffer_type>
ccl::event ccl_comm::alltoall_impl(const buffer_type& send_buf,
                                   buffer_type& recv_buf,
                                   size_t count,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::alltoall_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_alltoall(reinterpret_cast<const void*>(&send_buf),
                        reinterpret_cast<void*>(&recv_buf),
                        count,
                        ccl::native_type_info<buffer_type>::dtype,
                        internal_attr,
                        this,
                        get_stream_ptr(stream),
                        deps);
}

template <class buffer_type>
ccl::event ccl_comm::alltoall_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    size_t count,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoall_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");
    return {};
}

/* alltoallv */
template <class buffer_type>
ccl::event ccl_comm::alltoallv_impl(const buffer_type* send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    buffer_type* recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    return ccl_alltoallv(reinterpret_cast<const void*>(send_buf),
                         send_counts.data(),
                         reinterpret_cast<void*>(recv_buf),
                         recv_counts.data(),
                         ccl::native_type_info<buffer_type>::dtype,
                         attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

template <class buffer_type>
ccl::event ccl_comm::alltoallv_impl(const ccl::vector_class<buffer_type*>& send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    const ccl::vector_class<buffer_type*>& recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& dep) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;

    return ccl_alltoallv((void*)(send_buf.data()),
                         send_counts.data(),
                         (void*)(recv_buf.data()),
                         recv_counts.data(),
                         ccl::native_type_info<buffer_type>::dtype,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         dep);
}

template <class buffer_type>
ccl::event ccl_comm::alltoallv_impl(const buffer_type& send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    buffer_type& recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoallv_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL

    return ccl_alltoallv(reinterpret_cast<const void*>(&send_buf),
                         send_counts.data(),
                         reinterpret_cast<void*>(&recv_buf),
                         recv_counts.data(),
                         ccl::native_type_info<buffer_type>::dtype,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

template <class buffer_type>
ccl::event ccl_comm::alltoallv_impl(
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& send_buf,
    const ccl::vector_class<size_t>& send_counts,
    const ccl::vector_class<ccl::reference_wrapper_class<buffer_type>>& recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const ccl::stream::impl_value_t& stream,
    const ccl::alltoallv_attr& attr,
    const ccl::vector_class<ccl::event>& dep) {
    ccl_coll_attr internal_attr(attr);
    internal_attr.is_vector_buf = 1;
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_alltoallv((void*)(send_buf.data()),
                         send_counts.data(),
                         (void*)(recv_buf.data()),
                         recv_counts.data(),
                         ccl::native_type_info<buffer_type>::dtype,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         dep);
}

/* bcast */
template <class buffer_type>
ccl::event ccl_comm::broadcast_impl(buffer_type* buf,
                                    size_t count,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    return ccl_broadcast(reinterpret_cast<void*>(buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         root,
                         attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

template <class buffer_type>
ccl::event ccl_comm::broadcast_impl(buffer_type& buf,
                                    size_t count,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_broadcast(reinterpret_cast<void*>(&buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         root,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

/* broadcast */
template <class buffer_type>
ccl::event ccl_comm::broadcast_impl(buffer_type* send_buf,
                                    buffer_type* recv_buf,
                                    size_t count,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    return ccl_broadcast(reinterpret_cast<void*>(send_buf),
                         reinterpret_cast<void*>(recv_buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         root,
                         attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

template <class buffer_type>
ccl::event ccl_comm::broadcast_impl(buffer_type& send_buf,
                                    buffer_type& recv_buf,
                                    size_t count,
                                    int root,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::broadcast_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_broadcast(reinterpret_cast<void*>(&send_buf),
                         reinterpret_cast<void*>(&recv_buf),
                         count,
                         ccl::native_type_info<buffer_type>::dtype,
                         root,
                         internal_attr,
                         this,
                         get_stream_ptr(stream),
                         deps);
}

/* reduce */
template <class buffer_type>
ccl::event ccl_comm::reduce_impl(const buffer_type* send_buf,
                                 buffer_type* recv_buf,
                                 size_t count,
                                 ccl::reduction reduction,
                                 int root,
                                 const ccl::stream::impl_value_t& stream,
                                 const ccl::reduce_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    return ccl_reduce(reinterpret_cast<const void*>(send_buf),
                      reinterpret_cast<void*>(recv_buf),
                      count,
                      ccl::native_type_info<buffer_type>::dtype,
                      reduction,
                      root,
                      attr,
                      this,
                      get_stream_ptr(stream),
                      deps);
}

template <class buffer_type>
ccl::event ccl_comm::reduce_impl(const buffer_type& send_buf,
                                 buffer_type& recv_buf,
                                 size_t count,
                                 ccl::reduction reduction,
                                 int root,
                                 const ccl::stream::impl_value_t& stream,
                                 const ccl::reduce_attr& attr,
                                 const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_reduce(reinterpret_cast<const void*>(&send_buf),
                      reinterpret_cast<void*>(&recv_buf),
                      count,
                      ccl::native_type_info<buffer_type>::dtype,
                      reduction,
                      root,
                      internal_attr,
                      this,
                      get_stream_ptr(stream),
                      deps);
}

/* reduce_scatter */
template <class buffer_type>
ccl::event ccl_comm::reduce_scatter_impl(const buffer_type* send_buf,
                                         buffer_type* recv_buf,
                                         size_t recv_count,
                                         ccl::reduction reduction,
                                         const ccl::stream::impl_value_t& stream,
                                         const ccl::reduce_scatter_attr& attr,
                                         const ccl::vector_class<ccl::event>& deps) {
    return ccl_reduce_scatter(reinterpret_cast<const void*>(send_buf),
                              reinterpret_cast<void*>(recv_buf),
                              recv_count,
                              ccl::native_type_info<buffer_type>::dtype,
                              reduction,
                              attr,
                              this,
                              get_stream_ptr(stream),
                              deps);
}

template <class buffer_type>
ccl::event ccl_comm::reduce_scatter_impl(const buffer_type& send_buf,
                                         buffer_type& recv_buf,
                                         size_t recv_count,
                                         ccl::reduction reduction,
                                         const ccl::stream::impl_value_t& stream,
                                         const ccl::reduce_scatter_attr& attr,
                                         const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_reduce_scatter(reinterpret_cast<const void*>(&send_buf),
                              reinterpret_cast<void*>(&recv_buf),
                              recv_count,
                              ccl::native_type_info<buffer_type>::dtype,
                              reduction,
                              internal_attr,
                              this,
                              get_stream_ptr(stream),
                              deps);
}

/* recv */
template <class buffer_type>
ccl::event ccl_comm::recv_impl(buffer_type* recv_buf,
                               size_t recv_count,
                               int peer,
                               const ccl::stream::impl_value_t& stream,
                               const ccl::pt2pt_attr& attr,
                               const ccl::vector_class<ccl::event>& deps) {
    return ccl_recv(reinterpret_cast<void*>(recv_buf),
                    recv_count,
                    ccl::native_type_info<buffer_type>::dtype,
                    peer,
                    attr,
                    this,
                    get_stream_ptr(stream),
                    deps);
}

template <class buffer_type>
ccl::event ccl_comm::recv_impl(buffer_type& recv_buf,
                               size_t recv_count,
                               int peer,
                               const ccl::stream::impl_value_t& stream,
                               const ccl::pt2pt_attr& attr,
                               const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_recv(reinterpret_cast<void*>(&recv_buf),
                    recv_count,
                    ccl::native_type_info<buffer_type>::dtype,
                    peer,
                    internal_attr,
                    this,
                    get_stream_ptr(stream),
                    deps);
}

/* send */
template <class buffer_type>
ccl::event ccl_comm::send_impl(buffer_type* send_buf,
                               size_t send_count,
                               int peer,
                               const ccl::stream::impl_value_t& stream,
                               const ccl::pt2pt_attr& attr,
                               const ccl::vector_class<ccl::event>& deps) {
    return ccl_send(reinterpret_cast<void*>(send_buf),
                    send_count,
                    ccl::native_type_info<buffer_type>::dtype,
                    peer,
                    attr,
                    this,
                    get_stream_ptr(stream),
                    deps);
}

template <class buffer_type>
ccl::event ccl_comm::send_impl(buffer_type& send_buf,
                               size_t send_count,
                               int peer,
                               const ccl::stream::impl_value_t& stream,
                               const ccl::pt2pt_attr& attr,
                               const ccl::vector_class<ccl::event>& deps) {
    ccl_coll_attr internal_attr(attr);
#ifdef CCL_ENABLE_SYCL
    internal_attr.is_sycl_buf = 1;
#endif // CCL_ENABLE_SYCL
    return ccl_send(reinterpret_cast<void*>(&send_buf),
                    send_count,
                    ccl::native_type_info<buffer_type>::dtype,
                    peer,
                    internal_attr,
                    this,
                    get_stream_ptr(stream),
                    deps);
}

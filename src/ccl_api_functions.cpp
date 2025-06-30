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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"
#include "oneapi/ccl/api_functions.hpp"
#include "comm/comm.hpp"
#include "oneapi/ccl/exception.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "ccl_api_functions_generators.hpp"
#include "common/global/global.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"
// WA for broadcast to avoid scheduler completely
#include "coll/algorithms/broadcast/mpi_bcast_invoke.hpp"
namespace ccl {

namespace v1 {

/**
 * A structure that is a friend of the passed object
 * and which allows access to the internal representation of this object
 */
struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};

void init(const init_attr& attr) {
    auto& env = detail::environment::instance();
    (void)env;
}

/******************** ENVIRONMENT ********************/

library_version get_library_version() {
    return detail::environment::get_library_version();
}

/* datatype */
datatype register_datatype(const datatype_attr& attr) {
    return detail::environment::instance().register_datatype(attr);
}

void deregister_datatype(datatype dtype) {
    return detail::environment::instance().deregister_datatype(dtype);
}

size_t get_datatype_size(datatype dtype) {
    return detail::environment::instance().get_datatype_size(dtype);
}

/* KVS */
shared_ptr_class<kvs> create_main_kvs(const kvs_attr& attr) {
    return detail::environment::instance().create_main_kvs(attr);
}

shared_ptr_class<kvs> create_kvs(const kvs::address_type& addr, const kvs_attr& attr) {
    return detail::environment::instance().create_kvs(addr, attr);
}

/* device */
device create_device() {
    static empty_t empty{};
    return detail::environment::instance().create_device(empty);
}

/* context */
context create_context() {
    static empty_t empty{};
    return detail::environment::instance().create_context(empty);
}

/* stream */
stream create_stream() {
    return default_stream;
}

} // namespace v1

namespace preview {

/* communicator */
communicator create_communicator(const comm_attr& attr) {
    return ccl::detail::environment::instance().create_communicator(attr);
}

communicator create_communicator(const int size,
                                 shared_ptr_class<kvs_interface> kvs,
                                 const comm_attr& attr) {
    return ccl::detail::environment::instance().create_communicator(size, kvs, attr);
}

} // namespace preview

namespace v1 {
/***************** SPLIT COMMUNICATOR *****************/
communicator split_communicator(const communicator& comm, int color, int key) {
    return ccl::detail::environment::instance().split_communicator(comm, color, key);
}

/******************** COMMUNICATOR ********************/
communicator create_communicator(const int size,
                                 const int rank,
                                 shared_ptr_class<kvs_interface> kvs,
                                 const comm_attr& attr) {
    return detail::environment::instance().create_communicator(size, rank, kvs, attr);
}

/******************** GROUP CALLS ********************/

void group_start() {
    group_impl::start();
}

void group_end() {
    group_impl::end();
}

/******************** OPERATION ********************/

#define CHECK_DEPS(deps) \
    do { \
        if (!deps.empty()) { \
            throw ccl::exception( \
                std::string(__PRETTY_FUNCTION__) + \
                " - handling a vector of events that the operation should depend on is not implemented"); \
        } \
    } while (0)

/* allgather */
event allgather(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

event allgather(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(
        send_buf, recv_buf, count, dtype, disp(default_stream), attr, deps);
}

event allgather(const void* send_buf,
                const vector_class<void*>& recv_buf,
                size_t count,
                datatype dtype,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgather(const BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgather(const BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allgather(const BufferType* send_buf,
                vector_class<BufferType*>& recv_buf,
                size_t count,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgather(const BufferType* send_buf,
                vector_class<BufferType*>& recv_buf,
                size_t count,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgather(const BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgather(const BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgather(const BufferObjectType& send_buf,
                vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                size_t count,
                const communicator& comm,
                const stream& op_stream,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgather(const BufferObjectType& send_buf,
                vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                size_t count,
                const communicator& comm,
                const allgather_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgather(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

/* allgatherv */
event allgatherv(const void* send_buf,
                 size_t send_count,
                 void* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
}

event allgatherv(const void* send_buf,
                 size_t send_count,
                 void* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, dtype, disp(default_stream), attr, deps);
}

event allgatherv(const void* send_buf,
                 size_t send_count,
                 const vector_class<void*>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, dtype, disp(op_stream), attr, deps);
}

event allgatherv(const void* send_buf,
                 size_t send_count,
                 const vector_class<void*>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 datatype dtype,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, dtype, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                 size_t send_count,
                 BufferType* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                 size_t send_count,
                 BufferType* recv_buf,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                 size_t send_count,
                 vector_class<BufferType*>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allgatherv(const BufferType* send_buf,
                 size_t send_count,
                 vector_class<BufferType*>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                 size_t send_count,
                 BufferObjectType& recv_buf,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                 size_t send_count,
                 BufferObjectType& recv_buf,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                 size_t send_count,
                 vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const stream& op_stream,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allgatherv(const BufferObjectType& send_buf,
                 size_t send_count,
                 vector_class<ccl::reference_wrapper_class<BufferObjectType>>& recv_bufs,
                 const vector_class<size_t>& recv_counts,
                 const communicator& comm,
                 const allgatherv_attr& attr,
                 const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allgatherv(
        send_buf, send_count, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

/* allreduce */
event allreduce(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                reduction reduction,
                const communicator& comm,
                const stream& op_stream,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction, disp(op_stream), attr, deps);
}

event allreduce(const void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                reduction reduction,
                const communicator& comm,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, dtype, reduction, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event allreduce(const BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                reduction reduction,
                const communicator& comm,
                const stream& op_stream,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(send_buf, recv_buf, count, reduction, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event allreduce(const BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                reduction reduction,
                const communicator& comm,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, reduction, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allreduce(const BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                reduction reduction,
                const communicator& comm,
                const stream& op_stream,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(send_buf, recv_buf, count, reduction, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event allreduce(const BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                reduction reduction,
                const communicator& comm,
                const allreduce_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->allreduce(
        send_buf, recv_buf, count, reduction, disp(default_stream), attr, deps);
}

/* alltoall */
event alltoall(const void* send_buf,
               void* recv_buf,
               size_t count,
               datatype dtype,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

event alltoall(const void* send_buf,
               void* recv_buf,
               size_t count,
               datatype dtype,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(default_stream), attr, deps);
}

event alltoall(const vector_class<void*>& send_buf,
               const vector_class<void*>& recv_buf,
               size_t count,
               datatype dtype,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, dtype, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const BufferType* send_buf,
               BufferType* recv_buf,
               size_t count,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const BufferType* send_buf,
               BufferType* recv_buf,
               size_t count,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const vector_class<BufferType*>& send_buf,
               const vector_class<BufferType*>& recv_buf,
               size_t count,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoall(const vector_class<BufferType*>& send_buf,
               const vector_class<BufferType*>& recv_buf,
               size_t count,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const BufferObjectType& send_buf,
               BufferObjectType& recv_buf,
               size_t count,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const BufferObjectType& send_buf,
               BufferObjectType& recv_buf,
               size_t count,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
               const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
               size_t count,
               const communicator& comm,
               const stream& op_stream,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
               const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
               size_t count,
               const communicator& comm,
               const alltoall_attr& attr,
               const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoall(send_buf, recv_buf, count, disp(default_stream), attr, deps);
}

/* alltoallv */
event alltoallv(const void* send_buf,
                const vector_class<size_t>& send_counts,
                void* recv_buf,
                const vector_class<size_t>& recv_counts,
                datatype dtype,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, dtype, disp(op_stream), attr, deps);
}

event alltoallv(const void* send_buf,
                const vector_class<size_t>& send_counts,
                void* recv_buf,
                const vector_class<size_t>& recv_counts,
                datatype dtype,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, dtype, disp(default_stream), attr, deps);
}

event alltoallv(const vector_class<void*>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<void*>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                datatype dtype,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, dtype, disp(op_stream), attr, deps);
}

event alltoallv(const vector_class<void*>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<void*>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                datatype dtype,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, dtype, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const BufferType* send_buf,
                const vector_class<size_t>& send_counts,
                BufferType* recv_buf,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const BufferType* send_buf,
                const vector_class<size_t>& send_counts,
                BufferType* recv_buf,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const vector_class<BufferType*>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<BufferType*>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event alltoallv(const vector_class<BufferType*>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<BufferType*>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const BufferObjectType& send_buf,
                const vector_class<size_t>& send_counts,
                BufferObjectType& recv_buf,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const BufferObjectType& send_buf,
                const vector_class<size_t>& send_counts,
                BufferObjectType& recv_buf,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_buf, send_counts, recv_buf, recv_counts, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const stream& op_stream,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                const vector_class<size_t>& send_counts,
                const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                const vector_class<size_t>& recv_counts,
                const communicator& comm,
                const alltoallv_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->alltoallv(
        send_bufs, send_counts, recv_bufs, recv_counts, disp(default_stream), attr, deps);
}

/* barrier */
event barrier(const communicator& comm,
              const stream& op_stream,
              const barrier_attr& attr,
              const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->barrier(disp(op_stream), attr, deps);
}

event barrier(const communicator& comm, const barrier_attr& attr, const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->barrier(disp(default_stream), attr, deps);
}

/* broadcast */
event broadcast(void* buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    if (ccl::global_data::env().use_mpi_bcast_wa) {
        ccl_comm* global_comm = (ccl_comm*)(disp(comm).get());
        return invoke_mpi_bcast(buf, count, dtype, root, global_comm, op_stream, attr, deps);
    }
    return disp(comm)->bcast(buf, count, dtype, root, disp(op_stream), attr, deps);
}

event broadcast(void* buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, dtype, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* buf,
                size_t count,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* buf,
                size_t count,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& buf,
                size_t count,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& buf,
                size_t count,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->bcast(buf, count, root, disp(default_stream), attr, deps);
}

/* broadcast */
event broadcast(void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->broadcast(
        send_buf, recv_buf, count, dtype, root, disp(op_stream), attr, deps);
}

event broadcast(void* send_buf,
                void* recv_buf,
                size_t count,
                datatype dtype,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->broadcast(
        send_buf, recv_buf, count, dtype, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->broadcast(send_buf, recv_buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event broadcast(BufferType* send_buf,
                BufferType* recv_buf,
                size_t count,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps)

{
    impl_dispatch disp;
    return disp(comm)->broadcast(send_buf, recv_buf, count, root, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                int root,
                const communicator& comm,
                const stream& op_stream,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->broadcast(send_buf, recv_buf, count, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event broadcast(BufferObjectType& send_buf,
                BufferObjectType& recv_buf,
                size_t count,
                int root,
                const communicator& comm,
                const broadcast_attr& attr,
                const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->broadcast(send_buf, recv_buf, count, root, disp(default_stream), attr, deps);
}

/* reduce */
event reduce(const void* send_buf,
             void* recv_buf,
             size_t count,
             datatype dtype,
             reduction reduction,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, dtype, reduction, root, disp(op_stream), attr, deps);
}

event reduce(const void* send_buf,
             void* recv_buf,
             size_t count,
             datatype dtype,
             reduction reduction,
             int root,
             const communicator& comm,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, dtype, reduction, root, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce(const BufferType* send_buf,
             BufferType* recv_buf,
             size_t count,
             reduction reduction,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce(const BufferType* send_buf,
             BufferType* recv_buf,
             size_t count,
             reduction reduction,
             int root,
             const communicator& comm,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce(const BufferObjectType& send_buf,
             BufferObjectType& recv_buf,
             size_t count,
             reduction reduction,
             int root,
             const communicator& comm,
             const stream& op_stream,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce(const BufferObjectType& send_buf,
             BufferObjectType& recv_buf,
             size_t count,
             reduction reduction,
             int root,
             const communicator& comm,
             const reduce_attr& attr,
             const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce(
        send_buf, recv_buf, count, reduction, root, disp(default_stream), attr, deps);
}

/* reduce_scatter */
event reduce_scatter(const void* send_buf,
                     void* recv_buf,
                     size_t recv_count,
                     datatype dtype,
                     reduction reduction,
                     const communicator& comm,
                     const stream& op_stream,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, disp(op_stream), attr, deps);
}

event reduce_scatter(const void* send_buf,
                     void* recv_buf,
                     size_t recv_count,
                     datatype dtype,
                     reduction reduction,
                     const communicator& comm,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, dtype, reduction, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce_scatter(const BufferType* send_buf,
                     BufferType* recv_buf,
                     size_t recv_count,
                     reduction reduction,
                     const communicator& comm,
                     const stream& op_stream,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event reduce_scatter(const BufferType* send_buf,
                     BufferType* recv_buf,
                     size_t recv_count,
                     reduction reduction,
                     const communicator& comm,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce_scatter(const BufferObjectType& send_buf,
                     BufferObjectType& recv_buf,
                     size_t recv_count,
                     reduction reduction,
                     const communicator& comm,
                     const stream& op_stream,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event reduce_scatter(const BufferObjectType& send_buf,
                     BufferObjectType& recv_buf,
                     size_t recv_count,
                     reduction reduction,
                     const communicator& comm,
                     const reduce_scatter_attr& attr,
                     const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->reduce_scatter(
        send_buf, recv_buf, recv_count, reduction, disp(default_stream), attr, deps);
}

/* recv */
event recv(void* recv_buf,
           size_t recv_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, dtype, peer, disp(op_stream), attr, deps);
}

event recv(void* recv_buf,
           size_t recv_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, dtype, peer, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event recv(BufferType* recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event recv(BufferType* recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event recv(BufferObjectType& recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event recv(BufferObjectType& recv_buf,
           size_t recv_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->recv(recv_buf, recv_count, peer, disp(default_stream), attr, deps);
}

/* send */
event send(void* send_buf,
           size_t send_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, dtype, peer, disp(op_stream), attr, deps);
}

event send(void* send_buf,
           size_t send_count,
           datatype dtype,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, dtype, peer, disp(default_stream), attr, deps);
}

template <class BufferType, typename T>
event send(BufferType* send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(op_stream), attr, deps);
}

template <class BufferType, typename T>
event send(BufferType* send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(default_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event send(BufferObjectType& send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const stream& op_stream,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(op_stream), attr, deps);
}

template <class BufferObjectType, typename T>
event send(BufferObjectType& send_buf,
           size_t send_count,
           int peer,
           const communicator& comm,
           const pt2pt_attr& attr,
           const vector_class<event>& deps) {
    impl_dispatch disp;
    return disp(comm)->send(send_buf, send_count, peer, disp(default_stream), attr, deps);
}

} // namespace v1

namespace v1 {

// API force instantiations for Operations
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int8_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint8_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int16_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint16_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int32_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint32_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(int64_t);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(uint64_t);
/*API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(ccl::float16);*/
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(float);
API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(double);
/*API_COMM_OP_PTR_EXPLICIT_INSTANTIATION(ccl::bfloat16);*/

#ifdef CCL_ENABLE_SYCL
#ifndef COMMA
#define COMMA ,
#endif

API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int8_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint8_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int16_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint16_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int32_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint32_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<int64_t COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<uint64_t COMMA 1>);
/*API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<ccl::float16 COMMA 1>);*/
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<float COMMA 1>);
API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<double COMMA 1>);
/*API_COMM_OP_REF_EXPLICIT_INSTANTIATION(sycl::buffer<ccl::bfloat16 COMMA 1>);*/

#undef COMMA
#endif // CCL_ENABLE_SYCL

} // namespace v1

} // namespace ccl

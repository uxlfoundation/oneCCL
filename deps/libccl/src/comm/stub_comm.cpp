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

#ifdef CCL_ENABLE_STUB_BACKEND

#include "comm/stub_comm.hpp"
#include "common/event/impls/stub_event.hpp"

#include "stub_kvs_impl.hpp"

namespace ccl {

stub_comm::stub_comm(device_t device,
                     context_t context,
                     size_t rank,
                     size_t size,
                     std::shared_ptr<ccl::kvs> kvs,
                     const ccl::stub_kvs_impl* kvs_impl)
        : device_ptr(std::make_shared<ccl::device>(device)),
          context_ptr(std::make_shared<ccl::context>(context)),
          comm_rank(rank),
          comm_size(size),
          kvs(kvs),
          kvs_impl(kvs_impl) {}

stub_comm::~stub_comm() {}

stub_comm* stub_comm::create(device_t device,
                             context_t context,
                             size_t rank,
                             size_t size,
                             std::shared_ptr<ccl::kvs_interface> kvs_interface) {
    auto kvs_inst = std::dynamic_pointer_cast<ccl::kvs>(kvs_interface);
    CCL_THROW_IF_NOT(kvs_inst != nullptr, "only ccl::kvs is allowed with stub backend");

    auto kvs_impl = ccl::get_kvs_impl_typed<stub_kvs_impl>(kvs_inst);

    return new stub_comm(device, context, rank, size, std::move(kvs_inst), kvs_impl);
}

/* barrier */
ccl::event stub_comm::barrier_impl(const ccl::stream::impl_value_t& stream,
                                   const ccl::barrier_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* allgather */
ccl::event stub_comm::allgather_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgather_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

ccl::event stub_comm::allgather_impl(const void* send_buf,
                                     const ccl::vector_class<void*>& recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allgather_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* allgatherv */
ccl::event stub_comm::allgatherv_impl(const void* send_buf,
                                      size_t send_count,
                                      void* recv_buf,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::allgatherv_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

ccl::event stub_comm::allgatherv_impl(const void* send_buf,
                                      size_t send_count,
                                      const ccl::vector_class<void*>& recv_bufs,
                                      const ccl::vector_class<size_t>& recv_counts,
                                      ccl::datatype dtype,
                                      const ccl::stream::impl_value_t& stream,
                                      const ccl::allgatherv_attr& attr,
                                      const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* allreduce */
ccl::event stub_comm::allreduce_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::allreduce_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* alltoall */
ccl::event stub_comm::alltoall_impl(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    const ccl::stream::impl_value_t& stream,
                                    const ccl::alltoall_attr& attr,
                                    const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* alltoallv */
ccl::event stub_comm::alltoallv_impl(const void* send_buf,
                                     const ccl::vector_class<size_t>& send_counts,
                                     void* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     ccl::datatype dtype,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::alltoallv_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* bcast */
ccl::event stub_comm::broadcast_impl(void* buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::broadcast_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* broadcast */
ccl::event stub_comm::broadcast_impl(void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     const ccl::stream::impl_value_t& stream,
                                     const ccl::broadcast_attr& attr,
                                     const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* reduce */
ccl::event stub_comm::reduce_impl(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  ccl::reduction reduction,
                                  int root,
                                  const ccl::stream::impl_value_t& stream,
                                  const ccl::reduce_attr& attr,
                                  const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* reduce_scatter */
ccl::event stub_comm::reduce_scatter_impl(const void* send_buf,
                                          void* recv_buf,
                                          size_t recv_count,
                                          ccl::datatype dtype,
                                          ccl::reduction reduction,
                                          const ccl::stream::impl_value_t& stream,
                                          const ccl::reduce_scatter_attr& attr,
                                          const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* scatter */
ccl::event stub_comm::scatter_impl(const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   int root,
                                   const ccl::stream::impl_value_t& stream,
                                   const ccl::scatter_attr& attr,
                                   const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* recv */
ccl::event stub_comm::recv_impl(void* recv_buf,
                                size_t recv_count,
                                ccl::datatype dtype,
                                int peer,
                                const ccl::stream::impl_value_t& stream,
                                const ccl::pt2pt_attr& attr,
                                const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

/* send */
ccl::event stub_comm::send_impl(void* send_buf,
                                size_t send_count,
                                ccl::datatype dtype,
                                int peer,
                                const ccl::stream::impl_value_t& stream,
                                const ccl::pt2pt_attr& attr,
                                const ccl::vector_class<ccl::event>& deps) {
    return process_stub_backend();
}

ccl::event stub_comm::process_stub_backend() {
    std::stringstream s;
    s << "running stub communicator id: " << kvs_impl->get_id();
    std::cout << s.str() << std::endl;
    return std::unique_ptr<ccl::event_impl>(new ccl::stub_event_impl());
}

} // namespace ccl

#endif // CCL_ENABLE_STUB_BACKEND

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

#include "coll/algorithms/broadcast/sycl/broadcast_large_sycl_impl.hpp"

ccl::event broadcast_large(const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           int root,
                           ccl_comm* comm,
                           ccl_stream* global_stream,
                           const ccl::vector_class<ccl::event>& deps) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("invoking MT broadcast");
        ccl::global_data::env().sycl_broadcast_tmp_buf = 1;
        CCL_THROW_IF_NOT(ccl::global_data::env().sycl_broadcast_tmp_buf == 1,
                         "MT large kernel doesnt support disabled tmp buf");
    }
    else {
        LOG_DEBUG("invoking broadcast");
    }

    LOG_DEBUG("invoking broadcast_large");
    sycl_ptrs_type sycl_ptrs;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    const bool is_tmp_used = ccl::global_data::env().sycl_broadcast_tmp_buf;

    if (!is_tmp_used) {
        // For non-root ranks, send_buf may be nullptr (per spec, only root uses send_buf).
        // Use recv_buf as a placeholder for IPC exchange on non-root ranks since
        // only root's send_buf will actually be read from.
        void* send_buf_for_exchange = (send_buf != nullptr) ? (void*)send_buf : recv_buf;
        std::vector<void*> ptrs{ send_buf_for_exchange, recv_buf }; // index 0 and 1
        auto [sched, exchange_entry] = do_ipc_exchange(comm, global_stream, ptrs);

        sycl_ptrs.node_ptrs_rd =
            get_ipc_ptrs<void, MAX_NODE_RANKS>(node_comm, 0, send_buf_for_exchange, sched);
        sycl_ptrs.node_ptrs_wr = get_ipc_ptrs<void, MAX_NODE_RANKS>(node_comm, 1, recv_buf, sched);

        delete exchange_entry;
        delete sched;
    }
    else {
        sycl_ptrs.node_ptrs_rd = get_remote_node_tmp_buf(0, comm);
    }

    auto lambda = [&]<typename T>() {
        return broadcast_large_impl<T>(
            send_buf, recv_buf, count, dtype, root, comm, global_stream, sycl_ptrs, deps);
    };

    return invoke_collective(lambda, dtype);
}

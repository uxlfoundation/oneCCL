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

#include "coll/algorithms/scatter/sycl/scatter_large_sycl_impl.hpp"
#include "coll/algorithms/scatter/sycl/scatter_large_sycl.hpp"

#include <memory>

ccl::event scatter_large(const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         int root,
                         ccl_comm* comm,
                         const std::shared_ptr<ccl_comm>& p2p_comm,
                         ccl_stream* global_stream,
                         const ccl::vector_class<ccl::event>& deps,
                         sycl_scaleup_options coll_attr) {
    // MT mode requires tmp-buf staging — IPC exchange is not safe across threads.
    // Force via a local flag rather than mutating the shared env to avoid a data race.
    const bool force_tmp_buf = comm->is_multi_thread_instance();
    if (force_tmp_buf) {
        LOG_DEBUG("invoking MT scatter");
    }
    else {
        LOG_DEBUG("invoking scatter_large");
    }

    sycl_ptrs_type sycl_ptrs;
    const std::shared_ptr<ccl_comm>& local_comm = p2p_comm;
    std::shared_ptr<ccl_comm> registration_comm = comm->get_node_comm();
    const int local_size = local_comm->size();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    bool is_tmp_used = force_tmp_buf || coll_attr.force_use_tmp ||
                       ccl::global_data::env().sycl_scatter_large_tmp_buf;

    if (!is_tmp_used) {
        // Registered-pointer fast path: when the user has called
        // ccl::buffer_register / created a window covering send_buf on every
        // rank, get_registered_ptrs returns the IPC-mapped peer pointers
        // directly and we skip do_ipc_exchange (avoiding a sched build +
        // ze_handle_exchange round trip per call). Same shape as
        // allgatherv_large_sycl.cpp:79-94 and allreduce_large_sycl.cpp.
        //
        // Scatter-specific note: this path requires every local-domain rank (including
        // non-root) to have a real send_buf of size count*local_size*dsize
        // and to have registered it. Registration is collective, so if any
        // rank registered, all did and all pass real send_buf. If non-root
        // passes send_buf == nullptr the registration check returns empty
        // and we fall to the slow path consistently across ranks.
        std::vector<void*> registered_send_ptrs;
        if (send_buf != nullptr) {
            registered_send_ptrs = registration_comm->get_registered_ptrs(
                send_buf, count * static_cast<size_t>(local_size) * dsize);
        }

        if (registered_send_ptrs.size() != 0) {
            LOG_DEBUG("scatter_large: send_buf is registered, skipping do_ipc_exchange");
            sycl_ptrs.node_ptrs_rd = get_ipc_ptrs<void, MAX_NODE_RANKS>(
                registered_send_ptrs, comm, local_comm, (void*)send_buf);
        }
        else {
            // For non-root ranks, send_buf may be nullptr (per spec, only root uses
            // send_buf). Use recv_buf as a placeholder for IPC exchange — only
            // root's send_buf will actually be read from.
            void* send_buf_for_exchange = (send_buf != nullptr) ? (void*)send_buf : recv_buf;
            std::vector<void*> ptrs{ send_buf_for_exchange, recv_buf };
            auto [sched, exchange_entry] = do_ipc_exchange(local_comm.get(), global_stream, ptrs);

            sycl_ptrs.node_ptrs_rd =
                get_ipc_ptrs<void, MAX_NODE_RANKS>(local_comm, 0, send_buf_for_exchange, sched);
            sycl_ptrs.node_ptrs_wr =
                get_ipc_ptrs<void, MAX_NODE_RANKS>(local_comm, 1, recv_buf, sched);

            delete exchange_entry;
            delete sched;
        }

        // Validate alignment of all peer pointers + element block. `align`
        // here is the minimum-alignment gate for the IPC fast path, not the
        // GPU memcpy's actual vector width (which is HW/driver-decided and
        // can be wider than 4 when the inputs happen to be more strongly
        // aligned). Below this floor we conservatively drop to the tmp-buf
        // variant. Same threshold is used by allgatherv_large_sycl and
        // reduce_scatter_large_sycl. Full per-rank slice in scatter is
        // count*dsize bytes; the memcpy starts at rank * count * dsize, so
        // the offset must also align. all_aligned() folds both checks.
        const size_t align = 4;
        if (!all_aligned(sycl_ptrs.node_ptrs_rd.data(), local_size, count, dsize, align)) {
            LOG_DEBUG("scatter_large: IPC pointers not 4-byte aligned, falling back to tmp-buf");
            is_tmp_used = true;
            sycl_ptrs.node_ptrs_rd = get_remote_large_tmp_buf(0, local_comm.get());
        }
    }
    else {
        sycl_ptrs.node_ptrs_rd = get_remote_large_tmp_buf(0, local_comm.get());
    }

    // scatter_large is dtype-agnostic at the kernel level (pure byte movement),
    // so we skip invoke_collective's per-type dispatch and call the impl once.
    return scatter_large_impl(send_buf,
                              recv_buf,
                              count,
                              dtype,
                              root,
                              comm,
                              p2p_comm,
                              global_stream,
                              sycl_ptrs,
                              is_tmp_used,
                              deps);
}

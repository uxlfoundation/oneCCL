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

#include <algorithm>
#include <limits>

#include "coll/algorithms/broadcast/sycl/broadcast_large_sycl_impl.hpp"

broadcast_ring_slice_plan::broadcast_ring_slice_plan(size_t count,
                                                     size_t dsize,
                                                     int world,
                                                     size_t chunking_threshold)
        : max_slice_count(0),
          nslices(0),
          total_steps(0) {
    // Let K = broadcast_ring_auto_chunks_per_rank.
    // desired_slice_bytes = total_bytes / target_chunks, where target_chunks = world * K.
    //
    // Rationale for the target formula:
    //   For C slices and N ranks, total_steps = C + N - 2.
    //   The root injects one slice during each of the first C iterations.
    //   The final slice then needs N - 2 more iterations to reach the last rank.
    //   When C >> N, this pipeline-latency overhead is a small part of the run.
    //   Targeting C=K*N makes that overhead approximately 1/(K+1) of the iterations.
    //
    // Effective policy:
    //   threshold == 0  -> real_slice_bytes = desired_slice_bytes
    //   threshold > 0   -> real_slice_bytes = max(desired_slice_bytes, threshold)
    //
    // Theoretical optimum (for reference, not used directly):
    //   slice_bytes_opt ≈ sqrt(M * T_barrier * BW / (N - 2))
    const size_t total_bytes = count * dsize;
    const size_t target_chunks = static_cast<size_t>(world) * broadcast_ring_auto_chunks_per_rank;
    const size_t auto_slice_bytes =
        (target_chunks > 0) ? (total_bytes / target_chunks) : total_bytes;
    const size_t effective_slice_bytes = (chunking_threshold > 0)
                                             ? std::max(auto_slice_bytes, chunking_threshold)
                                             : auto_slice_bytes;

    max_slice_count = effective_slice_bytes / dsize;
    if (max_slice_count == 0)
        max_slice_count = 1;
    if (max_slice_count > count)
        max_slice_count = count;
    const size_t nslices_st = (count + max_slice_count - 1) / max_slice_count;
    // Ring-loop indices (slice_idx, total_steps, s) are int for the signed
    // "slice_idx >= 0" bounds guard. Guard against silent overflow at the
    // narrowing cast for pathological (huge count, tiny slice) inputs.
    CCL_THROW_IF_NOT(nslices_st <= static_cast<size_t>(std::numeric_limits<int>::max()),
                     "broadcast ring: nslices overflows int (count too large or slice too small)");
    nslices = static_cast<int>(nslices_st);
    total_steps = nslices + world - 2;
}

ccl::event broadcast_large(const void* send_buf,
                           void* recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           int root,
                           ccl_comm* comm,
                           std::shared_ptr<ccl_comm> p2p_comm,
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

    const bool is_tmp_used = ccl::global_data::env().sycl_broadcast_tmp_buf;

    // Both the IPC and ring paths run when !is_tmp_used. The ring and IPC additionally
    // need node_ptrs_wr (IPC pointers into each rank's recv_buf) which are set
    // up below. MT instances force is_tmp_used=1 above, so they never reach
    // here and never use the ring (which lacks the MT pthread_barrier_wait).
    if (!is_tmp_used) {
        // For non-root ranks, send_buf may be nullptr (per spec, only root uses send_buf).
        // Use recv_buf as a placeholder for IPC exchange on non-root ranks since
        // only root's send_buf will actually be read from.
        void* send_buf_for_exchange =
            (send_buf != nullptr) ? const_cast<void*>(send_buf) : recv_buf;
        std::vector<void*> ptrs{ send_buf_for_exchange, recv_buf }; // index 0 and 1
        auto [sched, exchange_entry] = do_ipc_exchange(p2p_comm.get(), global_stream, ptrs);

        sycl_ptrs.node_ptrs_rd =
            get_ipc_ptrs<void, MAX_NODE_RANKS>(p2p_comm, 0, send_buf_for_exchange, sched);
        sycl_ptrs.node_ptrs_wr = get_ipc_ptrs<void, MAX_NODE_RANKS>(p2p_comm, 1, recv_buf, sched);

        delete exchange_entry;
        delete sched;
    }
    else {
        sycl_ptrs.node_ptrs_rd = get_remote_large_tmp_buf(0, p2p_comm.get());
    }

    auto lambda = [&]<typename T>() {
        return broadcast_large_impl<T>(
            send_buf, recv_buf, count, dtype, root, comm, p2p_comm, global_stream, sycl_ptrs, deps);
    };

    return invoke_collective(lambda, dtype);
}

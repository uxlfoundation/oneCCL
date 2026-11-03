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

#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

#include <memory>

// Direct IPC large variant. Used when the IPC peer pointers exist and pass
// the minimum-alignment gate (4 bytes) used by the rest of the SYCL large
// kernels. The `4` is not "the vec-size of the GPU memcpy" — actual vector
// width is HW-/driver-decided per submission, and the driver may vectorize
// wider than 4 bytes when the source/destination happen to be more strongly
// aligned. The `4` is the floor below which we conservatively drop to the
// tmp-buf staged path. Same threshold is used by allgatherv_large_sycl
// and reduce_scatter_large_sycl.
//
// Pull semantics: each rank reads its own slice from root's send_buf via the
// IPC pointer. Two barriers — first to wait for root's data to be visible,
// second to let root know readers finished. Root never writes to its own
// IPC slot; if recv_buf already aliases its own block (in-place), the local
// copy is skipped. Pattern derived from broadcast_large_impl_ipc and from
// alltoall's per-peer memcpy plumbing — broadcast as the skeleton, alltoall
// as confirmation that q.memcpy(recv, peer_ipc_ptr + offset, bytes) is the
// correct shape for IPC-direct slice transfer.
inline ccl::event scatter_large_impl_ipc(const void* send_buf,
                                         void* recv_buf,
                                         size_t count,
                                         ccl::datatype dtype,
                                         int root,
                                         ccl_comm* comm,
                                         const std::shared_ptr<ccl_comm>& p2p_comm,
                                         sycl::queue q,
                                         sycl_ptrs_type& sycl_ptrs,
                                         const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("scatter large kernel ipc");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    const int comm_rank = p2p_comm->rank();
    const int comm_size = p2p_comm->size();

    // Root's send_buf as visible to every rank via IPC.
    void* root_send_buf = sycl_ptrs.node_ptrs_rd[root];

    // In-order queue: only the first submission needs explicit deps; later
    // submissions chain implicitly. So the first barrier carries dep_events,
    // and the memcpy + second barrier rely on in-order ordering.
    invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);

    const bool inplace =
        ccl::is_scatter_inplace(send_buf, recv_buf, count, dsize, comm_rank, root, comm_size);
    if (inplace) {
        LOG_DEBUG("scatter large inplace at root, no copy");
    }
    else {
        LOG_DEBUG("scatter large copy at root, count: ", count);
        const size_t my_offset_bytes = static_cast<size_t>(comm_rank) * count * dsize;
        const void* my_block = static_cast<const uint8_t*>(root_send_buf) + my_offset_bytes;
        q.memcpy(recv_buf, my_block, count * dsize);
    }

    // Second barrier so root knows readers are done before send_buf is reused.
    sycl::event ret_event = invoke_barrier(p2p_comm, q, {}, is_cpu_barrier);

    return ccl::event::create_from_native(ret_event);
}

// Tmp-buffer-staged large variant. Used when IPC pointers aren't available
// (PCIe-switchless, certain VM/isolation modes), when they fail the alignment
// check, or when CCL_SYCL_SCATTER_LARGE_TMP_BUF=1 forces it.
//
// Push semantics, chunked. Each chunk uses tmp_bufs[i%2] (one of two staging
// slots ping-pong'd across iterations, identical to broadcast_large_impl_tmp).
// Within each chunk, root packs comm_size per-peer sub-blocks back-to-back
// into its local tmp_bufs[i%2] slot; one barrier; non-root ranks pull their
// sub-block from root's remote tmp slot.
//
// Layout per chunk slot (size = get_tmp_buf_size_per_rank()):
//   [ peer 0 sub-block ][ peer 1 sub-block ]...[ peer N-1 sub-block ]
inline ccl::event scatter_large_impl_tmp(const void* send_buf,
                                         void* recv_buf,
                                         size_t count,
                                         ccl::datatype dtype,
                                         int root,
                                         ccl_comm* comm,
                                         const std::shared_ptr<ccl_comm>& p2p_comm,
                                         sycl::queue q,
                                         sycl_ptrs_type& sycl_ptrs,
                                         const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("scatter large kernel tmp buffer");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    const int comm_rank = p2p_comm->rank();
    const int comm_size = p2p_comm->size();
    if (comm->is_multi_thread_instance()) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }

    const bool inplace =
        ccl::is_scatter_inplace(send_buf, recv_buf, count, dsize, comm_rank, root, comm_size);

    // Two staging slots ping-pong'd across chunks. tmp_bufs[k] is our local
    // view; remote_tmp_bufs[k] is what we present to peers (and what we read
    // from peer 'root' below).
    std::vector<void*> tmp_bufs = { get_large_tmp_buf(0, comm), get_large_tmp_buf(1, comm) };
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs_0 =
        get_remote_large_tmp_buf(0, p2p_comm.get());
    std::array<void*, MAX_NODE_RANKS> remote_tmp_bufs_1 =
        get_remote_large_tmp_buf(1, p2p_comm.get());
    std::vector<void*> remote_root_tmp_bufs = { remote_tmp_bufs_0[root], remote_tmp_bufs_1[root] };

    // The full slot must accommodate one per-peer block per peer. Per-peer
    // sub-block size is therefore slot_bytes / comm_size, rounded down to
    // align (kernel_mem_align is preserved when chunking the pipeline).
    const size_t slot_bytes_full = (char*)tmp_bufs[1] - (char*)tmp_bufs[0];
    const size_t per_peer_slot_bytes_raw = slot_bytes_full / static_cast<size_t>(comm_size);
    const size_t align_bytes = ccl::global_data::env().kernel_mem_align;
    const size_t per_peer_slot_bytes = (per_peer_slot_bytes_raw / align_bytes) * align_bytes;
    CCL_THROW_IF_NOT(per_peer_slot_bytes >= dsize,
                     "scatter_large_tmp: per-peer slot (",
                     per_peer_slot_bytes,
                     " B) smaller than one element (dsize=",
                     dsize,
                     ")");
    const size_t chunk_count = per_peer_slot_bytes / dsize;
    const size_t num_chunks = count / chunk_count + (count % chunk_count != 0);

    // In-order queue: the user's input deps only need to be wired into the
    // very first submission; all later submissions chain implicitly. We
    // attach dep_events to the first memcpy on root (or to the first barrier
    // on non-root, since they don't issue the staging memcpys).
    bool first_submission = true;

    for (size_t i = 0; i < num_chunks; i++) {
        size_t copy_count = chunk_count;
        if (i == num_chunks - 1 && count % chunk_count != 0) {
            copy_count = count % chunk_count;
        }
        const size_t copy_bytes = copy_count * dsize;
        const size_t chunk_offset_bytes = i * chunk_count * dsize;

        void* my_tmp = tmp_bufs[i % 2];
        void* root_tmp_remote = remote_root_tmp_bufs[i % 2];

        // Stage: root packs N peer-blocks into its own tmp slot.
        if (comm_rank == root) {
            for (int p = 0; p < comm_size; p++) {
                const size_t src_offset_bytes =
                    static_cast<size_t>(p) * count * dsize + chunk_offset_bytes;
                void* dst =
                    static_cast<uint8_t*>(my_tmp) + static_cast<size_t>(p) * per_peer_slot_bytes;
                const void* src = static_cast<const uint8_t*>(send_buf) + src_offset_bytes;
                if (first_submission) {
                    q.memcpy(dst, src, copy_bytes, dep_events);
                    first_submission = false;
                }
                else {
                    q.memcpy(dst, src, copy_bytes);
                }
            }
        }
        if (comm->is_multi_thread_instance()) {
            pthread_barrier_wait(
                ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        }
        if (first_submission) {
            invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);
            first_submission = false;
        }
        else {
            invoke_barrier(p2p_comm, q, {}, is_cpu_barrier);
        }

        // Pull: each rank reads its sub-block out of root's remote tmp slot.
        if (comm_rank != root || !inplace) {
            void* recv_dst = static_cast<uint8_t*>(recv_buf) + chunk_offset_bytes;
            const void* tmp_src = static_cast<const uint8_t*>(root_tmp_remote) +
                                  static_cast<size_t>(comm_rank) * per_peer_slot_bytes;
            q.memcpy(recv_dst, tmp_src, copy_bytes);
        }
    }
    if (comm->is_multi_thread_instance()) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }
    sycl::event ret_event = invoke_barrier(p2p_comm, q, {}, is_cpu_barrier);
    return ccl::event::create_from_native(ret_event);
}

// scatter_large is a pure byte-movement kernel — both impl variants only
// touch dsize/byte-offsets, never sycl::vec<T>. Per Gengbin's review,
// dispatching on element type would compile 11 identical copies of the
// same memcpy code, so we keep this non-templated and let the body run
// once per call regardless of dtype.
inline ccl::event scatter_large_impl(const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     ccl_comm* comm,
                                     const std::shared_ptr<ccl_comm>& p2p_comm,
                                     ccl_stream* global_stream,
                                     sycl_ptrs_type& sycl_ptrs,
                                     bool is_tmp_used,
                                     const ccl::vector_class<ccl::event>& deps) {
    sycl::queue q = global_stream->get_native_stream();

    if (!is_tmp_used) {
        return scatter_large_impl_ipc(
            send_buf, recv_buf, count, dtype, root, comm, p2p_comm, q, sycl_ptrs, deps);
    }
    return scatter_large_impl_tmp(
        send_buf, recv_buf, count, dtype, root, comm, p2p_comm, q, sycl_ptrs, deps);
}

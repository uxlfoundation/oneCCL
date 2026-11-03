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
#include "coll/algorithms/utils/sycl_copy.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_barrier.hpp"

#include <memory>

// scatter_small: single-node, push-mode flat-tree SYCL scatter for small
// counts.
//
// High-level flow (3 phases per call):
//
//   1. Root parallel_for writes block i of `send_buf` into peer i's tmp-buf
//      slot via IPC remote pointers (peer 0 .. peer comm_size-1). The root
//      writes to its own slot too, so the read phase is uniform.
//   2. Node barrier: every rank waits for root's writes to commit before
//      anyone reads. Either CPU pthread barrier or SYCL kernel barrier
//      depending on CCL_SYCL_CCL_BARRIER.
//   3. Each rank memcpy's its own tmp slot into `recv_buf`. Skipped on
//      root if the call is in-place (recv_buf == send_buf + root*count*dsize).
//
// Two write-path variants:
//   - Vector path  (scatter_small_inner)        : sycl::vec<T, vec_size> bulk
//                                                 + scalar tail. Used when
//                                                 IPC bases + slot offset
//                                                 are alignof(T)-aligned.
//   - Byte path    (scatter_small_inner_bytes)  : 1 byte per work-item.
//                                                 Used when alignment fails.
//
// SYCL-graph (recording-path) handling is layered into the per-call
// generation index inside `tmp_buf` — see the `is_recording` branches.

// Empty kernel-name tag class. Required so each unique combination of
// template params gets its own SYCL kernel name (DPC++ can't deduce names
// from a generic lambda alone).
//   `use_block`         — true for the vec<T,VS> path, false for byte
//                          fallback (so VS=1 vec doesn't collide with the
//                          byte path's mangled name).
//   `use_global_barrier`— true for the single-kernel mode (write+barrier+
//                          read in one parallel_for, GB=1), false for the
//                          split-orchestrator mode (root-only write, host
//                          dispatches the barrier and read separately).
//                          Mirrors allreduce's `reduce_base<...,GB,...>`
//                          template flag.
// Not used at runtime.
template <typename T, int VS, bool use_block, bool use_global_barrier>
class oneccl_scatter_small {};

// Inner kernel body for scatter_small (vector path).
//
// Always runs phase 1: root pushes one element (or one vec-of-vec_size)
// from `in` into every peer's tmp slot in `out[]`. Source for peer i is
// `in + i * count_per_peer`; destination is `out[i]`. The write to
// `out[root]` is included so the read phase is uniform across all ranks.
//
// Each work-item handles either packed (idx < packed_count, vec stores) or
// tail (the trailing `count_per_peer % vec_size` elements via scalar
// stores).
//
// When `use_global_barrier == true`, the kernel additionally runs phase 2
// (`comm_barrier(comm_barrier_data, it)`) and phase 3 (read this rank's
// slot from `local_in` into `recv`), all inside the same parallel_for so
// we save two kernel-submission round-trips. When false, only phase 1 runs
// — the host driver schedules the barrier and the read-out as separate
// submissions. This mirrors allreduce's `reduce_base<...,GB,...>` flag.
//
// `local_in` and `recv` are unused on the GB=false path.
template <typename T, int vec_size, bool use_global_barrier>
inline void scatter_small_inner(const void* in,
                                std::array<void*, MAX_NODE_RANKS> out,
                                const void* local_in,
                                void* recv,
                                ccl_kernel_barrier_data kernel_barrier_data,
                                ccl_comm_barrier_data comm_barrier_data,
                                size_t comm_size,
                                int comm_rank,
                                int root,
                                bool skip_recv,
                                size_t count_per_peer,
                                sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    const size_t packed_count = count_per_peer / vec_size;

    // Split-orchestrator mode: kernel only runs phase 1 (root's writes). Host
    // already gates this submission to comm_rank == root, so no in-kernel
    // root check is needed here.
    if constexpr (!use_global_barrier) {
        if (idx < packed_count) {
#pragma unroll
            for (int i = 0; i < comm_size; i++) {
                AT* dst = static_cast<AT*>(out[i]);
                const AT* src = reinterpret_cast<const AT*>(in) + i * packed_count;
                dst[idx] = src[idx];
            }
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_count;
            if (new_idx < count_per_peer) {
#pragma unroll
                for (int i = 0; i < comm_size; i++) {
                    T* dst = static_cast<T*>(out[i]);
                    const T* src = static_cast<const T*>(in) + i * count_per_peer;
                    dst[new_idx] = src[new_idx];
                }
            }
        }
        return;
    }

    // Single-kernel mode: phase 1 (root only) + phase 2 (barrier) + phase 3 (read).

    // Phase 1: root pushes to every peer's slot.
    if (comm_rank == root) {
        if (idx < packed_count) {
#pragma unroll
            for (int i = 0; i < comm_size; i++) {
                AT* dst = static_cast<AT*>(out[i]);
                const AT* src = reinterpret_cast<const AT*>(in) + i * packed_count;
                dst[idx] = src[idx];
            }
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_count;
            if (new_idx < count_per_peer) {
#pragma unroll
                for (int i = 0; i < comm_size; i++) {
                    T* dst = static_cast<T*>(out[i]);
                    const T* src = static_cast<const T*>(in) + i * count_per_peer;
                    dst[new_idx] = src[new_idx];
                }
            }
        }
    }

    // Phase 2a: device-wide barrier across all work-groups of this kernel.
    // comm_barrier only signals the peers from the work-items with
    // idx < comm_size (i.e. work-group 0), so without this the root can
    // announce "my writes are done" while its other work-groups have not
    // issued their phase 1 stores yet, and peers read a stale tmp buf.
    // Same ordering as allgatherv_small / sycl_reductions.
    kernel_barrier(kernel_barrier_data.get_sync_ptr(), it);

    // Phase 2b: cross-rank barrier (GPU side, via remote atomics).
    comm_barrier(comm_barrier_data, it);

    // reset the barrier counter in the farthest slot for a later call
    if (idx == 0) {
        kernel_barrier_data.reset_sync_data();
    }

    // Phase 3: read out my slot. Skipped on root when in-place.
    if (skip_recv)
        return;
    if (idx < packed_count) {
        AT* dst = static_cast<AT*>(recv);
        const AT* src = static_cast<const AT*>(local_in);
        dst[idx] = src[idx];
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count_per_peer) {
            T* dst = static_cast<T*>(recv);
            const T* src = static_cast<const T*>(local_in);
            dst[new_idx] = src[new_idx];
        }
    }
}

// Byte-granularity inner kernel body. Same flag layout as the vec variant
// — `use_global_barrier=false` is write-only (alignment-fallback for the
// split orchestrator), `use_global_barrier=true` adds barrier+read-out.
template <bool use_global_barrier>
inline void scatter_small_inner_bytes(const uint8_t* in,
                                      std::array<void*, MAX_NODE_RANKS> out,
                                      const uint8_t* local_in,
                                      uint8_t* recv,
                                      ccl_kernel_barrier_data kernel_barrier_data,
                                      ccl_comm_barrier_data comm_barrier_data,
                                      size_t comm_size,
                                      int comm_rank,
                                      int root,
                                      bool skip_recv,
                                      size_t bytes_per_peer,
                                      sycl::nd_item<1> it) {
    const size_t byte_idx = it.get_global_linear_id();

    // Split-orchestrator mode: kernel only runs phase 1 (root's writes). Host
    // gates this submission to comm_rank == root.
    if constexpr (!use_global_barrier) {
        if (byte_idx < bytes_per_peer) {
#pragma unroll
            for (int i = 0; i < comm_size; i++) {
                uint8_t* dst = static_cast<uint8_t*>(out[i]);
                const uint8_t* src = in + i * bytes_per_peer;
                dst[byte_idx] = src[byte_idx];
            }
        }
        return;
    }

    // Single-kernel mode: phase 1 (root only) + phase 2 (barrier) + phase 3 (read).
    if (comm_rank == root) {
        if (byte_idx < bytes_per_peer) {
#pragma unroll
            for (int i = 0; i < comm_size; i++) {
                uint8_t* dst = static_cast<uint8_t*>(out[i]);
                const uint8_t* src = in + i * bytes_per_peer;
                dst[byte_idx] = src[byte_idx];
            }
        }
    }

    // see scatter_small_inner: order all work-groups' phase 1 stores before
    // work-group 0 signals the peers.
    kernel_barrier(kernel_barrier_data.get_sync_ptr(), it);

    comm_barrier(comm_barrier_data, it);

    if (byte_idx == 0) {
        kernel_barrier_data.reset_sync_data();
    }

    if (!skip_recv && byte_idx < bytes_per_peer) {
        recv[byte_idx] = local_in[byte_idx];
    }
}

// Per-dtype implementation. Sets up tmp-buf staging metadata, validates
// alignment, picks the vector vs. byte write path, then runs the 3-phase
// scatter_memcpy lambda below.
template <typename T>
ccl::event scatter_small_impl(const void* send_buf,
                              void* recv_buf,
                              size_t count,
                              ccl::datatype dtype,
                              int root,
                              ccl_comm* comm,
                              const std::shared_ptr<ccl_comm>& p2p_comm,
                              ccl_stream* global_stream,
                              const ccl::vector_class<ccl::event>& deps) {
    sycl::queue q = global_stream->get_native_stream();
    // Recording = SYCL-graph capture mode. When recording, we can't read
    // tmp_buf_idx from host because the value depends on prior recorded
    // submissions — the kernel itself reads/writes the index on-device.
    const bool is_recording = use_recording_path(q);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    // p2p_comm is the local communication domain for this scatter. The public
    // single-node path passes the full node communicator; domain-aware callers
    // may pass a smaller locality domain, such as a PCIe switch inside one physical node.
    const int comm_size = p2p_comm->size();
    const int comm_rank = p2p_comm->rank();

    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, count, dsize);
    const bool use_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    // hw thread count is the correctness floor for the single-kernel path:
    // comm_barrier inside the kernel only synchronises globally when all
    // sycl threads fit on the device at once, so kernel_size <= hw_threads.
    // CCL_SYCL_NUM_THREADS (default 0 = use hw_threads) lets users cap the
    // launch size below that for power / memory-pressure tuning. Same shape
    // as the large/ring kernels (e.g. allreduce_large_sycl_ring.hpp:215-220).
    const size_t hw_threads = get_total_threads(q);
    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    if (!num_threads)
        num_threads = hw_threads;
    const size_t sec_0 = std::min(num_threads, hw_threads);

    // In-place call detection — root's recv_buf points into the matching
    // slot of its own send_buf, so the trailing host->host memcpy on root
    // can be skipped.
    const bool inplace =
        ccl::is_scatter_inplace(send_buf, recv_buf, count, dsize, comm_rank, root, comm_size);

    // Local view + IPC-exchanged remote views of the per-rank tmp buf. The
    // _gpu variant returns device-resident pointers used by recording-path
    // submissions; the regular variant returns host-readable pointers.
    auto [local_tmp_buf, remote_ptrs] =
        is_recording ? p2p_comm->get_all_tmp_bufs_gpu(true) : p2p_comm->get_all_tmp_bufs(true);
    int* tmp_buf_idx = p2p_comm->get_tmp_buf_idx();
    int* tmp_buf_secondary_idx = p2p_comm->get_tmp_buf_secondary_idx();

    // Push: root writes each peer's block to the head of that peer's tmp buffer, and every rank reads
    // its block back from the head of its own. Tmp buffers are private per rank, so there's one block
    // per buffer and no per-rank slot is needed (same as broadcast_small / allgatherv_small).
    const size_t tmp_buf_size = p2p_comm->get_tmp_buf_size();
    const size_t total_bytes = count * dsize;

    // One block must fit in a tmp-buf generation; larger counts belong on the large kernel.
    CCL_THROW_IF_NOT(total_bytes <= tmp_buf_size,
                     "scatter_small: count*dsize (",
                     total_bytes,
                     ") exceeds tmp buf size (",
                     tmp_buf_size,
                     " B). Lower CCL_SYCL_SCATTER_SMALL_THRESHOLD or grow CCL_SYCL_TMP_BUF_SIZE.");

    // Only send_buf needs an alignof(T) check; the IPC bases are L0 page-aligned and we write at offset 0.
    const bool ptrs_aligned_T = is_aligned(send_buf, count, dsize, alignof(T));

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;
    // Multi-thread mode: ranks share the host process; serialise tmp-buf
    // generation index updates across threads with a pthread barrier
    // before/after the kernel submission.
    if (comm->is_multi_thread_instance()) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }

    // Vector kernel submission. SGS is also the wg size — kernel doesn't
    // need cross-SG sync. When `GB == true` the kernel additionally runs
    // the cross-rank comm_barrier and the read-out (single-kernel mode);
    // when `GB == false` it only does the root write (the host then
    // schedules the barrier and read as separate submissions, see
    // scatter_memcpy below). Mirrors allreduce's reduce_base<...,GB,...>.
    // VS : vec_size, SGS : sub_group_size, GB : use_global_barrier
    auto scatter_invoke =
        [=, &q]<int VS, int SGS, bool GB>(std::vector<sycl::event>& sycl_deps) {
            constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
            const size_t kernel_threads = count / vec_size + count % vec_size;
            const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
            const ccl_comm_barrier_data comm_barrier_data =
                GB ? p2p_comm->barrier_inc() : p2p_comm->barrier_data();
            // only the single-kernel path syncs across work-groups in-kernel
            const ccl_kernel_barrier_data kernel_barrier_data =
                GB ? get_kernel_barrier_data(comm).inc_slot() : get_kernel_barrier_data(comm);
            const bool skip_recv_root = (comm_rank == root) && inplace;

            sycl::event local_event = q.submit([=](sycl::handler& h) {
                h.depends_on(sycl_deps);
                h.parallel_for<oneccl_scatter_small<T, VS, true, GB>>(
                    sycl::nd_range<1>(kernel_size, wg_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                        auto remote_ptrs_cpy = remote_ptrs;
                        // Only the SYCL-graph generation offset applies here, and it's the same for every peer.
                        size_t base_offset_bytes = 0;
                        if (is_recording) {
                            base_offset_bytes = *tmp_buf_idx * tmp_buf_size;
                        }
                        for (size_t rem_idx = 0; rem_idx < remote_ptrs_cpy.size(); ++rem_idx) {
                            remote_ptrs_cpy[rem_idx] = static_cast<void*>(
                                static_cast<uint8_t*>(remote_ptrs[rem_idx]) + base_offset_bytes);
                        }
                        const void* my_slot = static_cast<const uint8_t*>(local_tmp_buf) +
                                              base_offset_bytes;
                        scatter_small_inner<T, VS, GB>(send_buf,
                                                       remote_ptrs_cpy,
                                                       my_slot,
                                                       recv_buf,
                                                       kernel_barrier_data,
                                                       comm_barrier_data,
                                                       comm_size,
                                                       comm_rank,
                                                       root,
                                                       skip_recv_root,
                                                       count,
                                                       it);
                        if (is_recording && it.get_global_linear_id() == 0) {
                            *tmp_buf_secondary_idx = *tmp_buf_idx;
                            *tmp_buf_idx = (*tmp_buf_idx + 1) % ccl_tmp_bufs::buf_count;
                        }
                    });
            });
            return local_event;
        };

    // Byte-fallback kernel submission. Same GB layout as scatter_invoke.
    // GB : use_global_barrier
    auto scatter_invoke_bytes = [=, &q]<bool GB>(std::vector<sycl::event>& sycl_deps) {
        constexpr int wg_size = 32, sg_size = 32;
        const size_t kernel_size = ((total_bytes + wg_size - 1) / wg_size) * wg_size;
        const ccl_comm_barrier_data comm_barrier_data =
            GB ? p2p_comm->barrier_inc() : p2p_comm->barrier_data();
        // only the single-kernel path syncs across work-groups in-kernel
        const ccl_kernel_barrier_data kernel_barrier_data =
            GB ? get_kernel_barrier_data(comm).inc_slot() : get_kernel_barrier_data(comm);
        const bool skip_recv_root = (comm_rank == root) && inplace;

        sycl::event local_event = q.submit([=](sycl::handler& h) {
            h.depends_on(sycl_deps);
            h.parallel_for<oneccl_scatter_small<T, 1, false, GB>>(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    auto remote_ptrs_cpy = remote_ptrs;
                    // Only the SYCL-graph generation offset applies here, and it's the same for every peer.
                    size_t base_offset_bytes = 0;
                    if (is_recording) {
                        base_offset_bytes = *tmp_buf_idx * tmp_buf_size;
                    }
                    for (size_t rem_idx = 0; rem_idx < remote_ptrs_cpy.size(); ++rem_idx) {
                        remote_ptrs_cpy[rem_idx] = static_cast<void*>(
                            static_cast<uint8_t*>(remote_ptrs[rem_idx]) + base_offset_bytes);
                    }
                    const uint8_t* my_slot =
                        static_cast<const uint8_t*>(local_tmp_buf) + base_offset_bytes;
                    scatter_small_inner_bytes<GB>(static_cast<const uint8_t*>(send_buf),
                                                  remote_ptrs_cpy,
                                                  my_slot,
                                                  static_cast<uint8_t*>(recv_buf),
                                                  kernel_barrier_data,
                                                  comm_barrier_data,
                                                  comm_size,
                                                  comm_rank,
                                                  root,
                                                  skip_recv_root,
                                                  total_bytes,
                                                  it);
                    if (is_recording && it.get_global_linear_id() == 0) {
                        *tmp_buf_secondary_idx = *tmp_buf_idx;
                        *tmp_buf_idx = (*tmp_buf_idx + 1) % ccl_tmp_bufs::buf_count;
                    }
                });
        });
        return local_event;
    };

    // The 3-phase orchestrator: root-only write -> node barrier ->
    // per-rank memcpy out of the local tmp slot. VS is the vec_size
    // chosen at the call site (1 = byte path, get_num_elements<T,8>() =
    // packed-vector path). Calls scatter_invoke[_bytes] with GB=false so
    // the kernel runs only the write phase; the host schedules the
    // barrier and read-out below.
    // VS : vec_size
    auto scatter_memcpy = [=, &q]<int VS>(std::vector<sycl::event> sycl_deps) {
        // Phase 1: only the root writes. Pick vector or byte path based
        // on alignment validation done above.
        if (comm_rank == root) {
            sycl::event local_event;
            if (ptrs_aligned_T) {
                local_event = scatter_invoke.template operator()<VS, 32, false>(sycl_deps);
            }
            else {
                local_event = scatter_invoke_bytes.template operator()<false>(sycl_deps);
            }
            sycl_deps.clear();
            sycl_deps.push_back(std::move(local_event));
        }
        // Phase 2: barrier so non-root ranks see root's writes before reading.
        sycl::event barrier_event = invoke_barrier(p2p_comm, q, sycl_deps, use_cpu_barrier);
        sycl::event end_event{};
        // Phase 3: each rank copies its block out of its tmp buffer into recv_buf, skipped on root when
        // in-place. In recording mode kernel_memcpy adds the generation offset, so tmp_src is the base.
        if (comm_rank != root || !inplace) {
            void* tmp_src = local_tmp_buf;
            sycl::event memcpy_event;
            if (is_recording) {
                memcpy_event = kernel_memcpy(q,
                                             tmp_src,
                                             recv_buf,
                                             tmp_buf_secondary_idx,
                                             nullptr,
                                             count,
                                             dsize,
                                             p2p_comm->get_tmp_buf_size(),
                                             { barrier_event });
            }
            else {
                memcpy_event = q.submit([=](sycl::handler& h) {
                    h.depends_on(barrier_event);
                    h.memcpy(recv_buf, tmp_src, total_bytes);
                });
            }
            end_event = memcpy_event;
        }
        else {
            end_event = barrier_event;
        }
        return end_event;
    };

    // Pick vec_size at the call site. get_num_elements<T, 8> = how many
    // elements of T fit in 8 bytes (e.g. 8 for char, 4 for fp16, 2 for
    // float, 1 for double). VS = 1 forces the kernel into pure scalar
    // tail-only mode, which is the byte-fallback shape.
    //
    // Prefer the single-kernel path (GB=true: write+barrier+read in one
    // parallel_for) when the data fits in hw threads and we're not
    // constrained by a host-side cpu barrier, recording-path device-side
    // index updates, or MT-mode cross-thread bracketing.
    const bool single_kernel_ok =
        !use_cpu_barrier && !is_recording && !comm->is_multi_thread_instance();
    const bool fits_hw = count <= sec_0;
    if (single_kernel_ok && fits_hw && ptrs_aligned_T && use_full_vector) {
        constexpr int vec_size = get_num_elements<T, 8>();
        kernel_event = scatter_invoke.template operator()<vec_size, 32, true>(dep_events);
    }
    else if (single_kernel_ok && fits_hw && !ptrs_aligned_T && total_bytes <= sec_0) {
        kernel_event = scatter_invoke_bytes.template operator()<true>(dep_events);
    }
    else if (ptrs_aligned_T && use_full_vector) {
        constexpr int vec_size = get_num_elements<T, 8>();
        kernel_event = scatter_memcpy.template operator()<vec_size>(dep_events);
    }
    else {
        kernel_event = scatter_memcpy.template operator()<1>(dep_events);
    }
    if (comm->is_multi_thread_instance()) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }
    return ccl::event::create_from_native(kernel_event);
}

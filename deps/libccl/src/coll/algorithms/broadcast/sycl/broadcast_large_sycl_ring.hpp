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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_copy.hpp"
// Pulls in the shared copy_kernel_wrap<T, full_vec_size, partial_vec_size> used
// by the copy-kernel ring below. Reusing it (instead of writing our own LSC
// kernel) keeps the LSC store assembly in one place and inherits the tuned
// L1WB_L3WB cache policy.
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_ring.hpp"

// ============================================================================
// This file holds BOTH pipelined-ring broadcast implementations:
//   * broadcast_large_impl_ring_copy_engine (backend=1) — copy engine (q.memcpy)
//   * broadcast_large_impl_ring_copy_kernel (backend=0) — copy via a SYCL kernel
// They share the same ring protocol and broadcast_ring_slice_plan; only the
// per-slice copy mechanism differs.
// ============================================================================

/**
 * Pipelined ring broadcast (simple protocol).
 *
 * The message is divided into slices (sized via the auto-heuristic below, with
 * CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD as an optional minimum-size floor).
 * At each pipeline step s, every rank except the final receiver forwards any
 * slice it holds one hop to the right neighbour (rank+1)%N. In the same step,
 * every non-root rank may receive a new slice from its left neighbour. All
 * writes happen in parallel; an invoke_p2p_barrier separates consecutive steps.
 *
 * Example: N=4 ranks (root=0), M=4 slices (s0..s3):
 *
 *   step 0:  rank0 ──s0──► rank1
 *   step 1:  rank0 ──s1──► rank1    rank1 ──s0──► rank2
 *   step 2:  rank0 ──s2──► rank1    rank1 ──s1──► rank2    rank2 ──s0──► rank3
 *   step 3:  rank0 ──s3──► rank1    rank1 ──s2──► rank2    rank2 ──s1──► rank3
 *   step 4:                         rank1 ──s3──► rank2    rank2 ──s2──► rank3
 *   step 5:                                                rank2 ──s3──► rank3
 *
 *   total steps = M + N - 2 = 6.
 *
 * After the first N-1 fill steps every intermediate rank simultaneously forwards
 * one slice right and receives a new slice from the left, keeping all links busy.
 * Rank 0 (root) only sends; rank N-1 only receives.
 *
 * Each rank's hop distance from root: d = (rank - root + world) % world.
 * At step s this rank sends slice index (s - d) if 0 <= (s-d) < nslices AND d < N-1.
 * Root (d=0) reads from send_buf; forwarding ranks read from own recv_buf
 * (the left neighbour wrote there via an IPC write pointer in the previous step).
 *
 * After all pipeline steps root copies send_buf -> recv_buf for out-of-place.
 *
 * Prerequisites:
 *   sycl_ptrs.node_ptrs_wr must be populated (IPC non-tmp path).
 *   CCL_SYCL_BROADCAST_TMP_BUF must be 0 (the default).
 *   CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD acts as a minimum-size floor on the
 *   auto-computed slice size:
 *     desired_slice_bytes = total_bytes / (world * broadcast_ring_auto_chunks_per_rank)
 *     threshold == 0  -> real_slice_bytes = desired_slice_bytes
 *     threshold > 0   -> real_slice_bytes = max(desired_slice_bytes, threshold)
 */

// Target this many slices per rank when auto-sizing.
//   More chunks => deeper pipeline fill, lower ramp-up fraction, higher barrier overhead.
//   Tune this value to balance pipeline utilization against barrier overhead.
inline constexpr size_t broadcast_ring_auto_chunks_per_rank = 4;

// Slice-size plan shared by both copy backends.
struct broadcast_ring_slice_plan {
    /**
     * Constructs the slice layout for a pipelined ring broadcast.
     *
     * @param count Number of elements in the broadcast buffer.
     * @param dsize Size of one element in bytes.
     * @param world Number of ranks in the ring communicator.
     * @param chunking_threshold Minimum slice size in bytes, or 0 for automatic sizing.
     */
    broadcast_ring_slice_plan(size_t count, size_t dsize, int world, size_t chunking_threshold);

    size_t max_slice_count; // elements per slice (last slice may be smaller)
    int nslices; // total slices
    int total_steps; // nslices + world - 2 synchronized iterations
};

// ----------------------------------------------------------------------------
// COPY-ENGINE backend (1): each slice copied by q.memcpy (hardware engine).
// ----------------------------------------------------------------------------
/**
 * Executes the pipelined ring broadcast using queue memcpy operations.
 *
 * @param send_buf Root's source buffer; may be null on non-root ranks.
 * @param recv_buf Destination buffer on the local rank.
 * @param count Number of elements to broadcast.
 * @param dtype Datatype of each element.
 * @param p2p_root Root rank in the p2p communicator.
 * @param p2p_comm Communicator that defines the ring rank space.
 * @param q Queue used for copies and synchronization.
 * @param sycl_ptrs IPC pointers to peer receive buffers.
 * @param deps Events that must complete before the broadcast starts.
 * @param threshold_auto_override_bytes Automatic-policy slice-size floor in bytes.
 * @param use_auto_threshold_override Whether to use the automatic-policy floor.
 * @return Event that completes after the broadcast and root's local copy.
 */
ccl::event broadcast_large_impl_ring_copy_engine(
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl::datatype dtype,
    int p2p_root,
    std::shared_ptr<ccl_comm> p2p_comm,
    sycl::queue q,
    sycl_ptrs_type& sycl_ptrs,
    const ccl::vector_class<ccl::event>& deps,
    // When the AUTO policy selects the ring it also picks a tuned
    // slice-floor threshold; those two values arrive here. When AUTO is off
    // (use_auto_threshold_override=false) this falls back to the env-var threshold.
    size_t threshold_auto_override_bytes = 0,
    bool use_auto_threshold_override = false) {
    const auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    const int rank = p2p_comm->rank();
    const int world = p2p_comm->size();
    CCL_THROW_IF_NOT(world >= 2,
                     "broadcast ring requires world >= 2 — caller must handle single-rank case");
    CCL_THROW_IF_NOT(p2p_root >= 0 && p2p_root < world,
                     "broadcast ring root must be relative to p2p_comm");

    const size_t chunking_threshold =
        use_auto_threshold_override ? threshold_auto_override_bytes
                                    : ccl::global_data::env().sycl_broadcast_chunking_threshold;
    const broadcast_ring_slice_plan plan{ count, dsize, world, chunking_threshold };

    // Hop distance of this rank from root (0 = root, 1 = first relay, N-1 = last receiver).
    const int hop_dist = (rank - p2p_root + world) % world;

    // Right neighbour in the pipeline direction; IPC write pointer into its recv_buf.
    const int right = (rank + 1) % world;
    char* const peer_recv_buf = static_cast<char*>(sycl_ptrs.node_ptrs_wr[right]);

    // Initial barrier: all ranks rendezvous before any IPC write begins.
    sycl::event e = invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);

    // ------------------------------------------------------------------
    // Pipelined ring loop.
    // At step s this rank sends slice (s - hop_dist) if it is a valid slice
    // index and this rank is not the last receiver.
    // All ranks call invoke_p2p_barrier every step to advance the pipeline.
    // ------------------------------------------------------------------
    for (int s = 0; s < plan.total_steps; s++) {
        const int slice_idx = s - hop_dist;
        if (slice_idx >= 0 && slice_idx < plan.nslices && hop_dist < world - 1) {
            const size_t slice_count =
                (slice_idx < plan.nslices - 1)
                    ? plan.max_slice_count
                    : count - static_cast<size_t>(slice_idx) * plan.max_slice_count;
            const size_t slice_bytes = slice_count * dsize;
            const size_t off = static_cast<size_t>(slice_idx) * plan.max_slice_count * dsize;

            // Root reads from send_buf; a forwarding rank reads from its own recv_buf
            // (written by the left neighbour via IPC in the previous step).
            const char* src = (hop_dist == 0 && send_buf != nullptr)
                                  ? static_cast<const char*>(send_buf) + off
                                  : static_cast<const char*>(recv_buf) + off;

            // Write directly into right neighbour's recv_buf via IPC write pointer.
            sycl::event copy_e = q.memcpy(peer_recv_buf + off, src, slice_bytes, e);
            e = invoke_p2p_barrier(p2p_comm, q, { copy_e }, is_cpu_barrier);
        }
        else {
            // No send this step: still must participate in the global barrier.
            e = invoke_p2p_barrier(p2p_comm, q, { e }, is_cpu_barrier);
        }
    }

    // Root out-of-place copy: root never receives through the ring, so copy the
    // entire send_buf into recv_buf so root's output buffer also holds the result.
    if (rank == p2p_root && send_buf != recv_buf && send_buf != nullptr) {
        e = q.memcpy(recv_buf, send_buf, count * dsize, e);
    }

    return ccl::event::create_from_native(e);
}

// ----------------------------------------------------------------------------
// COPY-KERNEL backend (0): each slice copied by a SYCL kernel reusing the
// shared copy_kernel_wrap<T, full_vec_size, partial_vec_size> from
// reduce_scatter_large_sycl_ring.hpp. Same ring protocol and slice-size math
// as the copy-engine backend; the differences from that path are:
//
//   1. The collective queue is in-order, so submission order alone guarantees
//      copy-then-barrier ordering — no manual sycl::event chaining between
//      the kernel launch and the p2p barrier (matches the allgatherv pattern).
//
//   2. Templated on T (element type). broadcast_large_impl<T> dispatches the
//      runtime dtype via invoke_collective(), so all dtypes work.
// ----------------------------------------------------------------------------
/**
 * Executes the pipelined ring broadcast using SYCL copy kernels.
 *
 * @tparam T Element type selected from dtype by the collective dispatcher.
 * @param send_buf Root's source buffer; may be null on non-root ranks.
 * @param recv_buf Destination buffer on the local rank.
 * @param count Number of elements to broadcast.
 * @param dtype Datatype of each element.
 * @param p2p_root Root rank in the p2p communicator.
 * @param p2p_comm Communicator that defines the ring rank space.
 * @param q In-order queue used for kernels and synchronization.
 * @param sycl_ptrs IPC pointers to peer receive buffers.
 * @param deps Events that must complete before the broadcast starts.
 * @param threshold_auto_override_bytes Automatic-policy slice-size floor in bytes.
 * @param use_auto_threshold_override Whether to use the automatic-policy floor.
 * @return Event that completes after the broadcast and root's local copy.
 */
template <typename T>
ccl::event broadcast_large_impl_ring_copy_kernel(const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 int p2p_root,
                                                 std::shared_ptr<ccl_comm> p2p_comm,
                                                 sycl::queue q,
                                                 sycl_ptrs_type& sycl_ptrs,
                                                 const ccl::vector_class<ccl::event>& deps,
                                                 size_t threshold_auto_override_bytes = 0,
                                                 bool use_auto_threshold_override = false) {
    const auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    const int rank = p2p_comm->rank();
    const int world = p2p_comm->size();
    CCL_THROW_IF_NOT(world >= 2,
                     "broadcast ring requires world >= 2 — caller must handle single-rank case");
    CCL_THROW_IF_NOT(p2p_root >= 0 && p2p_root < world,
                     "broadcast ring root must be relative to p2p_comm");

    const size_t chunking_threshold =
        use_auto_threshold_override ? threshold_auto_override_bytes
                                    : ccl::global_data::env().sycl_broadcast_chunking_threshold;
    const broadcast_ring_slice_plan plan{ count, dsize, world, chunking_threshold };

    // Kernel launch geometry — mirrors the allgatherv ring (one work-group sized
    // by sycl_work_group_size, kernel_threads chosen from slice element count).
    constexpr size_t copy_vector_bytes = 16;
    constexpr size_t copy_sub_group_size = 16;
    constexpr int full_vec_size = get_num_elements<T, copy_vector_bytes, true>();
    constexpr int partial_vec_size = get_num_elements<T, copy_vector_bytes, false>();
    const size_t max_work_group_size =
        q.get_device().get_info<sycl::info::device::max_work_group_size>();
    size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    if (work_group_size == 0 || work_group_size > max_work_group_size) {
        work_group_size = max_work_group_size;
    }

    const int hop_dist = (rank - p2p_root + world) % world;
    const int right = (rank + 1) % world;
    char* const peer_recv_buf = static_cast<char*>(sycl_ptrs.node_ptrs_wr[right]);

    // An out-of-place root stores every slice into both its peer's recv_buf and
    // its own recv_buf in the same kernel. In-place calls and relays use one store.
    const bool is_out_of_place_root =
        (rank == p2p_root) && (send_buf != recv_buf) && (send_buf != nullptr);

    // Initial barrier: all ranks rendezvous before any IPC write begins.
    invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);

    // Pipelined ring loop. In-order queue => no event chaining needed: each
    // submitted kernel/barrier waits for the previous one automatically.
    for (int s = 0; s < plan.total_steps; s++) {
        const int slice_idx = s - hop_dist;
        if (slice_idx >= 0 && slice_idx < plan.nslices && hop_dist < world - 1) {
            const size_t slice_count =
                (slice_idx < plan.nslices - 1)
                    ? plan.max_slice_count
                    : count - static_cast<size_t>(slice_idx) * plan.max_slice_count;
            const size_t off_bytes = static_cast<size_t>(slice_idx) * plan.max_slice_count * dsize;

            // Root reads from send_buf; a forwarder reads from its own recv_buf.
            const void* src = (hop_dist == 0 && send_buf != nullptr)
                                  ? static_cast<const char*>(send_buf) + off_bytes
                                  : static_cast<const char*>(recv_buf) + off_bytes;
            void* dst = peer_recv_buf + off_bytes;

            // Dual-store: only the out-of-place root. The root reads each
            // slice from send_buf and stores it to BOTH recv_buf AND the peer's
            // recv_buf in the SAME kernel launch — one load feeding two stores
            // from the same register. local_dst stays null for relays and
            // in-place calls.
            void* local_dst =
                is_out_of_place_root ? static_cast<char*>(recv_buf) + off_bytes : nullptr;

            // Vec choice — full vector when ALL kernel-touched pointers are
            // aligned and the slice count fits, otherwise scalar tail. When
            // dual-store fires, copy_2dst_wrap issues vector stores to
            // local_dst as well as dst, so local_dst must be checked too.
            // In-place / non-root cases have local_dst==nullptr and skip that
            // constraint (only src and dst matter).
            bool use_full_vector = can_use_full_vector(src, dst, slice_count, dsize);
            if (use_full_vector && local_dst != nullptr) {
                use_full_vector = can_use_full_vector(src, local_dst, slice_count, dsize);
            }
            const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;
            const size_t kernel_threads = slice_count / vec_size + slice_count % vec_size;
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
            sycl::nd_range<1> nd_range(kernel_size, work_group_size);

            // Pick the kernel by which branch local_dst landed in:
            //   != nullptr  -> root, dual-store mode: launch copy_2dst_wrap so
            //                  the SAME kernel does ONE load from send_buf and
            //                  TWO stores (peer's recv_buf + own recv_buf). The
            //                  local-HBM store overlaps the slow PCIe store.
            //   == nullptr  -> relay rank or in-place root: launch the plain
            //                  single-store copy_kernel_wrap.
            if (local_dst != nullptr) {
                q.parallel_for(nd_range,
                               [=](sycl::nd_item<1> it)
                                   [[sycl::reqd_sub_group_size(copy_sub_group_size)]] {
                                       copy_2dst_wrap<T, full_vec_size, partial_vec_size>(
                                           dst, local_dst, src, slice_count, use_full_vector);
                                   });
            }
            else {
                q.parallel_for(nd_range,
                               [=](sycl::nd_item<1> it)
                                   [[sycl::reqd_sub_group_size(copy_sub_group_size)]] {
                                       copy_kernel_wrap<T, full_vec_size, partial_vec_size>(
                                           dst, src, slice_count, use_full_vector);
                                   });
            }
            // In-order queue => empty deps; the barrier runs after the copy above.
            invoke_p2p_barrier(p2p_comm, q, {}, is_cpu_barrier);
        }
        else {
            invoke_p2p_barrier(p2p_comm, q, {}, is_cpu_barrier);
        }
    }

    // Return an event that completes when the in-order queue drains.
    sycl::event e_final = get_last_event(q);

    return ccl::event::create_from_native(e_final);
}

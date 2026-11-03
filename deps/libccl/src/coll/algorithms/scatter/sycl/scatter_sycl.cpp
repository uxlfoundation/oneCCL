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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/scatter/sycl/scatter_sycl.hpp"
#include "coll/algorithms/scatter/sycl/scatter_small_sycl.hpp"
#include "coll/algorithms/scatter/sycl/scatter_large_sycl.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_scaleup.hpp"
#include "coll/algorithms/utils/barrier_data.hpp"  // use_recording_path

#include <memory>
#include <optional>

namespace ccl {
namespace v1 {

// Run single-domain SYCL scatter inside the supplied p2p communicator.
// This is used both for normal node-local scatter and smaller PCIe-switch domains.
namespace {

// world == 1 fast path for single-domain scatter: root copies its own block
// out of `send_buf`; in-place is a no-op.
event scatter_scaleup_single_rank(const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  datatype dtype,
                                  int root,
                                  ccl_stream* global_stream,
                                  const vector_class<event>& deps) {
    auto ccl_dtype = global_data::get().dtypes->get(dtype);
    sycl::event sycl_e;
    auto sycl_q = global_stream->get_native_stream();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    size_t own_offset = static_cast<size_t>(root) * count * ccl_dtype.size();
    const void* own_block = static_cast<const char*>(send_buf) + own_offset;
    if (own_block != recv_buf) {
        LOG_DEBUG("single rank: out-of-place case, coll: scatter");
        sycl_e = sycl_q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, own_block, count * ccl_dtype.size());
        });
    }
    else {
        LOG_DEBUG("single rank: inplace case, coll: scatter");
        sycl_e = submit_wait_on_events(sycl_q, dep_events);
    }
    return event::create_from_native(sycl_e);
}

// Default (only) path: small kernel below threshold, large kernel otherwise.
event scatter_scaleup_default(const void* send_buf,
                              void* recv_buf,
                              size_t count,
                              datatype dtype,
                              int root,
                              ccl_comm* comm,
                              const std::shared_ptr<ccl_comm>& p2p_comm,
                              ccl_stream* global_stream,
                              const vector_class<event>& deps,
                              sycl_scaleup_options coll_attr) {
    auto ccl_dtype = global_data::get().dtypes->get(dtype);
    event e;
    if (count * ccl_dtype.size() <= ccl::global_data::env().sycl_scatter_small_threshold) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "scatter_small", "send_size", count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| scatter selects small kernel, count: ", count, " datatype: ", dtype);
        e = scatter_small(
            send_buf, recv_buf, count, dtype, root, comm, p2p_comm, global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| scatter selects small kernel, count: ",
                  count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "scatter_large", "send_size", count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| scatter selects large kernel, count: ", count, " datatype: ", dtype);
        e = scatter_large(
            send_buf, recv_buf, count, dtype, root, comm, p2p_comm, global_stream, deps, coll_attr);
        LOG_DEBUG("|CCL_SYCL| scatter selects large kernel, count: ",
                  count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    return e;
}

}  // namespace

event scatter_sycl_single_domain(sycl::queue& q,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 datatype dtype,
                                 int root,
                                 ccl_comm* comm,
                                 const std::shared_ptr<ccl_comm>& p2p_comm,
                                 ccl_stream* global_stream,
                                 const vector_class<event>& deps,
                                 bool& done,
                                 sycl_scaleup_options coll_attr) {
    done = true;

    if (p2p_comm->size() == 1) {
        return scatter_scaleup_single_rank(send_buf, recv_buf, count, dtype, root,
                                           global_stream, deps);
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const auto tune = ccl::sycl_policy_resolver::scaleup_resolver{
        ccl::sycl_policy::policy_query(
            ccl_coll_scatter, count * ccl_dtype.size(), ccl_dtype,
            /*r2r_size=*/0, comm->size(), use_recording_path(q))
            .with_comm(comm)
            .with_p2p_comm(p2p_comm.get())
            .with_stream(global_stream)
    }.scatter();
    (void)tune;  // scatter has no arc/LL split today; resolver always returns default_kernel

    return scatter_scaleup_default(send_buf, recv_buf, count, dtype, root, comm, p2p_comm,
                                   global_stream, deps, coll_attr);
}

// -----------------------------------------------------------------------------
// Multi-node scatter — shared helpers
// -----------------------------------------------------------------------------
//
// MPI-style scatter data layout: root's send_buf holds N*count elements
// ordered by global rank (block for rank g at offset g*count). Every rank
// returns with its own block in recv_buf.
//
namespace {

// Root-only repack for phase A. Reorders one chunk of send_buf from
// global-rank-major (MPI layout: block for rank g at offset g*count)
// into local-idx-major slots that node_comm's scatter expects

template <typename T, int vec_size>
static inline void pack_root_body(const T* src,
                                  T* out,
                                  size_t copy_count,
                                  size_t chunk_offset_elts,
                                  size_t src_stride,
                                  int node_size,
                                  int r2r_size,
                                  sycl::nd_item<1> it) {
    const size_t gid = it.get_global_linear_id();
    const size_t packed = copy_count / vec_size;
    const size_t tail = copy_count - packed * vec_size;
    const size_t slots = static_cast<size_t>(node_size) * r2r_size;
    const size_t per_slot = packed + tail; // work-items per slot
    const size_t total_threads = slots * per_slot;
    if (gid >= total_threads) {
        return;
    }

    const size_t slot = gid / per_slot;
    const size_t within_slot = gid - slot * per_slot;
    const size_t local_idx = slot / static_cast<size_t>(r2r_size);
    const size_t node_idx = slot - local_idx * static_cast<size_t>(r2r_size);
    const size_t global_rank = node_idx * static_cast<size_t>(node_size) + local_idx;
    const size_t src_slot_off = global_rank * src_stride + chunk_offset_elts;
    const size_t dst_slot_off = slot * copy_count;

    if (within_slot < packed) {
        using VT = sycl::vec<T, vec_size>;
        const VT* src_v =
            reinterpret_cast<const VT*>(src + src_slot_off);
        VT* out_v = reinterpret_cast<VT*>(out + dst_slot_off);
        out_v[within_slot] = src_v[within_slot];
    }
    else {
        const size_t within = packed * vec_size + (within_slot - packed);
        out[dst_slot_off + within] = src[src_slot_off + within];
    }
}

template <typename T, int vec_size>
static sycl::event pack_root_launch(sycl::queue& q,
                                    const void* send_buf,
                                    void* dst,
                                    size_t count,
                                    size_t chunk_offset_elts,
                                    size_t copy_count,
                                    int node_size,
                                    int r2r_size,
                                    const std::vector<sycl::event>& deps) {
    const size_t packed = copy_count / vec_size;
    const size_t tail = copy_count - packed * vec_size;
    const size_t per_slot = packed + tail;
    const size_t slots = static_cast<size_t>(node_size) * r2r_size;
    const size_t req_threads = slots * per_slot;

    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    const size_t kernel_size =
        ((req_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range(kernel_size, work_group_size);

    return q.submit([=](sycl::handler& h) {
        h.depends_on(deps);
        const T* src = static_cast<const T*>(send_buf);
        T* out = static_cast<T*>(dst);
        h.parallel_for(
            nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                pack_root_body<T, vec_size>(
                    src, out, copy_count, chunk_offset_elts, count,
                    node_size, r2r_size, it);
            });
    });
}

sycl::event pack_root_send_buf(sycl::queue& q,
                               const void* send_buf,
                               void* dst,
                               size_t count,
                               size_t chunk_offset_elts,
                               size_t copy_count,
                               ccl::datatype dtype,
                               int node_size,
                               int r2r_size,
                               const std::vector<sycl::event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    const bool base_aligned =
        can_use_full_vector(send_buf, dst, copy_count, dsize);

    auto lambda = [&]<typename T>() -> sycl::event {
        constexpr int full_vec = get_num_elements<T, 16, true>();
        constexpr int partial_vec = get_num_elements<T, 16, false>();

        // dst_slot_off = slot * copy_count. For slot >= 1 to have vector-aligned
        // stores, copy_count itself must be a multiple of full_vec.
        const bool full_ok = base_aligned && (count % full_vec == 0) &&
                             (chunk_offset_elts % full_vec == 0) &&
                             (copy_count % full_vec == 0);
        if (full_ok) {
            return pack_root_launch<T, full_vec>(
                q, send_buf, dst, count, chunk_offset_elts, copy_count,
                node_size, r2r_size, deps);
        }
        return pack_root_launch<T, partial_vec>(
            q, send_buf, dst, count, chunk_offset_elts, copy_count,
            node_size, r2r_size, deps);
    };
    return invoke_collective_sycl(lambda, dtype);
}

struct multi_node_layout {
    int node_size;
    int r2r_size;
    int root_node_idx;
    int root_local_idx;
    int this_local_idx;
    int this_node_idx;
    bool on_root_node;
    int r2r_root;
};

multi_node_layout compute_layout(ccl_comm* global_comm, int root) {
    ccl_comm* node_comm = global_comm->get_node_comm().get();
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();
    multi_node_layout l{};
    l.node_size = node_comm->size();
    l.r2r_size = r2r_comm->size();
    l.root_node_idx = root / l.node_size;
    l.root_local_idx = root - l.root_node_idx * l.node_size;

    const int global_rank = global_comm->rank();
    l.this_node_idx = global_rank / l.node_size;
    l.this_local_idx = global_rank - l.this_node_idx * l.node_size;
    l.on_root_node = (l.this_node_idx == l.root_node_idx);
    
    // In the same node-major numbering, r2r_comm[L] enumerates its
    // members in ascending node_idx order — so the r2r-comm rank whose
    // owner sits on root_node_idx is exactly root_node_idx.
    l.r2r_root = l.root_node_idx;
    return l;
}

// Effective RDMA usage: env-level opt-in AND per-device capability.
bool compute_use_rdma(sycl::queue& q) {
    if (!ccl::global_data::env().sycl_enable_direct_gpu_rdma) {
        return false;
    }
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    return !should_disable_rdma(ze_dev);
}

} // namespace

// -----------------------------------------------------------------------------
// Multi-node scatter — small-msg path (device scratch)
// -----------------------------------------------------------------------------
event scatter_sycl_multi_node_small_msg(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        datatype dtype,
                                        int root,
                                        ccl_comm* global_comm,
                                        ccl_stream* global_stream,
                                        const vector_class<event>& deps,
                                        bool& done) {
    // "scaleout first, then scaleup". Exploits
    // the node-major numbering — root's send_buf already has each
    // node's node_size*count payload laid out contiguously, so no pack
    // is required. 

    // Phase 1 (r2r_comm[root_local_idx] scatter, root=layout.r2r_root)
    // Phase 2 (node_comm scatter, root=layout.root_local_idx)
    // No chunking. If per-node slab exceeds one device scratch slot
    // the path falls back so the large-msg pack path can chunk it.

    done = true;
    event ev;

    auto ccl_dtype = global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    const auto layout = compute_layout(global_comm, root);
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();

    const bool is_node_root = (layout.this_local_idx == layout.root_local_idx);
    const size_t node_slab_bytes =
        static_cast<size_t>(layout.node_size) * count * dsize;


    const size_t scaleout_dev_buf_size = global_comm->get_scaleout_device_buf_size();
    const size_t scaleout_host_buf_size = global_comm->get_scaleout_host_buf_size();
    const size_t host_slab_bytes =
        static_cast<size_t>(layout.r2r_size) * node_slab_bytes;

    if (node_slab_bytes > scaleout_dev_buf_size) {
        LOG_INFO("scatter_sycl multi-node small: node slab (",
                 node_slab_bytes,
                 " B) exceeds device scratch slot (",
                 scaleout_dev_buf_size,
                 " B) -- falling back to large path");
        done = false;
        return ev;
    }

    // Compute effective RDMA before the host-slab guard so the pre-check
    // matches the condition scatter_scaleout_sycl will actually test. If
    // the pre-check disagreed, only node-root ranks would observe the
    // downstream done=false and non-node-roots would deadlock at Phase 2.
    const bool use_rdma = compute_use_rdma(q);
    const bool copy_to_host = !use_rdma;

    if (copy_to_host && host_slab_bytes > scaleout_host_buf_size) {
        LOG_INFO("scatter_sycl multi-node small: host slab (",
                 host_slab_bytes,
                 " B) exceeds host scratch slot (",
                 scaleout_host_buf_size,
                 " B) -- falling back to large path");
        done = false;
        return ev;
    }

    // Node-root ranks (one per node) acquire scratch — device for the
    // node's slab, host for staging when copy_to_host=true.
    void* scratch_dev = nullptr;
    void* scratch_host = nullptr;
    if (is_node_root) {
        scratch_dev = global_comm->get_scaleout_device_buf(q);
        if (copy_to_host) {
            scratch_host = global_comm->get_scaleout_host_buf();
        }
    }
    auto release_scratch = [&]() {
        if (scratch_dev) {
            global_comm->put_scaleout_device_buf(scratch_dev);
        }
        if (scratch_host) {
            global_comm->put_scaleout_host_buf(scratch_host);
        }
    };

    std::vector<sycl::event> initial_sycl_deps = get_sycl_events(deps);

    // -------- Phase 1: r2r scaleout from global root --------
    // Runs only on ranks with local_idx == root_local_idx (one per node).
    std::optional<sycl::event> scaleout_event;
    if (is_node_root) {
        vector_class<event> initial_deps;
        for (auto& se : initial_sycl_deps) {
            initial_deps.push_back(event::create_from_native(se));
        }

        bool sub_done = true;
        event scaleout_e = scatter_scaleout_sycl(
            q,
            (global_comm->rank() == root) ? send_buf : nullptr,
            scratch_dev,
            static_cast<size_t>(layout.node_size) * count, // per r2r peer
            dtype,
            layout.r2r_root,
            r2r_comm,
            initial_deps,
            sub_done,
            copy_to_host,
            /*is_cpu_buffers=*/false,
            /*aux_buf=*/scratch_host);
        if (!sub_done) {
            LOG_INFO("scatter_sycl multi-node small: r2r scaleout not done -- fallback");
            done = false;
            release_scratch();
            return scaleout_e;
        }
        scaleout_event = scaleout_e.get_native();
    }
    else if (!initial_sycl_deps.empty()) {
        scaleout_event = submit_wait_on_events(q, initial_sycl_deps);
    }

    // -------- Phase 2: intra-node scaleup on every node --------
    // Every node runs its own node_comm scatter with the local-root_local_idx
    // rank as sender. Each rank receives its `count` elements into recv_buf.
    vector_class<event> scaleup_deps;
    if (scaleout_event) {
        scaleup_deps.push_back(event::create_from_native(*scaleout_event));
    }

    bool sub_done = true;
    event scaleup_e = scatter_sycl_single_node(q,
                                               is_node_root ? scratch_dev : nullptr,
                                               recv_buf,
                                               count,
                                               dtype,
                                               layout.root_local_idx,
                                               global_comm,
                                               global_stream,
                                               scaleup_deps,
                                               sub_done);
    if (!sub_done) {
        LOG_INFO("scatter_sycl multi-node small: node_comm scaleup not done -- fallback");
        done = false;
        release_scratch();
        return scaleup_e;
    }

    if (scratch_host) {
        global_comm->put_scaleout_host_buf(scratch_host);
    }
    if (scratch_dev) {
        global_comm->put_scaleout_device_buf(scratch_dev);
    }
    return scaleup_e;
}

// -----------------------------------------------------------------------------
// Multi-node scatter — large-msg path (pack + chunked, optional overlap)
// -----------------------------------------------------------------------------

event scatter_sycl_multi_node_large_msg(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        datatype dtype,
                                        int root,
                                        ccl_comm* global_comm,
                                        ccl_stream* global_stream,
                                        const vector_class<event>& deps,
                                        bool& done) {
// Two-phase algorithm — "scaleup on root's node, then r2r scaleout":
//   Phase A (intra-node scatter on `node_comm`, root's node only)
//     Root repacks its send_buf into a local-idx-major slab. Root
//     then invokes scatter_sycl_single_node on `node_comm`
//
//   Phase B (inter-node scatter on `r2r_comm[local_idx]`)
//     On every node the local-local_idx rank enters r2r_comm[local_idx]
//     as one of its members then scatter_scaleout_sycl is invoked to complete
//     the inter-node scatter.

    done = true;
    event ev;

    auto ccl_dtype = global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    const auto layout = compute_layout(global_comm, root);
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();

    const bool use_rdma = compute_use_rdma(q);

    const bool overlap_enabled = ccl::global_data::env().sycl_scatter_scaleout_overlap;
    const size_t host_buf_size = global_comm->get_scaleout_host_buf_size();
    const size_t device_buf_size = global_comm->get_scaleout_device_buf_size();
    const size_t overlap_cap = ccl::global_data::env().sycl_scatter_scaleout_overlap_buf_size;

    const size_t per_element_slab_bytes =
        static_cast<size_t>(layout.node_size) * layout.r2r_size * dsize;

    size_t buf_cap = device_buf_size;
    if (!use_rdma) buf_cap = std::min(buf_cap, host_buf_size);
    if (overlap_enabled) buf_cap = std::min(buf_cap, overlap_cap);

    const size_t max_chunk_count =
        per_element_slab_bytes == 0
            ? count
            : std::min(count, buf_cap / per_element_slab_bytes);
    if (max_chunk_count == 0) {
        LOG_INFO("scatter_sycl multi-node large: scratch slot (",
                 buf_cap,
                 " B) too small for one packed element per rank -- falling back");
        done = false;
        return ev;
    }

    const size_t nchunks = (count + max_chunk_count - 1) / max_chunk_count;

    // Phase A runs on the caller's queue; phase B gets its own in-order
    // queue when overlap is enabled so its submissions pipeline with
    // phase A of the next chunk. Both paths rely on q being in-order to
    // serialize pack -> scaleup -> next pack without explicit deps.
    sycl::queue& q_a = q;
    sycl::queue q_b = overlap_enabled ? sycl::queue(q.get_context(),
                                                    q.get_device(),
                                                    sycl::property::queue::in_order{})
                                      : q;

    // Ping-pong scratch slots (fixes cross-chunk race).
    
    // scratch is always device USM (scaleup needs device memory + IPC-valid
    // pointers). host_aux is a pinned-host companion used only for scaleout
    // copy_to_host staging when RDMA is off; both are guarded by the same
    // last_scaleout event since they're touched by the same scaleout call.
    struct Slot {
        void* scratch = nullptr;
        void* host_aux = nullptr;
        std::optional<sycl::event> last_scaleout;
    };
    Slot slots[2];
    auto acquire = [&](int idx) -> Slot& {
        if (!slots[idx].scratch) {
            slots[idx].scratch = global_comm->get_scaleout_device_buf(q);
            if (!use_rdma) {
                slots[idx].host_aux = global_comm->get_scaleout_host_buf();
            }
        }
        return slots[idx];
    };
    auto release_all = [&]() {
        for (int i = 1; i >= 0; --i) {
            auto& s = slots[i];
            if (s.scratch) {
                global_comm->put_scaleout_device_buf(s.scratch);
                s.scratch = nullptr;
            }
            if (s.host_aux) {
                global_comm->put_scaleout_host_buf(s.host_aux);
                s.host_aux = nullptr;
            }
        }
    };

    std::vector<sycl::event> initial_sycl_deps = get_sycl_events(deps);

    std::optional<sycl::event> phase_scaleup_prev;   // scaleup[i-1] — chains next pack (q_a in-order)
    std::optional<sycl::event> phase_scaleout_prev;  // scaleout[i-1] — chains next scaleout when overlapping
    size_t chunk_offset = 0;

    for (size_t i = 0; i < nchunks; i++) {
        const size_t chunk_count = (i < nchunks - 1) ? max_chunk_count : count - chunk_offset;

        Slot& slot = acquire(i % 2);
        void* scratch = slot.scratch;

        // pack[i] deps:
        //   - initial deps (chunk 0 only — caller's events aren't on q_a)
        //   - scaleout that last read this slot — CRITICAL: crosses q_a→q_b,
        // No need to depend on phase_scaleup_prev: pack, scaleup, and the
        // non-root wait all submit to q_a (in-order), which serializes them.
        std::vector<sycl::event> pack_deps;
        if (i == 0) {
            pack_deps = initial_sycl_deps;
        }
        if (slot.last_scaleout) {
            pack_deps.push_back(*slot.last_scaleout);
        }

        if (global_comm->rank() == root) {
            phase_scaleup_prev = pack_root_send_buf(q_a,
                                                   send_buf,
                                                   scratch,
                                                   count,
                                                   chunk_offset,
                                                   chunk_count,
                                                   dtype,
                                                   layout.node_size,
                                                   layout.r2r_size,
                                                   pack_deps);
        }
        else if (!pack_deps.empty()) {
            phase_scaleup_prev = submit_wait_on_events(q_a, pack_deps);
        }

        // -------- Phase A: scaleup on root's node --------
        if (layout.on_root_node && layout.node_size > 1) {
            vector_class<event> scaleup_deps;
            if (phase_scaleup_prev) {
                scaleup_deps.push_back(event::create_from_native(*phase_scaleup_prev));
            }
            bool sub_done = true;
            event scaleup_e = scatter_sycl_single_node(
                q_a,
                (global_comm->rank() == root) ? scratch : nullptr,
                scratch,
                chunk_count * layout.r2r_size,
                dtype,
                layout.root_local_idx,
                global_comm,
                global_stream,
                scaleup_deps,
                sub_done);
            if (!sub_done) {
                LOG_INFO("scatter_sycl multi-node large: scaleup single-node not done -- fallback");
                done = false;
                release_all();
                return scaleup_e;
            }
            phase_scaleup_prev = scaleup_e.get_native();
        }

        // -------- Phase B: r2r scaleout on q_b --------
        vector_class<event> scaleout_deps;
        if (phase_scaleup_prev) {
            scaleout_deps.push_back(event::create_from_native(*phase_scaleup_prev));
        }
        // When overlapping, chain scaleout[i] on scaleout[i-1] so the r2r
        // library sees an ordered stream of collective calls on r2r_comm.
        // (Slot-reuse safety is enforced by slot.last_scaleout, not this.)
        if (overlap_enabled && phase_scaleout_prev) {
            scaleout_deps.push_back(event::create_from_native(*phase_scaleout_prev));
        }

        void* recv_slice = static_cast<uint8_t*>(recv_buf) + chunk_offset * dsize;
        bool sub_done = true;
        event scaleout_e = scatter_scaleout_sycl(
            q_b,
            layout.on_root_node ? scratch : nullptr,
            recv_slice,
            chunk_count,
            dtype,
            layout.r2r_root,
            r2r_comm,
            scaleout_deps,
            sub_done,
            /*copy_to_host=*/!use_rdma,
            /*is_cpu_buffers=*/false,
            /*aux_buf=*/slot.host_aux);
        if (!sub_done) {
            LOG_INFO("scatter_sycl multi-node large: r2r scaleout not done -- fallback");
            done = false;
            release_all();
            return scaleout_e;
        }

        phase_scaleout_prev = scaleout_e.get_native();
        slot.last_scaleout = phase_scaleout_prev;
        chunk_offset += chunk_count;
    }

    if (!phase_scaleout_prev) return ev;
    
    for (int i = 1; i >= 0; --i) {
        if (slots[i].host_aux) {
            global_comm->put_scaleout_host_buf(slots[i].host_aux);
        }
        if (slots[i].scratch) {
            global_comm->put_scaleout_device_buf(slots[i].scratch);
        }
    }
    return event::create_from_native(*phase_scaleout_prev);
}

// Preserve the existing single-node entry point by routing it through the
// domain-aware helper with node_comm as the p2p domain.
event scatter_sycl_single_node(sycl::queue& q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               datatype dtype,
                               int root,
                               ccl_comm* comm,
                               ccl_stream* global_stream,
                               const vector_class<event>& deps,
                               bool& done) {
    return scatter_sycl_single_domain(q,
                                      send_buf,
                                      recv_buf,
                                      count,
                                      dtype,
                                      root,
                                      comm,
                                      comm->get_node_comm(),
                                      global_stream,
                                      deps,
                                      done);
}

event scatter_sycl_multi_node(sycl::queue& q,
                              const void* send_buf,
                              void* recv_buf,
                              size_t count,
                              datatype dtype,
                              int root,
                              ccl_comm* global_comm,
                              ccl_stream* global_stream,
                              const vector_class<event>& deps,
                              bool& done) {
    auto ccl_dtype = global_data::get().dtypes->get(dtype);
    const size_t bytes = count * ccl_dtype.size();

    if (bytes > ccl::global_data::env().sycl_scatter_scaleout_threshold) {
        LOG_DEBUG("scatter_sycl multi-node bytes ",
                  bytes,
                  " > sycl_scatter_scaleout_threshold ",
                  ccl::global_data::env().sycl_scatter_scaleout_threshold,
                  " -- falling back to scheduler");
        done = false;
        event ev;
        return ev;
    }

    if (bytes <= ccl::global_data::env().sycl_scatter_scaleout_small_threshold) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "scatter_multi_node_small", "send_size", bytes, global_comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("scatter_sycl multi-node small path, count=", count);
        auto e = scatter_sycl_multi_node_small_msg(
            q, send_buf, recv_buf, count, dtype, root, global_comm, global_stream, deps, done);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        return e;
    }

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin(
        "scatter_multi_node_large", "send_size", bytes, global_comm->unique_id());
#endif // CCL_ENABLE_ITT
    LOG_DEBUG("scatter_sycl multi-node large path, count=", count);
    auto e = scatter_sycl_multi_node_large_msg(
        q, send_buf, recv_buf, count, dtype, root, global_comm, global_stream, deps, done);
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    return e;
}

event scatter_sycl(sycl::queue q,
                   const void* send_buf,
                   void* recv_buf,
                   size_t count,
                   datatype dtype,
                   int root,
                   ccl_comm* global_comm,
                   ccl_stream* global_stream,
                   const scatter_attr& attr,
                   const vector_class<event>& deps,
                   bool& done) {
    if (count == 0) {
        done = true;
        auto sycl_deps = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_deps);
        return event::create_from_native(e);
    }

    // Both scale-up kernels (small and large) implement only the simple
    // protocol -- IPC peer pointers plus device-side barriers. That protocol is
    // not reliable outside the ARC families: the other collectives switch to
    // their LL256 kernels there, but scatter has no LL256 path yet. Detect it up
    // front and fall back to the scheduler before anything is submitted -- a
    // done=false raised later would leave partial writes in recv_buf and hand
    // corrupted input to the fallback. is_arc_card() honours CCL_SYCL_FORCE_PCIE,
    // so the kernels can still be exercised on PVC for debugging.
    if (!is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()))) {
        LOG_DEBUG("|CCL_SYCL| scatter kernels require an ARC card, falling back to scheduler");
        done = false;
        event ev;
        return ev;
    }

    bool is_single_node = false;
    if (global_data::env().backend == backend_mode::native) {
        const topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node && global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("is_single_node");
        return scatter_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, root, global_comm, global_stream, deps, done);
    }

    return scatter_sycl_multi_node(
        q, send_buf, recv_buf, count, dtype, root, global_comm, global_stream, deps, done);
}

} // namespace v1
} // namespace ccl

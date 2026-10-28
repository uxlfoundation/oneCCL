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
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_scaleup.hpp"
#include "coll/algorithms/utils/barrier_data.hpp"  // use_recording_path

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/reduce/sycl/reduce_sycl.hpp"
#include "coll/algorithms/reduce/sycl/reduce_scaleout_sycl.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "coll/algorithms/utils/sycl_custom_preop.hpp"
#include "coll/algorithms/utils/transmit/ring_transmit.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

namespace ccl {
namespace v1 {

static bool do_fallback_to_scheduler(size_t size) {
#ifdef CCL_ENABLE_ESIMD
    const bool is_esimd_enabled = ccl::global_data::env().sycl_esimd;
#else
    const bool is_esimd_enabled = false;
#endif // CCL_ENABLE_ESIMD
    bool is_above_threshold = size > ccl::global_data::env().sycl_reduce_scaleout_threshold;
    bool exception_cases = is_esimd_enabled;
    return is_above_threshold || exception_cases;
}

namespace {

// world == 1 fast path: memcpy/wait-on-events, plus custom-reduction pre-op.
ccl::event reduce_scaleup_single_rank(const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      ccl::reduction reduction,
                                      ccl_comm* global_comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    sycl::event sycl_e;
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    auto sycl_q = global_stream->get_native_stream();
    if (send_buf != recv_buf) {
        LOG_DEBUG("single rank: out-of-place case, coll: reduce");
        sycl_e = sycl_q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
        });
    }
    else {
        LOG_DEBUG("single rank: inplace case, coll: reduce");
        sycl_e = submit_wait_on_events(sycl_q, dep_events);
    }
    if (ccl_reduction_type_storage::is_custom(reduction)) {
        ccl_reduction_data reduction_op = make_reduction_operation(reduction);
        auto lambda = [&]<typename T>() {
            sycl_e = pre_operation_invoke<T, 1, 32>(sycl_q, recv_buf, count, false, nullptr,
                                                    reduction_op,
                                                    global_comm->get_node_comm()->get_tmp_buf_size(),
                                                    { sycl_e });
            return ccl::event::create_from_native(sycl_e);
        };
        return invoke_collective(lambda, dtype);
    }
    return ccl::event::create_from_native(sycl_e);
}

// Arc LL256 ring, chunked. Only ring is supported today; fp64 on ARC 770 →
// hard-bail (done=false). Sets done=false if CCL_SYCL_REDUCE_LL is not "ring"
// or if any chunk fails.
ccl::event reduce_scaleup_ll256(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                int root,
                                ccl_comm* global_comm,
                                ccl_stream* global_stream,
                                bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    const size_t chunk_size = ccl::global_data::env().sycl_reduce_chunking_threshold;
    ccl::event e;
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin("reduce_ll_ring", "send_size",
                                  count * ccl_dtype.size(), global_comm->unique_id());
#endif // CCL_ENABLE_ITT
    size_t max_pack_count = chunk_size / dsize;
    size_t nchunks = calculate_chunking_pack_count(chunk_size, count, dsize, max_pack_count);
    size_t send_offset = 0;
    for (size_t iter = 0; iter < nchunks; iter++) {
        size_t pack_count = (iter < nchunks - 1) ? max_pack_count : count - send_offset;
        LOG_DEBUG("invoking reduce LL256 kernel reduce, count:", count,
                  " datatype: ", dtype, " algo: ",
                  ccl::global_data::env().sycl_reduce_ll_algo,
                  " iter: ", iter, " nchunks: ", nchunks);
        if (ccl::global_data::env().sycl_reduce_ll_algo == "ring") {
            e = reduce_ll<RingTransmit>((char*)send_buf + send_offset * dsize,
                                        (char*)recv_buf + send_offset * dsize,
                                        pack_count, dtype, root, reduction,
                                        global_comm, global_stream, done);
        }
        else {
            LOG_ERROR("CCL_SYCL_REDUCE_LL can only be ring");
            done = false;
        }
        if (!done) break;
        send_offset += pack_count;
    }
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    if (done) {
        LOG_DEBUG("invoking reduce LL256 kernel, count:", count,
                  " datatype: ", dtype, " done");
        return e;
    }
    // ARC 770 does not support fp64
    if (ccl::ze::get_device_family(global_stream->get_ze_device()) ==
            ccl::device_family::family6 &&
        dtype == ccl::datatype::float64) {
        LOG_DEBUG("arc_reduce does not support fp64");
        done = false;
    }
    return e;
}

}  // namespace

ccl::event reduce_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      ccl::reduction reduction,
                                      int root,
                                      ccl_comm* global_comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    done = true;

    if (global_comm->get_node_comm()->size() == 1) {
        return reduce_scaleup_single_rank(send_buf, recv_buf, count, dtype, reduction,
                                          global_comm, global_stream, deps);
    }

    const bool is_single_tile = global_comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected =
        global_comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| is_single_tile: ", is_single_tile,
              ", has_all_vertices_connected: ", has_all_vertices_connected);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const auto tune = ccl::sycl_policy_resolver::scaleup_resolver{
        ccl::sycl_policy::policy_query(
            ccl_coll_reduce, count * ccl_dtype.size(), ccl_dtype,
            /*r2r_size=*/0, global_comm->size(), use_recording_path(q))
            .with_comm(global_comm)
            .with_stream(global_stream)
    }.reduce();

    if (tune.protocol != ccl::sycl_policy::scaleup_protocol::ll) {
        // Non-Arc reduce scaleup is unsupported today; caller falls back.
        done = false;
        return {};
    }

    return reduce_scaleup_ll256(send_buf, recv_buf, count, dtype, reduction, root,
                                global_comm, global_stream, done);
}

ccl::event reduce_sycl_multi_node(sycl::queue& q,
                                  const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  ccl::reduction reduction,
                                  int root,
                                  ccl_comm* global_comm,
                                  ccl_stream* global_stream,
                                  const vector_class<ccl::event>& deps,
                                  bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_comm* node_comm = global_comm->get_node_comm().get();
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();
    int my_global_rank = global_comm->rank();

    done = true;

    // double check single node case
    if (r2r_comm->size() == 1) {
        LOG_DEBUG("reduce calls single node");
        return reduce_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, reduction, root, global_comm, global_stream, deps, done);
    }

    if (do_fallback_to_scheduler(count * ccl_dtype.size())) {
        LOG_DEBUG("reduce count size = ",
                  count * ccl_dtype.size(),
                  " is above scaleout SYCL threshold = ",
                  ccl::global_data::env().sycl_reduce_scaleout_threshold,
                  " or sycl::esimd mode is enabled -- falling back");
        done = false;
        return ev;
    }

    // The scale-up phase (reduce_sycl_single_node) only supports ARC GPUs; on
    // non-ARC it returns done=false. Detect that up front and fall back BEFORE
    // the scale-out phase runs -- otherwise scale-out would already stage into
    // recv_buf (and the custom pre-op would overwrite it), handing corrupted
    // input to the scheduler fallback, which is fatal for the in-place case.
    if (!is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()))) {
        LOG_DEBUG("reduce_sycl_multi_node: scale-up requires an ARC GPU; "
                  "falling back before scale-out");
        done = false;
        return ev;
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    std::vector<event> evs;
    for (auto& dep_event : dep_events) {
        ev = ccl::event::create_from_native(dep_event);
        evs.push_back(std::move(ev));
    }

    const int dsize = ccl_dtype.size();

    // Chunking: the scale-out staging (host) buffer bounds how much data can
    // cross the network at once. A custom pre-op narrows this further (below).
    size_t scaleout_buf_bytes = r2r_comm->get_scaleout_host_buf_size();

    // The scale-out (OFI) and scale-up transports perform a plain SUM and cannot
    // decode a custom reduction or perform the avg divide. So handle the special
    // reductions here and hand SUM to both phases:
    //  - avg:         reduce with SUM, then divide by the global rank count at the
    //                 end (only the root holds the final result).
    //  - pre_mul_sum: apply the per-rank scalar once up front, then reduce the
    //                 pre-scaled data with SUM.
    const bool reduce_has_pre_operation = ccl_reduction_type_storage::is_custom(reduction);
    ccl::reduction transport_reduction = reduction;
    // Device scratch borrowed for the custom pre-op; returned before any exit.
    void* preop_dev_buf = nullptr;
    if (reduction == ccl::reduction::avg) {
        transport_reduction = ccl::reduction::sum;
    }
    else if (reduce_has_pre_operation) {
        // For reduce, non-root ranks may pass an invalid/null recv_buf, so the
        // pre-op cannot prescale into recv_buf. Prescale into a borrowed device
        // scratch buffer instead (valid on every rank), then feed that to the
        // scale-out phase. The prescale is applied per chunk inside the loop below,
        // so only one chunk -- not the whole message -- has to fit the device
        // buffer: bound the chunk by the smaller of the host staging and device
        // scratch buffers (they default equal but are independent env vars).
        scaleout_buf_bytes =
            std::min(scaleout_buf_bytes, global_comm->get_scaleout_device_buf_size());

        // A chunk must hold at least one whole element. calculate_chunking_pack_count
        // rounds the chunk down to the dtype alignment, so anything below that yields
        // max_pack_count == 0 and a division by zero when it derives nchunks. Bail out
        // here instead -- before borrowing the scratch, so there is nothing to return.
        const size_t min_chunk_bytes = std::max<size_t>(4, dsize);
        if (scaleout_buf_bytes < min_chunk_bytes) {
            LOG_WARN_ONCE("reduce_sycl_multi_node: scaleout buffer = ",
                          scaleout_buf_bytes,
                          " bytes cannot hold one element of ",
                          min_chunk_bytes,
                          " bytes for the custom pre-op -- falling back");
            done = false;
            return ev;
        }

        // Borrow the scratch once here and reuse it each chunk; chunks are
        // serialized, so the reuse is safe.
        preop_dev_buf = global_comm->get_scaleout_device_buf(q);
        transport_reduction = ccl::reduction::sum;
    }

    // root_node_id is the root's index in the r2r dimension. It is compared
    // against r2r_comm->rank() below to decide which node runs the scale-up, and
    // passed as the scale-out (r2r) reduce root. This division assumes an equal
    // number of ranks per node; assert that precondition using the topology's
    // ppn check (same check can_use_sycl_kernels gates on).
    CCL_THROW_IF_NOT(global_comm->get_topo_manager().has_same_ppn(),
                     "reduce_sycl_multi_node assumes equal ranks per node");
    int root_node_id = root / node_comm->size();

    // Slice the message into chunks that each fit scaleout_buf_bytes (computed
    // above), and run both phases per chunk. For reduce this is a plain
    // byte-offset slice -- unlike reduce_scatter, the root simply receives
    // chunk 0, then chunk 1, ... so no per-rank data division is needed. This
    // also guarantees the scale-out size check never trips the fallback path.
    size_t max_pack_count = 0;
    size_t nchunks = calculate_chunking_pack_count(scaleout_buf_bytes, count, dsize, max_pack_count);

    LOG_DEBUG("reduce_sycl_multi_node: chunking count=",
              count,
              " into nchunks=",
              nchunks,
              " max_pack_count=",
              max_pack_count,
              " scaleout_buf_bytes=",
              scaleout_buf_bytes);

    size_t chunk_offset = 0; // in elements
    for (size_t iter = 0; iter < nchunks; iter++) {
        size_t chunk_count = (iter < nchunks - 1) ? max_pack_count : count - chunk_offset;
        const void* chunk_send = static_cast<const char*>(send_buf) + chunk_offset * dsize;
        void* chunk_recv = static_cast<char*>(recv_buf) + chunk_offset * dsize;

        // Custom pre-op: prescale this chunk into the device scratch and feed the
        // scaled data (not the raw input) to the scale-out phase. Done per chunk
        // so only one chunk needs to fit the scratch buffer.
        if (preop_dev_buf) {
            ev = stage_and_apply_custom_preop(q,
                                              preop_dev_buf,
                                              chunk_send,
                                              chunk_count,
                                              dtype,
                                              reduction,
                                              node_comm->get_tmp_buf_size(),
                                              get_sycl_events(evs));
            evs.clear();
            evs.push_back(std::move(ev));
            chunk_send = preop_dev_buf;
        }

        // Phase 1: Scale-out reduce across nodes using r2r_comm
        // Each rank reduces with its counterpart on other nodes
        // (rank i on each node forms its own r2r group)
        LOG_DEBUG("reduce_sycl_multi_node: Phase 1 - scale-out reduce using r2r_comm, root_node_id=",
                  root_node_id,
                  " iter=",
                  iter,
                  " chunk_count=",
                  chunk_count);

        ev = reduce_scaleout_sycl(q,
                                   chunk_send,
                                   chunk_recv,
                                   chunk_count,
                                   dtype,
                                   transport_reduction,
                                   root_node_id,
                                   r2r_comm,
                                   evs,
                                   true,  // original_deps
                                   done,
                                   false); // is_cpu_buffers

        if (!done) {
            LOG_INFO("reduce_sycl_multi_node: Phase 1 (scale-out) was not done -- falling back");
            if (preop_dev_buf) {
                global_comm->put_scaleout_device_buf(preop_dev_buf);
            }
            return ev;
        }

        // Phase 2: Scale-up reduce within the root node only
        // After Phase 1, each rank has reduced data from its counterparts
        // Now the root node needs to reduce locally to collect to the final root rank
        if (r2r_comm->rank() == root_node_id) {
            // This is the root node - do local reduce to collect to the final root rank.
            // root is relative to global_comm (the input communicator, which may be a
            // sub-communicator). get_rank_from_global expects a COMM_WORLD-absolute
            // rank, so bridge through get_global_rank first, then map into node_comm.
            int global_root = global_comm->get_global_rank(root);
            int local_root = node_comm->get_rank_from_global(global_root);
            LOG_DEBUG("reduce_sycl_multi_node: Phase 2 - scale-up reduce on root node, local_root=",
                      local_root,
                      " node_comm size=",
                      node_comm->size(),
                      " iter=",
                      iter);

            evs.clear();
            evs.push_back(std::move(ev));

            ev = reduce_sycl_single_node(q,
                                          chunk_recv,  // in-place
                                          chunk_recv,
                                          chunk_count,
                                          dtype,
                                          transport_reduction,
                                          local_root,
                                          node_comm,
                                          global_stream,
                                          evs,
                                          done);

            if (!done) {
                LOG_INFO("reduce_sycl_multi_node: Phase 2 (scale-up) was not done -- falling back");
                if (preop_dev_buf) {
                    global_comm->put_scaleout_device_buf(preop_dev_buf);
                }
                return ev;
            }
        }
        else {
            LOG_DEBUG("reduce_sycl_multi_node: non-root node, no Phase 2 needed, rank=",
                      my_global_rank);
        }

        // Serialize chunks: the next chunk depends on this chunk's completion so
        // the rotating scale-out staging buffers are safely reused. Keep `ev`
        // valid (copy the native event) so it still refers to the last chunk
        // after the loop.
        evs.clear();
        evs.push_back(ccl::event::create_from_native(ev.get_native()));
        chunk_offset += chunk_count;
    }

    // for avg: both phases summed the data; divide the final result by the global
    // rank count. Only the root rank holds the valid reduced buffer.
    if (reduction == ccl::reduction::avg && my_global_rank == root) {
        std::vector<sycl::event> avg_deps_evs{ ev.get_native() };
        int total_ranks = global_comm->size();
        LOG_DEBUG("reduce_sycl_multi_node: calculate average on counts: ",
                  count,
                  ", ranks: ",
                  total_ranks);
        sycl::event avg_e = sycl_average(q, recv_buf, count, total_ranks, dtype, avg_deps_evs);
        ev = ccl::event::create_from_native(avg_e);
    }

    // Return the borrowed pre-op scratch to the rotating pool. Same lifetime model
    // as the scale-out staging buffers: the pool has multiple slots so a later
    // borrow does not reuse this slot before the in-flight ops that read it drain.
    if (preop_dev_buf) {
        global_comm->put_scaleout_device_buf(preop_dev_buf);
    }

    return ev;
}

event reduce_sycl(sycl::queue q,
                     const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction reduction,
                      int root,
                     ccl_comm* global_comm,
                     ccl_stream* global_stream,
                     const reduce_attr& attr,
                     const vector_class<event>& deps,
                     bool& done) {
    done = true;
    ccl::event e;

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (count == 0) {
        auto sycl_deps = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_deps);
        return ccl::event::create_from_native(e);
    }

    if (is_single_node && ccl::global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("is_single_node");
        return reduce_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, reduction, root, global_comm, global_stream, deps, done);
    }

    event res =  reduce_sycl_multi_node(
        q, send_buf, recv_buf, count, dtype, reduction, root, global_comm, global_stream, deps, done);
    return res;
}

} // namespace v1
} // namespace ccl

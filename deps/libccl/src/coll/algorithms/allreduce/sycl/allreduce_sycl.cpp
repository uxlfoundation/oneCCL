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
#include "coll/algorithms/utils/sycl_policy_resolver_scaleup.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_scaleout.hpp"
#include "coll/algorithms/utils/barrier_data.hpp"  // use_recording_path
#include "coll/algorithms/utils/sycl_selection.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/allreduce/sycl/allreduce_sycl.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.hpp"
#include "coll/algorithms/broadcast/sycl/broadcast_sycl.hpp"
#include "allreduce_ring_ll256.hpp"
#include "coll/algorithms/utils/sycl_custom_preop.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "coll/algorithms/utils/transmit/ring_transmit.hpp"
#include "coll/algorithms/utils/transmit/twoshots_transmit.hpp"
#include "coll/algorithms/utils/transmit/oneshot_transmit.hpp"
#include "coll/selection/selector_impl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)


// Resolve the per-chunk LL transmit algorithm token for the SYCL allreduce
// scale-up path. CCL_SYCL_ALLREDUCE_LL picks one of
// "ring" / "twoshots" / "oneshot" per message-size range using the same
// "<algo>:<lo>-<hi>;..." syntax as the algorithm-selector framework
// (re-uses fill_table_from_str / get_value_from_table with std::string keys).
// When set to a single token, that algorithm is used for all sizes.
// When explicitly cleared, defaults to ring.
static std::string select_allreduce_scaleup_ll_algo(size_t bytes) {
    const std::string& spec = ccl::global_data::env().sycl_allreduce_ll_algo;
    if (spec.empty()) {
        return "ring";
    }

    // Build the table once on first call. The spec is parsed via the existing
    // selector framework (same syntax/semantics as CCL_OMP_ALLREDUCE_NUM_THREADS).
    static ccl_selection_table_t<std::string> table = [&]() {
        ccl_selection_table_t<std::string> t;
        fill_table_from_str(spec, t);
        return t;
    }();

    return ccl_algorithm_selector_base<std::string>::get_value_from_table(bytes, table);
}

namespace {

// world == 1 fast path: memcpy/wait-on-events, plus custom-reduction pre-op.
ccl::event allreduce_scaleup_single_rank(const void* send_buf,
                                         void* recv_buf,
                                         size_t count,
                                         ccl::datatype dtype,
                                         ccl::reduction reduction,
                                         std::shared_ptr<ccl_comm> p2p_domain_comm,
                                         ccl_stream* global_stream,
                                         const ccl::vector_class<ccl::event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    sycl::event sycl_e;
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    auto sycl_q = global_stream->get_native_stream();
    if (send_buf != recv_buf) {
        LOG_DEBUG("single rank: out-of-place case, coll: allreduce");
        sycl_e = sycl_q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
        });
    }
    else {
        LOG_DEBUG("single rank: inplace case, coll: allreduce");
        sycl_e = submit_wait_on_events(sycl_q, dep_events);
    }
    if (ccl_reduction_type_storage::is_custom(reduction)) {
        ccl_reduction_data reduction_op = make_reduction_operation(reduction);
        auto lambda = [&]<typename T>() {
            sycl_e = pre_operation_invoke<T, 1, 32>(sycl_q, recv_buf, count, false, nullptr,
                                                    reduction_op,
                                                    p2p_domain_comm->get_tmp_buf_size(),
                                                    { sycl_e });
            return ccl::event::create_from_native(sycl_e);
        };
        return invoke_collective(lambda, dtype);
    }
    return ccl::event::create_from_native(sycl_e);
}

// Arc family: LL256 chunked (when arc_allreduce is disabled) then, if LL256
// declines, arc_allreduce chunked. fp64 on ARC 770 → hard-bail (done=false).
// Returns unconditionally — this path never falls through to vec/esimd.
ccl::event allreduce_scaleup_arc_family(const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        ccl_comm* global_comm,
                                        std::shared_ptr<ccl_comm> p2p_domain_comm,
                                        ccl_stream* global_stream,
                                        const ccl::vector_class<ccl::event>& deps,
                                        bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    const size_t chunk_size = ccl::global_data::env().sycl_allreduce_chunking_threshold;
    ccl::event e;
    size_t max_pack_count = 0;
    if (!ccl::global_data::env().sycl_enable_arc_allreduce) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allreduce_ll_ring", "send_size",
                                      count * ccl_dtype.size(), global_comm->unique_id());
#endif // CCL_ENABLE_ITT
        size_t nchunks = calculate_chunking_pack_count(chunk_size, count, dsize, max_pack_count);
        size_t send_offset = 0;
        for (size_t iter = 0; iter < nchunks; iter++) {
            size_t pack_count = (iter < nchunks - 1) ? max_pack_count : count - send_offset;
            // Resolve LL transmit algorithm for this chunk via
            // CCL_SYCL_ALLREDUCE_LL (ring/twoshots/oneshot per byte range).
            const size_t chunk_bytes = pack_count * dsize;
            const std::string ll_algo = select_allreduce_scaleup_ll_algo(chunk_bytes);
            LOG_DEBUG("invoking allreduce LL256 kernel allreduce, count:", count,
                      " datatype: ", dtype, " algo: ", ll_algo, " iter: ", iter,
                      " nchunks: ", nchunks);
            if (ll_algo == "twoshots") {
                e = allreduce_ll<TwoShotsTransmit>((char*)send_buf + send_offset * dsize,
                                                   (char*)recv_buf + send_offset * dsize,
                                                   pack_count, dtype, reduction, global_comm,
                                                   p2p_domain_comm, global_stream, done);
            }
            else if (ll_algo == "oneshot" && chunk_bytes < (size_t)ONESHOT_MAX_MSGSIZE) {
                // In OneShot, the sender sub-group only sends data while the
                // receiver sub-group only receives, so the sender can overwrite
                // slots before the receiver reads them.
                // Safe limit: workSize <= nSlot * loopSize
                //   nSlot    = 8  (oneshot_transmit.hpp)
                //   loopSize = ipcbuf_size / (world * ringSize)
                // OneShot is designed for small messages, so it is hard-capped
                // at ONESHOT_MAX_MSGSIZE; above that it falls back to ring.
                // Use CCL_SYCL_ALLREDUCE_LL range syntax to tune where oneshot runs.
                e = allreduce_ll<OneShotTransmit>((char*)send_buf + send_offset * dsize,
                                                  (char*)recv_buf + send_offset * dsize,
                                                  pack_count, dtype, reduction, global_comm,
                                                  p2p_domain_comm, global_stream, done);
            }
            else if (ll_algo == "ring" || ll_algo == "oneshot") {
                // Ring is the default, also used as fallback when oneshot exceeds threshold
                e = allreduce_ll<RingTransmit>((char*)send_buf + send_offset * dsize,
                                               (char*)recv_buf + send_offset * dsize,
                                               pack_count, dtype, reduction, global_comm,
                                               p2p_domain_comm, global_stream, done);
            }
            else {
                LOG_ERROR("CCL_SYCL_ALLREDUCE_LL must be "
                          "ring, twoshots or oneshot, got: ", ll_algo);
                done = false;
            }
            if (!done) break;
            send_offset += pack_count;
        } // end for
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done) {
            LOG_DEBUG("invoking allreduce LL256 kernel, count:", count,
                      " datatype: ", dtype, " done");
            return e;
        }
    }
    // ARC 770 does not support fp64
    if (ccl::ze::get_device_family(global_stream->get_ze_device()) ==
            ccl::device_family::family6 &&
        dtype == ccl::datatype::float64) {
        LOG_DEBUG("arc_allreduce does not support fp64");
        done = false;
        return e;
    }
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin("arc_allreduce", "send_size",
                                  count * ccl_dtype.size(), global_comm->unique_id());
#endif // CCL_ENABLE_ITT
    done = true;
    // arc_ll256_allreduce uses a hardcoded sum kernel and ignores the actual `reduction`
    // (except for the avg post-pass), so it cannot apply the per-rank scalar for custom
    // (pre_mul_sum) reductions. Stage send_buf -> recv_buf with the scalar applied once
    // outside the chunked loop -- send_buf and recv_buf are both `count` elements, so a
    // single stage_and_apply_custom_preop call covers the entire range -- then run the
    // existing chunked loop with recv_buf as both source and destination, with SUM.
    const bool reduce_has_pre_operation = ccl_reduction_type_storage::is_custom(reduction);
    const void *transport_send_buf = send_buf;
    ccl::reduction transport_reduction = reduction;
    if (reduce_has_pre_operation) {
        sycl::queue q = global_stream->get_native_stream();
        (void)stage_and_apply_custom_preop(q, recv_buf, send_buf, count, dtype, reduction,
                                           p2p_domain_comm->get_tmp_buf_size(),
                                           get_sycl_events(deps));
        transport_send_buf = recv_buf;
        transport_reduction = ccl::reduction::sum;
    }
    size_t nchunks = calculate_chunking_pack_count(chunk_size, count, dsize, max_pack_count);
    size_t send_offset = 0;
    for (size_t iter = 0; iter < nchunks; iter++) {
        size_t pack_count = (iter < nchunks - 1) ? max_pack_count : count - send_offset;
        LOG_DEBUG("invoking allreduce LL256 kernel arc_allreduce, count:", count,
                  " datatype: ", dtype, " iter: ", iter, " nchunks: ", nchunks);
        e = arc_allreduce((char*)transport_send_buf + send_offset * dsize,
                          (char*)recv_buf + send_offset * dsize,
                          pack_count, dtype, transport_reduction,
                          global_comm, p2p_domain_comm, global_stream);
        LOG_DEBUG("invoking allreduce LL256 kernel arc_allreduce, count:", count,
                  " datatype: ", dtype, " done");
        send_offset += pack_count;
    }
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    return e;
}

// Non-ESIMD vec path: small kernel below threshold, else large kernel.
ccl::event allreduce_scaleup_default(const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 ccl::reduction reduction,
                                 ccl_comm* global_comm,
                                 std::shared_ptr<ccl_comm> p2p_domain_comm,
                                 ccl_stream* global_stream,
                                 const ccl::vector_class<ccl::event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl::event e;
    if (count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_small_threshold) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "allreduce_small", "send_size", count * ccl_dtype.size(), global_comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("invoking small allreduce kernel, count:", count, " datatype: ", dtype);
        e = allreduce_small(
            send_buf, recv_buf, count, dtype, reduction, global_comm, p2p_domain_comm, global_stream, deps);
        LOG_DEBUG("invoking small allreduce kernel, count:", count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "allreduce_large", "send_size", count * ccl_dtype.size(), global_comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("invoking large allreduce kernel, count:", count, " datatype: ", dtype);
        e = allreduce_large(
            send_buf, recv_buf, count, dtype, reduction, global_comm, p2p_domain_comm, global_stream, deps);
        LOG_DEBUG("invoking large allreduce kernel, count:", count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    return e;
}

#ifdef CCL_ENABLE_ESIMD
// ESIMD path: small (if in-threshold and all-vertices-connected), then medium
// (if not single-tile, or 2-rank without tmp buf), then large (if not
// single-tile); else done=false.
ccl::event allreduce_scaleup_esimd(sycl::queue& q,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   int rank,
                                   uint32_t world,
                                   ccl::datatype dtype,
                                   ccl::reduction reduction,
                                   ccl_comm* global_comm,
                                   std::shared_ptr<ccl_comm> p2p_domain_comm,
                                   ccl_stream* global_stream,
                                   bool is_single_tile,
                                   bool has_all_vertices_connected,
                                   const ccl::vector_class<ccl::event>& deps,
                                   bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl::event e;
    if (count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_small_threshold &&
        has_all_vertices_connected) {
        init_allreduce_small(dtype, q, p2p_domain_comm.get(), global_stream, rank, world);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "allreduce_small", "send_size", count * ccl_dtype.size(), global_comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allreduce selects small kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_small(dtype, q, send_buf, recv_buf, count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects small kernel, count:",
                  count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done) return e;
        // continue to medium kernel if not done
    }
    if ((count * ccl_dtype.size() <= ccl::global_data::env().sycl_allreduce_medium_threshold ||
         (p2p_domain_comm->size() == 2 && !ccl::global_data::env().sycl_allreduce_tmp_buf)) &&
        !is_single_tile) { // medium message sizes
        init_allreduce_medium(dtype, q, p2p_domain_comm.get(), global_stream, rank, world);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "allreduce_medium", "send_size", count * ccl_dtype.size(), global_comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allreduce selects medium kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_medium(dtype, q, send_buf, recv_buf, count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects medium kernel, count:",
                  count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        return e;
    }
    if (!is_single_tile) { // large message sizes
        init_allreduce_large(dtype, q, p2p_domain_comm.get(), global_stream, rank, world);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "allreduce_large", "send_size", count * ccl_dtype.size(), global_comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| allreduce selects large kernel, count:", count, " datatype: ", dtype);
        e = run_allreduce_large(dtype, q, send_buf, recv_buf, count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| allreduce selects large kernel, count:",
                  count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        return e;
    }
    done = false;
    return e;
}
#endif // CCL_ENABLE_ESIMD

}  // namespace

namespace ccl {
namespace v1 {

ccl::event allreduce_scaleup_sycl(sycl::queue& q,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   ccl::reduction reduction,
                                   ccl_comm* global_comm,
                                   std::shared_ptr<ccl_comm> p2p_domain_comm,
                                   ccl_stream* global_stream,
                                   const vector_class<event>& deps,
                                   bool& done) {
    done = true;

    if (p2p_domain_comm->size() == 1) {
        return allreduce_scaleup_single_rank(send_buf, recv_buf, count, dtype, reduction,
                                             p2p_domain_comm, global_stream, deps);
    }

    const uint32_t world = p2p_domain_comm->size();
    const int rank = p2p_domain_comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const bool is_single_tile = global_comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = p2p_domain_comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| is_single_tile: ", is_single_tile,
              ", has_all_vertices_connected: ", has_all_vertices_connected);

    const auto tune = ccl::sycl_policy_resolver::scaleup_resolver{
        ccl::sycl_policy::policy_query(
            ccl_coll_allreduce, count * ccl_dtype.size(), ccl_dtype,
            /*r2r_size=*/0, global_comm->size(), use_recording_path(q))
            .with_comm(global_comm)
            .with_p2p_comm(p2p_domain_comm.get())
            .with_stream(global_stream)
    }.allreduce();

    switch (tune.protocol) {
        case ccl::sycl_policy::scaleup_protocol::ll:
            return allreduce_scaleup_arc_family(send_buf, recv_buf, count, dtype, reduction,
                                                global_comm, p2p_domain_comm, global_stream,
                                                deps, done);
#ifdef CCL_ENABLE_ESIMD
        case ccl::sycl_policy::scaleup_protocol::esimd:
            return allreduce_scaleup_esimd(q, send_buf, recv_buf, count, rank, world, dtype,
                                           reduction, global_comm, p2p_domain_comm, global_stream,
                                           is_single_tile, has_all_vertices_connected,
                                           deps, done);
#endif
        case ccl::sycl_policy::scaleup_protocol::default_kernel:
        default:
            return allreduce_scaleup_default(send_buf, recv_buf, count, dtype, reduction,
                                         global_comm, p2p_domain_comm, global_stream, deps);
    }
}


static bool do_fallback_to_scheduler(size_t size) {
#ifdef CCL_ENABLE_ESIMD
    const bool is_esimd_enabled = ccl::global_data::env().sycl_esimd;
#else
    const bool is_esimd_enabled = false;
#endif // CCL_ENABLE_ESIMD
    bool is_above_threshold = size > ccl::global_data::env().sycl_allreduce_scaleout_threshold;
    bool exception_cases = is_esimd_enabled;
    return is_above_threshold || exception_cases;
}

bool is_rs_remainder_supported(size_t recv_count,
                               size_t rem_count,
                               size_t comm_size,
                               ccl_datatype ccl_dtype) {
    return rem_count != 0 && (recv_count * comm_size + rem_count) * ccl_dtype.size() <=
                                 ccl::global_data::env().sycl_reduce_scatter_small_threshold;
}

bool is_ag_remainder_supported(size_t send_count, size_t rem_count, ccl_datatype ccl_dtype) {
    return rem_count != 0 && (send_count + rem_count) * ccl_dtype.size() <=
                                 ccl::global_data::env().sycl_allgatherv_small_threshold;
}

ccl::event allreduce_sycl_multi_domain_rs_phase(sycl::queue& q,
                                              const void* send_buf,
                                              void* recv_buf,
                                              size_t recv_count,
                                              const void* remainder_send_buf,
                                              void* remainder_recv_buf,
                                              size_t remainder_count,
                                              ccl::datatype dtype,
                                              ccl::reduction reduction,
                                              ccl_comm* global_comm,
                                              std::shared_ptr<ccl_comm> p2p_comm,
                                              ccl_stream* global_stream,
                                              const vector_class<ccl::event>& deps,
                                              bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    if (is_rs_remainder_supported(recv_count, remainder_count, p2p_comm->size(), ccl_dtype)) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("reduce_scatter_small",
                                      "send_size",
                                      recv_count * p2p_comm->size() * ccl_dtype.size(),
                                      global_comm->unique_id());
#endif // CCL_ENABLE_ITT
        ev = reduce_scatter_small(send_buf,
                                  recv_buf,
                                  recv_count,
                                  remainder_count,
                                  dtype,
                                  reduction,
                                  global_comm,
                                  p2p_comm,
                                  global_stream,
                                  deps);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
        sycl_scaleup_options coll_attr{/*force_tmp=*/true};
        ev = reduce_scatter_scaleup_sycl(q,
                                         send_buf,
                                         recv_buf,
                                         recv_count,
                                         dtype,
                                         reduction,
                                         global_comm,
                                         p2p_comm,
                                         global_stream,
                                         deps,
                                         done,
                                         coll_attr);
        if (!done) {
            LOG_INFO("allreduce_sycl reduce_scatter was not done -- falling back");
            // fallback
            return ev;
        }

        // use allreduce to handle reduction for the remainder
        if (remainder_count) {
            std::vector<event> evs;
            evs.push_back(std::move(ev));

            ev = allreduce_scaleup_sycl(q,
                                        remainder_send_buf,
                                        remainder_recv_buf,
                                        remainder_count,
                                        dtype,
                                        reduction,
                                        global_comm,
                                        p2p_comm,
                                        global_stream,
                                        evs,
                                        done);

            if (!done) {
                LOG_INFO("allreduce_sycl reduce_scatter was not done -- falling back");
                // fallback
                return ev;
            }
        }
    }
    return ev;
}

ccl::event allreduce_sycl_multi_domain_ag_phase(sycl::queue& q,
                                                const void* send_buf,
                                                void* recv_buf,
                                                size_t send_count,
                                                const void* remainder_send_buf,
                                                void* remainder_recv_buf,
                                                size_t remainder_count,
                                                ccl::datatype dtype,
                                                ccl_comm* global_comm,
                                                std::shared_ptr<ccl_comm> p2p_comm,
                                                ccl_stream* global_stream,
                                                const vector_class<event>& deps,
                                                bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int last_p2p_comm_rank = p2p_comm->size() - 1;

    if (is_ag_remainder_supported(send_count, remainder_count, ccl_dtype)) {
        const size_t last_block_count = send_count + remainder_count;

        size_t ag_send_count =
            p2p_comm->rank() == last_p2p_comm_rank ? last_block_count : send_count;
        std::vector<size_t> recv_counts(p2p_comm->size() - 1, send_count);
        recv_counts.push_back(last_block_count);

#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("allgatherv_small",
                                      "send_size",
                                      ag_send_count * ccl_dtype.size(),
                                      p2p_comm->get_parent_comm()->unique_id());
#endif // CCL_ENABLE_ITT
        ev = allgatherv_small(q,
                              send_buf,
                              ag_send_count,
                              recv_buf,
                              recv_counts,
                              {},
                              dtype,
                              p2p_comm.get(),
                              global_stream,
                              deps,
                              done);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (!done) {
            LOG_INFO("allreduce_sycl allgatherv_small was not done -- falling back");
            // fallback
            return ev;
        }
    }
    else {
        std::vector<size_t> recv_counts(p2p_comm->size(), send_count);

        sycl_scaleup_options coll_attr{/*force_tmp=*/true, /*wait_deps=*/true};
        ev = allgather_scaleup_sycl(q,
                                    send_buf,
                                    send_count,
                                    recv_buf,
                                    recv_counts,
                                    {},
                                    dtype,
                                    global_comm,
                                    p2p_comm,
                                    global_stream,
                                    deps,
                                    done,
                                    coll_attr);
        if (!done) {
            // fallback
            LOG_INFO("allreduce_sycl allgatherv was not done -- falling back");
            return ev;
        }

        // last rank copies/broadcasts the remainder to other ranks
        if (remainder_count) {
            std::vector<event> rem_evs;
            rem_evs.push_back(std::move(ev));

            ev = broadcast_scaleup_sycl(q,
                                        remainder_recv_buf,
                                        remainder_recv_buf,
                                        remainder_count,
                                        dtype,
                                        last_p2p_comm_rank,
                                        global_comm,
                                        p2p_comm,
                                        global_stream,
                                        rem_evs,
                                        done);
        }
    }
    return ev;
}

ccl::event allreduce_sycl_multi_domain(sycl::queue& q,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       ccl::datatype dtype,
                                       ccl::reduction reduction,
                                       ccl_comm* global_comm,
                                       std::shared_ptr<ccl_comm> p2p_comm,
                                       std::shared_ptr<ccl_comm> r2r_comm,
                                       ccl_stream* global_stream,
                                       const vector_class<ccl::event>& deps,
                                       bool& done) {
    ccl::event ev;
    std::shared_ptr<ccl_comm> node_comm = global_comm->get_node_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();

    done = true;

    const int last_p2p_comm_rank = p2p_comm->size() - 1;

    // double check single node case
    if (r2r_comm->size() == 1) {
        LOG_DEBUG("allreduce calls single p2p domain");
        return allreduce_scaleup_sycl(
            q, send_buf, recv_buf, count, dtype, reduction, global_comm, p2p_comm, global_stream, deps, done);
    }

    if (do_fallback_to_scheduler(count * dsize)) {
        LOG_DEBUG("allreduce count size = ",
                  count * dsize,
                  " is above scaleout SYCL threshold = ",
                  ccl::global_data::env().sycl_allreduce_scaleout_threshold,
                  " or sycl::esimd mode is enabled, or other conditions not met -- falling back");
        done = false;
        return ev;
    }

    // for the scale-out case, use sum reduction to calculate the total sum,
    // then submit average kernel
    ccl::reduction ar_scaleup_reduction = reduction;
    if (reduction == ccl::reduction::avg) {
        ar_scaleup_reduction = ccl::reduction::sum;
    }
    // NOTE: for user-defined reduction, use sum for scale-out,
    // cause direct algorithm cannot decode correct reduction storage index yet.
    const bool reduce_has_pre_operation = ccl_reduction_type_storage::is_custom(reduction);
    ccl::reduction ar_scaleout_reduction = ar_scaleup_reduction;
    if (reduction != ccl::reduction::avg && reduce_has_pre_operation) {
        ar_scaleout_reduction = ccl::reduction::sum;
    }

    // TODO: Sycl allgatherv does not support counts that are non-divisible by the node_comm size.
    //       Once this support is enabled, the algorithm will be simplified.

    size_t line_size = ccl::global_data::env().sycl_kernels_line_size;
    CCL_THROW_IF_NOT(
        !(line_size % dsize), "datatype size not divisible by line_size=", line_size);

    size_t counts_per_line = line_size / dsize;

    const int buf_size = global_comm->get_scaleout_device_buf_size();
    size_t max_iter_count;
    if (count * dsize <= buf_size) {
        max_iter_count = count;
    }
    else {
        max_iter_count = buf_size / ccl_dtype.size();
        // Round the per-iteration count down to a whole number of lines *and* a whole
        // number of per-rank shares, so each chunk divides evenly across the node ranks
        // and stays line-size aligned.
        const size_t align = counts_per_line * p2p_comm->size();
        max_iter_count = (max_iter_count / align) * align;
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    std::vector<event> evs;
    for (int i = 0; i < dep_events.size(); i++) {
        ev = ccl::event::create_from_native(dep_events[i]);
        evs.push_back(std::move(ev));
    }

    // need a barrier between scale-up/out phases on PCIe systems
    // because both phases compete on the PCIe bandwidth and
    // interference with each other
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    const bool is_arc = is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()));

    size_t displ = 0, displ_count = 0;
    int nchunks = (count + max_iter_count - 1) / max_iter_count;
    for (int i = 0; i < nchunks; i++) {
        size_t iter_count = i < nchunks - 1 ? max_iter_count : count - displ_count;

        size_t counts_per_rank = iter_count / p2p_comm->size();
        size_t remainder_count = iter_count % p2p_comm->size();

        {
            size_t total_lines = iter_count / counts_per_line;
            remainder_count = iter_count % counts_per_line;

            size_t lines_per_rank = total_lines / p2p_comm->size();
            size_t remainder_lines = total_lines % p2p_comm->size();

            counts_per_rank = lines_per_rank * counts_per_line;
            remainder_count += remainder_lines * counts_per_line;

            CCL_THROW_IF_NOT(iter_count == remainder_count + (counts_per_rank * p2p_comm->size()),
                             "Incorrect calculations for lines_per_rank.");
        }

        LOG_DEBUG("allreduce_sycl count=",
                  iter_count,
                  " counts_per_rank=",
                  counts_per_rank,
                  " remainder_count=",
                  remainder_count);

        // single remainder case (which is the last chunk) is handled by the
        // direct MPI call due to performance reasons
        if (counts_per_rank == 0 && remainder_count) {
            CCL_ASSERT(i == nchunks - 1);
            LOG_DEBUG("using CPU-side algorithm for the remainder count=", remainder_count);
            auto direct_send_buffer = (char*)send_buf + displ;

            if (reduce_has_pre_operation) {
                ccl_reduction_data reduction_op = make_reduction_operation(reduction);
                sycl::event sycl_e;
                if (send_buf != recv_buf) {
                    sycl_e = q.submit([=](sycl::handler& h) {
                        h.depends_on(dep_events);
                        h.memcpy((char*)recv_buf + displ,
                                 (char*)send_buf + displ,
                                 iter_count * dsize);
                    });
                }
                auto lambda = [&]<typename T>() {
                    sycl_e = pre_operation_invoke<T, 1, 32>(q,
                                                            (char*)recv_buf + displ,
                                                            iter_count,
                                                            false,
                                                            nullptr,
                                                            reduction_op,
                                                            p2p_comm->get_tmp_buf_size(),
                                                            { sycl_e });

                    return ccl::event::create_from_native(sycl_e);
                };
                ev = invoke_collective(lambda, dtype);
                evs.clear();
                evs.push_back(std::move(ev));
                direct_send_buffer = (char*)recv_buf + displ;
            }

            ev = allreduce_scaleout_sycl(q,
                                         direct_send_buffer,
                                         (char*)recv_buf + displ,
                                         iter_count,
                                         dtype,
                                         ar_scaleout_reduction,
                                         global_comm,
                                         evs,
                                         true, /* original_deps */
                                         sycl_scaleout_tune_attr{ccl::sycl_policy::algorithm::direct}, /* tune_attr */
                                         done,
                                         false /*is_cpu_buffers*/);
            if (!done) {
                LOG_INFO("allreduce_sycl allreduce_scaleout_sycl for remainder count"
                         " was not done -- falling back");
                // fallback
                return ev;
            }

            if (reduction == ccl::reduction::avg) {
                // set dependencies
                std::vector<sycl::event> avg_deps_evs;
                avg_deps_evs.push_back(ev.get_native());
                // average divisor
                int total_ranks = global_comm->size();
                LOG_DEBUG("allreduce_sycl calculate average on counts: ",
                          count,
                          ", ranks: ",
                          total_ranks);

                sycl::event reduce_event = sycl_average(
                    q, (char*)recv_buf + displ, iter_count, total_ranks, dtype, avg_deps_evs);
                ev = ccl::event::create_from_native(reduce_event);
            }
            return ev;
        }

        bool need_barrier = false;
        if (is_arc && counts_per_rank * dsize >= 32 * 1024 * 1024)
            need_barrier = true;

        // prepare current node rank offset inside recv buffer
        const size_t recv_count_offset_bytes =
            p2p_comm->rank() * counts_per_rank * dsize;
        auto recv_rank_ptr = ptr_offset((char*)recv_buf + displ, recv_count_offset_bytes);

        // prepare buffers offset to handle the remainder
        const size_t remainder_offset_count = iter_count - remainder_count;
        const size_t remainder_offset_bytes = remainder_offset_count * dsize;
        auto remainder_send_buf = ptr_offset((char*)send_buf + displ, remainder_offset_bytes);
        auto remainder_recv_buf = ptr_offset((char*)recv_buf + displ, remainder_offset_bytes);

        // prepare scale-out data
        // last rank is a bit bigger to handle the remainder
        size_t scaleout_counts = p2p_comm->rank() == last_p2p_comm_rank
                                     ? counts_per_rank + remainder_count
                                     : counts_per_rank;

        //#define PRINT_TIMING

#ifdef PRINT_TIMING
        q.wait();
        cpu_timer<1> ctimer;
        int rank = global_comm->rank();
#endif // PRINT_TIMING

        // scale-up reduce-scatter phase
        if (p2p_comm->size() > 1) {
#ifdef PRINT_TIMING
            ctimer.start(0);
#endif // PRINT_TIMING
            ev = allreduce_sycl_multi_domain_rs_phase(q,
                                                      (char*)send_buf + displ,
                                                      recv_rank_ptr,
                                                      counts_per_rank,
                                                      remainder_send_buf,
                                                      remainder_recv_buf,
                                                      remainder_count,
                                                      dtype,
                                                      ar_scaleup_reduction,
                                                      global_comm,
                                                      p2p_comm,
                                                      global_stream,
                                                      evs,
                                                      done);
            if (!done) {
                LOG_INFO("allreduce_sycl reduce_scatter phase was not done -- falling back");
                // fallback
                return ev;
            }
            if (need_barrier)
                invoke_barrier(node_comm, q, {}, is_cpu_barrier);
#ifdef PRINT_TIMING
            ev.wait();
            q.wait(); // for multiple kernels with out-of-order queue
            ctimer.stop(0);
            fprintf(
                stderr,
                "[%d] allreduce_sycl_multi_node_rs_phase takes %f us on data: counts_per_rank: %ld displ: %ld\n",
                rank,
                ctimer.get_us(0),
                counts_per_rank,
                displ);
#endif // PRINT_TIMING

            evs.clear();
            evs.push_back(std::move(ev));
        }
        else if (reduce_has_pre_operation) {
            ccl_reduction_data reduction_op = make_reduction_operation(reduction);
            std::vector<sycl::event> sycl_evs = get_sycl_events(evs);
            sycl::event sycl_e;
            if (send_buf != recv_buf) {
                sycl_e = q.submit([=](sycl::handler& h) {
                    h.depends_on(sycl_evs);
                    h.memcpy((char*)recv_buf + displ,
                             (char*)send_buf + displ,
                             scaleout_counts * dsize);
                });
            }

            auto lambda = [&]<typename T>() {
                sycl_e = pre_operation_invoke<T, 1, 32>(q,
                                                        (char*)recv_buf + displ,
                                                        scaleout_counts,
                                                        false,
                                                        nullptr,
                                                        reduction_op,
                                                        p2p_comm->get_tmp_buf_size(),
                                                        { sycl_e });

                return ccl::event::create_from_native(sycl_e);
            };
            ev = invoke_collective(lambda, dtype);
            evs.clear();
            evs.push_back(std::move(ev));
        }

        // scaleout allreduce phase
        {
            sycl_scaleout_tune_attr scaleout_tune_attr = ccl::sycl_policy_resolver::scaleout_resolver{
                ccl::sycl_policy::policy_query(ccl_coll_allreduce, counts_per_rank * dsize, ccl_dtype,
                    r2r_comm->size(), global_comm->size(), use_recording_path(q))
            }.allreduce();
            LOG_DEBUG("allreduce_sycl scaleout count: ",
                      count,
                      " and scaleout_count: ",
                      scaleout_counts,
                      " and pipeline_chunk_size: ",
                      scaleout_tune_attr.pipeline_chunk_size(),
                      " and #chunk: ",
                      i,
                      " of nchunks: ",
                      nchunks,
                      " p2p_comm: ",
                      p2p_comm->size(),
                      " - allreduce_scaleout_sycl");

            void* scaleout_send_ptr;
            if (p2p_comm->size() > 1) {
                scaleout_send_ptr = recv_rank_ptr;
            }
            else if (reduce_has_pre_operation) {
                scaleout_send_ptr = (char*)recv_buf + displ;
            }
            else {
                scaleout_send_ptr = (char*)send_buf + displ;
            }
            bool original_deps = p2p_comm->size() == 1 ? (evs.size() == 0) : false;
#ifdef PRINT_TIMING
            ctimer.start(0);
#endif // PRINT_TIMING
            ev = allreduce_scaleout_sycl(q,
                                         scaleout_send_ptr,
                                         recv_rank_ptr,
                                         scaleout_counts,
                                         dtype,
                                         ar_scaleout_reduction,
                                         r2r_comm.get(),
                                         evs,
                                         original_deps,
                                         scaleout_tune_attr,
                                         done,
                                         false /*is_cpu_buffers*/);
            if (!done) {
                LOG_INFO("allreduce_sycl scaleout was not done -- falling back");
                return ev;
            }
            if (need_barrier)
                invoke_barrier(node_comm, q, {}, is_cpu_barrier);
#ifdef PRINT_TIMING
            ev.wait();
            q.wait(); // for multiple kernels with out-of-order queue
            ctimer.stop(0);
            fprintf(stderr,
                    "[%d] allreduce_scaleout_sycl takes %f us on data: scaleout_counts: %ld barrier: %d\n",
                    rank,
                    ctimer.get_us(0),
                    scaleout_counts,
                    need_barrier);
#endif // PRINT_TIMING

            if (reduction == ccl::reduction::avg) {
                // set dependencies
                std::vector<sycl::event> avg_deps_evs;
                avg_deps_evs.push_back(ev.get_native());
                // average divisor
                int total_ranks = global_comm->size();
                LOG_DEBUG("allreduce_sycl calculate average on counts: ",
                          scaleout_counts,
                          ", ranks: ",
                          total_ranks);

                sycl::event reduce_event = sycl_average(
                    q, recv_rank_ptr, scaleout_counts, total_ranks, dtype, avg_deps_evs);
                ev = ccl::event::create_from_native(reduce_event);
            }
        }

        if (p2p_comm->size() > 1) {
            evs.clear();
            evs.push_back(std::move(ev));

#ifdef PRINT_TIMING
            ctimer.start(0);
#endif // PRINT_TIMING \
    // scale-up allgatherv phase
            ev = allreduce_sycl_multi_domain_ag_phase(q,
                                                      recv_rank_ptr,
                                                      (char*)recv_buf + displ,
                                                      counts_per_rank,
                                                      remainder_send_buf,
                                                      remainder_recv_buf,
                                                      remainder_count,
                                                      dtype,
                                                      global_comm,
                                                      p2p_comm,
                                                      global_stream,
                                                      evs,
                                                      done);
            if (!done) {
                // fallback
                LOG_INFO("allreduce_sycl allgatherv phase was not done -- falling back");
                return ev;
            }
            if (need_barrier)
                invoke_barrier(node_comm, q, {}, is_cpu_barrier);
#ifdef PRINT_TIMING
            ev.wait();
            q.wait(); // for multiple kernels with out-of-order queue
            ctimer.stop(0);
            fprintf(
                stderr,
                "[%d] allreduce_sycl_multi_node_ag_phase takes %f us on data: counts_per_rank: %ld \n",
                rank,
                ctimer.get_us(0),
                counts_per_rank);
#endif // PRINT_TIMING
        }

        if (i < nchunks - 1) {
            evs.clear();
            evs.push_back(std::move(ev));
            displ_count += iter_count;
            displ += iter_count * dsize;
        }
    } // for chunking

    return ev;
}

event allreduce_sycl(sycl::queue q,
                     const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction reduction,
                     ccl_comm* global_comm,
                     ccl_stream* global_stream,
                     const allreduce_attr& attr,
                     const vector_class<event>& deps,
                     bool& done) {
    done = true;

    if (count == 0) {
        auto sycl_deps = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_deps);
        return ccl::event::create_from_native(e);
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    auto p2p_level = ccl_p2p_domain::sycl_get_allreduce_p2p_level(global_comm, count * ccl_dtype.size(), ccl::global_data::env().sycl_p2p_level_allreduce_threshold);

    auto [ p2p_comm, r2r_comm ] = ccl_p2p_domain::get_p2p_comms(global_comm, p2p_level);

    // single p2p domain
    if (r2r_comm->is_uniform && r2r_comm->size() == 1) {
        LOG_DEBUG("is_single_domain");
        return allreduce_scaleup_sycl(q, send_buf, recv_buf, count, dtype, reduction, global_comm, p2p_comm, global_stream, deps, done);
    }
 
    return allreduce_sycl_multi_domain(
        q, send_buf, recv_buf, count, dtype, reduction, global_comm, p2p_comm, r2r_comm, global_stream, deps, done);
}

} // namespace v1
} // namespace ccl

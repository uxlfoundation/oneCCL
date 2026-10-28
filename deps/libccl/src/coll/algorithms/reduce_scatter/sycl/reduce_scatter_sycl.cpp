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
#include "common/api_wrapper/mpi_api_wrapper.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_sycl.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/algorithms/utils/transmit/ring_transmit.hpp"
#include "coll/algorithms/utils/transmit/oneshot_transmit.hpp"
#include "coll/selection/selector_impl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

// Resolve the per-message-size LL transmit algorithm token for the SYCL
// reduce-scatter scale-up path. CCL_SYCL_REDUCE_SCATTER_LL picks one of
// "ring" / "oneshot" per message-size range using the same
// "<algo>:<lo>-<hi>;..." syntax as the algorithm-selector framework
// (re-uses fill_table_from_str / get_value_from_table with std::string keys).
// When set to a single token, that algorithm is used for all sizes.
// When explicitly cleared, defaults to ring.
static std::string select_reduce_scatter_scaleup_ll_algo(size_t bytes) {
    const std::string& spec = ccl::global_data::env().sycl_reduce_scatter_ll_algo;
    if (spec.empty()) {
        return "ring";
    }

    static ccl_selection_table_t<std::string> table = [&]() {
        ccl_selection_table_t<std::string> t;
        fill_table_from_str(spec, t);
        return t;
    }();

    return ccl_algorithm_selector_base<std::string>::get_value_from_table(bytes, table);
}

namespace {

// world == 1 fast path: memcpy/wait-on-events, plus custom-reduction pre-op.
ccl::event reduce_scatter_scaleup_single_rank(sycl::queue& /*q*/,
                                              const void* send_buf,
                                              void* recv_buf,
                                              size_t recv_count,
                                              ccl::datatype dtype,
                                              ccl::reduction reduction,
                                              const std::shared_ptr<ccl_comm> p2p_comm,
                                              ccl_stream* global_stream,
                                              const ccl::vector_class<ccl::event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    sycl::event sycl_e;
    auto sycl_q = global_stream->get_native_stream();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    if (send_buf != recv_buf) {
        LOG_DEBUG("single rank: out-of-place case, coll: reduce_scatter");
        sycl_e = sycl_q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, recv_count * ccl_dtype.size());
        });
    }
    else {
        LOG_DEBUG("single rank: inplace case, coll: reduce_scatter");
        sycl_e = submit_wait_on_events(sycl_q, dep_events);
    }
    if (ccl_reduction_type_storage::is_custom(reduction)) {
        ccl_reduction_data reduction_op = make_reduction_operation(reduction);
        auto lambda = [&]<typename T>() {
            sycl_e = pre_operation_invoke<T, 1, 32>(sycl_q, recv_buf, recv_count, false, nullptr,
                                                    reduction_op, p2p_comm->get_tmp_buf_size(),
                                                    { sycl_e });
            return ccl::event::create_from_native(sycl_e);
        };
        return invoke_collective(lambda, dtype);
    }
    return ccl::event::create_from_native(sycl_e);
}

// Arc LL256 path: single-shot; per-msg selects `oneshot` vs `ring` from
// CCL_SYCL_REDUCE_SCATTER_LL via `select_reduce_scatter_scaleup_ll_algo`.
ccl::event reduce_scatter_scaleup_ll256(const void* send_buf,
                                        void* recv_buf,
                                        size_t recv_count,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        ccl_comm* comm,
                                        const std::shared_ptr<ccl_comm> p2p_comm,
                                        ccl_stream* global_stream,
                                        bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl::event e;
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin(
        "reduce_scatter_ll", "recv_size", recv_count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
    LOG_DEBUG("invoking reduce_scatter LL kernel reduce_scatter_ll, recv_count:",
              recv_count, " datatype: ", dtype);
    const size_t msg_bytes = recv_count * ccl_dtype.size();
    const std::string ll_algo = select_reduce_scatter_scaleup_ll_algo(msg_bytes);
    if (ll_algo == "oneshot" && msg_bytes < (size_t)ONESHOT_MAX_MSGSIZE) {
        // OneShot does the scatter+reduce in a single shot (minimum latency);
        // hard-capped at ONESHOT_MAX_MSGSIZE, above which it falls back to ring.
        e = reduce_scatter_ll<OneShotTransmit>(
            send_buf, recv_buf, recv_count, dtype, reduction, comm, p2p_comm, global_stream, done);
    }
    else if (ll_algo == "ring" || ll_algo == "oneshot") {
        // Ring is the default, also used as fallback when oneshot exceeds threshold.
        e = reduce_scatter_ll<RingTransmit>(
            send_buf, recv_buf, recv_count, dtype, reduction, comm, p2p_comm, global_stream, done);
    }
    else {
        LOG_ERROR("CCL_SYCL_REDUCE_SCATTER_LL must be ring or oneshot, got: ", ll_algo);
        done = false;
    }
    LOG_DEBUG("invoking reduce_scatter LL kernel, recv_count:", recv_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    return e;
}

// Non-ESIMD vec path: small kernel below threshold, else large kernel.
ccl::event reduce_scatter_scaleup_default(sycl::queue& /*q*/,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t recv_count,
                                      uint32_t world,
                                      ccl::datatype dtype,
                                      ccl::reduction reduction,
                                      ccl_comm* comm,
                                      const std::shared_ptr<ccl_comm> p2p_comm,
                                      ccl_stream* global_stream,
                                      const ccl::vector_class<ccl::event>& deps,
                                      sycl_scaleup_options coll_attr) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl::event e;
    if (recv_count * world * ccl_dtype.size() <= ccl::global_data::env().sycl_reduce_scatter_small_threshold) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "reduce_scatter_small", "send_size", recv_count * world * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("invoking small reduce_scatter: recv_count:", recv_count, " datatype: ", dtype);
        e = reduce_scatter_small(
            send_buf, recv_buf, recv_count, 0 /* rem_count */, dtype, reduction, comm, p2p_comm, global_stream, deps);
        LOG_DEBUG("invoking small reduce_scatter: recv_count:", recv_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "reduce_scatter_large", "send_size", recv_count * world * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("invoking large reduce_scatter: recv_count:", recv_count, " datatype: ", dtype);
        e = reduce_scatter_large(
            send_buf, recv_buf, recv_count, dtype, reduction, comm, p2p_comm, global_stream, deps, coll_attr);
        LOG_DEBUG("invoking large reduce_scatter: recv_count:", recv_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    return e;
}

#ifdef CCL_ENABLE_ESIMD
// ESIMD path: small (if in-threshold and all-vertices-connected), then medium
// (if not single-tile), then large (if not single-tile); else done=false.
ccl::event reduce_scatter_scaleup_esimd(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t recv_count,
                                        int rank,
                                        uint32_t world,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        const std::shared_ptr<ccl_comm> p2p_comm,
                                        ccl_comm* comm,
                                        ccl_stream* global_stream,
                                        bool is_single_tile,
                                        bool has_all_vertices_connected,
                                        const ccl::vector_class<ccl::event>& deps,
                                        bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl::event e;
    if (recv_count * world * ccl_dtype.size() <= ccl::global_data::env().sycl_reduce_scatter_small_threshold &&
        has_all_vertices_connected) {
        init_reduce_scatter_small(dtype, q, p2p_comm, global_stream, rank, world);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "reduce_scatter_small", "send_size", recv_count * world * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects small kernel, recv_count:", recv_count, " datatype: ", dtype);
        e = run_reduce_scatter_small(dtype, q, send_buf, recv_buf, recv_count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects small kernel, recv_count:",
                  recv_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done) return e;
    }
    if (recv_count * world * ccl_dtype.size() <= ccl::global_data::env().sycl_reduce_scatter_medium_threshold &&
        !is_single_tile) {
        init_reduce_scatter_medium(dtype, q, p2p_comm, global_stream, rank, world);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "reduce_scatter_medium", "send_size", recv_count * world * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects medium kernel: count:", recv_count, " datatype: ", dtype);
        e = run_reduce_scatter_medium(dtype, q, send_buf, recv_buf, recv_count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects medium kernel: count:",
                  recv_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        return e;
    }
    if (!is_single_tile) {
        init_reduce_scatter_large(dtype, q, p2p_comm, global_stream, rank, world);
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "reduce_scatter_large", "send_size", recv_count * world * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects large kernel: count:", recv_count, " datatype: ", dtype);
        e = run_reduce_scatter_large(dtype, q, send_buf, recv_buf, recv_count, reduction, deps, done);
        LOG_DEBUG("|CCL_SYCL| reduce_scatter selects large kernel: count:",
                  recv_count, " datatype: ", dtype, " done");
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

ccl::event reduce_scatter_scaleup_sycl(sycl::queue& q,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t recv_count,
                                       datatype dtype,
                                       reduction reduction,
                                       ccl_comm* comm,
                                       const std::shared_ptr<ccl_comm> p2p_comm,
                                       ccl_stream* global_stream,
                                       const vector_class<event>& deps,
                                       bool& done,
                                       sycl_scaleup_options coll_attr) {
    ccl::event e;
    done = true;

    // Fast path: no scaleup needed when the local p2p domain has one rank.
    if (p2p_comm->size() == 1) {
        return reduce_scatter_scaleup_single_rank(q, send_buf, recv_buf, recv_count, dtype,
                                                  reduction, p2p_comm, global_stream, deps);
    }

    const uint32_t world = p2p_comm->size();
    const int rank = p2p_comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected",
              comm->get_topo_manager().has_all_vertices_connected());

    // Threshold for the LL256 gate uses total bytes across the local domain.
    const auto tune = ccl::sycl_policy_resolver::scaleup_resolver{
        ccl::sycl_policy::policy_query(
            ccl_coll_reduce_scatter, recv_count * world * ccl_dtype.size(), ccl_dtype,
            /*r2r_size=*/0, comm->size(), use_recording_path(q))
            .with_comm(comm)
            .with_p2p_comm(p2p_comm.get())
            .with_stream(global_stream)
    }.reduce_scatter();

    switch (tune.protocol) {
        case ccl::sycl_policy::scaleup_protocol::ll:
            return reduce_scatter_scaleup_ll256(send_buf, recv_buf, recv_count, dtype, reduction,
                                                comm, p2p_comm, global_stream, done);
#ifdef CCL_ENABLE_ESIMD
        case ccl::sycl_policy::scaleup_protocol::esimd:
            return reduce_scatter_scaleup_esimd(q, send_buf, recv_buf, recv_count, rank, world,
                                                dtype, reduction, p2p_comm, comm, global_stream,
                                                comm->get_pair_comm()->size() == 1,
                                                comm->get_topo_manager().has_all_vertices_connected(),
                                                deps, done);
#endif
        case ccl::sycl_policy::scaleup_protocol::default_kernel:
        default:
            return reduce_scatter_scaleup_default(q, send_buf, recv_buf, recv_count, world, dtype,
                                              reduction, comm, p2p_comm, global_stream, deps, coll_attr);
    }
}

// calculate the location to copy from for a given index
static inline int pred_loc(int idx, int N, int M) {
    return idx == M * N - 1 ? idx : (M * idx) % (M * N - 1);
}

template <typename T>
inline void copy_kernel(void* out, const void* in) {
    *(T*)out = *(T*)in;
}

template <typename T, int vec_size>
void inline copy_vec(const void* in,
                     void* out,
                     int out_idx,
                     int b_idx,
                     int in_idx,
                     size_t recv_count,
                     size_t displ,
                     size_t block_count) {
    using AT = sycl::vec<T, vec_size>;
    const size_t packed_count = recv_count / vec_size;
    if (b_idx < packed_count) {
        int offset = b_idx * vec_size;
        T* in_ptr = (T*)in + in_idx * block_count + displ + offset;
        T* out_ptr = (T*)out + out_idx * recv_count + offset;
        copy_kernel<AT>(out_ptr, in_ptr);
    }
    else {
        int offset = vec_size * packed_count + (b_idx - packed_count);
        T* in_ptr = (T*)in + in_idx * block_count + displ + offset;
        T* out_ptr = (T*)out + out_idx * recv_count + offset;
        if (offset < recv_count) {
            *out_ptr = *in_ptr;
        }
    }
}

// single kernel with sycl::vec
template <typename T, int vec_size, int SGS>
class oneccl_transposeT {};

template <typename T, int vec_size, int SGS>
sycl::event transposeT(sycl::queue& q,
                       const void* send_buf,
                       void* out_buf,
                       size_t recv_count,
                       size_t displ,
                       size_t block_count,
                       datatype dtype,
                       int nodes,
                       int ppn,
                       std::vector<sycl::event>& dep_events) {
    sycl::event e;
    constexpr int wg_size = SGS;
    constexpr int sg_size = SGS;
    size_t vec_count = recv_count / vec_size + recv_count % vec_size;
    int kernel_threads = vec_count * nodes * ppn;
    int kernel_size = (kernel_threads + wg_size - 1) / wg_size * wg_size;
    e = q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);
        h.parallel_for<oneccl_transposeT<T, vec_size, SGS>>(
            sycl::nd_range<1>(kernel_size, wg_size), [=](sycl::nd_item<1> it) {
                const size_t idx = it.get_global_linear_id();
                if (idx >= kernel_threads)
                    return;
                int a = idx / vec_count;
                int b = idx % vec_count;
                int a_from = pred_loc(a, nodes, ppn);
                copy_vec<T, vec_size>(send_buf, out_buf, a, b, a_from, recv_count, displ, block_count);
            });
    });
    return e;
}

// multiple sycl kernel version
// support chunking, pack into contiguous output buffer
static sycl::event transpose(sycl::queue& q,
                             const void* send_buf,
                             void* out_buf,
                             size_t recv_count,
                             size_t displ,
                             size_t block_count,
                             datatype dtype,
                             int nodes,
                             int ppn,
                             std::vector<sycl::event>& dep_events) {
    sycl::event e;

/* almost same performance */
#define USE_OOO_QUEUE 0
#if USE_OOO_QUEUE
    // out of order queue
    static sycl::queue q_worker(q.get_context(), q.get_device());
    std::vector<sycl::event> evs;
#else
    sycl::queue q_worker = q;
#endif // USE_OOO_QUEUE

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t chunk_size = recv_count * ccl_dtype.size();
    size_t block_size = block_count * ccl_dtype.size();
    size_t displ_size = displ * ccl_dtype.size();
    int n = 0;
    for (int i = 0; i < ppn; i++) {
        for (int j = 0; j < nodes; j++) {
            char* src = (char*)send_buf + (j * ppn + i) * block_size + displ_size;
            char* dest = (char*)out_buf + n * chunk_size;
            n++;
            e = q_worker.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(dest, src, chunk_size);
            });
#if USE_OOO_QUEUE
            evs.push_back(std::move(e));
#endif // USE_OOO_QUEUE
        }
    }
#if USE_OOO_QUEUE
    e = submit_wait_on_events(q, evs);
#endif // USE_OOO_QUEUE
    return e;
}

static sycl::event rearrange(sycl::queue& q,
                             const void* send_buf,
                             void* staging_buf,
                             size_t recv_count, /* actual recv count */
                             size_t displ, /* offset */
                             size_t block_count, /* block count, or stride for input buffer */
                             datatype dtype,
                             int nodes,
                             int ppn,
                             std::vector<sycl::event>& dep_events) {
    sycl::event e;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t total_size = recv_count * ccl_dtype.size() * nodes * ppn;

    if (total_size <= 1073741824) {
        // single sycl kernel only works on the full buffer
        bool align4 = is_aligned(send_buf, recv_count, ccl_dtype.size(), 4);
        auto lambda = [&]<typename T>() {
            if (recv_count <= 65536) {
                // can not use full vector size (8) if not 4-byte aligned
                if (align4) {
                    constexpr int vec_size = get_num_elements<T, 8>();
                    return transposeT<T, vec_size, 8>(
                        q, send_buf, staging_buf, recv_count, displ, block_count, dtype, nodes, ppn, dep_events);
                }
                else {
                    constexpr int vec_size = get_num_elements<T, 8, false>();
                    return transposeT<T, vec_size, 32>(
                        q, send_buf, staging_buf, recv_count, displ, block_count, dtype, nodes, ppn, dep_events);
                }
            }
            else {
                if (align4) {
                    constexpr int vec_size = get_num_elements<T, 8>();
                    return transposeT<T, vec_size, 32>(
                        q, send_buf, staging_buf, recv_count, displ, block_count, dtype, nodes, ppn, dep_events);
                }
                else {
                    constexpr int vec_size = get_num_elements<T, 8, false>();
                    return transposeT<T, vec_size, 32>(
                        q, send_buf, staging_buf, recv_count, displ, block_count, dtype, nodes, ppn, dep_events);
                }
            }
        };
        e = invoke_scaleout_collective(lambda, dtype);
    }
    else {
        // multiple sycl kernels
        e = transpose(q, send_buf, staging_buf, recv_count, displ, block_count, dtype, nodes, ppn, dep_events);
    }
    return e;
}

//#define PRINT_TIMING

static bool do_fallback_to_scheduler(size_t size) {
    bool is_above_threshold = size > ccl::global_data::env().sycl_reduce_scatter_scaleout_threshold;
    /*
    bool exception_cases = (ccl::global_data::env().atl_transport == ccl_atl_ofi &&
                            (ccl::global_data::env().sycl_reduce_scatter_scaleout_algo == "auto" ||
                             ccl::global_data::env().sycl_reduce_scatter_scaleout_algo == "direct"));
			     */

    bool exception_cases = false;
    return is_above_threshold || exception_cases;
}

ccl::event reduce_scatter_sycl_multi_domain(sycl::queue& q,
                                            const void* send_buf,
                                            void* recv_buf,
                                            size_t recv_count,
                                            datatype dtype,
                                            reduction reduction,
                                            ccl_comm* global_comm,
                                            std::shared_ptr<ccl_comm> p2p_comm,
                                            std::shared_ptr<ccl_comm> r2r_comm,
                                            ccl_stream* global_stream,
                                            const vector_class<event>& deps,
                                            bool& done) {
    ccl::event ev;
    sycl::event e;
    done = true;

    int rank = global_comm->rank();
    int world = global_comm->size();

    if (r2r_comm->size() == 1) {
        LOG_DEBUG("reduce_scatter calls single node");
        return reduce_scatter_scaleup_sycl(
            q, send_buf, recv_buf, recv_count, dtype, reduction, global_comm, p2p_comm, global_stream, deps, done);
    }

    LOG_DEBUG("reduce_scatter calls multi node: recv_count: ",
              recv_count,
              " type : ",
              dtype,
              " recording: ",
              use_recording_path(q));

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t total_size = recv_count * ccl_dtype.size() * world;
    CCL_ASSERT(recv_count > 0);

    // check if to fallback
    if (do_fallback_to_scheduler(recv_count * ccl_dtype.size())) {
        done = false;
        return ev;
    }

    // for the scale-out case, use sum reduction to calculate the total sum,
    // then submit average kernel
    ccl::reduction rs_scaleup_reduction = reduction;
    if (reduction == ccl::reduction::avg) {
        rs_scaleup_reduction = ccl::reduction::sum;
    }
    const bool reduce_has_pre_operation = ccl_reduction_type_storage::is_custom(reduction);
    ccl::reduction rs_scaleout_reduction = rs_scaleup_reduction;
    if (reduction != ccl::reduction::avg && reduce_has_pre_operation) {
        rs_scaleout_reduction = ccl::reduction::sum;
    }

    bool __attribute__((unused)) in_place = (recv_buf == (char*)send_buf + recv_count * rank * ccl_dtype.size());

    const int buf_size = global_comm->get_scaleout_device_buf_size();
    size_t max_pack_count;
    if (total_size <= buf_size) {
        max_pack_count = recv_count;
    }
    else {
        max_pack_count = buf_size / world;
        int typesize = std::max(4, (int)ccl_dtype.size());
        max_pack_count = max_pack_count / typesize * typesize;
        max_pack_count = max_pack_count / ccl_dtype.size();
        CCL_ASSERT(max_pack_count > 0);
    }

    size_t displ = 0;
    int nchunks = (recv_count + max_pack_count - 1) / max_pack_count;

    bool need_rearrange = nchunks > 1 || p2p_comm->size() > 1 || reduce_has_pre_operation;
    void* staging_buf = need_rearrange ? global_comm->get_scaleout_device_buf(q) : (void*)send_buf;

    std::vector<ccl::event> evs;
    // copy deps to evs
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    for (auto& e : dep_events) {
        ev = ccl::event::create_from_native(e);
        evs.push_back(std::move(ev));
    }

#ifdef PRINT_TIMING
    q.wait();
    cpu_timer<1> ctimer;
#endif // PRINT_TIMING

    for (int i = 0; i < nchunks; i++) {
        size_t pack_count = i < nchunks - 1 ? max_pack_count : recv_count - displ;

#ifdef PRINT_TIMING
        ctimer.start(0);
#endif // PRINT_TIMING

        if (need_rearrange) {
            // rearrange data to a staging buffer of same size
            std::vector<sycl::event> dep_events = get_sycl_events(evs);
            e = rearrange(q,
                          send_buf,
                          staging_buf,
                          pack_count,
                          displ,
                          recv_count,
                          dtype,
                          r2r_comm->size(),
                          p2p_comm->size(),
                          dep_events);

#ifdef PRINT_TIMING
            e.wait();
            q.wait(); // for multiple kernels with out-of-order queue
            ctimer.stop(0);
            fprintf(stderr,
                    "[%d] rearrange takes %f us on data: %ld displ: %ld\n",
                    rank,
                    ctimer.get_us(0),
                    recv_count * world,
                    displ);

            ctimer.start(0);
#endif // PRINT_TIMING

            evs.clear();
            ev = ccl::event::create_from_native(e);
            evs.push_back(std::move(ev));
        }

        // scale up on each node
        void* scaleup_buf;
        if (p2p_comm->size() > 1) {
            size_t scaleup_recv_count = pack_count * r2r_comm->size();
            scaleup_buf = (char*)staging_buf + scaleup_recv_count * p2p_comm->rank() * ccl_dtype.size();
            sycl_scaleup_options coll_attr{/*force_tmp=*/true};
            ev = reduce_scatter_scaleup_sycl(q,
                                             staging_buf,
                                             scaleup_buf,
                                             scaleup_recv_count,
                                             dtype,
                                             rs_scaleup_reduction,
                                             global_comm,
                                             p2p_comm,
                                             global_stream,
                                             evs,
                                             done,
                                             coll_attr);
            if (!done) {
                // fallback
                LOG_INFO("allreduce_sycl allgatherv was not done -- falling back");
                return ev;
            }

#ifdef PRINT_TIMING
            ev.wait();
            ctimer.stop(0);
            fprintf(stderr,
                    "[%d] scale up takes %f us on %ld displ: %ld\n",
                    rank,
                    ctimer.get_us(0),
                    recv_count * r2r_comm->size(),
                    displ);

            ctimer.start(0);
#endif // PRINT_TIMING

            evs.clear();
            evs.push_back(std::move(ev));
        }
        else {
            scaleup_buf = staging_buf;

            if (reduce_has_pre_operation) {
                ccl_reduction_data reduction_op = make_reduction_operation(reduction);
                std::vector<sycl::event> sycl_evs = get_sycl_events(evs);
                sycl::event sycl_e;

                auto lambda = [&]<typename T>() {
                    sycl_e = pre_operation_invoke<T, 1, 32>(q,
                                                            scaleup_buf,
                                                            pack_count * r2r_comm->size(),
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
        }

        // scale out
        sycl_scaleout_tune_attr scaleout_tune_attr = ccl::sycl_policy_resolver::scaleout_resolver{
            ccl::sycl_policy::policy_query(ccl_coll_reduce_scatter, pack_count * ccl_dtype.size(), ccl_dtype,
                r2r_comm->size(), global_comm->size(), use_recording_path(q))
        }.reduce_scatter();
        void* scaleout_recv_buf = (char*)recv_buf + displ * ccl_dtype.size();
        ev = reduce_scatter_scaleout_sycl(q,
                                          scaleup_buf,
                                          scaleout_recv_buf,
                                          pack_count,
                                          dtype,
                                          rs_scaleout_reduction,
                                          r2r_comm.get(),
                                          evs,
                                          false,
                                          scaleout_tune_attr,
                                          done);
        CCL_ASSERT(done);

#ifdef PRINT_TIMING
        ev.wait();
        ctimer.stop(0);
        fprintf(stderr,
                "[%d] scale out takes %f us on %ld count node %d displ:%ld\n",
                rank,
                ctimer.get_us(0),
                recv_count,
                r2r_comm->size(),
                displ);
#endif // PRINT_TIMING

        if (reduction == ccl::reduction::avg) {
            // set dependencies
            std::vector<sycl::event> avg_deps_evs;
            avg_deps_evs.push_back(ev.get_native());
            // average divisor
            int total_ranks = global_comm->size();
            LOG_DEBUG("reduce_scatter_sycl calculate average on counts: ", pack_count, ", ranks: ", total_ranks);

            sycl::event reduce_event =
                sycl_average(q, scaleout_recv_buf, pack_count, total_ranks, dtype, avg_deps_evs);
            ev = ccl::event::create_from_native(reduce_event);
        }

        //ev.wait();
        if (i < nchunks - 1) {
            evs.clear();
            evs.push_back(std::move(ev));
            displ += pack_count;
        }
    } // end of for

    if (need_rearrange)
        global_comm->put_scaleout_device_buf(staging_buf);

    return ev;
}

ccl::event reduce_scatter_scaleout_only(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t recv_count,
                                        datatype dtype,
                                        reduction reduction,
                                        ccl_comm* global_comm,
                                        ccl_stream* global_stream,
                                        const vector_class<event>& deps,
                                        bool& done) {
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    sycl_scaleout_tune_attr scaleout_tune_attr = ccl::sycl_policy_resolver::scaleout_resolver{
        ccl::sycl_policy::policy_query(ccl_coll_reduce_scatter, recv_count * ccl_dtype.size(), ccl_dtype,
            /*r2r_size=*/0, global_comm->size(), use_recording_path(q))
    }.reduce_scatter();
    LOG_DEBUG("reduce_scatter_sycl scaleout only");
    ev = reduce_scatter_scaleout_sycl(q,
                                      send_buf,
                                      recv_buf,
                                      recv_count,
                                      dtype,
                                      reduction,
                                      global_comm,
                                      deps,
                                      true,
                                      scaleout_tune_attr,
                                      done);
    if (!done) {
        LOG_INFO("reduce_scatter_sycl scaleout was not done -- falling back");
    }
    return ev;
}

ccl::event reduce_scatter_sycl(sycl::queue q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t recv_count,
                               datatype dtype,
                               reduction reduction,
                               ccl_comm* comm,
                               ccl_stream* op_stream,
                               const reduce_scatter_attr& attr,
                               const vector_class<event>& deps,
                               bool& done) {
    if (recv_count == 0) {
        done = true;
        auto sycl_events = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_events);
        return ccl::event::create_from_native(e);
    }

#ifdef CCL_ENABLE_ESIMD
    // ESIMD scale-up does not support sub-communicators
    if (ccl::global_data::env().sycl_esimd) {
        int ppn = is_single_node ? comm->size() : comm->get_node_comm()->size();
        if (ppn != ccl::global_data::get().get_local_proc_count() && ppn != 1) {
            done = false;
            ccl::event ev;
            return ev;
        }
    }
#endif // CCL_ENABLE_ESIMD

    CCL_THROW_IF_NOT(q.is_in_order(), "SYCL queue must be in-order");

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    auto p2p_level = ccl_p2p_domain::sycl_get_reduce_scatter_p2p_level(comm, recv_count * ccl_dtype.size(), ccl::global_data::env().sycl_p2p_level_reduce_scatter_threshold);

    auto [ p2p_comm, r2r_comm ] = ccl_p2p_domain::get_p2p_comms(comm, p2p_level);

    if (r2r_comm->is_uniform && r2r_comm->size() == 1) {
        LOG_DEBUG("is_single_domain");
        return reduce_scatter_scaleup_sycl(q, send_buf, recv_buf, recv_count, dtype, reduction, comm, p2p_comm, op_stream, deps, done);
    }

    return reduce_scatter_sycl_multi_domain(q, send_buf, recv_buf, recv_count, dtype, reduction, comm, p2p_comm, r2r_comm, op_stream, deps, done);
}

} // namespace v1
} // namespace ccl

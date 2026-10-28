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

#include "coll/algorithms/alltoall/sycl/alltoall_sycl.hpp"
#include "coll/algorithms/alltoall/sycl/alltoall_ll256.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_scaleup.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_scaleout.hpp"
#include "coll/algorithms/utils/barrier_data.hpp"  // use_recording_path
#include "coll/algorithms/utils/transmit/transmit.hpp"
#include "proxy/proxy.hpp"

namespace ccl {
namespace v1 {

namespace {

// world == 1 fast path: memcpy or wait-on-events.
ccl::event alltoall_scaleup_single_rank(const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        ccl_stream* global_stream,
                                        const ccl::vector_class<ccl::event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    sycl::event sycl_e;
    auto sycl_q = global_stream->get_native_stream();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    if (send_buf != recv_buf) {
        LOG_DEBUG("single rank: out-of-place case, coll: alltoall");
        sycl_e = sycl_q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
        });
    }
    else {
        LOG_DEBUG("single rank: inplace case, coll: alltoall");
        sycl_e = submit_wait_on_events(sycl_q, dep_events);
    }
    return ccl::event::create_from_native(sycl_e);
}

// Arc family: alltoall_ll chunked (unless CCL_SYCL_ENABLE_ARC_ALLTOALL_LL
// disables it), then arc_alltoall (single call) when shape aligns. Returns
// with done=true on success. If neither runs, falls back via `fell_through`
// so the caller can try the vec path.
ccl::event alltoall_scaleup_arc_family(sycl::queue& q,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       ccl::datatype dtype,
                                       ccl_comm* comm,
                                       const std::shared_ptr<ccl_comm> p2p_comm,
                                       bool is_numa_comm,
                                       int numa_split,
                                       ccl_stream* global_stream,
                                       const ccl::vector_class<ccl::event>& deps,
                                       bool& done,
                                       bool& fell_through) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    uint32_t world = p2p_comm->size();
    int rank = p2p_comm->rank();
    fell_through = false;
    ccl::event e;

    if (!ccl::global_data::env().sycl_enable_arc_alltoall_ll) {
        int node_size = p2p_comm->size();
        const int chunk_size = ccl::global_data::env().sycl_alltoall_chunking_threshold;
        size_t max_pack_count;

        if (send_buf != recv_buf) {
            q.memcpy((char*)recv_buf + rank * count * ccl_dtype.size(),
                     (char*)send_buf + rank * count * ccl_dtype.size(),
                     count * ccl_dtype.size());
        }

        if (chunk_size == 0 || count * ccl_dtype.size() <= chunk_size) {
            max_pack_count = count;
        }
        else {
            max_pack_count = chunk_size;
            int typesize = std::max(4, (int)ccl_dtype.size());
            max_pack_count = max_pack_count / typesize * typesize;
            max_pack_count = max_pack_count / ccl_dtype.size();
            CCL_ASSERT(max_pack_count > 0);
        }
        size_t send_offset = 0;
        int nchunks = divUp(count, max_pack_count);
        for (int iter = 0; iter < nchunks; iter++) {
            int pack_count = (iter < nchunks - 1) ? max_pack_count : count - send_offset;
            std::vector<size_t> offsets(node_size);
            for (int i = 0; i < node_size; i++) {
                offsets[i] = i * count * ccl_dtype.size();
            }
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                "alltoall_ll", "send_size", pack_count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("invoking alltoall LL256 kernel alltoall_ll, count:",
                      pack_count, " datatype: ", dtype);
            e = alltoall_ll((char*)send_buf + send_offset * ccl_dtype.size(),
                            (char*)recv_buf + send_offset * ccl_dtype.size(),
                            pack_count, offsets, dtype, comm, p2p_comm,
                            global_stream, deps, done);
            if (!done) break;
            send_offset += pack_count;
        }
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        if (done) {
            LOG_DEBUG("invoking alltoall LL256 kernel, count:", count, " datatype: ", dtype, " done");
            return e;
        }
    }

    if (((count * dt_sz) % LS_SZ == 0) && ((world & (world - 1)) == 0)) {
        done = true;
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "arc_alltoall", "send_size", count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| alltoall selects arc_alltoall, count: ", count, " datatype: ", dtype);
        e = arc_alltoall(
            send_buf, recv_buf, count, dtype, comm, is_numa_comm, numa_split, global_stream);
        LOG_DEBUG("|CCL_SYCL| alltoall selects arc_alltoall, count: ",
                  count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        return e;
    }

    // Neither arc_alltoall_ll succeeded nor shape aligns for arc_alltoall.
    fell_through = true;
    return e;
}

// Non-arc vec path: alltoall_large chunked (out-of-place only; in-place is
// unsupported and returns done=false).
ccl::event alltoall_scaleup_default(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl_comm* comm,
                                const std::shared_ptr<ccl_comm> p2p_comm,
                                ccl_stream* global_stream,
                                const ccl::vector_class<ccl::event>& deps,
                                bool is_arc,
                                bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    uint32_t world = p2p_comm->size();
    ccl::event e;

    // ARC does chunking as needed
    const size_t chunk_size =
        is_arc ? ccl::global_data::env().sycl_alltoall_chunking_threshold : 0;
    size_t max_pack_count;
    size_t nchunks = calculate_chunking_pack_count(chunk_size, count, dt_sz, max_pack_count);
    size_t offset = 0;
    for (size_t iter = 0; iter < nchunks; iter++) {
        size_t pack_count = (iter < nchunks - 1) ? max_pack_count : count - offset;
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin(
            "alltoall_large", "send_size", pack_count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| alltoall selects large kernel, count: ",
                  pack_count, " datatype: ", dtype, "iter:", iter, "nchunks:", nchunks);
        std::vector<size_t> scaleup_offsets(world);
        for (size_t r = 0; r < world; r++) {
            scaleup_offsets[r] = r * count + offset;
        }
        e = alltoall_large(send_buf, recv_buf, pack_count, scaleup_offsets, dtype,
                           comm, p2p_comm, global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| alltoall selects large kernel, count: ",
                  pack_count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
        offset += pack_count;
    }
    return e;
}

}  // namespace

ccl::event alltoall_scaleup_sycl(sycl::queue& q,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 ccl_comm* comm,
                                 const std::shared_ptr<ccl_comm> p2p_comm,
                                 bool is_numa_comm,
                                 int numa_split,
                                 ccl_stream* global_stream,
                                 const vector_class<event>& deps,
                                 bool& done) {
    done = true;

    if (p2p_comm->size() == 1) {
        return alltoall_scaleup_single_rank(send_buf, recv_buf, count, dtype, global_stream, deps);
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t total_size = count * ccl_dtype.size();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected",
              comm->get_topo_manager().has_all_vertices_connected());
    const bool is_arc = is_arc_card(ccl::global_data::get().ze_data->devices[0].family);

    const auto tune = ccl::sycl_policy_resolver::scaleup_resolver{
        ccl::sycl_policy::policy_query(
            ccl_coll_alltoall, total_size, ccl_dtype,
            /*r2r_size=*/0, comm->size(), use_recording_path(q))
            .with_comm(comm)
            .with_stream(global_stream)
    }.alltoall();

    if (tune.protocol == ccl::sycl_policy::scaleup_protocol::ll) {
        bool fell_through = false;
        auto ev = alltoall_scaleup_arc_family(q, send_buf, recv_buf, count, dtype, comm,
                                              p2p_comm, is_numa_comm, numa_split,
                                              global_stream, deps, done, fell_through);
        if (!fell_through) return ev;
    }

#ifdef CCL_ENABLE_ESIMD
    if (ccl::global_data::env().sycl_esimd) {
        LOG_WARN(
            "|CCL_SYCL| sycl ESIMD requested for alltoall collective; ESIMD not supported, falling back to alltoall sycl implementation");
    }
#endif // CCL_ENABLE_ESIMD

    return alltoall_scaleup_default(send_buf, recv_buf, count, dtype, comm, p2p_comm,
                                global_stream, deps, is_arc, done);
}

ccl::event alltoall_scaleout_only_sycl(sycl::queue& q,
                                       const void* send_buf,
                                       void* recv_buf,
                                       size_t count,
                                       ccl::datatype dtype,
                                       ccl_comm* comm,
                                       std::shared_ptr<ccl_comm> r2r_comm,
                                       ccl_stream* global_stream,
                                       const vector_class<event>& deps,
                                       bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    sycl_scaleout_tune_attr scaleout_tune_attr = ccl::sycl_policy_resolver::scaleout_resolver{
        ccl::sycl_policy::policy_query(ccl_coll_alltoall, count * ccl_dtype.size(), ccl_dtype,
            comm->size(), comm->size(), use_recording_path(q))
    }.alltoall();

    LOG_DEBUG("alltoall scaleout only");
    return alltoall_scaleout_sycl(q,
                                  send_buf,
                                  recv_buf,
                                  count,
                                  dtype,
                                  comm,
                                  global_stream,
                                  deps,
                                  true,
                                  scaleout_tune_attr,
                                  done);
}

ccl::event alltoall_sycl_multi_domain(sycl::queue& q,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      ccl_comm* comm,
                                      std::shared_ptr<ccl_comm> p2p_comm,
                                      std::shared_ptr<ccl_comm> r2r_comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    if (p2p_comm->size() == 1 || !transport_rdma_enabled()) {
        return alltoall_scaleout_only_sycl(
            q, send_buf, recv_buf, count, dtype, comm, r2r_comm, global_stream, deps, done);
    }

    // pair-wise exchange in p2p_comm, followed by pairwise exchange
    // on all ranks out of the p2p_comm
    bool batch_mode = true;

    // when the proxy service is enabled, run the scale-out exchange on the proxy thread;
    // a service that failed to start leaves the non-proxy implementation
    const bool use_proxy = ccl::global_data::env().sycl_enable_alltoall_scaleout_proxy &&
                           ccl::global_data::get().proxy_service->is_started();

    LOG_DEBUG("|CCL_SYCL| alltoall selects numa pairwise only kernel, count:",
              count,
              " datatype: ",
              dtype,
              " use_proxy: ",
              use_proxy);

    return alltoall_sycl_numa_pairwise_rdma_oneshot(q,
                                                    send_buf,
                                                    recv_buf,
                                                    count,
                                                    dtype,
                                                    comm,
                                                    p2p_comm,
                                                    r2r_comm,
                                                    global_stream,
                                                    deps,
                                                    batch_mode,
                                                    use_proxy,
                                                    done);
}

ccl::event alltoall_sycl(sycl::queue q,
                         const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         ccl_comm* comm,
                         ccl_stream* op_stream,
                         const alltoall_attr& attr,
                         const vector_class<event>& deps,
                         bool& done) {
    if (count == 0) {
        done = true;
        auto sycl_events = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_events);
        return ccl::event::create_from_native(e);
    }

    if (send_buf == recv_buf) {
        LOG_WARN_ONCE("|CCL_SYCL| oneCCL does not support in-place Alltoall; result may be wrong");
        // continue to run but results may be wrong
        // this is to support CCL benchmark
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    auto p2p_level = ccl_p2p_domain::sycl_get_alltoall_p2p_level(
        comm,
        count * ccl_dtype.size(),
        ccl::global_data::env().sycl_p2p_level_alltoall_small_threshold,
        ccl::global_data::env().sycl_p2p_level_alltoall_threshold);

    auto [p2p_comm, r2r_comm] = ccl_p2p_domain::get_p2p_comms(comm, p2p_level);

    // single p2p domain
    if (r2r_comm->is_uniform && r2r_comm->size() == 1) {
        LOG_DEBUG("is_single_domain");
        return alltoall_scaleup_sycl(
            q, send_buf, recv_buf, count, dtype, comm, p2p_comm, false, 0, op_stream, deps, done);
    }

    return alltoall_sycl_multi_domain(
        q, send_buf, recv_buf, count, dtype, comm, p2p_comm, r2r_comm, op_stream, deps, done);
}

} // namespace v1
} // namespace ccl

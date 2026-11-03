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

#include "coll/algorithms/alltoallv/sycl/alltoallv_sycl.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_scaleup.hpp"
#include "coll/algorithms/utils/sycl_policy_resolver_scaleout.hpp"
#include "coll/algorithms/utils/barrier_data.hpp"  // use_recording_path
#include "coll/algorithms/utils/transmit/transmit.hpp"

namespace ccl {
namespace v1 {

namespace {

// world == 1 fast path: memcpy or wait-on-events.
ccl::event alltoallv_scaleup_single_rank(const void* send_buf,
                                         void* recv_buf,
                                         size_t max_count,
                                         ccl::datatype dtype,
                                         ccl_stream* global_stream,
                                         const ccl::vector_class<ccl::event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dt_sz = ccl_dtype.size();
    sycl::event sycl_e;
    auto sycl_q = global_stream->get_native_stream();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    if (send_buf != recv_buf) {
        LOG_DEBUG("single rank: out-of-place case, coll: alltoallv");
        sycl_e = sycl_q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(recv_buf, send_buf, max_count * dt_sz);
        });
    }
    else {
        LOG_DEBUG("single rank: inplace case, coll: alltoallv");
        sycl_e = submit_wait_on_events(sycl_q, dep_events);
    }
    return ccl::event::create_from_native(sycl_e);
}

// Arc LL256 for alltoallv. On success returns done=true; on !done sets
// `fell_through` so the caller falls back to vec.
ccl::event alltoallv_scaleup_arc_ll(sycl::queue& q,
                                    const void* send_buf,
                                    const ccl::vector_class<size_t>& send_counts,
                                    const ccl::vector_class<size_t>& send_displs,
                                    void* recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    const ccl::vector_class<size_t>& recv_displs,
                                    ccl::datatype dtype,
                                    ccl_comm* comm,
                                    const std::shared_ptr<ccl_comm> p2p_comm,
                                    ccl_stream* global_stream,
                                    const ccl::vector_class<ccl::event>& deps,
                                    size_t max_count,
                                    bool& done,
                                    bool& fell_through) {
    fell_through = false;
    ccl::event e;
    if (ccl::global_data::env().sycl_enable_arc_alltoall_ll) {
        fell_through = true;
        return e;
    }
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dt_sz = ccl_dtype.size();
    const int rank = p2p_comm->rank();

    if (send_buf != recv_buf) {
        q.memcpy((char*)recv_buf + recv_displs[rank],
                 (char*)send_buf + send_displs[rank],
                 send_counts[rank] * dt_sz);
    }

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin(
        "alltoallv_ll", "send_size", max_count * dt_sz, comm->unique_id());
#endif // CCL_ENABLE_ITT
    LOG_DEBUG("invoking alltoallv LL256 kernel alltoallv_ll, count:",
              max_count, " datatype: ", dtype);
    e = alltoallv_ll((char*)send_buf, send_counts, send_displs,
                     (char*)recv_buf, recv_counts, recv_displs,
                     dtype, comm, p2p_comm, global_stream, deps, max_count, done);
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    if (done) {
        LOG_DEBUG("invoked alltoallv LL256 kernel, count:", max_count, " datatype: ", dtype, " done");
        return e;
    }
    fell_through = true;
    return e;
}

// Non-arc vec path: alltoallv_large (out-of-place only).
ccl::event alltoallv_scaleup_default(const void* send_buf,
                                 const ccl::vector_class<size_t>& send_counts,
                                 const ccl::vector_class<size_t>& send_displs,
                                 void* recv_buf,
                                 const ccl::vector_class<size_t>& recv_counts,
                                 const ccl::vector_class<size_t>& recv_displs,
                                 ccl::datatype dtype,
                                 ccl_comm* comm,
                                 ccl_stream* global_stream,
                                 const ccl::vector_class<ccl::event>& deps,
                                 const alltoallv_ctx& options,
                                 bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dt_sz = ccl_dtype.size();
    ccl::event e;
    if (send_buf == recv_buf) {
        LOG_WARN_ONCE("|CCL_SYCL| oneCCL does not support in-place Alltoallv; falling back");
        done = false;
        return e;
    }
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin(
        "alltoallv_large", "send_size", options.max_count * dt_sz, comm->unique_id());
#endif // CCL_ENABLE_ITT
    LOG_DEBUG("|CCL_SYCL| alltoallv selects large kernel, count: ",
              options.max_count, " datatype: ", dtype);
    e = alltoallv_large(send_buf, send_counts, send_displs, recv_buf, recv_counts, recv_displs,
                        dtype, comm, global_stream, deps, options);
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    return e;
}

}  // namespace

ccl::event alltoallv_scaleup_sycl(sycl::queue& q,
                                  const void* send_buf,
                                  const ccl::vector_class<size_t>& send_counts,
                                  const ccl::vector_class<size_t>& send_displs,
                                  void* recv_buf,
                                  const ccl::vector_class<size_t>& recv_counts,
                                  const ccl::vector_class<size_t>& recv_displs,
                                  ccl::datatype dtype,
                                  ccl_comm* comm,
                                  const std::shared_ptr<ccl_comm> p2p_comm,
                                  bool is_numa_comm,
                                  int numa_split,
                                  ccl_stream* global_stream,
                                  const vector_class<event>& deps,
                                  const alltoallv_ctx& options,
                                  bool& done) {
    done = true;

    if (p2p_comm->size() == 1) {
        return alltoallv_scaleup_single_rank(send_buf, recv_buf, options.max_count, dtype,
                                             global_stream, deps);
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    (void)ccl_dtype;
    const bool has_all_vertices_connected = comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected ", has_all_vertices_connected);

    // `msg_bytes` here carries ELEMENTS (options.max_count), not bytes — the
    // alltoallv gate compares element count directly against the threshold.
    auto ccl_dtype_q = ccl::global_data::get().dtypes->get(dtype);
    const auto tune = ccl::sycl_policy_resolver::scaleup_resolver{
        ccl::sycl_policy::policy_query(
            ccl_coll_alltoallv, options.max_count, ccl_dtype_q,
            /*r2r_size=*/0, comm->size(), use_recording_path(q))
            .with_comm(comm)
            .with_stream(global_stream)
    }.alltoallv();

    if (tune.protocol == ccl::sycl_policy::scaleup_protocol::ll) {
        bool fell_through = false;
        auto ev = alltoallv_scaleup_arc_ll(q, send_buf, send_counts, send_displs,
                                           recv_buf, recv_counts, recv_displs, dtype,
                                           comm, p2p_comm, global_stream, deps,
                                           options.max_count, done, fell_through);
        if (!fell_through) return ev;
    }

#ifdef CCL_ENABLE_ESIMD
    if (ccl::global_data::env().sycl_esimd) {
        LOG_WARN(
            "|CCL_SYCL| sycl ESIMD requested for alltoallv collective; ESIMD not supported, falling back to alltoallv sycl implementation");
    }
#endif // CCL_ENABLE_ESIMD

    return alltoallv_scaleup_default(send_buf, send_counts, send_displs, recv_buf, recv_counts,
                                 recv_displs, dtype, comm, global_stream, deps, options, done);
}

ccl::event alltoallv_sycl_multi_domain(sycl::queue& q,
                                       const void* send_buf,
                                       const ccl::vector_class<size_t>& send_counts,
                                       const ccl::vector_class<size_t>& send_displs,
                                       void* recv_buf,
                                       const ccl::vector_class<size_t>& recv_counts,
                                       const ccl::vector_class<size_t>& recv_displs,
                                       ccl::datatype dtype,
                                       ccl_comm* comm,
                                       std::shared_ptr<ccl_comm> p2p_comm,
                                       std::shared_ptr<ccl_comm> r2r_comm,
                                       ccl_stream* global_stream,
                                       const vector_class<event>& deps,
                                       alltoallv_ctx* options,
                                       bool& done) {
    if (send_buf == recv_buf) {
        LOG_WARN_ONCE("|CCL_SYCL| oneCCL does not support in-place Alltoallv; falling back");
        done = false;
        return ccl::event();
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t dt_sz = ccl_dtype.size();
    uint32_t world = comm->size();

    sycl_scaleout_tune_attr scaleout_tune_attr = ccl::sycl_policy_resolver::scaleout_resolver{
        ccl::sycl_policy::policy_query(ccl_coll_alltoallv, options->max_count * dt_sz, ccl_dtype,
            /*r2r_size=*/0, comm->size(), use_recording_path(q))
    }.alltoall();

    if (scaleout_tune_attr.algo() == ccl::sycl_policy::algorithm::direct) {
        options->copy_to_host = !ccl::global_data::env().sycl_enable_direct_gpu_rdma;
        {
            ze_device_handle_t ze_dev =
                sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
            if (should_disable_rdma(ze_dev))
                options->copy_to_host = true;
        }

        if (options->copy_to_host) {
            size_t host_buf_size = comm->get_scaleout_host_buf_size();
            size_t size_to_compare = options->total_max_count * dt_sz;
            if (host_buf_size < size_to_compare) {
                options->is_pipeline = true;
            }

            LOG_DEBUG("scaleout alltoallv: global_bytes=",
                      size_to_compare,
                      " host_buf_size=",
                      host_buf_size,
                      " is_pipeline=",
                      options->is_pipeline);

            // pipelined per-rank chunking for simple_direct algo
            if (options->is_pipeline) {
                // use a separate copy queue to allow overlapping 1.copy+2.compute and 3.copy
                auto copy_q =
                    sycl::queue(q.get_context(), q.get_device(), sycl::property::queue::in_order{});
                // When the total message is too large to fit in the host buffer,
                // we split each rank's data into fixed-size chunks and loop:
                // 1. D2H copy of this chunk's send data into host staging
                // 2. MPI alltoallv for this chunk
                // 3. H2D copy of received data back into recv_buf
                // Host buffer layout for alltoallv:
                //   [ send_slot[0] | send_slot[1] | ... | send_slot[world-1] |
                //     recv_slot[0] | recv_slot[1] | ... | recv_slot[world-1] ]

                // choose chunk size
                size_t max_safe_chunk_bytes = host_buf_size / world / 2;
                if (max_safe_chunk_bytes == 0) {
                    LOG_DEBUG(
                        "scaleout alltoallv pipeline: scaleout_host_buf_size ",
                        host_buf_size,
                        " is too small to handle even one chunk of data for scaleout phase. Falling back.");
                    done = false;
                    ccl::event e;
                    return e;
                }
                size_t chunk_bytes = ccl::global_data::env().sycl_alltoall_chunking_threshold;
                if (chunk_bytes == 0 || chunk_bytes > max_safe_chunk_bytes) {
                    chunk_bytes = max_safe_chunk_bytes;
                }

                void* scaleout_send_buf = comm->get_scaleout_host_buf();
                void* scaleout_recv_buf = (char*)scaleout_send_buf + world * chunk_bytes;

                // how many bytes already transferred
                std::vector<size_t> send_progress(world, 0);
                std::vector<size_t> recv_progress(world, 0);
                // bytes and offsets per chunk for direct alltoallv call
                std::vector<size_t> chunk_send_bytes(world, 0);
                std::vector<size_t> chunk_recv_bytes(world, 0);

                std::vector<size_t> send_bytes(world, 0);
                std::vector<size_t> recv_bytes(world, 0);
                for (auto i = 0; i < world; i++) {
                    send_bytes[i] = send_counts[i] * dt_sz;
                    recv_bytes[i] = recv_counts[i] * dt_sz;
                }

                sycl::event copy_e, op_e;
                std::vector<ccl::event> copy_events;

                // options stay the same for all iterations
                options->chunk_bytes = chunk_bytes;
                options->copy_to_host = false;
                options->is_cpu_buffers = true;
                std::vector<sycl::event> dep_events = get_sycl_events(deps);

                // loop until all data is transferred
                for (;;) {
                    bool has_data_to_transfer = false;
                    size_t max_send_bytes = 0;
                    for (auto rank = 0; rank < world; rank++) {
                        size_t send_leftover = send_bytes[rank] - send_progress[rank];
                        size_t recv_leftover = recv_bytes[rank] - recv_progress[rank];
                        chunk_send_bytes[rank] = std::min(send_leftover, chunk_bytes);
                        if (chunk_send_bytes[rank] > max_send_bytes) {
                            max_send_bytes = chunk_send_bytes[rank];
                        }
                        chunk_recv_bytes[rank] = std::min(recv_leftover, chunk_bytes);
                        if (chunk_send_bytes[rank] > 0 || chunk_recv_bytes[rank] > 0) {
                            has_data_to_transfer = true;
                        }
                    }
                    if (!has_data_to_transfer) {
                        break;
                    }

                    // 1: D2H copy, each rank copies a data chunk into the host_buf send area
                    for (auto rank = 0; rank < world; rank++) {
                        if (chunk_send_bytes[rank] == 0)
                            continue;
                        q.submit([=](sycl::handler& h) {
                            h.depends_on(dep_events);
                            h.memcpy((char*)scaleout_send_buf + rank * chunk_bytes,
                                     (char*)send_buf + send_displs[rank] + send_progress[rank],
                                     chunk_send_bytes[rank]);
                        });
                    }
                    dep_events.clear();

                    // 2: direct alltoallv
#ifdef CCL_ENABLE_ITT
                    ccl::profile::itt::task_begin("alltoallv_scaleout_sycl_simple",
                                                  "send_size",
                                                  max_send_bytes,
                                                  comm->unique_id());
#endif // CCL_ENABLE_ITT
                    LOG_DEBUG(
                        "|CCL_SYCL| alltoallv scaleout selects simple (direct) kernel in pipeline mode, max count:",
                        max_send_bytes / dt_sz,
                        " datatype: ",
                        dtype);
                    op_e = alltoallv_scaleout_sycl_direct(q,
                                                          scaleout_send_buf,
                                                          chunk_send_bytes,
                                                          scaleout_recv_buf,
                                                          chunk_recv_bytes,
                                                          dtype,
                                                          comm,
                                                          copy_events,
                                                          false,
                                                          *options,
                                                          done);
#ifdef CCL_ENABLE_ITT
                    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
                    copy_events.clear();

                    // 3: H2D copy received data from the staging recv area back into recv_buf
                    for (auto rank = 0; rank < world; rank++) {
                        if (chunk_recv_bytes[rank] == 0)
                            continue;
                        copy_e = copy_q.submit([=](sycl::handler& h) {
                            h.depends_on(op_e);
                            h.memcpy((char*)recv_buf + recv_displs[rank] + recv_progress[rank],
                                     (char*)scaleout_recv_buf + rank * chunk_bytes,
                                     chunk_recv_bytes[rank]);
                        });
                    }
                    copy_events.push_back(ccl::event::create_from_native(copy_e));

                    // update progress
                    for (auto rank = 0; rank < world; rank++) {
                        send_progress[rank] += chunk_send_bytes[rank];
                        recv_progress[rank] += chunk_recv_bytes[rank];
                    }
                } // main for loop of sending msg

                done = true;
                auto sycl_deps = get_sycl_events(copy_events);
                op_e = submit_wait_on_events(q, sycl_deps);
                return ccl::event::create_from_native(op_e);
            }
        }
    }

    // default path
    return alltoallv_scaleout_sycl(q,
                                   send_buf,
                                   send_counts,
                                   send_displs,
                                   recv_buf,
                                   recv_counts,
                                   recv_displs,
                                   dtype,
                                   comm,
                                   global_stream,
                                   deps,
                                   true,
                                   scaleout_tune_attr,
                                   *options,
                                   done);
}

ccl::event alltoallv_sycl(sycl::queue q,
                          const void* send_buf,
                          const ccl::vector_class<size_t>& send_counts,
                          void* recv_buf,
                          const ccl::vector_class<size_t>& recv_counts,
                          ccl::datatype dtype,
                          ccl_comm* comm,
                          ccl_stream* op_stream,
                          const alltoallv_attr& attr,
                          const vector_class<event>& deps,
                          bool& done) {
    if (send_buf == recv_buf) {
        LOG_WARN_ONCE("|CCL_SYCL| oneCCL does not support in-place Alltoallv; falling back");
        done = false;
        return ccl::event();
    }

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    auto world = comm->size();
    alltoallv_ctx options{};
    if (world == 1) {
        options.max_count = send_counts[0];
        options.total_max_count = send_counts[0];
    }
    else {
        // since alltoallv version is used, we choose the max count among
        // all the ranks send_counts and recv_counts values to determine
        // the chunk size that should be the same across all the ranks for future reference. as a result only one iteration is needed
        // critical for alltoallv where each rank has different send_counts/recv_counts,
        // otherwise deadlock can happen when some ranks decide to pipeline and some don't
        // blocking because we need the global_max_bytes result for the next steps
        size_t max_send = *std::max_element(send_counts.begin(), send_counts.end());
        size_t max_recv = *std::max_element(recv_counts.begin(), recv_counts.end());
        size_t total_recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0ul);
        size_t total_send_bytes = std::accumulate(send_counts.begin(), send_counts.end(), 0ul);
        size_t local_counts[2] = { std::max(max_send, max_recv),
                                   total_send_bytes + total_recv_count };
        size_t global_counts[2] = { 0, 0 };
        int ep_idx = 0;
        atl_req_t req;
        std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
        ATL_CALL_THROW_IF_ERROR(atl_comm->allreduce(
            ep_idx, local_counts, global_counts, 2, ATL_DTYPE_UINT64, ATL_REDUCTION_MAX, req));
        ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
        if (!req.is_completed) {
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
        }

        options.max_count = global_counts[0];
        options.total_max_count = global_counts[1];
    }

    if (options.max_count == 0) {
        done = true;
        auto sycl_events = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_events);
        return ccl::event::create_from_native(e);
    }

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    std::vector<size_t> send_displs(world, 0);
    std::vector<size_t> recv_displs(world, 0);
    for (auto i = 1; i < world; i++) {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1] * ccl_dtype.size();
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1] * ccl_dtype.size();
    }

    auto p2p_level = ccl_p2p_domain::sycl_get_default_p2p_level(comm, options.max_count * ccl_dtype.size(), ccl::global_data::env().sycl_p2p_level_threshold);

    auto [ p2p_comm, r2r_comm ] = ccl_p2p_domain::get_p2p_comms(comm, p2p_level);

    // single p2p domain
    if (r2r_comm->is_uniform && r2r_comm->size() == 1) {
        LOG_DEBUG("is_single_domain");
        return alltoallv_scaleup_sycl(q,
                                      send_buf,
                                      send_counts,
                                      send_displs,
                                      recv_buf,
                                      recv_counts,
                                      recv_displs,
                                      dtype,
                                      comm,
                                      p2p_comm,
                                      false,
                                      0,
                                      op_stream,
                                      deps,
                                      options,
                                      done);
    }

    return alltoallv_sycl_multi_domain(q,
                                       send_buf,
                                       send_counts,
                                       send_displs,
                                       recv_buf,
                                       recv_counts,
                                       recv_displs,
                                       dtype,
                                       comm,
                                       p2p_comm,
                                       r2r_comm,
                                       op_stream,
                                       deps,
                                       &options,
                                       done);
}

} // namespace v1
} // namespace ccl

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
#include "coll/algorithms/utils/host_task.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/alltoallv/sycl/alltoallv_sycl.hpp"
#include "coll/algorithms/alltoallv/sycl/alltoallv_pairwise.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

sycl::event alltoallv_scaleout_sycl_simple_blocking(sycl::queue& q,
                                                    const void* send_buf,
                                                    const ccl::vector_class<size_t>& send_counts,
                                                    void* recv_buf,
                                                    const ccl::vector_class<size_t>& recv_counts,
                                                    ccl::datatype dtype,
                                                    ccl_comm* comm,
                                                    const ccl::vector_class<ccl::event>& deps,
                                                    bool original_deps,
                                                    const alltoallv_ctx& options,
                                                    bool& done) {
    sycl::event op_end, copy_e;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    uint32_t world = comm->size();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    void* scaleout_send_buf = const_cast<void*>(send_buf);
    void* scaleout_recv_buf = recv_buf;

    if (original_deps)
        q.wait();
    for (auto& dep : deps) {
        auto e = dep.get_native();
        e.wait();
    }

    std::vector<size_t> send_bytes(world, 0);
    std::vector<size_t> recv_bytes(world, 0);
    // if pipelining is enabled, we can use send/recv counts as is
    // because these vectors already have the proper bytes values for each chunk
    if (!options.is_pipeline) {
        for (uint32_t i = 0; i < world; i++) {
            send_bytes[i] = send_counts[i] * ccl_dtype.size();
            recv_bytes[i] = recv_counts[i] * ccl_dtype.size();
        }
    }

    std::vector<size_t> send_displs(world, 0);
    std::vector<size_t> recv_displs(world, 0);
    if (options.is_pipeline && options.chunk_bytes) {
        // in pipeline mode we use host_buf offsets:
        // ranks have equal slots, each calculated as rank * chunk_bytes
        for (uint32_t i = 0; i < world; i++) {
            send_displs[i] = i * options.chunk_bytes;
            recv_displs[i] = i * options.chunk_bytes;
        }
    }
    else {
        for (uint32_t i = 1; i < world; i++) {
            send_displs[i] = send_displs[i - 1] + send_bytes[i - 1];
            recv_displs[i] = recv_displs[i - 1] + recv_bytes[i - 1];
        }
    }

    if (options.copy_to_host) {
        // MPI standard explicitly prohibits IN_PLACE mode for Alltoallv
        // because there is no guaranteed relationship between send and
        // receive layouts. So we need one more tmp_buf for this case.
        size_t total_send_count = std::accumulate(send_counts.begin(), send_counts.end(), 0ul);
        scaleout_send_buf = comm->get_scaleout_host_buf();
        scaleout_recv_buf = (char*)scaleout_send_buf + total_send_count * ccl_dtype.size();
        copy_e = q.submit([=](sycl::handler& h) {
            h.memcpy(scaleout_send_buf, send_buf, total_send_count * ccl_dtype.size());
        });
        copy_e.wait();
    }
    else if (!options.is_cpu_buffers) {
        if (!check_mpi_supports_rdma()) {
            LOG_WARN("copy_to_host=false with a GPU buffer. "
                     "TODO: make sure I_MPI_OFFLOAD is set or GPU RDMA is enabled");
        }
    }

    // call ccl::wrapper for MPI/OFI.
    int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
    atl_req_t req;
    ATL_CALL_THROW_IF_ERROR(
        atl_comm->alltoallv(ep_idx,
                            scaleout_send_buf,
                            options.is_pipeline ? send_counts.data() : send_bytes.data(),
                            send_displs.data(),
                            scaleout_recv_buf,
                            options.is_pipeline ? recv_counts.data() : recv_bytes.data(),
                            recv_displs.data(),
                            req));

    ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
    if (!req.is_completed) {
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
    }
    else {
        // The operation was probably blocking, since it finished really quickly
    }

    if (options.copy_to_host) {
        size_t total_recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0ul);
        op_end = q.memcpy(recv_buf, scaleout_recv_buf, total_recv_count * ccl_dtype.size());
        op_end.wait();
    }

    done = true;
    return op_end;
}

sycl::event alltoallv_scaleout_sycl_simple_nonblocking(sycl::queue& q,
                                                       const void* send_buf,
                                                       const ccl::vector_class<size_t>& send_counts,
                                                       void* recv_buf,
                                                       const ccl::vector_class<size_t>& recv_counts,
                                                       ccl::datatype dtype,
                                                       ccl_comm* comm,
                                                       const ccl::vector_class<ccl::event>& deps,
                                                       bool original_deps,
                                                       const alltoallv_ctx& options,
                                                       bool& done) {
    sycl::event op_end, copy_e;
    sycl::context ctx = q.get_context();
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    uint32_t world = comm->size();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    void* scaleout_send_buf = const_cast<void*>(send_buf);
    void* scaleout_recv_buf = recv_buf;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    bool src_do_fast_gdrcopy = false;
    bool dst_do_fast_gdrcopy = false;
    void *mmap_src_buf, *mmap_dst_buf;

    std::vector<size_t> send_bytes(world, 0);
    std::vector<size_t> recv_bytes(world, 0);
    // if pipelining is enabled, we can use send/recv counts as is
    // because these vectors already have the proper bytes values for each chunk
    if (!options.is_pipeline) {
        for (uint32_t i = 0; i < world; i++) {
            send_bytes[i] = send_counts[i] * ccl_dtype.size();
            recv_bytes[i] = recv_counts[i] * ccl_dtype.size();
        }
    }

    std::vector<size_t> send_displs(world, 0);
    std::vector<size_t> recv_displs(world, 0);
    if (options.is_pipeline && options.chunk_bytes) {
        // in pipeline mode we use host_buf offsets:
        // ranks have equal slots, each calculated as rank * chunk_bytes
        for (uint32_t i = 0; i < world; i++) {
            send_displs[i] = i * options.chunk_bytes;
            recv_displs[i] = i * options.chunk_bytes;
        }
    }
    else {
        for (uint32_t i = 1; i < world; i++) {
            send_displs[i] = send_displs[i - 1] + send_bytes[i - 1];
            recv_displs[i] = recv_displs[i - 1] + recv_bytes[i - 1];
        }
    }

    if (options.copy_to_host) {
        // MPI standard explicitly prohibits IN_PLACE mode for Alltoallv
        // because there is no guaranteed relationship between send and
        // receive layouts. Requesting 2 buffers exceeds the GPU memory
        // limits so we need to be sure that tmp buffer is big enough to
        // contain data for both send and recv buffers
        size_t total_send_count = std::accumulate(send_counts.begin(), send_counts.end(), 0ul);
        scaleout_send_buf = comm->get_scaleout_host_buf();
        scaleout_recv_buf = (char*)scaleout_send_buf + total_send_count * ccl_dtype.size();
        copy_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(scaleout_send_buf, send_buf, total_send_count * ccl_dtype.size());
        });
        dep_events.clear();
        dep_events.push_back(std::move(copy_e));
    }
    else if (!options.is_cpu_buffers) {
        if (!check_mpi_supports_rdma()) {
            LOG_WARN("copy_to_host=false with a GPU buffer. "
                     "TODO: make sure I_MPI_OFFLOAD is set or GPU RDMA is enabled");
        }
    }

    op_end = ccl_host_task(q, dep_events, [=]() {
        // call ccl::wrapper for MPI/OFI.
        int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
        atl_req_t req;
        std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
        ATL_CALL_THROW_IF_ERROR(
            atl_comm->alltoallv(ep_idx,
                                scaleout_send_buf,
                                options.is_pipeline ? send_counts.data() : send_bytes.data(),
                                send_displs.data(),
                                scaleout_recv_buf,
                                options.is_pipeline ? recv_counts.data() : recv_bytes.data(),
                                recv_displs.data(),
                                req));

        atl_comm->check_wait_throw(ep_idx, req);
    });

    if (options.copy_to_host) {
        size_t total_recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0ul);
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(op_end);
            h.memcpy(recv_buf, scaleout_recv_buf, total_recv_count * ccl_dtype.size());
        });
    }

    done = true;
    return op_end;
}

ccl::event alltoallv_scaleout_sycl(sycl::queue& q,
                                   const void* send_buf,
                                   const ccl::vector_class<size_t>& send_counts,
                                   const ccl::vector_class<size_t>& send_displs,
                                   void* recv_buf,
                                   const ccl::vector_class<size_t>& recv_counts,
                                   const ccl::vector_class<size_t>& recv_displs,
                                   ccl::datatype dtype,
                                   ccl_comm* comm,
                                   ccl_stream* global_stream,
                                   const ccl::vector_class<ccl::event>& deps,
                                   bool original_deps,
                                   sycl_scaleout_tune_attr tune_attr,
                                   const alltoallv_ctx& options,
                                   bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    sycl_scaleout_algo algo = tune_attr.algo();

    sycl::event ev;
    ccl::event e;
    done = false;

    switch (algo) {
        case ccl::sycl_policy::algorithm::direct: {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("alltoallv_scaleout_sycl_simple",
                                          "send_size",
                                          options.max_count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| alltoallv scaleout selects simple (direct) kernel, count:",
                      options.max_count,
                      " datatype: ",
                      dtype);
            ev = alltoallv_scaleout_sycl_direct(q,
                                                send_buf,
                                                send_counts,
                                                recv_buf,
                                                recv_counts,
                                                dtype,
                                                comm,
                                                deps,
                                                original_deps,
                                                options,
                                                done);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            return ccl::event::create_from_native(ev);
        }
        case ccl::sycl_policy::algorithm::scatter:
            e = alltoallv_sycl_scatter_rdma(q,
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
                                            options,
                                            done);
            return e;
        case ccl::sycl_policy::algorithm::gdr_only_pairwise:
            e = alltoallv_sycl_global_pairwise_rdma(q,
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
                                                    options,
                                                    done);
            return e;
        case ccl::sycl_policy::algorithm::fallback:
        default: done = false; return e;
    }
}

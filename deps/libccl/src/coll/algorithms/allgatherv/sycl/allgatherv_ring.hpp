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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/coll_util.hpp"

#include "proxy/proxy_types.hpp"
#include "proxy/proxy_staging.hpp"
#include "proxy/host_initiated/cmd_ring_host.hpp"
#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

namespace ccl {
namespace v1 {

// require GPU RDMA
template <typename T>
inline sycl::event allgatherv_ring_blocking(sycl::queue& q,
                                            const void* send_buf,
                                            size_t send_count,
                                            void* recv_buf,
                                            const ccl::vector_class<size_t>& recv_counts,
                                            std::vector<size_t>& offsets,
                                            ccl::datatype dtype,
                                            ccl_comm* comm,
                                            const ccl::vector_class<ccl::event>& deps,
                                            bool& done) {
    sycl::event sycl_e;
    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t send_size = send_count * ccl_dtype.size();
    // calculate count offsets
    if (offsets.empty()) {
        offsets.resize(world);
        offsets[0] = 0;
        for (int rank_idx = 1; rank_idx < world; rank_idx++) {
            offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * ccl_dtype.size();
        }
    }

    bool in_place = ccl::is_allgatherv_inplace(
        send_buf, send_count, recv_buf, recv_counts.data(), offsets.data(), ccl_dtype.size(), rank, world);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    if (!in_place) {
        sycl_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy((char*)recv_buf + offsets[rank], send_buf, send_size);
        });
    }
    else {
        sycl_e = submit_wait_on_events(q, dep_events);
    }

    // blocking
    q.wait();

    // ring buffer pointer to operate
    void *send_ptr, *recv_ptr;
    size_t send_block_size, recv_block_size;

    // ring left-right indexes
    int right = (rank + 1) % world;
    int left = (rank - 1 + world) % world;

    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t sched_id = comm->get_sched_id(true, false);
    int64_t tag = atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), sched_id, 0);

    // loop starting indexes
    int s = rank;
    int r = (s - 1 + world) % world;

    int ep_idx = 0;
    int iter = 0;
    while (iter < world - 1) {
        send_block_size = recv_counts[s] * ccl_dtype.size();
        recv_block_size = recv_counts[r] * ccl_dtype.size();

        send_ptr = (char*)recv_buf + offsets[s];
        recv_ptr = (char*)recv_buf + offsets[r];

        atl_req_t send_req, recv_req;
        ATL_CALL_THROW_IF_ERROR(atl_comm->recv(ep_idx, recv_ptr, recv_block_size, left, tag, recv_req));
        ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx, send_ptr, send_block_size, right, tag, send_req));
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));

        s = r;
        r = (r - 1 + world) % world;
        iter++;
    }

    done = true;
    return sycl_e;
}

template <typename T>
inline sycl::event allgatherv_ring_nonblocking(sycl::queue& q,
                                               const void* send_buf,
                                               size_t send_count,
                                               void* recv_buf,
                                               const ccl::vector_class<size_t>& recv_counts,
                                               std::vector<size_t>& offsets,
                                               ccl::datatype dtype,
                                               ccl_comm* comm,
                                               const ccl::vector_class<ccl::event>& deps,
                                               bool original_deps,
                                               sycl_scaleout_tune_attr tune_attr,
                                               bool& done) {
    sycl::event sycl_e, copy_event;
    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    // tuning parameters
    size_t pipeline_chunk_size = tune_attr.pipeline_chunk_size();

    size_t send_size = send_count * ccl_dtype.size();
    // calculate recv counts offsets
    if (offsets.empty()) {
        offsets.resize(world);
        offsets[0] = 0;
        for (int rank_idx = 1; rank_idx < world; rank_idx++) {
            offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * ccl_dtype.size();
        }
    }

    bool in_place = ccl::is_allgatherv_inplace(
        send_buf, send_count, recv_buf, recv_counts.data(), offsets.data(), ccl_dtype.size(), rank, world);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    if (!in_place) {
        copy_event = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy((char*)recv_buf + offsets[rank], send_buf, send_size);
        });
        dep_events.clear();
        dep_events.push_back(std::move(copy_event));
    }
    else if (original_deps) {
        sycl::event e = submit_wait_on_events(q, dep_events);
        dep_events.clear();
        dep_events.push_back(std::move(e));
    }

    // use an out-of-order queue for GPU pipelining only
    bool use_ooq = !ccl::global_data::env().sycl_enable_pipeline_gpu_rdma;
    sycl::queue q_worker = use_ooq ? sycl::queue(q.get_context(), q.get_device()) : q;

    // send my buffer
    void *send_ptr, *recv_ptr;
    size_t send_block_count, recv_block_count;

    // ring left-right indexes
    int right = (rank + 1) % world;
    int left = (rank - 1 + world) % world;

    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t sched_id = comm->get_sched_id(true, false);
    int64_t tag = atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), sched_id, 100);

    // loop starting indexes
    int s = rank;
    int r = (s - 1 + world) % world;

    // calculate the number of chunks required for pipeline
    size_t nchunks;
    auto min_count = std::min_element(recv_counts.begin(), recv_counts.end());
    auto max_count = std::max_element(recv_counts.begin(), recv_counts.end());
    pipe_prep(*min_count, *max_count, ccl_dtype.size(), pipeline_chunk_size, nchunks);
    LOG_DEBUG("Allgatherv ring min_block_count: ",
              *min_count,
              " max_block_count: ",
              *max_count,
              " nchunks: ",
              nchunks,
              " pipeline_chunk_size: ",
              pipeline_chunk_size);

    int iter = 0;
    sycl::event sendrecv_e;
    while (iter < world - 1) {
        send_block_count = recv_counts[s];
        recv_block_count = recv_counts[r];

        send_ptr = (char*)recv_buf + offsets[s];
        recv_ptr = (char*)recv_buf + offsets[r];

        sendrecv_e = pipe_sendrecv(q_worker,
                                   send_ptr,
                                   send_block_count,
                                   right,
                                   tag,
                                   recv_ptr,
                                   recv_block_count,
                                   left,
                                   tag,
                                   dtype,
                                   nchunks,
                                   comm,
                                   dep_events,
                                   ccl::global_data::env().sycl_enable_pipeline_gpu_rdma); // GPU RDMA

        if (use_ooq) {
            dep_events.clear();
            dep_events.push_back(std::move(sendrecv_e));
        }

        s = r;
        r = (r - 1 + world) % world;
        iter++;
    }

    // submit to in-order queue
    if (use_ooq) {
        sycl_e = submit_wait_on_events(q, dep_events);
    }
    else {
        sycl_e = sendrecv_e;
    }

    done = true;
    return sycl_e;
}

inline sycl::event allgatherv_scaleout_sycl_ring(sycl::queue& q,
                                                 const void* send_buf,
                                                 size_t send_count,
                                                 void* recv_buf,
                                                 const ccl::vector_class<size_t>& recv_counts,
                                                 std::vector<size_t>& recv_offsets,
                                                 ccl::datatype dtype,
                                                 ccl_comm* comm,
                                                 const ccl::vector_class<ccl::event>& deps,
                                                 bool original_deps,
                                                 sycl_scaleout_tune_attr& tune_attr,
                                                 bool& done) {
    auto lambda = [&]<typename T>() {
        if (ccl::global_data::env().enable_op_sync && transport_rdma_enabled()) {
            return allgatherv_ring_blocking<T>(
                q, send_buf, send_count, recv_buf, recv_counts, recv_offsets, dtype, comm, deps, done);
        }
        return allgatherv_ring_nonblocking<T>(q,
                                              send_buf,
                                              send_count,
                                              recv_buf,
                                              recv_counts,
                                              recv_offsets,
                                              dtype,
                                              comm,
                                              deps,
                                              original_deps,
                                              tune_attr,
                                              done);
    };

    return invoke_scaleout_collective(lambda, dtype);
}

// require GPU RDMA or MPI copy pipelining
inline void allgatherv_proxy_ring_blocking(ccl_proxy_context& ctx,
                                           const void* send_buf,
                                           size_t send_count,
                                           void* recv_buf,
                                           std::vector<size_t>& recv_counts,
                                           std::vector<size_t>& offsets,
                                           ccl::datatype dtype,
                                           ccl_comm* comm,
                                           size_t pipeline_chunk_size) {
    int world = comm->size();
    int rank = comm->rank();

    const bool is_rdma_supported = ctx.staging->m_is_rdma_supported;
    const size_t num_work_groups = ctx.num_work_groups;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t send_size = send_count * ccl_dtype.size();
    // calculate count offsets
    if (offsets.empty()) {
        offsets.resize(world);
        offsets[0] = 0;
        for (int rank_idx = 1; rank_idx < world; rank_idx++) {
            offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * ccl_dtype.size();
        }
    }

    LOG_DEBUG("allgatherv_proxy_ring_blocking: rank ",
              rank,
              "/",
              world,
              ", send_count ",
              send_count,
              ", recv_counts [",
              recv_counts[0],
              ", ...], chunk_size ",
              pipeline_chunk_size);

    bool in_place = ccl::is_allgatherv_inplace(
        send_buf, send_count, recv_buf, recv_counts.data(), offsets.data(), ccl_dtype.size(), rank, world);
    if (!in_place) {
        // same as sycl::queue::memcpy((char*)recv_buf + offsets[rank], send_buf, send_size).wait();
        // device-to-device copy via proxy (unified API handles splitting internally)
        ctx.producer->submit_and_wait(
            ccl_ring_cmd_type::copy, (void*)send_buf, (char*)recv_buf + offsets[rank], nullptr, send_size, dtype, num_work_groups);
    }

    // ring buffer pointers to operate
    void *send_ptr, *recv_ptr;
    size_t send_block_size, recv_block_size;

    // ring left-right indexes
    int right = (rank + 1) % world;
    int left = (rank - 1 + world) % world;

    // tag creation
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    ccl_sched_id_t pt2pt_sched_id = atl_comm->tag_creator->get_pt2pt_sched_id();
    int64_t tag = atl_comm->tag_creator->create(0 /* rank */, comm->get_comm_id(), pt2pt_sched_id, 0);

    // loop starting indexes
    int s = rank;
    int r = (s - 1 + world) % world;

    int ep_idx = 0;
    int iter = 0;
    while (iter < world - 1) {
        send_block_size = recv_counts[s] * ccl_dtype.size();
        recv_block_size = recv_counts[r] * ccl_dtype.size();

        send_ptr = (char*)recv_buf + offsets[s];
        recv_ptr = (char*)recv_buf + offsets[r];

        if (is_rdma_supported) {
            atl_req_t send_req, recv_req;
            ATL_CALL_THROW_IF_ERROR(atl_comm->recv(ep_idx, recv_ptr, recv_block_size, left, tag, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->send(ep_idx, send_ptr, send_block_size, right, tag, send_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req));
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req));
        }
        else {
            ccl_proxy_staging::do_sendrecv(ctx.staging,
                                           send_ptr,
                                           send_block_size,
                                           right,
                                           recv_ptr,
                                           recv_block_size,
                                           left,
                                           pipeline_chunk_size,
                                           comm,
                                           tag);
        }

        s = r;
        r = (r - 1 + world) % world;
        iter++;
    }
}

// Proxy service wrapper for ring allgatherv
inline void proxy_allgatherv_ring(const ccl_proxy_op& op, ccl_proxy_context& ctx) {
    const size_t chunk_size = op.pipeline_chunk_size
        ? op.pipeline_chunk_size
        : ccl::global_data::env().sycl_max_pipeline_chunk_size;

    const size_t send_count = op.count;
    std::vector<size_t> recv_counts =
        op.recv_counts.empty() ? std::vector<size_t>(op.comm->size(), send_count) : op.recv_counts;
    std::vector<size_t> offsets = op.offsets;

    allgatherv_proxy_ring_blocking(ctx,
                                   op.src,
                                   send_count,
                                   op.dst,
                                   recv_counts,
                                   offsets,
                                   op.dtype,
                                   op.comm,
                                   chunk_size);
}

// Persistent-kernel based allgatherv ring scale-out algorithm
inline sycl::event allgatherv_scaleout_sycl_proxy_ring(
    sycl::queue& q,
    const void* send_buf,
    size_t send_count,
    void* recv_buf,
    const ccl::vector_class<size_t>& recv_counts,
    const std::vector<size_t>& recv_offsets,
    ccl::datatype dtype,
    ccl_comm* comm,
    const ccl::vector_class<ccl::event>& deps,
    sycl_scaleout_tune_attr tune_attr,
    bool& done) {
    const int comm_size = comm->size();
    const int comm_rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    const size_t send_size = send_count * ccl_dtype.size();

    // Compute byte offsets for each rank's block if the caller did
    // not provide them, so we can locate this rank's slot in recv_buf.
    std::vector<size_t> offsets = recv_offsets;
    if (offsets.empty()) {
        offsets.resize(comm_size);
        offsets[0] = 0;
        for (int rank_idx = 1; rank_idx < comm_size; rank_idx++) {
            offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * ccl_dtype.size();
        }
    }

    LOG_DEBUG("allgatherv_scaleout_sycl_proxy_ring: rank ",
              comm_rank,
              "/",
              comm_size,
              ", send_count ",
              send_count);

    // The only on-device work the allgatherv proxy ring needs is copying this
    // rank's own send block into its slot of recv_buf when not already in place;
    // the ring send/recv itself runs entirely on the host. Do that copy here on
    // the SYCL queue (chained on the input deps) and point the proxy op's source
    // at the in-place slot. This makes the proxy algorithm see an in-place layout
    // and skip the on-device copy, so the persistent kernel does no device compute
    // and we can launch it as a minimal sync-only kernel.
    void* proxy_send_buf = (void*)send_buf;
    bool in_place = ccl::is_allgatherv_inplace(
        send_buf, send_count, recv_buf, recv_counts.data(), offsets.data(), ccl_dtype.size(), comm_rank, comm_size);
    if (!in_place) {
        void* own_slot = (char*)recv_buf + offsets[comm_rank];
        sycl::event copy_event = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(own_slot, send_buf, send_size);
        });
        dep_events.clear();
        dep_events.push_back(std::move(copy_event));
        // Source is now the in-place slot, so the proxy algo sees an in-place layout.
        proxy_send_buf = own_slot;
    }

    ccl_proxy_op op{ ccl_proxy_op_type::allgatherv_ring,
                     proxy_send_buf,
                     recv_buf,
                     /*count=*/send_count,
                     dtype,
                     ccl::reduction::sum, // unused by allgatherv
                     comm,
                     /*aux=*/nullptr };
    op.pipeline_chunk_size = tune_attr.pipeline_chunk_size();
    op.recv_counts.assign(recv_counts.begin(), recv_counts.end());
    op.offsets = std::move(offsets);
    // No on-device compute remains, so a single sync-only work-group is fine.
    op.sync_only = true;

    sycl::event scaleout_event = submit_and_launch_persistent_kernel(q, op, dep_events);

    done = true;
    return scaleout_event;
}

} // namespace v1
} // namespace ccl

/*
 Copyright 2016-2020 Intel Corporation
 
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

namespace ccl {
namespace v1 {

template <typename T>
inline sycl::event allgatherv_ring_blocking(sycl::queue& q,
                                            const void* send_buf,
                                            size_t send_count,
                                            void* recv_buf,
                                            const ccl::vector_class<size_t>& recv_counts,
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
    std::vector<size_t> offsets(world);
    offsets[0] = 0;
    for (int rank_idx = 1; rank_idx < world; rank_idx++) {
        offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * ccl_dtype.size();
    }

    bool in_place = ccl::is_allgatherv_inplace(
        send_buf, send_count, recv_buf, recv_counts.data(), ccl_dtype.size(), rank, world);

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
                                               ccl::datatype dtype,
                                               ccl_comm* comm,
                                               const ccl::vector_class<ccl::event>& deps,
                                               bool original_deps,
                                               sycl_allgatherv_tune_attr tune_attr,
                                               bool& done) {
    sycl::event sycl_e, copy_event;
    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    // tuning parameters
    size_t pipeline_chunk_size = tune_attr.pipeline_chunk_size;

    size_t send_size = send_count * ccl_dtype.size();
    // calculate recv counts offsets
    std::vector<size_t> offsets(world);
    offsets[0] = 0;
    for (int rank_idx = 1; rank_idx < world; rank_idx++) {
        offsets[rank_idx] = offsets[rank_idx - 1] + recv_counts[rank_idx - 1] * ccl_dtype.size();
    }

    bool in_place = ccl::is_allgatherv_inplace(
        send_buf, send_count, recv_buf, recv_counts.data(), ccl_dtype.size(), rank, world);

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

    // use an out-of-order queue
    sycl::queue q_worker(q.get_device());

    // send my buffer
    void *send_ptr, *recv_ptr;
    size_t send_block_count, recv_block_count;

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

        dep_events.clear();
        dep_events.push_back(std::move(sendrecv_e));

        s = r;
        r = (r - 1 + world) % world;
        iter++;
    }

    // submit to in-order queue
    sycl_e = submit_wait_on_events(q, dep_events);

    done = true;
    return sycl_e;
}

inline sycl::event allgatherv_scaleout_sycl_ring(sycl::queue& q,
                                                 const void* send_buf,
                                                 size_t send_count,
                                                 void* recv_buf,
                                                 const ccl::vector_class<size_t>& recv_counts,
                                                 ccl::datatype dtype,
                                                 ccl_comm* comm,
                                                 const ccl::vector_class<ccl::event>& deps,
                                                 bool original_deps,
                                                 sycl_allgatherv_tune_attr& tune_attr,
                                                 bool& done) {
    auto lambda = [&]<typename T>() {
        if (ccl::global_data::env().enable_op_sync) {
            return allgatherv_ring_blocking<T>(
                q, send_buf, send_count, recv_buf, recv_counts, dtype, comm, deps, done);
        }
        else {
            return allgatherv_ring_nonblocking<T>(
                q, send_buf, send_count, recv_buf, recv_counts, dtype, comm, deps, original_deps, tune_attr, done);
        }
    };

    return invoke_scaleout(lambda, dtype);
}

} // namespace v1
} // namespace ccl

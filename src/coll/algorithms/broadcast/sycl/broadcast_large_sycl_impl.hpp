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
#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_kernels.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

ccl::event broadcast_large_impl_ipc(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    int root,
                                    ccl_comm* comm,
                                    sycl::queue q,
                                    sycl_ptrs_type& sycl_ptrs,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("broadcast large kernel no tmp buffer");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    void* root_send_buf = sycl_ptrs.node_ptrs_rd[root];

    sycl::event barrier_event = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    sycl::event copy_event;
    if (comm->rank() == root && send_buf == recv_buf) {
        copy_event = barrier_event;
        LOG_DEBUG("broadcast large inplace, no copy at root", root);
    }
    else {
        copy_event = q.memcpy(recv_buf, root_send_buf, count * dsize, barrier_event);
    }

    // root waits for all ranks to finish reading
    sycl::event ret_event = invoke_barrier(node_comm, q, { copy_event }, is_cpu_barrier);

    return ccl::event::create_from_native(ret_event);
}

ccl::event broadcast_large_impl_tmp(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    int root,
                                    ccl_comm* comm,
                                    sycl::queue q,
                                    sycl_ptrs_type& sycl_ptrs,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("broadcast large kernel tmp buffer");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }
    void* root_send_buf = sycl_ptrs.node_ptrs_rd[root];

    std::vector<void*> tmp_bufs = { get_tmp_buf(0, comm), get_tmp_buf(1, comm) };
    std::vector<void*> remote_tmp_bufs = { get_remote_node_tmp_buf(0, comm)[root],
                                           get_remote_node_tmp_buf(1, comm)[root] };
    const size_t chunk_bytes = (char*)tmp_bufs[1] - (char*)tmp_bufs[0];
    const size_t chunk_count = chunk_bytes / dsize;
    const size_t num_chunks = count / chunk_count + (count % chunk_count != 0);

    for (int i = 0; i < num_chunks; i++) {
        void* send_buf_chunk = (char*)send_buf + i * chunk_bytes;
        void* recv_buf_chunk = (char*)recv_buf + i * chunk_bytes;

        size_t copy_count = chunk_count;
        // last chunk may have lower count
        if (i == num_chunks - 1 && count % chunk_count != 0) {
            copy_count = count % chunk_count;
        }
        const size_t copy_bytes = copy_count * dsize;

        // copy send buffer to tmp buffer
        sycl::event e;
        if (comm->rank() == root) {
            e = q.memcpy(tmp_bufs[i % 2], send_buf_chunk, copy_bytes, dep_events);
            dep_events.clear();
            dep_events.push_back(e);
        }
        if (comm->is_multi_thread_instance() == true) {
            pthread_barrier_wait(
                &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        }
        e = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);
        dep_events.clear();
        dep_events.push_back(e);

        // copy data from tmp buffer in root to recv buffer
        if (comm->rank() != root || send_buf != recv_buf) {
            e = q.memcpy(recv_buf_chunk, remote_tmp_bufs[i % 2], copy_bytes, dep_events);
            dep_events.clear();
            dep_events.push_back(e);
        }
    }
    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }
    sycl::event ret_event = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);
    return ccl::event::create_from_native(ret_event);
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP>
ccl::event broadcast_large_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                int root,
                                ccl_comm* comm,
                                ccl_stream* global_stream,
                                sycl_ptrs_type& sycl_ptrs,
                                const ccl::vector_class<ccl::event>& deps) {
    sycl::queue q = global_stream->get_native_stream();

    const bool is_tmp_used = ccl::global_data::env().sycl_broadcast_tmp_buf;

    ccl::event e;
    if (!is_tmp_used) {
        e = broadcast_large_impl_ipc(
            send_buf, recv_buf, count, dtype, root, comm, q, sycl_ptrs, deps);
    }
    else {
        e = broadcast_large_impl_tmp(
            send_buf, recv_buf, count, dtype, root, comm, q, sycl_ptrs, deps);
    }

    return e;
}

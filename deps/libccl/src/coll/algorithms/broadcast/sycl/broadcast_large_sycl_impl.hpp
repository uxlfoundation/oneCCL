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
#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/broadcast/sycl/broadcast_large_sycl_ring.hpp"
// AUTO policy table for the pipelined ring (lookup by (world, total_bytes)).
#include "coll/algorithms/broadcast/sycl/broadcast_ring_auto_policy.hpp"

ccl::event broadcast_large_impl_ipc(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    int root,
                                    ccl_comm* comm,
                                    std::shared_ptr<ccl_comm> p2p_comm,
                                    sycl::queue q,
                                    sycl_ptrs_type& sycl_ptrs,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("broadcast large kernel no tmp buffer");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    void* root_send_buf = sycl_ptrs.node_ptrs_rd[root];

    sycl::event barrier_event = invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);

    sycl::event copy_event;
    if (p2p_comm->rank() == root && send_buf == recv_buf) {
        copy_event = barrier_event;
        LOG_DEBUG("broadcast large inplace, no copy at root", root);
    }
    else {
        copy_event = q.memcpy(recv_buf, root_send_buf, count * dsize, barrier_event);
    }

    // root waits for all ranks to finish reading
    sycl::event ret_event = invoke_barrier(p2p_comm, q, { copy_event }, is_cpu_barrier);

    return ccl::event::create_from_native(ret_event);
}

ccl::event broadcast_large_impl_tmp(const void* send_buf,
                                    void* recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    int root,
                                    ccl_comm* comm,
                                    std::shared_ptr<ccl_comm> p2p_comm,
                                    sycl::queue q,
                                    sycl_ptrs_type& sycl_ptrs,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("broadcast large kernel tmp buffer");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }
    void* root_send_buf = sycl_ptrs.node_ptrs_rd[root];

    std::vector<void*> tmp_bufs = { get_large_tmp_buf(0, comm), get_large_tmp_buf(1, comm) };
    std::vector<void*> remote_tmp_bufs = { get_remote_large_tmp_buf(0, p2p_comm.get())[root],
                                           get_remote_large_tmp_buf(1, p2p_comm.get())[root] };
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
        if (p2p_comm->rank() == root) {
            e = q.memcpy(tmp_bufs[i % 2], send_buf_chunk, copy_bytes, dep_events);
            dep_events.clear();
            dep_events.push_back(e);
        }
        if (comm->is_multi_thread_instance() == true) {
            pthread_barrier_wait(
                ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        }
        e = invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);
        dep_events.clear();
        dep_events.push_back(e);

        // copy data from tmp buffer in root to recv buffer
        if (p2p_comm->rank() != root || send_buf != recv_buf) {
            e = q.memcpy(recv_buf_chunk, remote_tmp_bufs[i % 2], copy_bytes, dep_events);
            dep_events.clear();
            dep_events.push_back(e);
        }
    }
    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
    }
    sycl::event ret_event = invoke_barrier(p2p_comm, q, dep_events, is_cpu_barrier);
    return ccl::event::create_from_native(ret_event);
}

template <typename T>
ccl::event broadcast_large_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                int root,
                                ccl_comm* comm,
                                std::shared_ptr<ccl_comm> p2p_comm,
                                ccl_stream* global_stream,
                                sycl_ptrs_type& sycl_ptrs,
                                const ccl::vector_class<ccl::event>& deps) {
    sycl::queue q = global_stream->get_native_stream();

    // MT instances force is_tmp_used=1 in broadcast_large(); the tmp path is
    // the only one with pthread_barrier_wait MT synchronisation, so route
    // there first before evaluating any of the ring / IPC alternatives.
    const bool is_tmp_used = ccl::global_data::env().sycl_broadcast_tmp_buf;
    if (is_tmp_used) {
        return broadcast_large_impl_tmp(
            send_buf, recv_buf, count, dtype, root, comm, p2p_comm, q, sycl_ptrs, deps);
    }

    // ARC/BMG default (ring ON unless CCL_SYCL_BROADCAST_RING was set
    // explicitly) is resolved centrally in env_data::adjust_device_defaults,
    // so this field already carries the effective value.
    const bool is_arc = is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()));
    const bool use_ring = ccl::global_data::env().sycl_broadcast_ring;
    const bool use_ring_auto = ccl::global_data::env().sycl_broadcast_ring_auto;
    const bool use_copy_engine = ccl::global_data::env().sycl_broadcast_ring_copy_engine;
    const size_t total_bytes = count * ccl::global_data::get().dtypes->get(dtype).size();

    // AUTO policy: when both RING=1 and RING_AUTO=1, the (world, size) lookup
    // table can either decline the ring (route back to baseline IPC) or accept
    // it with a tuned slice-floor threshold. The table is calibrated for the
    // DMA copy-engine ring; when the copy-kernel ring is selected we still
    // consult it for the use_ring decision but discard the threshold override
    // (the copy-kernel path has its own performance characteristics).
    bool effective_use_ring = use_ring && is_arc;
    bool use_auto_threshold_override = false;
    size_t threshold_auto_override_bytes = 0;
    if (effective_use_ring && use_ring_auto) {
        const broadcast_ring_auto_decision decision{ total_bytes, p2p_comm->size() };
        effective_use_ring = decision.use_ring;
        if (use_copy_engine) {
            threshold_auto_override_bytes = decision.threshold_bytes;
            use_auto_threshold_override = decision.use_ring;
        }
    }

    if (effective_use_ring && use_copy_engine) {
        return broadcast_large_impl_ring_copy_engine(send_buf,
                                                     recv_buf,
                                                     count,
                                                     dtype,
                                                     root,
                                                     p2p_comm,
                                                     q,
                                                     sycl_ptrs,
                                                     deps,
                                                     threshold_auto_override_bytes,
                                                     use_auto_threshold_override);
    }
    if (effective_use_ring) {
        return broadcast_large_impl_ring_copy_kernel<T>(send_buf,
                                                        recv_buf,
                                                        count,
                                                        dtype,
                                                        root,
                                                        p2p_comm,
                                                        q,
                                                        sycl_ptrs,
                                                        deps,
                                                        threshold_auto_override_bytes,
                                                        use_auto_threshold_override);
    }
    return broadcast_large_impl_ipc(
        send_buf, recv_buf, count, dtype, root, comm, p2p_comm, q, sycl_ptrs, deps);
}

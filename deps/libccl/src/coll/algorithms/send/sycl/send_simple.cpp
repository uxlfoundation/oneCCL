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

#include "common/global/global.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/send/sycl/send_simple.hpp"
#endif // CCL_ENABLE_SYCL

ccl::event send_simple(const void *send_buf,
                       size_t send_count,
                       ccl::datatype dtype,
                       int peer_rank,
                       ccl_comm *comm,
                       ccl_stream *global_stream,
                       const ccl::vector_class<ccl::event> &deps,
                       bool &done) {
    sycl::queue q = global_stream->get_native_stream();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int node_peer_rank = node_comm->get_rank_from_global(peer_rank);

    LOG_DEBUG("send_simple send_count: ", send_count, " peer rank: ", peer_rank);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    // Reuse the small tmp buffer already IPC-exchanged by coll_init.
    // remote_ptrs[node_peer_rank] is a device pointer into the peer's
    // tmp buffer; that is the staging slot we write into.
    // No usage of cpu/gpu based counters, no need to use tmp_bufs_gpu.
    auto [local_tmp_buf, remote_ptrs] = node_comm->get_all_tmp_bufs(true);
    void *staging = remote_ptrs[node_peer_rank];
    CCL_THROW_IF_NOT(staging,
                     "send_simple: remote tmp buf is null for peer ",
                     node_peer_rank);

    sycl::device device = q.get_device();
    const size_t max_work_group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 && work_group_size <= max_work_group_size);

    const size_t max_threads = get_total_threads(q) * 4;
    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    if (!num_threads)
        num_threads = max_threads;
    num_threads = std::min(num_threads, max_threads);

    // Staging slot capacity per peer (bytes). The kernel chunks the user
    // message into pieces of at most this size so the slot is never overflowed.
    const size_t staging_bytes = node_comm->get_tmp_buf_size();

    auto lambda = [&]<typename T>() -> sycl::event {
        constexpr int vec_size = get_num_elements<T, 16, true>();

        const size_t chunk_count = staging_bytes / sizeof(T);
        CCL_THROW_IF_NOT(chunk_count > 0,
                         "send_simple: staging slot smaller than one element of type, ",
                         "staging_bytes=", staging_bytes, " sizeof(T)=", sizeof(T));

        // Size the launch for one chunk, not the full message — the kernel
        // loops over chunks internally and reuses the same launch shape.
        const size_t threads_for_one_chunk =
            chunk_count / vec_size + chunk_count % vec_size;
        size_t kernel_threads = std::min(threads_for_one_chunk, num_threads);
        // For messages smaller than one chunk, scale the launch down to the
        // actual work; copy_kernel_wrap handles thread/element imbalance.
        const size_t threads_for_message = send_count / vec_size + send_count % vec_size;
        kernel_threads = std::min(kernel_threads, std::max(threads_for_message, size_t{ 1 }));
        // p2p_barrier_pair uses a uniform group barrier, so every work-item
        // must reach it. Ensure at least MAX_NODE_RANKS work-items.
        kernel_threads = std::max(kernel_threads, size_t{ MAX_NODE_RANKS });
        const size_t kernel_size =
            ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
        sycl::nd_range<1> nd_range(kernel_size, work_group_size);

        SendSimple<T, vec_size> k(
            staging, send_buf, send_count, chunk_count, node_peer_rank, node_comm.get());
        return q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.parallel_for(nd_range, k);
        });
    };

    sycl::event e = invoke_collective_sycl(lambda, dtype);
    done = true;
    return ccl::event::create_from_native(e);
}

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
#include "coll/algorithms/recv/sycl/recv_memcpy.hpp"
#endif // CCL_ENABLE_SYCL

ccl::event recv_memcpy(void *recv_buf,
                       size_t recv_count,
                       ccl::datatype dtype,
                       int peer_rank,
                       ccl_comm *comm,
                       ccl_stream *global_stream,
                       const ccl::vector_class<ccl::event> &deps,
                       bool &done) {
    sycl::queue q = global_stream->get_native_stream();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int node_peer_rank = node_comm->get_rank_from_global(peer_rank);

    LOG_DEBUG("recv_memcpy recv_count: ", recv_count, " peer rank: ", peer_rank);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    // Same staging slot as recv_simple — our own tmp buffer (the sender
    // writes into it via its IPC view of remote_ptrs[our_rank]). The
    // remote_ptrs half of the pair is unused on the receiver side.
    // No usage of cpu/gpu based counters, no need to use tmp_bufs_gpu.
    void *local_tmp_buf = node_comm->get_all_tmp_bufs(true).first;
    CCL_THROW_IF_NOT(local_tmp_buf, "recv_memcpy: local tmp buf is null");

    const size_t dtype_size = ccl::global_data::get().dtypes->get(dtype).size();
    const size_t total_bytes = recv_count * dtype_size;
    const size_t staging_bytes = node_comm->get_tmp_buf_size();

    // Per chunk:
    //   1. invoke_p2p_barrier_pair                          -- wait for sender
    //   2. q.memcpy(dst+offset, staging, this_chunk_bytes)  -- driver copy
    //   3. invoke_p2p_barrier_pair                          -- release slot
    //
    // `last` carries the previous chunk's barrier #2 event into the next
    // chunk's first barrier as a dep; the first chunk uses the caller's
    // deps directly.
    sycl::event last;
    std::vector<sycl::event> bdeps;

    for (size_t off = 0; off < total_bytes; off += staging_bytes) {
        const size_t this_chunk = std::min(staging_bytes, total_bytes - off);
        char *dst_off = static_cast<char *>(recv_buf) + off;

        bdeps.clear();
        if (off == 0) {
            bdeps = dep_events;
        } else {
            bdeps.push_back(last);
        }
        // Barrier #1: wait for sender to publish chunk.
        sycl::event b1 = invoke_p2p_barrier_pair(node_comm, node_peer_rank, q, bdeps);
        sycl::event copy_ev = q.memcpy(dst_off, local_tmp_buf, this_chunk, { b1 });
        // Barrier #2: signal slot drained.
        last = invoke_p2p_barrier_pair(node_comm, node_peer_rank, q, { copy_ev });
    }

    done = true;
    return ccl::event::create_from_native(last);
}

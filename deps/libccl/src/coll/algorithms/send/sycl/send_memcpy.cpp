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
#include "coll/algorithms/send/sycl/send_memcpy.hpp"
#endif // CCL_ENABLE_SYCL

ccl::event send_memcpy(const void *send_buf,
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

    LOG_DEBUG("send_memcpy send_count: ", send_count, " peer rank: ", peer_rank);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    // Same staging slot as send_simple — peer's IPC-mapped tmp buffer.
    // Only the peer's slot pointer is needed on the sender side; local_tmp_buf
    // would be the slot we receive into, which is irrelevant here.
    // No usage of cpu/gpu based counters, no need to use tmp_bufs_gpu.
    void *staging = node_comm->get_all_tmp_bufs(true).second[node_peer_rank];
    CCL_THROW_IF_NOT(staging,
                     "send_memcpy: remote tmp buf is null for peer ",
                     node_peer_rank);

    const size_t dtype_size = ccl::global_data::get().dtypes->get(dtype).size();
    const size_t total_bytes = send_count * dtype_size;
    const size_t staging_bytes = node_comm->get_tmp_buf_size();

    // Stream the message through the staging slot in chunks of at most
    // staging_bytes. Per chunk:
    //   1. q.memcpy(staging, src+offset, this_chunk_bytes)  -- driver copy
    //   2. invoke_p2p_barrier_pair                          -- data ready
    //   3. invoke_p2p_barrier_pair                          -- slot drained
    //
    // `last` carries the previous chunk's barrier #2 event into the next
    // chunk's memcpy as a dep; the first chunk uses the caller's deps
    // directly (the in-order queue + h.depends_on chain in
    // invoke_p2p_barrier_pair gives ordering across submits).
    sycl::event last;
    std::vector<sycl::event> mdeps;

    for (size_t off = 0; off < total_bytes; off += staging_bytes) {
        const size_t this_chunk = std::min(staging_bytes, total_bytes - off);
        const char *src_off = static_cast<const char *>(send_buf) + off;

        mdeps.clear();
        if (off == 0) {
            mdeps = dep_events;
        } else {
            mdeps.push_back(last);
        }
        sycl::event copy_ev = q.memcpy(staging, src_off, this_chunk, mdeps);
        // Barrier #1: data visible to peer.
        sycl::event b1 = invoke_p2p_barrier_pair(node_comm, node_peer_rank, q, { copy_ev });
        // Barrier #2: peer drained, slot is reusable.
        last = invoke_p2p_barrier_pair(node_comm, node_peer_rank, q, { b1 });
    }

    done = true;
    return ccl::event::create_from_native(last);
}

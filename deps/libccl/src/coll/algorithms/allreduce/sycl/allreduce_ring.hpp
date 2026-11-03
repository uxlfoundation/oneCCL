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

#include "proxy/proxy.hpp"
#include "proxy/proxy_types.hpp"
#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_ring.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_ring.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

namespace ccl {
namespace v1 {

template <typename T>
inline sycl::event allreduce_ring_blocking(sycl::queue &q,
                                           const void *send_buf,
                                           void *recv_buf,
                                           size_t count,
                                           datatype dtype,
                                           reduction reduction,
                                           ccl_comm *comm,
                                           const ccl::vector_class<ccl::event> &deps,
                                           bool &done) {
    const int world = comm->size();
    const int rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const size_t main_block_count = count / world;
    const size_t last_block_count = main_block_count + count % world;

    void *recv_local_buf = (char *)recv_buf + rank * main_block_count * ccl_dtype.size();
    sycl::event rs_event = reduce_scatter_ring_blocking_impl<T>(
        q, send_buf, recv_local_buf, count, dtype, reduction, comm, deps, done);

    ccl::event rs_ccl_event = ccl::event::create_from_native(rs_event);
    std::vector<ccl::event> vector_events;
    vector_events.push_back(std::move(rs_ccl_event));

    size_t ag_send_count = rank == world - 1 ? last_block_count : main_block_count;
    std::vector<size_t> recv_counts(world - 1, main_block_count);
    recv_counts.push_back(last_block_count);

    void *send_local_buf = recv_local_buf;
    std::vector<size_t> offsets;
    sycl::event ag_event = allgatherv_ring_blocking<T>(q,
                                                       send_local_buf,
                                                       ag_send_count,
                                                       recv_buf,
                                                       recv_counts,
                                                       offsets,
                                                       dtype,
                                                       comm,
                                                       vector_events,
                                                       done);

    done = true;
    return ag_event;
}

template <typename T>
inline sycl::event allreduce_ring_nonblocking(sycl::queue &q,
                                              const void *send_buf,
                                              void *recv_buf,
                                              size_t count,
                                              datatype dtype,
                                              reduction reduction,
                                              ccl_comm *comm,
                                              const ccl::vector_class<ccl::event> &deps,
                                              bool original_deps,
                                              sycl_scaleout_tune_attr tune_attr,
                                              bool &done) {
    int world = comm->size();
    int rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const size_t main_block_count = count / world;
    const size_t last_block_count = main_block_count + count % world;

    void *recv_local_buf = (char *)recv_buf + rank * main_block_count * ccl_dtype.size();

    sycl_scaleout_tune_attr rs_tune_attr = tune_attr;
    rs_tune_attr.set_algo(ccl::sycl_policy::algorithm::ring);
    sycl::event rs_event = reduce_scatter_ring_nonblocking_impl<T>(q,
                                                                   send_buf,
                                                                   recv_local_buf,
                                                                   count,
                                                                   dtype,
                                                                   reduction,
                                                                   comm,
                                                                   deps,
                                                                   original_deps,
                                                                   rs_tune_attr,
                                                                   done);

    ccl::event rs_ccl_event = ccl::event::create_from_native(rs_event);
    std::vector<ccl::event> vector_events;
    vector_events.push_back(std::move(rs_ccl_event));

    size_t ag_send_count = rank == world - 1 ? last_block_count : main_block_count;
    std::vector<size_t> recv_counts(world - 1, main_block_count);
    recv_counts.push_back(last_block_count);

    void *send_local_buf = recv_local_buf;
    sycl_scaleout_tune_attr ag_tune_attr = tune_attr;
    ag_tune_attr.set_algo(ccl::sycl_policy::algorithm::ring);
    std::vector<size_t> offsets;
    sycl::event ag_event = allgatherv_ring_nonblocking<T>(q,
                                                          send_local_buf,
                                                          ag_send_count,
                                                          recv_buf,
                                                          recv_counts,
                                                          offsets,
                                                          dtype,
                                                          comm,
                                                          vector_events,
                                                          false /* original_deps */,
                                                          ag_tune_attr,
                                                          done);

    done = true;
    return ag_event;
}

inline sycl::event allreduce_scaleout_sycl_ring(sycl::queue &q,
                                                const void *send_buf,
                                                void *recv_buf,
                                                size_t count,
                                                datatype dtype,
                                                reduction reduction,
                                                ccl_comm *comm,
                                                const ccl::vector_class<ccl::event> &deps,
                                                bool original_deps,
                                                sycl_scaleout_tune_attr tune_attr,
                                                bool &done) {
    auto lambda = [&]<typename T>() {
        if (ccl::global_data::env().enable_op_sync) {
            return allreduce_ring_blocking<T>(
                q, send_buf, recv_buf, count, dtype, reduction, comm, deps, done);
        }
        else {
            return allreduce_ring_nonblocking<T>(q,
                                                 send_buf,
                                                 recv_buf,
                                                 count,
                                                 dtype,
                                                 reduction,
                                                 comm,
                                                 deps,
                                                 original_deps,
                                                 tune_attr,
                                                 done);
        }
    };

    return invoke_scaleout_collective(lambda, dtype);
}

/* ======== Proxy service path ======== */

// Proxy-adapted blocking allreduce ring for use by the proxy thread
// Executes reduce-scatter + allgather phases with explicit staging and command ring
inline void allreduce_proxy_ring_blocking(ccl_proxy_context& ctx,
                                          const void* send_buf,
                                          void* recv_buf,
                                          size_t count,
                                          ccl::datatype dtype,
                                          ccl::reduction reduction,
                                          ccl_comm* comm,
                                          void* aux_buffer,
                                          size_t pipeline_chunk_size) {
    // === Reduce-scatter phase ===
    const int world = comm->size();
    const int rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const size_t main_block_count = count / world;
    const size_t last_block_count = main_block_count + count % world;

    void *recv_local_buf = (char *)recv_buf + rank * main_block_count * ccl_dtype.size();

    // Run actual reduce_scatter algorithm using the provided parameters
    reduce_scatter_proxy_ring_blocking_impl(ctx,
                                            send_buf,
                                            recv_local_buf,
                                            count,
                                            dtype,
                                            reduction,
                                            comm,
                                            aux_buffer,
                                            pipeline_chunk_size);

    // === Allgather phase ===
    size_t ag_send_count = rank == world - 1 ? last_block_count : main_block_count;
    std::vector<size_t> recv_counts(world - 1, main_block_count);
    recv_counts.push_back(last_block_count);

    void* send_local_buf = recv_local_buf;
    std::vector<size_t> offsets;

    // Run actual allgather algorithm using the provided parameters
    allgatherv_proxy_ring_blocking(ctx,
                                   send_local_buf,
                                   ag_send_count,
                                   recv_buf,
                                   recv_counts,
                                   offsets,
                                   dtype,
                                   comm,
                                   pipeline_chunk_size);
}

// Proxy service wrapper for ring allreduce
// Unpacks ccl_proxy_op and calls allreduce_proxy_ring_blocking
inline void proxy_allreduce_ring(const ccl_proxy_op& op, ccl_proxy_context& ctx) {
    // Helper to get pipeline chunk size
    const size_t chunk_size = op.pipeline_chunk_size
        ? op.pipeline_chunk_size
        : ccl::global_data::env().sycl_max_pipeline_chunk_size;

    // Unpack operation descriptor and call the algorithm
    allreduce_proxy_ring_blocking(ctx,
                                  op.src,
                                  op.dst,
                                  op.count,
                                  op.dtype,
                                  op.reduction,
                                  op.comm,
                                  op.aux_dev,
                                  chunk_size);
}

// Note: example of using persistent kernel for current ccl algorithms
inline sycl::event allreduce_scaleout_sycl_proxy_ring(
    sycl::queue& q,
    const void* send_buf,
    void* recv_buf,
    size_t count,
    ccl::datatype dtype,
    ccl::reduction reduction,
    ccl_comm* comm,
    const ccl::vector_class<ccl::event>& deps,
    sycl_scaleout_tune_attr tune_attr,
    bool& done) {
    bool is_recording = use_recording_path(q);
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    // Obtain rank and communicator size
    const int rank = comm->rank();
    const int comm_size = comm->size();

    // Uneven decomposition into "size" blocks:
    // blocks [0..comm_size-2] have main_block_count
    // block  [comm_size-1]    has last_block_count
    const size_t main_block_count = count / static_cast<size_t>(comm_size);
    const size_t last_block_count = main_block_count + (count % static_cast<size_t>(comm_size));
    const size_t last_block_size = last_block_count * ccl_dtype.size();

    // Decide how the reduce-scatter aux (double-buffer) memory is provided so it
    // can hold the largest possible block.
    // Common case: the scaleout device-buffer pool slot is large enough.
    // Large-block fallback: the block exceeds half a pool slot, so we allocate a
    // dedicated device buffer here and hand it to the algorithm via the op. The
    // algorithm borrows it (non-null aux) and does not touch its lifetime; we
    // free it below once the kernel has finished.
    // TODO: maybe just drop it an do chunking on scaleout_device_buf inside algo
    bool to_free = false;
    void* aux_buffer = nullptr;
    if (last_block_size > comm->get_scaleout_device_buf_size() / 2) {
        CCL_THROW_IF_NOT(!is_recording, "large-block allocation does not support recording");
        aux_buffer = sycl::malloc_device(last_block_size * 2, q);
        to_free = true;
    }

    // Decide work-group count based on minimum block size in reduce-scatter
    const size_t min_block_size = main_block_count * ccl_dtype.size();
    const size_t num_work_groups = select_persistent_work_group_num(min_block_size);

    LOG_DEBUG("allreduce_scaleout_sycl_proxy_ring: rank ",
              rank,
              "/",
              comm_size,
              ", count ",
              count,
              ", main_block_count ",
              main_block_count,
              ", last_block_count ",
              last_block_count,
              ", num_work_groups ",
              num_work_groups);

    ccl_proxy_op op{ ccl_proxy_op_type::allreduce_ring,
                     (void*)send_buf,
                     recv_buf,
                     count,
                     dtype,
                     reduction,
                     comm,
                     /*aux=*/aux_buffer };
    // Optional
    op.pipeline_chunk_size = tune_attr.pipeline_chunk_size();
    op.num_work_groups = num_work_groups;

    sycl::event scaleout_event = submit_and_launch_persistent_kernel(q, op, dep_events);

    // Free the large-block fallback allocation once the kernel has finished.
    if (to_free) {
        scaleout_event = q.submit([=](sycl::handler& h) {
            h.depends_on(scaleout_event);
            h.host_task([=]() {
                sycl::free(aux_buffer, q);
            });
        });
    }

    done = true;
    return scaleout_event;
}

} // namespace v1
} // namespace ccl

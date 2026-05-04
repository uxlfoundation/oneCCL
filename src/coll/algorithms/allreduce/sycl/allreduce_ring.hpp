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
                                              sycl_allreduce_tune_attr tune_attr,
                                              bool &done) {
    int world = comm->size();
    int rank = comm->rank();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const size_t main_block_count = count / world;
    const size_t last_block_count = main_block_count + count % world;

    void *recv_local_buf = (char *)recv_buf + rank * main_block_count * ccl_dtype.size();
    sycl_reduce_scatter_tune_attr rs_tune_attr = { reduce_scatter_scaleout_algo::ring,
                                                   tune_attr.pipeline_chunk_size };
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
    sycl_allgatherv_tune_attr ag_tune_attr = { allgatherv_scaleout_algo::ring,
                                               tune_attr.pipeline_chunk_size };
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
                                                sycl_allreduce_tune_attr tune_attr,
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

} // namespace v1
} // namespace ccl

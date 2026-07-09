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

#include <vector>
#include <optional>

#include "common/utils/enums.hpp"
#include "common/utils/buffer.hpp"
#include "oneapi/ccl/types.hpp"
#include "internal_types.hpp"

#define CCL_COLL_LIST \
    ccl_coll_allgather, ccl_coll_allgatherv, ccl_coll_allreduce, ccl_coll_alltoall, \
        ccl_coll_alltoallv, ccl_coll_barrier, ccl_coll_bcast, ccl_coll_broadcast, ccl_coll_recv, \
        ccl_coll_reduce, ccl_coll_reduce_scatter, ccl_coll_send, ccl_coll_scatter

enum ccl_coll_allgather_algo {
    ccl_coll_allgather_undefined = 0,

    ccl_coll_allgather_direct,
    ccl_coll_allgather_naive,
    ccl_coll_allgather_ring,
    ccl_coll_allgather_flat,
    ccl_coll_allgather_multi_bcast,
    ccl_coll_allgather_topo
};

enum ccl_coll_allgatherv_algo {
    ccl_coll_allgatherv_undefined = 0,

    ccl_coll_allgatherv_direct,
    ccl_coll_allgatherv_naive,
    ccl_coll_allgatherv_ring,
    ccl_coll_allgatherv_flat,
    ccl_coll_allgatherv_multi_bcast,
    ccl_coll_allgatherv_topo
};

enum ccl_coll_allreduce_algo {
    ccl_coll_allreduce_undefined = 0,

    ccl_coll_allreduce_direct,
    ccl_coll_allreduce_rabenseifner,
    ccl_coll_allreduce_nreduce,
    ccl_coll_allreduce_ring,
    ccl_coll_allreduce_ring_rma,
    ccl_coll_allreduce_double_tree,
    ccl_coll_allreduce_recursive_doubling,
    ccl_coll_allreduce_2d,
    ccl_coll_allreduce_topo
};

enum ccl_coll_alltoall_algo {
    ccl_coll_alltoall_undefined = 0,

    ccl_coll_alltoall_direct,
    ccl_coll_alltoall_naive,
    ccl_coll_alltoall_scatter,
    ccl_coll_alltoall_topo
};

enum ccl_coll_alltoallv_algo {
    ccl_coll_alltoallv_undefined = 0,

    ccl_coll_alltoallv_direct,
    ccl_coll_alltoallv_naive,
    ccl_coll_alltoallv_scatter,
    ccl_coll_alltoallv_topo
};

enum ccl_coll_barrier_algo {
    ccl_coll_barrier_undefined = 0,

    ccl_coll_barrier_direct,
    ccl_coll_barrier_ring
};

enum ccl_coll_bcast_algo {
    ccl_coll_bcast_undefined = 0,

    ccl_coll_bcast_direct,
    ccl_coll_bcast_ring,
    ccl_coll_bcast_double_tree,
    ccl_coll_bcast_naive,
    ccl_coll_bcast_topo
};

enum ccl_coll_broadcast_algo {
    ccl_coll_broadcast_undefined = 0,

    ccl_coll_broadcast_direct,
    ccl_coll_broadcast_ring,
    ccl_coll_broadcast_double_tree,
    ccl_coll_broadcast_naive,
    ccl_coll_broadcast_topo
};

enum ccl_coll_recv_algo {
    ccl_coll_recv_undefined = 0,

    ccl_coll_recv_direct,
    ccl_coll_recv_offload,
    ccl_coll_recv_topo
};

enum ccl_coll_reduce_algo {
    ccl_coll_reduce_undefined = 0,

    ccl_coll_reduce_direct,
    ccl_coll_reduce_rabenseifner,
    ccl_coll_reduce_ring,
    ccl_coll_reduce_tree,
    ccl_coll_reduce_double_tree,
    ccl_coll_reduce_topo
};

enum ccl_coll_reduce_scatter_algo {
    ccl_coll_reduce_scatter_undefined = 0,

    ccl_coll_reduce_scatter_direct,
    ccl_coll_reduce_scatter_naive,
    ccl_coll_reduce_scatter_ring,
    ccl_coll_reduce_scatter_topo
};

enum ccl_coll_send_algo {
    ccl_coll_send_undefined = 0,

    ccl_coll_send_direct,
    ccl_coll_send_offload,
    ccl_coll_send_topo
};

enum ccl_coll_scatter_algo {
    ccl_coll_scatter_undefined = 0,

    ccl_coll_scatter_naive
};

union ccl_coll_algo {
    ccl_coll_allgather_algo allgather;
    ccl_coll_allgatherv_algo allgatherv;
    ccl_coll_allreduce_algo allreduce;
    ccl_coll_alltoall_algo alltoall;
    ccl_coll_alltoallv_algo alltoallv;
    ccl_coll_barrier_algo barrier;
    ccl_coll_bcast_algo bcast;
    ccl_coll_broadcast_algo broadcast;
    ccl_coll_recv_algo recv;
    ccl_coll_reduce_algo reduce;
    ccl_coll_reduce_scatter_algo reduce_scatter;
    ccl_coll_send_algo send;
    ccl_coll_scatter_algo scatter;
    int value;

    ccl_coll_algo() : value(0) {}
    bool has_value() const {
        return (value != 0);
    }
};

enum ccl_coll_type {
    ccl_coll_allgather,
    ccl_coll_allgatherv,
    ccl_coll_allreduce,
    ccl_coll_alltoall,
    ccl_coll_alltoallv,
    ccl_coll_barrier,
    ccl_coll_bcast,
    ccl_coll_broadcast,
    ccl_coll_recv,
    ccl_coll_reduce,
    ccl_coll_reduce_scatter,
    ccl_coll_send,
    ccl_coll_scatter,
    ccl_coll_last_regular = ccl_coll_scatter,

    ccl_coll_partial,
    ccl_coll_undefined,

    ccl_coll_last_value
};

/*
 * Chunking mode describes how an operation should be divided into smaller parts
 * to achieve either better performance through parallel execution or to limit
 * used resources by executing smaller parts sequentially.
 * 
 * Currently available chunking modes:
 *  1) `ccl_pipeline_none` - operation is NON divisible and no chunking should happen
 *  2) `ccl_buffer_implicit` - operation can be ran in multiple chunks, the operation
 *                             is built the same way as without chunking by providing
 *                             `send_buf` and `recv_buf` with appropriate offsets.
 *  3) `ccl_offset_explicit` - operation can be ran in multiple chunks, each chunk of
 *                             the operation receives the SAME `send_buf` and `recv_buf`,
 *                             so implementation of building function has to take into
 *                             account supplied `recv_buf` offset and other parameters.
 *  
*/
enum ccl_chunking_mode { ccl_pipeline_none = 0, ccl_buffer_implicit, ccl_offset_explicit };
std::string to_string(ccl_chunking_mode& mode);

const char* ccl_coll_type_to_str(ccl_coll_type type);

void ccl_get_segment_sizes(size_t dtype_size,
                           size_t elem_count,
                           size_t requested_seg_size,
                           std::vector<size_t>& seg_sizes);

class ccl_sched;

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
class ccl_comm;

std::optional<size_t> ccl_get_pipe_size(const size_t buf_size,
                                        const size_t dtype_size,
                                        const size_t chunk_count);
bool ccl_is_ptr_aligned(uintptr_t start_ptr, size_t mem_align);
ccl::status ccl_build_topo_uniform_buff_size_op(
    ccl_sched* sched,
    ccl_buffer send_buf,
    ccl_buffer recv_buf,
    size_t count,
    size_t dtype_size,
    size_t pipe_nof_chunks,
    const std::string& op_name,
    ccl::profile::metrics_counter& metrics,
    ccl_comm* comm,
    std::function<ccl::status(ccl_sched* sched,
                              ccl_buffer send_buf,
                              ccl_buffer recv_buf,
                              size_t count,
                              size_t offset,
                              size_t combined_count)> fill_op_lambda,
    ccl_chunking_mode mode,
    ccl_coll_type coll);
uint32_t ccl_submit_ze_commands_in_subsched_entries(ccl_sched* sched);
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

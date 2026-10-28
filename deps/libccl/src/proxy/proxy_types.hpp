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

/* proxy_types.hpp
 *
 * Common types shared across the proxy infrastructure.
 * Contains data types, communicator wrapper, and staging memory.
 */

#pragma once

#include "oneapi/ccl/types.hpp"
#include "comm/comm.hpp"
#include "common/utils/utils.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <stdexcept>
#include <vector>

// Forward declarations
class ccl_cmd_ring_producer;
class ccl_proxy_staging;

/* ======== Proxy operation types ======== */

enum class ccl_proxy_op_type : uint8_t {
    none,
    allreduce_ring,
    allreduce_direct,
    reduce_scatter_ring,
    allgatherv_ring,
    alltoall_pairwise,

    // Keep this last - used for array sizing
    _count
};

/* ======== Proxy context - infrastructure resources for algorithms ======== */

// Groups proxy service infrastructure resources passed to algorithm implementations
struct ccl_proxy_context {
    ccl_cmd_ring_producer* producer;  // Command ring producer for device operations
    ccl_proxy_staging* staging;       // Staging copy manager for pipelined operations
    sycl::queue* queue;               // Proxy-thread in-order queue (device buffer allocation, etc.)
    size_t num_work_groups;           // Number of work-groups for multi-WG kernel (0 = single-WG)

    ccl_proxy_context(ccl_cmd_ring_producer* prod,
                      ccl_proxy_staging* stg,
                      sycl::queue* q,
                      size_t num_wg = 1)
        : producer(prod), staging(stg), queue(q), num_work_groups(num_wg) {}
};

/* ======== Host-side algorithm operation descriptor ======== */

struct ccl_proxy_op {
    ccl_proxy_op_type type;
    void* src;
    void* dst;
    size_t count;
    ccl::datatype dtype;
    ccl::reduction reduction;
    ccl_comm* comm;

    // additional fields
    void* aux_dev; // auxiliary device pointer (e.g. for allreduce recv buffer)
    size_t pipeline_chunk_size;
    size_t num_work_groups; // Number of work-groups for multi-WG persistent kernel (0 = auto-detect)

    // Per-rank element counts. Used by allgatherv, where blocks may differ in size.
    // Empty means every rank contributes the same block (derived from op.count).
    std::vector<size_t> recv_counts;

    // Byte offsets into dst for each rank's block. Used by allgatherv, where the
    // caller (e.g. hierarchical multi-node allgatherv) may place blocks at
    // global-rank-based offsets that are not a simple prefix sum of recv_counts.
    // Empty means the algorithm computes a contiguous prefix-sum layout itself.
    std::vector<size_t> offsets;

    // Sync-only kernel: the persistent kernel performs no device compute and is
    // used purely as a start/exit synchronization point (e.g. allreduce_direct,
    // where the reduction runs on the host via MPI). Launches a minimal kernel
    // (1 work-group, one sub-group wide) instead of the full-width heuristic.
    bool sync_only;

    // First peer index of the pairwise exchange loop. Used by alltoall, where the
    // ranks in [0, peer_start) are handled outside the proxy (scale-up phase or
    // self-copy on the caller's queue). The numa-pairwise variant passes
    // p2p_comm->size(), the scatter variant passes 0 or 1 depending on whether the
    // self-copy is fused into the exchange.
    int peer_start;

    // Monotonic generation assigned by proxy_service at submit time. Shared as
    // the single source of truth between the proxy op and its persistent kernel
    // so they rendezvous on the lifecycle handshake without any state reset.
    uint32_t generation;

    // note: add your additional fields as needed
};

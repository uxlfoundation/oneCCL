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

/* proxy_func.cpp
*
*  Proxy algorithm dispatch table and registration.
*  Algorithm wrappers are defined in their respective algorithm header files.
*/

#include "proxy/proxy_func.hpp"

#include "proxy/proxy.hpp"
#include "proxy/proxy_staging.hpp"
#include "proxy/proxy_types.hpp"
#include "proxy/proxy_utils.hpp"
#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_ring_host.hpp"

#include "coll/algorithms/utils/sycl_coll_base.hpp"

// Include algorithm headers that define proxy_ wrapper functions
#include "coll/algorithms/allreduce/sycl/allreduce_direct.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_ring.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_ring.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_ring.hpp"
#include "coll/algorithms/alltoall/sycl/alltoall_sycl.hpp"

#include <sycl/sycl.hpp>

/* ======== Proxy algorithm dispatch table ======== */

// Global upcall table - initialized by init_proxy_algo_table()
std::array<proxy_algo_func_t, static_cast<size_t>(ccl_proxy_op_type::_count)> proxy_algo_upcall;

/* ======== Upcall table initialization ======== */

void init_proxy_algo_table() {
    // Initialize all entries to nullptr
    proxy_algo_upcall.fill(nullptr);

    // Register algorithm implementations by operation type
    // Each proxy_ wrapper is defined in its algorithm's header file
    proxy_algo_upcall[static_cast<size_t>(ccl_proxy_op_type::allreduce_ring)] =
        ccl::proxy_allreduce_ring;
    proxy_algo_upcall[static_cast<size_t>(ccl_proxy_op_type::allreduce_direct)] =
        proxy_allreduce_direct;
    proxy_algo_upcall[static_cast<size_t>(ccl_proxy_op_type::reduce_scatter_ring)] =
        ccl::proxy_reduce_scatter_ring;
    proxy_algo_upcall[static_cast<size_t>(ccl_proxy_op_type::allgatherv_ring)] =
        ccl::proxy_allgatherv_ring;
    proxy_algo_upcall[static_cast<size_t>(ccl_proxy_op_type::alltoall_pairwise)] =
        proxy_alltoall_pairwise;

    // To add a new algorithm:
    // 1. Add new operation type to ccl_proxy_op_type enum in proxy_types.hpp
    // 2. Implement proxy_<algo_name> wrapper in the algorithm's header file
    //    - Function signature: void proxy_<algo_name>(const ccl_proxy_op& op, ccl_proxy_context& ctx)
    //    - Unpack ccl_proxy_op fields and call the actual algorithm implementation
    //    - Use ctx.producer and ctx.staging to access infrastructure resources
    // 3. Register it here: proxy_algo_upcall[static_cast<size_t>(ccl_proxy_op_type::your_algo)] = proxy_your_algo;
}

/* ======== RAII guard for kernel lifecycle ======== */

// RAII guard that wraps algorithm execution with wait_kernel_start/exit signaling
class ccl_kernel_lifecycle_guard {
public:
    explicit ccl_kernel_lifecycle_guard(ccl_kernel_lifecycle* lifecycle,
                                        uint32_t generation,
                                        ccl_cmd_ring_producer* producer = nullptr,
                                        size_t num_work_groups = 1)
            : m_lifecycle(lifecycle),
              m_producer(producer),
              m_num_work_groups(num_work_groups),
              m_generation(generation) {
        CCL_ASSERT(m_lifecycle != nullptr, "lifecycle pointer is null");
        CCL_ASSERT(m_producer != nullptr, "producer pointer is null");
        // Wait until the kernel for THIS generation is alive on the device.
        // A stale started_gen from a prior op cannot satisfy this (wrap-safe
        // >= compare), so we never start the algorithm before our kernel runs.
        ccl_kernel_lifecycle::wait_kernel_start(m_lifecycle, m_generation);
    }

    ~ccl_kernel_lifecycle_guard() {
        // Every persistent kernel — including sync-only ones (allreduce_direct) —
        // terminates via a ring exit command. One uniform termination protocol,
        // so there is no unconsumed-ring / no-op-signal race to hang on.
        //
        // Fire-and-forget on purpose: we submit the exit commands and do not wait
        // for their completions. Waiting would be a deadline-free spin in a
        // destructor that also runs during exception unwinding — exactly when the
        // kernel may be the thing that failed — turning a reported error into a hang.
        if (m_num_work_groups > 1) {
            // Multi-WG path: send N exit commands (one per work-group)
            m_producer->submit_exit_multi(m_num_work_groups);
        }
        else {
            // Single-WG path: send one exit command
            m_producer->submit_exit();
        }
    }

    ccl_kernel_lifecycle_guard(const ccl_kernel_lifecycle_guard&) = delete;
    ccl_kernel_lifecycle_guard& operator=(const ccl_kernel_lifecycle_guard&) = delete;

private:
    ccl_kernel_lifecycle* m_lifecycle;
    ccl_cmd_ring_producer* m_producer;
    size_t m_num_work_groups;
    uint32_t m_generation;
};

/* ======== Operation dispatch ======== */

void process_host_initiated_op(const ccl_proxy_op& op,
                               ccl_cmd_ring_producer* producer,
                               ccl_kernel_lifecycle* lifecycle,
                               ccl_proxy_staging* staging,
                               sycl::queue* queue) {
    const uint8_t op_index = static_cast<uint8_t>(op.type);

    // Bounds check
    if (op_index >= static_cast<uint8_t>(ccl_proxy_op_type::_count)) {
        LOG_WARN("Invalid proxy service operation type: ", op_index);
        return;
    }
    // Lookup algorithm in dispatch table
    proxy_algo_func_t algo_func = proxy_algo_upcall[op_index];
    if (algo_func == nullptr) {
        LOG_WARN("Unregistered proxy service operation type: ", op_index);
        return;
    }

    // wait_kernel_start(generation) on construction, submit_exit on destruction.
    ccl_kernel_lifecycle_guard lifecycle_guard(
        lifecycle, op.generation, producer, op.num_work_groups);

    // Create proxy context wrapping infrastructure resources
    ccl_proxy_context ctx(producer, staging, queue, op.num_work_groups);

    // Dispatch to the registered algorithm
    algo_func(op, ctx);
}

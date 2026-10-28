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

/* persistent_kernel.cpp
 *
 * Implementation of host-side submission logic for the CCL proxy service
 */

#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

#include "proxy/proxy.hpp"
#include "proxy/proxy_utils.hpp"
#include "proxy/proxy_types.hpp"

#include "proxy/host_initiated/cmd_ring_types.hpp"

#include "common/env/env.hpp"
#include "common/global/global.hpp"

#include "coll/algorithms/utils/sycl_reductions.hpp"

#include <sycl/sycl.hpp>

/* ======== Work-group selection heuristic ======== */

// Select number of work-groups for persistent kernel based on data size
// Heuristic: balance parallelism vs GPU utilization
// For small data (<128KB), single WG avoids atomic contention.
// For large data (>=8MB), many WGs maximize GPU utilization.
// Thresholds tuned for typical collective communication patterns.
size_t select_persistent_work_group_num(size_t block_size) {
    if (block_size >= 16 * 1024 * 1024) {
        // Very large data (>=16MB per block): maximize throughput.
        // Capped by MAX_PERSISTENT_WORK_GROUPS, which CMD_RING_SIZE is sized against.
        return MAX_PERSISTENT_WORK_GROUPS;
    } else if (block_size >= 8 * 1024 * 1024) {
        return 16;  // Large data (8-16MB per block): high parallelism
    } else if (block_size >= 2 * 1024 * 1024) {
        return 8;   // Medium-large data (2-8MB): good parallelism
    } else if (block_size >= 512 * 1024) {
        return 4;   // Medium data (512KB-2MB): balanced
    } else if (block_size >= 128 * 1024) {
        return 2;   // Small-medium data (128KB-512KB): light parallelism
    } else {
        return 1;   // Small data (<128KB): avoid atomic overhead
    }
}

// Helper function to submit the host task that uploads the proxy operation and launches the multi-WG persistent kernel
// Note: launches N work-groups that independently compete for command slots via atomic counter
sycl::event submit_and_launch_persistent_kernel(
    sycl::queue& q,
    ccl_proxy_op& op,
    const std::vector<sycl::event>& dep_events) {
    bool is_recording = use_recording_path(q);
    // Treat num_work_groups == 0 as 1 work-group
    // In case op.sync_only launch only 1 work-group, since no computation is required.
    const size_t num_work_groups = (op.num_work_groups == 0 || op.sync_only) ? 1 : op.num_work_groups;
    op.num_work_groups = num_work_groups;

    // Obtain reduction operation data for the specified reduction type.
    ccl_reduction_data reduction_op = make_reduction_operation(op.reduction);

    // Obtain pointers to the proxy thread's shared lifecycle, command ring, and completions
    ccl_kernel_lifecycle* lifecycle = ccl::global_data::get().proxy_service->lifecycle();
    ccl_cmd_ring* ring = ccl::global_data::get().proxy_service->ring();
    ccl_cmd_ring_completions* completions = ccl::global_data::get().proxy_service->completions();

    // Assign this op/kernel pair a monotonic generation (single source of truth).
    // The same value is carried in the op (for the proxy guard) and passed as a
    // kernel argument, so proxy and kernel rendezvous on the lifecycle handshake.
    const uint32_t gen = ccl::global_data::get().proxy_service->next_generation();
    op.generation = gen;

    // Upload the proxy operation to the proxy thread.
    //
    // Non-recording (direct) path: submit directly from this (main) thread. Each
    // launch draws a fresh, strictly increasing generation via next_generation(),
    // so started_gen is monotonic and the wrap-safe >= handshake never mistakes a
    // prior op's value for this one. No re-arm is needed (and adding one would
    // race a host store against the kernel's device store).
    //
    // Recording path: the op is submitted from a host task so it is ordered by
    // graph replay relative to the kernel. Here `gen` is frozen at record time and
    // reused on every replay, so started_gen must be re-armed to a value below
    // `gen` before each replay's kernel republishes it — otherwise a stale
    // started_gen left by the previous replay would satisfy wait_kernel_start
    // immediately, before this replay's kernel has actually launched. We arm to
    // exactly `gen - 1`: wrap-safe one-below the awaited value, so the guard's
    // (int32_t)(started_gen - gen) < 0 spin holds until the kernel writes `gen`,
    // regardless of the counter's magnitude.
    std::vector<sycl::event> kernel_deps;
    if (is_recording) {
        sycl::event upload_plan_event = q.submit([&](sycl::handler& h) {
            h.depends_on(dep_events);
            h.host_task([=]() {
                ccl_kernel_lifecycle::reset_kernel_start(lifecycle, gen);
                ccl::global_data::get().proxy_service->submit(op);
            });
        });
        kernel_deps.push_back(upload_plan_event);
    }
    else {
        ccl::global_data::get().proxy_service->submit(op);
        kernel_deps = dep_events;
    }

    constexpr int sg_size = 32; // should be default sub-group size for most devices
    constexpr int wg_size = 1024; // should be default work-group size for most devices
    const size_t max_wg_size = q.get_device().get_info<sycl::info::device::max_work_group_size>();
    // Sync-only ops do no device work, so one sub-group is enough
    size_t local_size = op.sync_only ? static_cast<size_t>(sg_size) : static_cast<size_t>(wg_size);
    if (local_size > max_wg_size)
        local_size = max_wg_size;

    LOG_DEBUG("[proxy] launching multi-WG persistent kernel: ",
              num_work_groups,
              " work-groups, local_size: ",
              local_size,
              ", total threads: ",
              num_work_groups * local_size);

    sycl::nd_range<1> ndr{ sycl::range<1>(num_work_groups * local_size),
                           sycl::range<1>(local_size) };
    sycl::event kernel_event = q.submit([&](sycl::handler& h) {
        h.depends_on(kernel_deps);
        h.parallel_for(
            ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                persistent_kernel_main(it, lifecycle, ring, completions, reduction_op, gen);
            });
    });

    return kernel_event;
}
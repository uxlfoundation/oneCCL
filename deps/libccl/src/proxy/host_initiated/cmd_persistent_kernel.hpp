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

/* persistent_kernel.hpp
 *
 * The persistent GPU kernel.
 */

#pragma once

#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_ring_device.hpp"

#include "coll/algorithms/utils/sycl_reductions.hpp"

#include <sycl/sycl.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
namespace syclext = sycl::ext::oneapi;

/* ======== Operation handlers ======== */

template <typename T>
static inline void device_op_reduce(sycl::nd_item<1> it,
                                    const ccl_device_cmd& cmd,
                                    const ccl_reduction_data& reduction) {
    // Use local IDs for multi-WG: each WG processes its own chunk independently
    const auto lid = it.get_local_id(0);
    const auto lsz = it.get_local_range(0);

    const T* src = reinterpret_cast<const T*>(cmd.src);
    T* dst = reinterpret_cast<T*>(cmd.dst);
    const T* aux = reinterpret_cast<const T*>(cmd.aux);
    // count is a typed count (e.g. number of elements, not number of bytes)
    const size_t count = cmd.count;

    for (size_t i = lid; i < count; i += lsz) {
        // dst[i] = src[i] + aux[i];
        dst[i] = apply_reduction<T>(reduction, src[i], aux[i]);
    }
}

// Grid-stride copy in units of vector type V (a whole WG cooperates on one
// command via local IDs), plus a byte tail for the sub-V remainder.
template <typename V>
static inline void device_copy_vec(sycl::nd_item<1> it,
                                   const char* src,
                                   char* dst,
                                   size_t nbytes) {
    const auto lid = it.get_local_id(0);
    const auto lsz = it.get_local_range(0);

    const size_t n_vec = nbytes / sizeof(V);
    const V* vsrc = reinterpret_cast<const V*>(src);
    V* vdst = reinterpret_cast<V*>(dst);

    for (size_t i = lid; i < n_vec; i += lsz) {
        vdst[i] = vsrc[i];
    }
    // Tail bytes that do not fill a whole V.
    const size_t tail_start = n_vec * sizeof(V);
    for (size_t i = tail_start + lid; i < nbytes; i += lsz) {
        dst[i] = src[i];
    }
}

static inline void device_op_copy(sycl::nd_item<1> it, const ccl_device_cmd& cmd) {
    const char* src = static_cast<const char*>(cmd.src);
    char* dst = static_cast<char*>(cmd.dst);
    // count is a number of bytes, not elements, for copy command
    const size_t nbytes = cmd.count;

    // Copy with the widest access for which both pointers are aligned
    const uintptr_t addr_bits =
        reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst);
    if ((addr_bits & 0xF) == 0) {
        device_copy_vec<sycl::vec<uint32_t, 4>>(it, src, dst, nbytes); // 16 bytes
    }
    else if ((addr_bits & 0x3) == 0) {
        device_copy_vec<uint32_t>(it, src, dst, nbytes); // 4 bytes
    }
    else {
        device_copy_vec<char>(it, src, dst, nbytes); // 1 byte fallback
    }
}

/* ======== Command dispatch ======== */

static inline void device_dispatch(sycl::nd_item<1> it,
                                   const ccl_device_cmd& cmd,
                                   const ccl_reduction_data& reduction) {
    switch (cmd.op) {
        case ccl_ring_cmd_type::reduce:
            switch (static_cast<ccl::datatype>(cmd.dtype)) {
                case ccl::datatype::int16: device_op_reduce<short>(it, cmd, reduction); break;
                case ccl::datatype::float16:
#ifdef CCL_SYCL_VEC_SUPPORT_FP16
                    device_op_reduce<sycl::half>(it, cmd, reduction);
#else
                    // Fallback to int16 reduction if half is not supported
                    device_op_reduce<short>(it, cmd, reduction);
#endif
                    break;
                case ccl::datatype::bfloat16:
#ifdef CCL_SYCL_VEC_SUPPORT_BF16
                    device_op_reduce<sycl::ext::oneapi::bfloat16>(it, cmd, reduction);
#else
                    // Fallback to int16 reduction if bfloat16 is not supported
                    device_op_reduce<short>(it, cmd, reduction);
#endif
                    break;
                case ccl::datatype::int8: device_op_reduce<int8_t>(it, cmd, reduction); break;
                case ccl::datatype::uint8: device_op_reduce<uint8_t>(it, cmd, reduction); break;
                case ccl::datatype::int32: device_op_reduce<int>(it, cmd, reduction); break;
                case ccl::datatype::uint32: device_op_reduce<uint32_t>(it, cmd, reduction); break;
                case ccl::datatype::float32: device_op_reduce<float>(it, cmd, reduction); break;
                case ccl::datatype::float64: device_op_reduce<double>(it, cmd, reduction); break;
                case ccl::datatype::int64: device_op_reduce<int64_t>(it, cmd, reduction); break;
                case ccl::datatype::uint64: device_op_reduce<uint64_t>(it, cmd, reduction); break;
                default: device_op_reduce<int>(it, cmd, reduction); break;
            }
            break;

        case ccl_ring_cmd_type::copy: device_op_copy(it, cmd); break;

        default: break;
    }
}

/* ======== Multi-WG persistent kernel work loop ======== */

static inline void persistent_kernel_work_loop(sycl::nd_item<1> it,
                                               ccl_kernel_lifecycle* lifecycle,
                                               ccl_cmd_ring* ring,
                                               ccl_cmd_ring_completions* completions,
                                               const ccl_reduction_data reduction) {
    auto group = it.get_group();

    // SLM state shared within each work-group
#if defined(__INTEL_LLVM_COMPILER) && __INTEL_LLVM_COMPILER >= 20260000
    syclexp::work_group_static<ccl_shared_kernel_state> slm_state_static;
    ccl_shared_kernel_state& slm_state = slm_state_static;
#else
    auto slm_state_ptr = syclext::group_local_memory<ccl_shared_kernel_state>(group);
    ccl_shared_kernel_state& slm_state = *slm_state_ptr;
#endif

    // Atomic reference to the ring's slot allocation counter
    sycl::atomic_ref<uint32_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
        next_slot_atomic(ring->next_slot);

    for (;;) {
        uint32_t my_slot_seq = 0;

        // Leader-only: atomically claim next slot from shared counter
        if (group.leader()) {
            // Atomic fetch_add returns OLD value, which is our slot sequence
            my_slot_seq = next_slot_atomic.fetch_add(1);

            // Poll until the command at this slot is ready
            // (sequence in slot matches what we claimed)
            cmd_ring_poll_atomic(ring, my_slot_seq);

            // Command ready - read it into SLM
            slm_state.cmd = cmd_ring_read(ring, my_slot_seq);

            // Check for exit command
            if (slm_state.cmd.op == ccl_ring_cmd_type::exit) {
                slm_state.has_work = false;
                slm_state.should_exit = true;
            } else {
                slm_state.has_work = true;
                slm_state.should_exit = false;
            }
        }

        // Barrier: wait for leader to populate SLM
        sycl::group_barrier(group);

        // All threads read SLM state
        ccl_shared_kernel_state st = slm_state;

        if (st.should_exit) {
            // Exit command - complete its ring slot before terminating.
            // This is required: the slot's completion must be advanced so that
            // when the producer wraps around and reuses this slot, its
            // backpressure (wait completion == seq - CMD_RING_SIZE) is satisfied.
            // Without it the exit slot is never released and a later submit()/
            // submit_exit() hangs.
            if (group.leader()) {
                cmd_ring_complete(completions, st.cmd.sequence);
            }
            break;
        }

        if (st.has_work) {
            // Execute command in parallel
            device_dispatch(it, st.cmd, reduction);

            // Barrier: ensure all work-items finished
            sycl::group_barrier(group);

            // Leader marks completion
            if (group.leader()) {
                cmd_ring_complete(completions, st.cmd.sequence);
            }
        }
    }
}


/* ======== Lifecycle management ======== */

// Publish this launch's generation to the host, notifying the proxy thread that
// the kernel is alive.
// atomics on USM are invalid).
template <typename Group>
static inline void kernel_proxy_notify(Group group, ccl_kernel_lifecycle* lifecycle, uint32_t gen) {
    if (group.get_group_linear_id() == 0 && group.leader()) {
        lifecycle->started_gen = gen;
        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    }
}

/* ======== Multi-WG persistent kernel main function ======== */

static inline void persistent_kernel_main(sycl::nd_item<1> it,
                                          ccl_kernel_lifecycle* lifecycle,
                                          ccl_cmd_ring* ring,
                                          ccl_cmd_ring_completions* completions,
                                          const ccl_reduction_data reduction,
                                          uint32_t gen) {
    auto group = it.get_group();
    // Notify proxy thread that the kernel (generation `gen`) has started.
    kernel_proxy_notify(group, lifecycle, gen);
    // Multi-WG work loop: each WG independently claims and processes commands
    persistent_kernel_work_loop(it, lifecycle, ring, completions, reduction);
    // Work loop exits when this WG receives its exit command
}

/* === Utility functions for submiting host task and launching persistent kernels === */

// Select number of work-groups for persistent kernel based on data size
// Heuristic: balance parallelism vs atomic contention overhead
size_t select_persistent_work_group_num(size_t block_size);

sycl::event submit_and_launch_persistent_kernel(sycl::queue& q, ccl_proxy_op& op, const std::vector<sycl::event>& dep_events);

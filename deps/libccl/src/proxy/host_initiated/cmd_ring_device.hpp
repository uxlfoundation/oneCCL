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

/* cmd_ring_device.hpp
 *
 * Device-side consumer for the command ring.
 */

#pragma once

#include "proxy/host_initiated/cmd_ring_types.hpp"

#include <sycl/sycl.hpp>

// Optimized atomic poll using acquire semantics for device memory
static inline void cmd_ring_poll_atomic(ccl_cmd_ring* ring, uint32_t next_slot) {
    const uint32_t slot_idx = next_slot % CMD_RING_SIZE;

    // Create atomic reference to sequence field with acq_rel semantics
    // This ensures proper memory ordering when ring is in device memory
    sycl::atomic_ref<uint32_t,
                     sycl::memory_order::acq_rel,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space>
        seq_atomic(ring->slots[slot_idx].sequence);

    // Blocking poll - wait until sequence matches expected value
    uint32_t observed_seq;
    do {
        observed_seq = seq_atomic.load();
    } while (observed_seq != next_slot);
}

// Legacy polling (non-atomic, for compatibility)
static inline bool cmd_ring_poll(ccl_cmd_ring* ring, uint32_t next_slot) {
    const uint32_t slot_idx = next_slot % CMD_RING_SIZE;
    const uint32_t observed_seq = ring->slots[slot_idx].sequence;
    return (observed_seq == next_slot);
}

// Read command from ring slot
static inline ccl_device_cmd cmd_ring_read(ccl_cmd_ring* ring, uint32_t next_slot) {
    const uint32_t slot_idx = next_slot % CMD_RING_SIZE;
    return ring->slots[slot_idx];
}

// Write completion with release fence
// Note: completions array is in USM host memory, cannot use atomic_ref with global_space
static inline void cmd_ring_complete(ccl_cmd_ring_completions* completions, uint32_t seq) {
    const uint32_t slot_idx = seq % CMD_RING_SIZE;

    // Regular write followed by release fence
    completions->entries[slot_idx].completion = seq;

    // Release fence ensures completion is published
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}
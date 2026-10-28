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

/* cmd_ring_types.hpp
 *
 * Types for the host-initiated command ring (CPU → GPU).
 *
 * The CPU proxy thread produces commands (reduce, copy) into the
 * command ring. The persistent GPU kernel consumes and executes them.
 */

#pragma once

#include "proxy/proxy_types.hpp"
#include "proxy/proxy_utils.hpp"

#include "common/utils/utils.hpp"

/* ======== Kernel lifecycle ======== */

struct ccl_kernel_lifecycle {
    // Monotonic generation counter (GPU → CPU). It only ever increases and is
    // never reset, so no host-side zeroing (the old reset() race) is possible.
    // A stale write carrying an older generation is a no-op against the
    // wrap-safe >= predicate below. Lives in host-pinned USM and is accessed
    // with plain load/store + system fence (device atomics are not valid on
    // USM). Kernels terminate via a ring exit command, so there is no CPU → GPU
    // finish channel — the proxy never needs to signal the kernel back.
    uint32_t started_gen; // generation of the kernel that has begun

    // Reset the start handshake to one generation below `gen`, so a subsequent
    // wait_kernel_start(gen) blocks until the kernel republishes `gen`. Needed on
    // the SYCL-graph recording path, where `gen` is frozen at record time and
    // reused on every replay: without re-arming, a stale started_gen left by the
    // previous replay would satisfy the wait immediately. `gen - 1` is wrap-safe
    // one-below the awaited value, so the wait's (int32_t)(started_gen - gen) < 0
    // spin holds regardless of the counter's magnitude. Plain host store; ordering
    // to the proxy/kernel is provided by the caller (queue mutex + host-task edge).
    // Static to keep the struct trivial and POD-compatible.
    static void reset_kernel_start(ccl_kernel_lifecycle* lc, uint32_t gen) {
        lc->started_gen = gen - 1;
    }

    // Block until the persistent kernel for generation `gen` signals it is alive.
    // Static to keep the struct trivial and POD-compatible.
    static void wait_kernel_start(const ccl_kernel_lifecycle* lc, uint32_t gen) {
        wait_backoff_ge(&lc->started_gen, gen);
    }
};

/* ======== Device command types ======== */

enum class ccl_ring_cmd_type : uint32_t {
    nop = 0,
    reduce = 1,
    copy = 2,
    exit = 3, // Signal kernel to exit (replaces lifecycle->finish polling)
};

/* ======== Single command slot (one cache line) ======== */

struct ccl_device_cmd {
    ccl_ring_cmd_type op;
    void* src;
    void* dst;
    void* aux;
    uint32_t count;
    uint32_t dtype;

    uint32_t sequence; // set by CPU when command is submitted

    uint32_t _pad[5]; // pad to 64 bytes (cache line)
};

static_assert(sizeof(ccl_device_cmd) == CACHELINE_SIZE,
              "ccl_device_cmd must be exactly one cache line");

/* ======== Command ring buffer ======== */

constexpr uint32_t CMD_RING_SIZE = 64;

// Upper bound returned by select_persistent_work_group_num() (see
// cmd_persistent_kernel.cpp). submit_split() issues one command per work-group,
// so this many slots are in flight for a single op.
constexpr uint32_t MAX_PERSISTENT_WORK_GROUPS = 32;

// Power-of-two is load-bearing for wrap-safety, not just for a cheap modulo.
// Back-pressure computes old_seq = seq - CMD_RING_SIZE on a uint32_t that is
// meant to wrap, so the seq % CMD_RING_SIZE -> slot mapping only survives the
// 2^32 wrap if CMD_RING_SIZE divides 2^32. With e.g. 48 (2^32 % 48 == 16) the
// producer and a work-group would derive different slots for the same sequence
// after the wrap, and cmd_ring_poll_atomic (no timeout) would spin forever.
static_assert((CMD_RING_SIZE & (CMD_RING_SIZE - 1)) == 0,
              "CMD_RING_SIZE must be a power of two: seq % CMD_RING_SIZE must stay "
              "consistent across the uint32_t wrap for the seq - CMD_RING_SIZE backpressure");

// The ring must hold a whole op's chunks plus headroom, otherwise submit_split()
// starts stalling mid-op with no build error and no log.
static_assert(CMD_RING_SIZE >= 2 * MAX_PERSISTENT_WORK_GROUPS,
              "ring must hold at least one op's chunks plus headroom");

struct ccl_cmd_ring {
    ccl_device_cmd slots[CMD_RING_SIZE];

    /* Multi-WG persistent kernel support:
     * Atomic counter for work-group slot allocation.
     * Each WG atomically increments this to claim next slot.
     * Monotonic: never reset, grows across all ops. */
    alignas(CACHELINE_SIZE) uint32_t next_slot;

    /* Initialize ring slots to pre-submitted state.
     * Slot i gets sequence=i.
     * Producer starts at CMD_RING_SIZE.
     * NOTE: only called once at context init. The ring is a monotonic,
     * wrap-safe protocol and must never be reset between ops. */
    static void init_slots(ccl_cmd_ring* ring) {
        std::memset(ring, 0, sizeof(ccl_cmd_ring));
        for (uint32_t i = 0; i < CMD_RING_SIZE; ++i) {
            ring->slots[i].sequence = i;
        }
        ring->next_slot = CMD_RING_SIZE; // Start after pre-initialized slots
    }
};

/* ======== Completion tracking (separate host memory array) ======== */

struct ccl_ring_completion {
    uint32_t completion; // set by GPU when command is completed
};

struct ccl_cmd_ring_completions {
    ccl_ring_completion entries[CMD_RING_SIZE];

    /* Initialize completions to pre-completed state.
     * Entry i gets completion=i. */
    static void init_completions(ccl_cmd_ring_completions* completions) {
        std::memset(completions, 0, sizeof(ccl_cmd_ring_completions));
        for (uint32_t i = 0; i < CMD_RING_SIZE; ++i) {
            completions->entries[i].completion = i;
        }
    }
};

/* ======== Shared state for persistent kernel commands ======== */

struct ccl_shared_kernel_state {
    ccl_device_cmd cmd;
    bool has_work;
    bool should_exit;
};


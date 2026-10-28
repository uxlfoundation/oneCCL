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

#include <cstddef>

// AUTO policy for the broadcast pipelined ring.
//
// Maps (world_size, total_bytes) -> {use_ring, slice-floor threshold}. Built
// from empirical sweep on real PCIe (CRI/BMG) hardware. Active only when
// CCL_SYCL_BROADCAST_RING=1 AND CCL_SYCL_BROADCAST_RING_AUTO=1; in that case
// the table overrides both the user's RING flag (we may decline the ring for
// sizes where the baseline IPC pull still wins) and the user's
// CCL_SYCL_BROADCAST_CHUNKING_THRESHOLD (the table picks the slice floor that
// performed best at this (N, size) point).
//
// Sizes outside the calibrated set fall through to the last bucket. World
// sizes not covered by any table get use_ring=false — caller keeps its
// default path.
//
// To re-tune: re-run the broadcast sweep on the target hardware, then update
// the constants in the tables below.

struct broadcast_ring_auto_decision {
    bool use_ring;
    std::size_t threshold_bytes;

    /**
     * Selects the ring and its slice-size floor for a message shape.
     *
     * @param total_bytes Total broadcast message size in bytes.
     * @param world Number of ranks in the node communicator.
     */
    broadcast_ring_auto_decision(std::size_t total_bytes, int world);
};

namespace broadcast_ring_auto_policy_detail {

struct policy_entry {
    // Upper bound (exclusive) on total_bytes for this bucket. Entries are
    // ordered by ascending upper_bound; the tail bucket uses SIZE_MAX.
    std::size_t upper_bound;
    bool use_ring;
    std::size_t threshold_bytes;
};

inline constexpr std::size_t KB = 1024;
inline constexpr std::size_t MB = 1024 * KB;

// world = 2 (measured)
inline constexpr policy_entry world_2_policy[] = {
    { 16 * MB, false, 0 },         { 32 * MB, true, 16 * MB },   { 64 * MB, true, 32 * MB },
    { 128 * MB, true, 256 * KB },  { 256 * MB, true, 16 * MB },  { 512 * MB, true, 128 * KB },
    { 1024 * MB, true, 256 * KB }, { SIZE_MAX, true, 512 * KB },
};

// world = 4 (measured)
inline constexpr policy_entry world_4_policy[] = {
    { 32 * MB, false, 0 },        { 64 * MB, true, 512 * KB },  { 128 * MB, true, 128 * KB },
    { 256 * MB, true, 512 * KB }, { 512 * MB, true, 128 * KB }, { SIZE_MAX, true, 1 * MB },
};

// world = 8 (measured)
inline constexpr policy_entry world_8_policy[] = {
    { 64 * MB, false, 0 },        { 128 * MB, true, 128 * KB }, { 256 * MB, true, 4 * MB },
    { 512 * MB, true, 256 * KB }, { 1024 * MB, true, 16 * MB }, { SIZE_MAX, true, 512 * KB },
};

// world = 16 (BD, PROVISIONAL - NOT SWEPT).
// Shape mirrors world_8_policy with roughly doubled thresholds to reflect
// the additional ring hops. Re-sweep on 16-GPU hardware and replace.
inline constexpr policy_entry world_16_policy[] = {
    { 128 * MB, false, 0 },        { 256 * MB, true, 128 * KB }, { 512 * MB, true, 4 * MB },
    { 1024 * MB, true, 256 * KB }, { 2048 * MB, true, 16 * MB }, { SIZE_MAX, true, 512 * KB },
};

/**
 * Applies the first policy bucket whose upper bound contains the message size.
 *
 * @tparam N Number of entries in the policy table.
 * @param table Policy buckets ordered by increasing upper bound.
 * @param total_bytes Total broadcast message size in bytes.
 * @param out Decision updated from the matching bucket.
 */
template <std::size_t N>
inline void apply(const policy_entry (&table)[N],
                  std::size_t total_bytes,
                  broadcast_ring_auto_decision& out) {
    for (const auto& e : table) {
        if (total_bytes < e.upper_bound) {
            out.use_ring = e.use_ring;
            out.threshold_bytes = e.threshold_bytes;
            return;
        }
    }
}

} // namespace broadcast_ring_auto_policy_detail

inline broadcast_ring_auto_decision::broadcast_ring_auto_decision(std::size_t total_bytes,
                                                                  int world)
        : use_ring(false),
          threshold_bytes(0) {
    using namespace broadcast_ring_auto_policy_detail;
    switch (world) {
        case 2: apply(world_2_policy, total_bytes, *this); break;
        case 4: apply(world_4_policy, total_bytes, *this); break;
        case 8: apply(world_8_policy, total_bytes, *this); break;
        // NOTE: world=16 thresholds are PROVISIONAL — need to tune on real
        // 16-GPU hardware. See world_16_policy definition above.
        case 16: apply(world_16_policy, total_bytes, *this); break;
        default: break;
    }
}

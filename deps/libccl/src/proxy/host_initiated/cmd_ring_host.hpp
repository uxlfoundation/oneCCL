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

/* cmd_ring_host.hpp
 *
 * Host-side producer for the device command ring.
 */

#pragma once

#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/proxy_utils.hpp"

#include "common/global/global.hpp"

#include <atomic>
#include <cstring>

#include <immintrin.h>

/* ======== Cache line write optimization ======== */

// Write a single 64-byte command to GDR-mapped memory.
__attribute__((target("movdir64b")))
inline void write_cmd_single(ccl_device_cmd* dst, const ccl_device_cmd* src) {
// TODO: add fallback for intrinsic, __MOVDIR64B__ doesn't work well
// #if defined(__MOVDIR64B__) && defined(__x86_64__)
    // _movdir64b: Single 64-byte direct store instruction
    // PCIe write combining, single transaction
    _movdir64b(dst, src);
// #else
//     // Fallback: compiler-optimized struct copy
//     // Modern compilers will generate vector stores for aligned 64-byte copy
//     *dst = *src;
// #endif
}

class ccl_cmd_ring_producer {
public:
    ccl_cmd_ring_producer() = default;

    void init(ccl_cmd_ring* ring, ccl_cmd_ring_completions* completions) {
        m_ring = ring;
        m_ring_host = ring;
        m_completions = completions;
        m_next_send = CMD_RING_SIZE;
    }

    void init(ccl_cmd_ring* ring_device,
              ccl_cmd_ring* ring_host,
              ccl_cmd_ring_completions* completions) {
        m_ring = ring_device;
        m_ring_host = ring_host;
        m_completions = completions;
        m_next_send = CMD_RING_SIZE;
    }

    uint32_t submit(ccl_ring_cmd_type op,
                    void* src,
                    void* dst,
                    void* aux,
                    uint32_t count,
                    uint32_t dtype) {
        const uint32_t seq = m_next_send++;
        const uint32_t slot_idx = seq % CMD_RING_SIZE;
        ccl_device_cmd* slot = &m_ring_host->slots[slot_idx];

        // Backpressure: wait for consumer to complete the previous
        // occupant. On the first pass, old_seq = seq - RING_SIZE
        // matches the pre-initialized completion value.
        // Poll host memory completions array for optimal cache locality.
        const uint32_t old_seq = seq - CMD_RING_SIZE;
        wait_backoff_eq(&m_completions->entries[slot_idx].completion, old_seq);

        // Build command in temporary (stack-local) and write as single cache line.
        // This is critical for GDR-mapped memory: a single 64-byte write over PCIe
        // is much faster than 8 separate 8-byte writes.
        ccl_device_cmd cmd;
        cmd.op = op;
        cmd.src = src;
        cmd.dst = dst;
        cmd.aux = aux;
        cmd.count = count;
        cmd.dtype = dtype;
        cmd.sequence = seq;
        std::memset(cmd._pad, 0, sizeof(cmd._pad)); // Zero padding for cleaner PCIe traffic

        // Write entire cache line at once using optimal method for the platform:
        // - _movdir64b (single 64-byte PCIe write)
        // - Optional: fallback to compiler-optimized struct copy
        write_cmd_single(slot, &cmd);

        return seq;
    }

    void wait_completion(uint32_t seq) {
        const uint32_t slot_idx = seq % CMD_RING_SIZE;
        // Poll host memory completions array for optimal cache locality.
        wait_backoff_eq(&m_completions->entries[slot_idx].completion, seq);
    }
    
    // Submit work split across N chunks for multi-WG consumption
    // Returns vector of sequence numbers (one per chunk)
    std::vector<uint32_t> submit_split(ccl_ring_cmd_type op,
                                       void* src,
                                       void* dst,
                                       void* aux,
                                       uint32_t total_count,
                                       uint32_t dtype,
                                       uint32_t num_chunks,
                                       size_t elem_size) {
        std::vector<uint32_t> sequences;
        sequences.reserve(num_chunks);

        if (op == ccl_ring_cmd_type::copy) {
            elem_size = 1; // copy is byte-based, not element-based
        }

        // Calculate chunk size (typed count)
        const uint32_t chunk_count = (total_count + num_chunks - 1) / num_chunks;

        uint32_t remaining = total_count;
        for (uint32_t i = 0; i < num_chunks; ++i) {
            const uint32_t this_chunk = std::min(chunk_count, remaining);
            if (this_chunk == 0) break;

            const size_t offset = i * chunk_count * elem_size;

            void* chunk_src = src ? static_cast<char*>(src) + offset : nullptr;
            void* chunk_dst = dst ? static_cast<char*>(dst) + offset : nullptr;
            void* chunk_aux = aux ? static_cast<char*>(aux) + offset : nullptr;

            uint32_t seq = submit(op, chunk_src, chunk_dst, chunk_aux, this_chunk, dtype);
            sequences.push_back(seq);

            remaining -= this_chunk;
        }

        return sequences;
    }

    // Wait for all sequences from split submission
    void wait_split_completions(const std::vector<uint32_t>& sequences) {
        for (uint32_t seq : sequences) {
            wait_completion(seq);
        }
    }

    void submit_and_wait(ccl_ring_cmd_type op,
                        void* src,
                        void* dst,
                        void* aux,
                        uint32_t count,
                        ccl::datatype dtype,
                        size_t num_work_groups = 1) {
        auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
        if (num_work_groups > 1) {
            // Multi-WG path: split work across work-groups
            auto sequences = submit_split(op, src, dst, aux, count, static_cast<uint32_t>(dtype), num_work_groups, ccl_dtype.size());
            wait_split_completions(sequences);
        } else {
            // Single-WG path: submit as one command
            uint32_t seq = submit(op, src, dst, aux, count, static_cast<uint32_t>(dtype));
            wait_completion(seq);
        }
    }

    // Submit exit command to signal persistent kernel to terminate
    // This eliminates the need for polling lifecycle->finish in USM host memory
    uint32_t submit_exit() {
        const uint32_t seq = m_next_send++;
        const uint32_t slot_idx = seq % CMD_RING_SIZE;
        ccl_device_cmd* slot = &m_ring_host->slots[slot_idx];

        // Wait for previous command at this slot to complete
        const uint32_t old_seq = seq - CMD_RING_SIZE;
        wait_backoff_eq(&m_completions->entries[slot_idx].completion, old_seq);

        // Build exit command - only op and sequence matter, rest are zeroed
        ccl_device_cmd cmd;
        cmd.op = ccl_ring_cmd_type::exit;
        cmd.src = nullptr;
        cmd.dst = nullptr;
        cmd.aux = nullptr;
        cmd.count = 0;
        cmd.dtype = 0;
        cmd.sequence = seq;
        std::memset(cmd._pad, 0, sizeof(cmd._pad));

        // Write exit command as single cache line
        write_cmd_single(slot, &cmd);

        return seq;
    }

    // Submit N exit commands (one per work-group)
    std::vector<uint32_t> submit_exit_multi(uint32_t num_work_groups) {
        std::vector<uint32_t> sequences;
        sequences.reserve(num_work_groups);

        for (uint32_t i = 0; i < num_work_groups; ++i) {
            sequences.push_back(submit_exit());
        }

        return sequences;
    }

    uint32_t next_sequence() const {
        return m_next_send;
    }

private:
    ccl_cmd_ring* m_ring = nullptr;                    // device memory pointer (passed to kernel)
    ccl_cmd_ring* m_ring_host = nullptr;               // host-mapped pointer (for producer access)
    ccl_cmd_ring_completions* m_completions = nullptr; // host memory completions array
    uint32_t m_next_send = CMD_RING_SIZE;
};
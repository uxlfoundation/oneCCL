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

/* cmd_ring_context.hpp
 *
 * Owns host-pinned allocations for the command ring and kernel lifecycle.
 * Uses device memory + GDR mmap for optimal polling performance.
 */

#pragma once

#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_ring_host.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <sycl/sycl.hpp>
#include <cstring>
#include <stdexcept>

class ccl_cmd_ring_context {
public:
    ccl_cmd_ring_context() = default;
    // Note: destroy() must be called explicitly with the queue before destructor
    ~ccl_cmd_ring_context() = default;

    ccl_cmd_ring_context(const ccl_cmd_ring_context&) = delete;
    ccl_cmd_ring_context& operator=(const ccl_cmd_ring_context&) = delete;

    void init(sycl::queue& q) {
        // Lifecycle: host-pinned memory (accessed by both host and device)
        m_lifecycle =
            static_cast<ccl_kernel_lifecycle*>(sycl::malloc_host(sizeof(ccl_kernel_lifecycle), q));
        if (!m_lifecycle) {
            throw std::runtime_error(
                "Failed to allocate ccl_kernel_lifecycle in host-pinned memory");
        }
        std::memset(m_lifecycle, 0, sizeof(ccl_kernel_lifecycle));

        // Command ring: device memory for optimal device polling + GDR mmap for host access
        m_ring = static_cast<ccl_cmd_ring*>(sycl::aligned_alloc_device(CACHELINE_SIZE, sizeof(ccl_cmd_ring), q));
        if (!m_ring) {
            sycl::free(m_lifecycle, q);
            m_lifecycle = nullptr;
            throw std::runtime_error("Failed to allocate ccl_cmd_ring in device memory");
        }

        // Initialize ring slots via temporary host buffer
        ccl_cmd_ring temp_ring;
        ccl_cmd_ring::init_slots(&temp_ring);
        q.memcpy(m_ring, &temp_ring, sizeof(ccl_cmd_ring)).wait();

        // Create GDR-backed host mapping for the device ring
        try {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
            m_ring_host = static_cast<ccl_cmd_ring*>(ccl::ze::ze_ipc_mmap_device_allocation(
                ze_ctx, m_ring, sizeof(ccl_cmd_ring)));
        }
        catch (const std::exception& e) {
            sycl::free(m_ring, q);
            m_ring = nullptr;
            sycl::free(m_lifecycle, q);
            m_lifecycle = nullptr;
            throw std::runtime_error(std::string("Failed to mmap device ring: ") + e.what());
        }

        // Completions: host memory for optimal host polling (no cache snooping overhead)
        m_completions = static_cast<ccl_cmd_ring_completions*>(
            sycl::aligned_alloc_host(CACHELINE_SIZE, sizeof(ccl_cmd_ring_completions), q));
        if (!m_completions) {
            ccl::ze::ze_ipc_munmap_device_allocation(m_ring_host, sizeof(ccl_cmd_ring));
            m_ring_host = nullptr;
            sycl::free(m_ring, q);
            m_ring = nullptr;
            sycl::free(m_lifecycle, q);
            m_lifecycle = nullptr;
            throw std::runtime_error(
                "Failed to allocate ccl_cmd_ring_completions in host memory");
        }
        ccl_cmd_ring_completions::init_completions(m_completions);

        // Initialize producer with device ring, host-mapped ring, and host completions
        m_producer.init(m_ring, m_ring_host, m_completions);
    }

    // NOTE: intentionally NOT resetting the ring / lifecycle / completions /
    // producer between ops. These are monotonic, wrap-safe protocols:
    //   - ring->next_slot, producer.m_next_send  : slot allocation counters
    //   - slots[].sequence, completions[].completion : per-slot sequence numbers
    //   - lifecycle->started_gen                  : per-op start handshake
    //                                               (generation is the m_gen
    //                                               counter, single source of truth)
    // At every op boundary the producer and consumer counters are provably back
    // in lockstep (N data+exit commands produced == N consumed), so the next op
    // continues seamlessly. Resetting them raced with in-flight kernels and the
    // proxy thread (the old bug). One-time initialization happens in init().
    void reset() {
        // Deliberately empty. Kept for API compatibility with callers.

        // // Reset lifecycle to pre-started state
        // std::memset(m_lifecycle, 0, sizeof(ccl_kernel_lifecycle));
        // // Reset ring slots via host-mapped pointer
        // ccl_cmd_ring::init_slots(m_ring_host);
        // // Reset completions array in host memory
        // ccl_cmd_ring_completions::init_completions(m_completions);
        // // reset producer state (keeps the same ring pointers)
        // m_producer.init(m_ring, m_ring_host, m_completions);
    }

    void destroy(sycl::queue& q) {
        if (m_completions) {
            sycl::free(m_completions, q);
            m_completions = nullptr;
        }

        // Unmap the GDR-backed host pointer before freeing device memory
        if (m_ring_host) {
            ccl::ze::ze_ipc_munmap_device_allocation(m_ring_host, sizeof(ccl_cmd_ring));
            m_ring_host = nullptr;
        }

        if (m_ring) {
            sycl::free(m_ring, q);
            m_ring = nullptr;
        }

        if (m_lifecycle) {
            sycl::free(m_lifecycle, q);
            m_lifecycle = nullptr;
        }
    }

    ccl_kernel_lifecycle* lifecycle() const {
        return m_lifecycle;
    }
    ccl_cmd_ring* ring() const {
        return m_ring;
    }
    ccl_cmd_ring_completions* completions() const {
        return m_completions;
    }
    ccl_cmd_ring_producer& producer() {
        return m_producer;
    }

    // Monotonic generation counter — the single source of truth for the
    // kernel/proxy lifecycle handshake. Assigned at submit time, carried in the
    // op and passed as a kernel argument. Never reset.
    uint32_t next_generation() {
        return ++m_gen;
    }

private:
    ccl_kernel_lifecycle* m_lifecycle = nullptr;
    ccl_cmd_ring* m_ring = nullptr;                     // device memory pointer
    ccl_cmd_ring* m_ring_host = nullptr;                // GDR-backed host-mapped pointer
    ccl_cmd_ring_completions* m_completions = nullptr;  // host memory completions
    ccl_cmd_ring_producer m_producer;
    uint32_t m_gen = 0;                                 // generation counter (1-based)
};
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

/* proxy_utils.hpp
 *
 * Common host-side utilities used across proxy infrastructure:
 *   - Memory fenced load/store
 *   - Backoff wait
 *   - Thread affinity
 *   - Block geometry helpers
 */

#pragma once

#include "proxy/proxy_types.hpp"
#include "proxy/host_initiated/cmd_ring_types.hpp"

#include <atomic>
#include <thread>
#include <chrono>
#include <cstdio>

#include <pthread.h>
#include <sched.h>
#include <errno.h>

#include <immintrin.h>

/* ======== Backoff wait ======== */

/* Backoff strategy for proxy thread idle loop
 *
 * Three phases:
 *   1. Hot spin  (_mm_pause):   lowest latency, highest power
 *   2. Warm spin (yield):       gives up timeslice, moderate latency
 *   3. Cold sleep (usleep):     lowest power, highest latency
 *
 * Any successful poll resets to hot phase immediately.
 */
struct ccl_proxy_backoff {
    static constexpr int hot_spins = 4096; // ~40us of _mm_pause
    static constexpr int warm_spins = 16384; // yield every 64 iters after hot
    static constexpr int cold_us = 10; // sleep granularity in cold phase

    int idle_count = 0;

    void reset() {
        idle_count = 0;
    }

    void idle() {
        if (idle_count < hot_spins) {
            _mm_pause();
        }
        else if (idle_count < warm_spins) {
            if ((idle_count & 0x3F) == 0)
                std::this_thread::yield();
            else
                _mm_pause();
        }
        else {
            std::this_thread::sleep_for(std::chrono::microseconds(cold_us));
        }
        ++idle_count;
    }
};

/* Spin thresholds for the completion/handshake waits below.
 *
 * Single place to tune the wait path: the _ge and _eq waits differ only in the
 * comparison, so both go through wait_backoff_until() and share these numbers.
 * Optimized for the low-latency path — a tight spin for ~80us covers most small
 * ops, then yields sparsely to keep context switches down, then sleeps.
 *
 * Deliberately separate from ccl_proxy_backoff (the idle loop) and from the
 * worker's CCL_SPIN_COUNT / CCL_YIELD_TYPE.
 */
struct ccl_proxy_wait_backoff {
    static constexpr int hot_spins = 8000; // ~80us of _mm_pause
    static constexpr int warm_spins = 20000; // yield every 256 iters after hot
    static constexpr int cold_us = 1; // sleep granularity in cold phase
};

// Spin on *p until `ready(observed)` holds, escalating pause -> yield -> sleep.
template <typename Predicate>
static inline void wait_backoff_until(const volatile uint32_t* p, Predicate ready) {
    int it = 0;
    while (!ready(*p)) {
        if (it < ccl_proxy_wait_backoff::hot_spins) {
            _mm_pause();
        }
        else if (it < ccl_proxy_wait_backoff::warm_spins) {
            if ((it & 0xFF) == 0)
                std::this_thread::yield();
            else
                _mm_pause();
        }
        else {
            std::this_thread::sleep_for(std::chrono::microseconds(ccl_proxy_wait_backoff::cold_us));
        }
        ++it;
    }
}

// Wrap-safe monotonic wait: spins until *p has advanced to at least `target`.
// Uses "before" comparison (int32_t)(observed - target) >= 0 so the counter can grow
// forever and wrap harmlessly.
static inline void wait_backoff_ge(const volatile uint32_t* p, uint32_t target) {
    wait_backoff_until(p, [target](uint32_t observed) {
        return static_cast<int32_t>(observed - target) >= 0;
    });
}

static inline void wait_backoff_eq(const volatile uint32_t* p, uint32_t target) {
    wait_backoff_until(p, [target](uint32_t observed) {
        return observed == target;
    });
}

/* ======== Thread affinity ======== */

static inline void set_thread_affinity(pthread_t thr, int cpu_id) {
    if (cpu_id < 0)
        return;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    // Core pinning is a performance contract: an unpinned proxy shows up as
    // unexplained latency variance, so report the failure rather than hiding it
    // behind CCL_LOG_LEVEL=debug (matches ccl_base_thread::set_affinity).
    const int rc = pthread_setaffinity_np(thr, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        LOG_ERROR("[proxy] pthread_setaffinity_np(cpu=", cpu_id, ") failed, rc=", rc);
    }
    else {
        LOG_DEBUG("[proxy] thread pinned to cpu=", cpu_id);
    }
}

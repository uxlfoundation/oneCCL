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

// SYCL kernel memory-copy rate limiter (variable-byte acquire).
//
// A per-device, in-process token bucket in device-global memory. Participating
// kernels call `ccl::ccl_rate_limiter::acquire(handle, it, bytes)` from
// work-group-uniform control flow, charging exactly the number of bytes
// they are about to move. This eliminates the over-accounting inherent in
// fixed-quantum designs.
//
// Disabled paths:
//   * Runtime: configured rate is 0 -> handle.enabled is 0; hook returns
//     immediately after a single uniform-branch check.
//
// Runtime stats: set CCL_SYCL_RATE_LIMIT_STATS=1 to allocate device-side
// counters and have acquire()/refill() update them. Off by default.

#include <cstdint>
#include <vector>
#include <sycl/sycl.hpp>
#include "oneapi/ccl/config.h"

namespace ccl {
namespace ccl_rate_limiter {

// Device-side counters for observability. Allocated alongside the bucket in
// device USM when CCL_SYCL_RATE_LIMIT_STATS=1. All fields are atomically
// updated from device code (relaxed ordering — counters are best-effort).
struct rate_stats {
    // Acquire path
    uint64_t acquire_count; // total acquire() calls (leader path entries)
    uint64_t acquire_total_bytes; // sum of all bytes requested
    uint64_t acquire_immediate_count; // satisfied on first CAS (no wait)
    uint64_t acquire_wait_count; // needed at least one refill/wait cycle
    uint64_t acquire_cas_fail_count; // CAS failures in the consume loop
    uint64_t insufficient_tokens_count; // leader found tokens < requested bytes

    // Acquire overhead (device ticks) — min/max/avg
    uint64_t min_acquire_ticks; // shortest single acquire duration
    uint64_t max_acquire_ticks; // longest single acquire duration
    uint64_t total_acquire_ticks; // cumulative (avg = total / acquire_count)

    // Refill path
    uint64_t refill_count; // successful cooperative refills (clock CAS won)
    uint64_t refill_skip_count; // skipped (interval not yet elapsed)
    uint64_t refill_cas_fail_count; // clock-advance CAS failures
    uint64_t refill_force_count; // force=true refills (stalled path)
    uint64_t refill_total_earned; // total bytes credited

    // Misc
    uint64_t max_acquire_bytes; // largest single request
    uint64_t backoff_count; // short_backoff() invocations
};

// Device-side bucket. Allocated in device USM, one per physical device.
// All fields are written only by the limiter machinery.
struct rate_bucket {
    // Mutable fields (written on every refill / acquire).
    int64_t tokens; // bytes available; 0 <= tokens <= capacity_bytes
    uint64_t last_refill_ticks; // device-clock ticks of last credit-bearing refill

    // Immutable after init.
    int64_t capacity_bytes; // burst size
    int64_t ticks_to_bytes_scaled; // fixed-point: bytes_per_tick << shift
    int ticks_to_bytes_shift; // paired shift
    int pad0;
    int64_t max_safe_elapsed_ticks; // INT64_MAX / ticks_to_bytes_scaled
    int64_t max_safe_shift_earned; // INT64_MAX >> ticks_to_bytes_shift
    uint64_t refill_interval_ticks; // minimum gap between cooperative refills
    int enabled; // 0 -> hooks are no-ops
    int stats_enabled; // 0 -> stats collection is off
    int backoff_cycles; // short_backoff delay in device clock cycles

    // Pointer to device-side stats (null when stats disabled).
    rate_stats* stats;
};

// Captured by value into kernel lambdas. Tiny by design.
struct bucket_handle {
    rate_bucket* bucket; // null when disabled
    int64_t max_acquire_bytes; // host-side validation cap; not used on device hot path
    int stats_enabled;
    int backoff_cycles; // cached from bucket — avoids global mem load on fast path
};

// ---- Host-side API ----------------------------------------------------------

// Returns a handle for the device backing `q`. Lazily initializes the per-device
// bucket on first call. Returns a disabled handle when the configured rate is 0
// and throws on invalid config.
CCL_API bucket_handle get_handle(sycl::queue& q);


// Returns true when the rate limiter is active for this handle.
// Callers can use this to skip setup work when the limiter is off.
inline bool is_enabled(const bucket_handle& h) {
    return h.bucket != nullptr;
}

// Returns true when stats collection is active for this handle.
inline bool is_stats_enabled(const bucket_handle& h) {
    return h.stats_enabled != 0;
}

// Copies the device-side stats to host and returns them. Returns a zeroed
// struct if stats are disabled or the handle is inactive.
CCL_API rate_stats read_stats(sycl::queue& q, const bucket_handle& h);

// Atomically resets all stats counters to zero on the device.
CCL_API void reset_stats(sycl::queue& q, const bucket_handle& h);

// Reads and prints a formatted summary of the rate limiter stats to LOG_INFO.
// Also returns the snapshot. No-op (returns zeroed struct) if stats are disabled.
CCL_API rate_stats print_stats(sycl::queue& q, const bucket_handle& h);

// Same as print_stats(q, h), but allows routing output to stdout instead of
// the logger when print_to_stdout is true.
CCL_API rate_stats print_stats(sycl::queue& q, const bucket_handle& h, bool print_to_stdout);

// ---- Profiling (debug aid for kernel timing) --------------------------------
//
// Per-call accumulators. Each profile_host_start/stop pair records one entry
// in the per-call vectors; for device mode, all events added between start
// and stop are grouped under that call. Reset between measurement windows.
// Modes are advisory: callers in the collective code check the mode and skip
// the event capture entirely when device profiling is off.

enum class profile_mode {
    none = 0,
    host = 1, // CPU wall clock around the work region
    device = 2, // sum of SYCL event command_end - command_start
    both = 3,
};

CCL_API void set_profile_mode(profile_mode m);
CCL_API profile_mode get_profile_mode();

// Reset all per-call vectors and pending events.
CCL_API void profile_reset();

// Bracket one "call". Each start/stop pair appends one entry to the per-call
// vectors. Events registered between them are grouped under that call.
CCL_API void profile_host_start();
CCL_API void profile_host_stop();

// Register a kernel event for later device-time accumulation. Cheap; the wait
// and profiling-info read happen lazily in profile_get_device_us_per_call.
CCL_API void profile_add_device_event(const sycl::event& ev);

// Return per-call host wall times (microseconds). One entry per host_start/stop pair.
CCL_API std::vector<double> profile_get_host_us_per_call();

// Wait on all pending device events, return per-call device-time totals
// (microseconds). One entry per host_start/stop pair; entry i is the sum of
// all event durations registered during call i. Clears pending events.
CCL_API std::vector<double> profile_get_device_us_per_call();

// ---- Device-side hook -------------------------------------------------------

namespace detail {

// SPIR-V Khronos clock builtin. Scope 1 = Device. Returns a monotonic
// 64-bit tick counter from the device clock (Intel card timestamp register
// on Intel GPUs). The function attribute keeps the call from being Common subexpression eliminated.
// Guard with __SYCL_DEVICE_ONLY__ so the symbol is never referenced from
// the host-side translation unit, which would cause an undefined-reference
// linker error.
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" uint64_t __spirv_ReadClockKHR(int) noexcept;
#endif

// Reads the device's monotonic clock (Intel card timestamp register).
// Scope 1 = Device. Wrap-safe when elapsed < 2^63 ticks.
// Only callable from device code; __spirv_ReadClockKHR is not a host symbol.
#ifdef __SYCL_DEVICE_ONLY__
inline uint64_t device_clock() {
    return __spirv_ReadClockKHR(1);
}
#else
inline uint64_t device_clock() {
    return 0;
}
#endif

// IGC built-in: yield the EU for ~N clocks; not exposed as a SYCL symbol so
// it needs an explicit extern "C" declaration before first use.
#if defined(__INTEL_LLVM_COMPILER) && defined(__SYCL_DEVICE_ONLY__)
extern "C" SYCL_EXTERNAL void __builtin_IB_eu_thread_pause(unsigned int);
#endif

inline void short_backoff(uint32_t cycles) {
#if defined(__INTEL_LLVM_COMPILER) && defined(__SYCL_DEVICE_ONLY__)
    __builtin_IB_eu_thread_pause(cycles);
#elif defined(__SYCL_DEVICE_ONLY__)
    uint64_t t0 = __spirv_ReadClockKHR(3);
    while ((__spirv_ReadClockKHR(3) - t0) < static_cast<uint64_t>(cycles)) {
    }
#endif
}

constexpr sycl::access::address_space gmem = sycl::access::address_space::global_space;
using mo = sycl::memory_order;
using ms = sycl::memory_scope;

// Stats helpers. All take a possibly-null `rate_stats*`; when null (stats
// disabled at runtime via CCL_SYCL_RATE_LIMIT_STATS=0) they return immediately
// so callers don't need to branch.
inline void stat_inc(rate_stats* stats_ptr, uint64_t rate_stats::* field) {
    if (stats_ptr == nullptr)
        return;
    sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem>(stats_ptr->*field).fetch_add(1);
}

inline void stat_add(rate_stats* stats_ptr, uint64_t rate_stats::* field, uint64_t val) {
    if (stats_ptr == nullptr)
        return;
    sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem>(stats_ptr->*field).fetch_add(val);
}

inline void stat_max(rate_stats* stats_ptr, uint64_t rate_stats::* field, uint64_t val) {
    if (stats_ptr == nullptr)
        return;
    sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem>(stats_ptr->*field).fetch_max(val);
}

inline void stat_min(rate_stats* stats_ptr, uint64_t rate_stats::* field, uint64_t val) {
    if (stats_ptr == nullptr)
        return;
    sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem>(stats_ptr->*field).fetch_min(val);
}

// Fast path for per-acquire request accounting. Performs one null check and
// then updates all three counters without additional branching.
inline void stat_record_acquire_request(rate_stats* stats_ptr, uint64_t bytes) {
    if (stats_ptr == nullptr) {
        return;
    }
    sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem>(stats_ptr->acquire_count)
        .fetch_add(1);
    sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem>(stats_ptr->acquire_total_bytes)
        .fetch_add(bytes);
    sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem>(stats_ptr->max_acquire_bytes)
        .fetch_max(bytes);
}

// Returns an atomic reference to the bucket's token counter.
inline sycl::atomic_ref<int64_t, mo::relaxed, ms::device, gmem> tokens_ref(rate_bucket* b) {
    return sycl::atomic_ref<int64_t, mo::relaxed, ms::device, gmem>(b->tokens);
}

// Returns an atomic reference to the bucket's last-refill timestamp.
inline sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem> clock_ref(rate_bucket* b) {
    return sycl::atomic_ref<uint64_t, mo::relaxed, ms::device, gmem>(b->last_refill_ticks);
}

// Converts elapsed device-clock ticks into bytes earned and the matching
// tick advance to record. Handles overflow, bucket-clamp, and sub-byte cases.
// Out-params: earned bytes to credit; used_ticks to advance last_refill_ticks.
inline void compute_credit(int64_t elapsed_ticks,
                           const rate_bucket* b,
                           int64_t& earned,
                           int64_t& used_ticks) {
    if (elapsed_ticks > b->max_safe_elapsed_ticks) {
        // Long-idle: fill to burst, discard surplus. Advance clock by full elapsed.
        earned = b->capacity_bytes;
        used_ticks = elapsed_ticks;
        return;
    }
    int64_t e = (elapsed_ticks * b->ticks_to_bytes_scaled) >> b->ticks_to_bytes_shift;
    if (e <= 0) {
        // Sub-byte elapsed; caller leaves clock untouched so the deficit accrues.
        earned = 0;
        used_ticks = 0;
        return;
    }
    if (e >= b->capacity_bytes) {
        // Bucket-clamp: discard surplus beyond burst.
        earned = b->capacity_bytes;
        used_ticks = elapsed_ticks;
        return;
    }
    if (e > b->max_safe_shift_earned) {
        // Pathological shift / earned combo: back-shift would overflow. Give up
        // the rounding-cancellation property; preserve correctness.
        earned = e;
        used_ticks = elapsed_ticks;
        return;
    }
    // Common path: time-equivalent of `earned` whole bytes.
    earned = e;
    used_ticks = (e << b->ticks_to_bytes_shift) / b->ticks_to_bytes_scaled;
}

// Atomically advances last_refill_ticks from `last` to `last + used_ticks`.
// Returns true if this thread won the CAS (i.e., is responsible for crediting).
inline bool advance_clock(rate_bucket* b, uint64_t last, int64_t used_ticks) {
    auto r = clock_ref(b);
    uint64_t expected = last;
    return r.compare_exchange_strong(expected, last + static_cast<uint64_t>(used_ticks));
}

// Adds `earned` bytes to the token counter, clamped to capacity_bytes.
// Uses a CAS loop to handle concurrent credits from multiple work-groups.
inline void credit_tokens(rate_bucket* b, int64_t earned) {
    auto t = tokens_ref(b);
    int64_t cur = t.load();
    while (true) {
        int64_t want = cur + earned;
        if (want > b->capacity_bytes) {
            want = b->capacity_bytes;
        }
        if (t.compare_exchange_strong(cur, want)) {
            return;
        }
        // CAS reloads `cur` on failure; loop body recomputes the clamp.
    }
}

// Attempts a cooperative token refill. Reads the device clock, computes earned
// bytes since the last refill, and credits them if this thread wins the clock CAS.
// Skips the refill if elapsed < refill_interval_ticks and `force` is false.
inline void try_refill(rate_bucket* b, bool force) {
    uint64_t now = device_clock();
    uint64_t last = clock_ref(b).load();
    uint64_t elapsed_ticks = now - last;

    if (!force && elapsed_ticks < b->refill_interval_ticks) {
        stat_inc(b->stats, &rate_stats::refill_skip_count);
        return;
    }
    if (elapsed_ticks == 0) {
        return;
    }

    int64_t earned = 0;
    int64_t used_ticks = 0;
    compute_credit(static_cast<int64_t>(elapsed_ticks), b, earned, used_ticks);

    if (earned <= 0) {
        return;
    }

    if (!advance_clock(b, last, used_ticks)) {
        stat_inc(b->stats, &rate_stats::refill_cas_fail_count);
        return;
    }

    stat_inc(b->stats, &rate_stats::refill_count);
    stat_add(b->stats, &rate_stats::refill_total_earned, static_cast<uint64_t>(earned));
    credit_tokens(b, earned);
}

// Busy-polls the token counter until >= `bytes` tokens are available or
// max_wait_ticks have elapsed (wrap-safe). Exits early on either condition
// so the caller can attempt a refill and re-enter if still empty.
inline void wait_for_tokens_or_interval(rate_bucket* b,
                                        int64_t bytes,
                                        uint64_t max_wait_ticks,
                                        uint32_t backoff_cycles) {
    uint64_t start = device_clock();

    while (true) {
        if (tokens_ref(b).load() >= bytes) {
            return;
        }
        if ((device_clock() - start) >= max_wait_ticks) {
            return;
        }
        short_backoff(backoff_cycles);
        stat_inc(b->stats, &rate_stats::backoff_count);
    }
}

// Core acquire loop executed by work-group lane 0. Spins until exactly `bytes`
// tokens have been atomically subtracted from the bucket, triggering
// cooperative refills whenever the bucket is found empty.
// use_fetch_sub, backoff_cycles, and s are passed from the bucket_handle
// (register-resident) to avoid loading them from global memory on every call.
// s is null when stats are disabled.
inline void acquire_leader_path(rate_bucket* b,
                                int64_t bytes,
                                uint32_t backoff_cycles,
                                rate_stats* stats_ptr) {
    uint64_t acquire_start = (stats_ptr != nullptr) ? device_clock() : 0;
    bool waited = false;
    uint64_t entered_empty_at = 0;
    auto t = tokens_ref(b);

    while (true) {
        // Speculative fetch_sub: one unconditional atomic on the success branch.
        // On failure we restore and fall through to refill/wait.
        int64_t old = t.fetch_sub(bytes);
        if (old >= bytes) {
            if (stats_ptr != nullptr) {
                uint64_t elapsed = device_clock() - acquire_start;
                stat_add(stats_ptr, &rate_stats::total_acquire_ticks, elapsed);
                stat_max(stats_ptr, &rate_stats::max_acquire_ticks, elapsed);
                stat_min(stats_ptr, &rate_stats::min_acquire_ticks, elapsed);
                if (waited) {
                    stat_inc(stats_ptr, &rate_stats::acquire_wait_count);
                }
                else {
                    stat_inc(stats_ptr, &rate_stats::acquire_immediate_count);
                }
            }
            return;
        }
        // Restore: we took tokens we weren't entitled to.
        t.fetch_add(bytes);
        stat_inc(stats_ptr, &rate_stats::acquire_cas_fail_count);

        waited = true;
        stat_inc(stats_ptr, &rate_stats::insufficient_tokens_count);
        uint64_t now = device_clock();
        if (entered_empty_at == 0) {
            entered_empty_at = now;
        }
        bool stalled = (now - entered_empty_at) > b->refill_interval_ticks;
        if (stalled) {
            stat_inc(stats_ptr, &rate_stats::refill_force_count);
        }
        try_refill(b, stalled);

        wait_for_tokens_or_interval(b, bytes, b->refill_interval_ticks, backoff_cycles);
    }
}

} // namespace detail

// Work-group-uniform rate-limiter hook. Call from uniform control flow before
// moving `bytes` of data. Lane 0 runs the token-acquire loop; all lanes
// then synchronize via a group barrier before proceeding.
// Precondition (host-enforced): bytes <= capacity_bytes.
inline void acquire(const bucket_handle& h, const sycl::nd_item<1>& it, int64_t bytes) {
    if (!h.bucket) {
        return;
    }
    if (it.get_local_id(0) == 0) {
        // Load stats pointer once — null when stats disabled, avoiding CL1 fetch entirely.
        rate_stats* stats_ptr = (h.stats_enabled != 0) ? h.bucket->stats : nullptr;
        detail::stat_record_acquire_request(stats_ptr, static_cast<uint64_t>(bytes));
        detail::acquire_leader_path(
            h.bucket, bytes, static_cast<uint32_t>(h.backoff_cycles), stats_ptr);
    }
    sycl::group_barrier(it.get_group());
}

} // namespace ccl_rate_limiter
} // namespace ccl

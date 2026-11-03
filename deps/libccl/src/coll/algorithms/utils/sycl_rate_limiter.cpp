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

// Host side of the SYCL kernel memory-copy rate limiter.
// See sycl_rate_limiter_design.md (§8 Initialization) for the full policy.

#include "coll/algorithms/utils/sycl_rate_limiter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <limits>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <iostream>
#include <vector>

#include "common/global/global.hpp"
#include "common/log/log.hpp"

namespace ccl {
namespace ccl_rate_limiter {

namespace {

// Process-local map: device id -> bucket pointer (in that device's USM).
// Devices are identified by sycl::device equality.
struct device_entry {
    rate_bucket* bucket;
    rate_stats* dev_stats_ptr;
    sycl::context context;
    int64_t max_acquire_bytes;
    bool stats_enabled;
    int backoff_cycles;
};

uint64_t get_rate_limiter_env_value(const char* env_name,
                                    uint64_t min_value,
                                    uint64_t max_value,
                                    uint64_t default_value,
                                    const char* value_desc) {
    const char* v = std::getenv(env_name);
    if (!v || !*v) {
        return default_value;
    }

    // strtoull accepts optional sign; explicitly reject negatives.
    if (v[0] == '-') {
        LOG_WARN("sycl_rate_limiter: invalid ",
                 env_name,
                 "=\"",
                 v,
                 "\"; using default ",
                 default_value,
                 " (allowed range: ",
                 min_value,
                 "..",
                 max_value,
                 ", ",
                 value_desc,
                 ")");
        return default_value;
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(v, &end, 10);
    if (errno == ERANGE || end == v || *end != '\0' || parsed < min_value || parsed > max_value) {
        LOG_WARN("sycl_rate_limiter: invalid ",
                 env_name,
                 "=\"",
                 v,
                 "\"; using default ",
                 default_value,
                 " (allowed range: ",
                 min_value,
                 "..",
                 max_value,
                 ", ",
                 value_desc,
                 ")");
        return default_value;
    }

    return static_cast<uint64_t>(parsed);
}

size_t read_rate_bytes_per_sec_env(size_t default_value) {
    return static_cast<size_t>(
        get_rate_limiter_env_value("CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC",
                                   uint64_t{ 0 },
                                   static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
                                   static_cast<uint64_t>(default_value),
                                   "bytes per second"));
}

size_t read_burst_bytes_env(size_t default_value) {
    return static_cast<size_t>(
        get_rate_limiter_env_value("CCL_SYCL_RATE_LIMIT_BURST_BYTES",
                                   uint64_t{ 0 },
                                   static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
                                   static_cast<uint64_t>(default_value),
                                   "burst bytes"));
}

size_t read_max_acquire_bytes_env(size_t default_value) {
    return static_cast<size_t>(
        get_rate_limiter_env_value("CCL_SYCL_RATE_LIMIT_MAX_ACQUIRE_BYTES",
                                   uint64_t{ 0 },
                                   static_cast<uint64_t>(std::numeric_limits<size_t>::max()),
                                   static_cast<uint64_t>(default_value),
                                   "max acquire bytes"));
}

uint64_t read_device_clock_hz_env(uint64_t default_value) {
    return get_rate_limiter_env_value("CCL_SYCL_RATE_LIMIT_DEVICE_CLOCK_HZ",
                                      uint64_t{ 0 },
                                      std::numeric_limits<uint64_t>::max(),
                                      default_value,
                                      "device clock Hz");
}

uint64_t read_refill_interval_ns_env(uint64_t default_value) {
    return get_rate_limiter_env_value("CCL_SYCL_RATE_LIMIT_REFILL_INTERVAL_NS",
                                      uint64_t{ 0 },
                                      std::numeric_limits<uint64_t>::max(),
                                      default_value,
                                      "refill interval ns");
}

int read_stats_env(int default_value) {
    return static_cast<int>(get_rate_limiter_env_value("CCL_SYCL_RATE_LIMIT_STATS",
                                                        uint64_t{ 0 },
                                                        uint64_t{ 1 },
                                                        static_cast<uint64_t>(default_value),
                                                        "stats toggle"));
}

int read_burst_divisor_env() {
    constexpr int default_divisor = 500; // 1/500 s = 2 ms
    constexpr int min_divisor = 1;
    constexpr int max_divisor = 1000000;
    return static_cast<int>(
        get_rate_limiter_env_value("CCL_SYCL_RATE_LIMIT_BURST_DIVISOR",
                                   static_cast<uint64_t>(min_divisor),
                                   static_cast<uint64_t>(max_divisor),
                                   static_cast<uint64_t>(default_divisor),
                                   "burst divisor"));
}

int read_backoff_cycles_env() {
    constexpr int default_cycles = 32;
    constexpr int min_cycles = 1;
    constexpr int max_cycles = 1000000;
    return static_cast<int>(
        get_rate_limiter_env_value("CCL_SYCL_RATE_LIMIT_BACKOFF_CYCLES",
                                   static_cast<uint64_t>(min_cycles),
                                   static_cast<uint64_t>(max_cycles),
                                   static_cast<uint64_t>(default_cycles),
                                   "device backoff cycles"));
}

std::mutex devices_mutex;
std::unordered_map<sycl::device, std::vector<device_entry>> devices_by_sycl_device;

// ---- Calibration ----------------------------------------------------------
//
// Both timestamps are captured inside a SINGLE kernel invocation with a
// device-clock spin between them. This avoids the cross-kernel clock reset
// seen on Intel BMG/Xe2 when the GPU enters a low-power state between
// separate kernel launches (which would make t1 < t0).
//
// The kernel spins for SPIN_TICKS device-clock ticks, holding the GPU active.
// The host measures the wall-clock time around the kernel wait; the ratio
// dev_delta / host_ns gives ticks-per-second. We take the median of several
// samples to reduce launch-latency jitter.

double calibrate_ticks_per_sec(sycl::queue& q) {
    using clk = std::chrono::steady_clock;
    constexpr int n_samples = 5;
    // Spin ~16M ticks per sample. At 1 GHz ≈ 16 ms; at 3 GHz ≈ 5 ms.
    // Large enough to dwarf kernel-launch overhead (~0.1 ms), small enough to be fast.
    constexpr uint64_t SPIN_TICKS = 1ULL << 24; // 16,777,216

    uint64_t* dev_ts = sycl::malloc_device<uint64_t>(2, q);
    if (!dev_ts) {
        throw std::runtime_error("sycl_rate_limiter: failed to allocate calibration buffer");
    }

    // Adaptive warmup: spin until consecutive readings agree within 5%.
    // One iteration (~11 ms at 1.5 GHz) is not enough to pull the GPU from a
    // low-power P-state to its boost clock; keep spinning until stable.
    // Cap at MAX_WARMUP iterations to bound startup time.
    constexpr int MAX_WARMUP = 12;
    constexpr double STABLE_THRESHOLD = 0.05; // 5% relative change → stable

    double prev_warmup_tps = 0.0;
    double min_warmup_tps = 0.0;
    double max_warmup_tps = 0.0;
    for (int warmup_index = 0; warmup_index < MAX_WARMUP; ++warmup_index) {
        auto wt0 = clk::now();
        q.single_task([=]() {
             uint64_t t0 = detail::device_clock();
             while (detail::device_clock() - t0 < SPIN_TICKS) {
             }
             dev_ts[0] = t0;
             dev_ts[1] = detail::device_clock();
         }).wait();
        auto wt1 = clk::now();

        double warmup_ns = std::chrono::duration<double, std::nano>(wt1 - wt0).count();
        double warmup_tps = (warmup_ns > 0.0) ? (SPIN_TICKS / warmup_ns * 1e9) : 0.0;
        if ((warmup_index == 0) || (warmup_tps < min_warmup_tps)) {
            min_warmup_tps = warmup_tps;
        }
        if ((warmup_index == 0) || (warmup_tps > max_warmup_tps)) {
            max_warmup_tps = warmup_tps;
        }

        if (prev_warmup_tps > 0.0 &&
            std::abs(warmup_tps - prev_warmup_tps) / prev_warmup_tps < STABLE_THRESHOLD) {
            LOG_DEBUG("sycl_rate_limiter: warmup stable after ",
                      warmup_index + 1,
                      " iterations (",
                      warmup_tps / 1e6,
                      " MHz, min=",
                      min_warmup_tps / 1e6,
                      " MHz, max=",
                      max_warmup_tps / 1e6,
                      " MHz)");
            break;
        }
        prev_warmup_tps = warmup_tps;
    }

    std::array<double, n_samples> samples;
    int n_valid = 0;
    for (int sample_index = 0; sample_index < n_samples; ++sample_index) {
        auto host_t0 = clk::now();
        q.single_task([=]() {
             uint64_t t0 = detail::device_clock();
             while (detail::device_clock() - t0 < SPIN_TICKS) {
             }
             dev_ts[0] = t0;
             dev_ts[1] = detail::device_clock();
         }).wait();
        auto host_t1 = clk::now();

        uint64_t t0 = 0, t1 = 0;
        q.memcpy(&t0, &dev_ts[0], sizeof(uint64_t)).wait();
        q.memcpy(&t1, &dev_ts[1], sizeof(uint64_t)).wait();

        uint64_t dev_delta = (t1 >= t0) ? (t1 - t0) : 0;
        double host_ns = std::chrono::duration<double, std::nano>(host_t1 - host_t0).count();

        LOG_TRACE("sycl_rate_limiter: calibration sample ",
                  sample_index,
                  ": t0=",
                  t0,
                  " t1=",
                  t1,
                  " dev_delta=",
                  dev_delta,
                  " host_ns=",
                  static_cast<int64_t>(host_ns));

        if (dev_delta == 0 || host_ns <= 0.0)
            continue;
        samples[n_valid++] = static_cast<double>(dev_delta) / host_ns * 1e9;
    }

    sycl::free(dev_ts, q);
    if (n_valid == 0) {
        throw std::runtime_error(
            "sycl_rate_limiter: device clock calibration failed "
            "(all samples had dev_delta=0; __spirv_ReadClockKHR may be unsupported)");
    }

    // Median of valid samples.
    std::sort(samples.begin(), samples.begin() + n_valid);
    double median = samples[n_valid / 2];
    if (n_valid % 2 == 0) {
        median = 0.5 * (samples[n_valid / 2 - 1] + samples[n_valid / 2]);
    }

    LOG_DEBUG("sycl_rate_limiter: calibrated device clock = ",
              static_cast<int64_t>(median),
              " ticks/s (",
              median / 1e6,
              " MHz), ",
              n_valid,
              "/",
              n_samples,
              " samples");
    return median;
}

// ---- Shift selection (design §8.3) ---------------------------------------

struct shift_choice {
    int shift;
    int64_t scaled;
    // max_safe_elapsed: largest elapsed tick delta that can be multiplied by
    // ticks_to_bytes_scaled without int64 overflow.
    int64_t max_safe_elapsed;
    // max_safe_shift_earned: largest earned-byte value that can be left-shifted
    // by `shift` while reconstructing used_ticks.
    int64_t max_safe_shift_earned;
};

bool select_shift(double rate_bps,
                  int64_t capacity_bytes,
                  uint64_t refill_interval_ticks,
                  double ticks_per_sec,
                  shift_choice& out) {
    constexpr int PREFERRED_SHIFT = 30;
    constexpr int MIN_SHIFT = 8;
    constexpr int64_t HEADROOM = 16;

    for (int shift = PREFERRED_SHIFT; shift >= MIN_SHIFT; --shift) {
        // Back-multiply: capacity << shift must fit in int64.
        if (capacity_bytes > 0 &&
            shift >
                63 - static_cast<int>(std::ceil(std::log2(static_cast<double>(capacity_bytes))))) {
            continue;
        }

        // ticks_to_bytes_scaled = round(rate * (1<<shift) / ticks_per_sec)
        long double scaled_d = static_cast<long double>(rate_bps) *
                               static_cast<long double>(1ULL << shift) /
                               static_cast<long double>(ticks_per_sec);
        if (scaled_d < 1.0L) {
            // Representability: rate too small at this shift; try smaller shift.
            continue;
        }
        if (scaled_d > static_cast<long double>(INT64_MAX / 2)) {
            // Scaled value too large; try smaller shift.
            continue;
        }
        int64_t scaled = static_cast<int64_t>(scaled_d + 0.5L);
        if (scaled <= 0)
            continue;

        // Forward-multiply headroom under normal operation.
        int64_t max_safe_elapsed = INT64_MAX / scaled;
        int64_t max_safe_shift_earned = INT64_MAX >> shift;
        if (refill_interval_ticks != 0 &&
            scaled > INT64_MAX / HEADROOM /
                         static_cast<int64_t>(std::max<uint64_t>(refill_interval_ticks, 1))) {
            continue;
        }

        // Long-idle headroom: avoid overflow on very long idle intervals.
        if (max_safe_elapsed < static_cast<int64_t>(refill_interval_ticks) * HEADROOM) {
            continue;
        }

        out.shift = shift;
        out.scaled = scaled;
        out.max_safe_elapsed = max_safe_elapsed;
        out.max_safe_shift_earned = max_safe_shift_earned;
        LOG_DEBUG("sycl_rate_limiter: select_shift chose"
                  " shift=",
                  shift,
                  " scaled=",
                  scaled,
                  " max_safe_elapsed=",
                  max_safe_elapsed,
                  " max_safe_shift_earned=",
                  max_safe_shift_earned,
                  " (rate_bps=",
                  static_cast<int64_t>(rate_bps),
                  ", capacity_bytes=",
                  capacity_bytes,
                  ", refill_interval_ticks=",
                  refill_interval_ticks,
                  ", ticks_per_sec=",
                  static_cast<int64_t>(ticks_per_sec),
                  ")");
        return true;
    }
    return false;
}

// ---- One-time per-device init --------------------------------------------

device_entry create_for_device(sycl::queue& q) {
    int backoff_cycles = read_backoff_cycles_env();
    const auto& env = ccl::global_data::env();
    size_t rate_bps = read_rate_bytes_per_sec_env(env.sycl_rate_limit_bytes_per_sec);
    if (rate_bps == 0) {
        return device_entry{ nullptr, nullptr, q.get_context(), 0, 0, 0 };
    }

    // Validate / derive parameters per design §8.2.
    int burst_divisor = read_burst_divisor_env();
    CCL_THROW_IF_NOT(burst_divisor != 0,
                     "sycl_rate_limiter: CCL_SYCL_RATE_LIMIT_BURST_DIVISOR must be non-zero");
    size_t burst_bytes = read_burst_bytes_env(env.sycl_rate_limit_burst_bytes);
    if (burst_bytes == 0) {
        // Default: rate / burst_divisor  (divisor=500 => 2 ms worth; env: CCL_SYCL_RATE_LIMIT_BURST_DIVISOR).
        burst_bytes = static_cast<size_t>((rate_bps + burst_divisor - 1) / burst_divisor);
        if (burst_bytes == 0) {
            burst_bytes = 1024;
        }
    }
    if (burst_bytes <= 1) {
        throw std::runtime_error("sycl_rate_limiter: invalid burst_bytes (must be > 1)");
    }

    size_t max_acquire_bytes = read_max_acquire_bytes_env(env.sycl_rate_limit_max_acquire_bytes);
    if (max_acquire_bytes == 0) {
        max_acquire_bytes = burst_bytes / 2;
    }
    if (max_acquire_bytes > burst_bytes / 2) {
        throw std::runtime_error(
            "sycl_rate_limiter: invalid config: max_acquire_bytes (" +
            std::to_string(max_acquire_bytes) + ") must be <= burst_bytes/2 (" +
            std::to_string(burst_bytes / 2) +
            "); set CCL_SYCL_RATE_LIMIT_MAX_ACQUIRE_BYTES <= CCL_SYCL_RATE_LIMIT_BURST_BYTES/2");
    }
    if (max_acquire_bytes == 0) {
        throw std::runtime_error("sycl_rate_limiter: derived max_acquire_bytes == 0");
    }

    // Calibrate the device clock (__spirv_ReadClockKHR spin measurement). If
    // the user pinned a value via CCL_SYCL_RATE_LIMIT_DEVICE_CLOCK_HZ, use
    // that but still calibrate and print both for comparison.
    double ticks_per_sec;
    double calibrated_ticks_per_sec = calibrate_ticks_per_sec(q);
    uint64_t device_clock_hz = read_device_clock_hz_env(env.sycl_rate_limit_device_clock_hz);
    if (device_clock_hz != 0) {
        ticks_per_sec = static_cast<double>(device_clock_hz);
        LOG_DEBUG("sycl_rate_limiter: user-supplied device clock = ",
                  static_cast<int64_t>(ticks_per_sec),
                  " ticks/s (",
                  ticks_per_sec / 1e6,
                  " MHz) [",
                  CCL_SYCL_RATE_LIMIT_DEVICE_CLOCK_HZ,
                  "]"
                  " vs calibrated = ",
                  static_cast<int64_t>(calibrated_ticks_per_sec),
                  " ticks/s (",
                  calibrated_ticks_per_sec / 1e6,
                  " MHz)"
                  " -- using user-supplied value");
    }
    else {
        ticks_per_sec = calibrated_ticks_per_sec;
    }

    // Refill interval: env (ns) or default to ~burst/(8*rate).
    uint64_t refill_interval_ns =
        read_refill_interval_ns_env(env.sycl_rate_limit_refill_interval_ns);
    if (refill_interval_ns == 0) {
        // Aim for ~burst/(8*rate) so the bucket refills smoothly.
        double s = static_cast<double>(burst_bytes) / static_cast<double>(rate_bps) / 8.0;
        refill_interval_ns = static_cast<uint64_t>(std::max(s * 1e9, 1000.0));
    }
    uint64_t refill_interval_ticks =
        static_cast<uint64_t>(static_cast<double>(refill_interval_ns) * ticks_per_sec / 1e9);
    constexpr uint64_t MIN_INTERVAL_TICKS = 8;
    if (refill_interval_ticks < MIN_INTERVAL_TICKS) {
        refill_interval_ticks = MIN_INTERVAL_TICKS;
    }

    // Pick a fixed-point shift.
    shift_choice sc{};
    if (!select_shift(static_cast<double>(rate_bps),
                      static_cast<int64_t>(burst_bytes),
                      refill_interval_ticks,
                      ticks_per_sec,
                      sc)) {
        throw std::runtime_error(
            "sycl_rate_limiter: no viable fixed-point shift for "
            "(rate=" +
            std::to_string(rate_bps) + ", burst=" + std::to_string(burst_bytes) +
            ", clock_hz=" + std::to_string(static_cast<int64_t>(ticks_per_sec)) + ")");
    }

    // Allocate the bucket in device USM.
    rate_bucket* bucket = sycl::malloc_device<rate_bucket>(1, q);
    if (!bucket) {
        throw std::runtime_error("sycl_rate_limiter: failed to allocate device bucket");
    }

    int stats_on = (read_stats_env(env.sycl_rate_limit_stats) != 0) ? 1 : 0;
    rate_stats* dev_stats = nullptr;
    if (stats_on) {
        dev_stats = sycl::malloc_device<rate_stats>(1, q);
        if (!dev_stats) {
            throw std::runtime_error("sycl_rate_limiter: failed to allocate device stats");
        }
        rate_stats zeroed{};
        zeroed.min_acquire_ticks = UINT64_MAX;
        q.memcpy(dev_stats, &zeroed, sizeof(rate_stats)).wait();
    }

    rate_bucket initial{};
    initial.tokens = static_cast<int64_t>(burst_bytes);
    initial.last_refill_ticks = 0;
    initial.capacity_bytes = static_cast<int64_t>(burst_bytes);
    initial.ticks_to_bytes_scaled = sc.scaled;
    initial.ticks_to_bytes_shift = sc.shift;
    initial.max_safe_elapsed_ticks = sc.max_safe_elapsed;
    initial.max_safe_shift_earned = sc.max_safe_shift_earned;
    initial.refill_interval_ticks = refill_interval_ticks;
    initial.enabled = 1;
    initial.stats_enabled = stats_on;
    initial.backoff_cycles = backoff_cycles;
    initial.stats = dev_stats;

    q.memcpy(bucket, &initial, sizeof(rate_bucket)).wait();
    // Stamp last_refill_ticks from the device's own clock.
    rate_bucket* b_ptr = bucket;
    q.single_task([=]() {
         b_ptr->last_refill_ticks = detail::device_clock();
     }).wait();

    LOG_DEBUG("sycl_rate_limiter: enabled rate=",
              rate_bps,
              " B/s, burst=",
              burst_bytes,
              " B, max_acquire=",
              max_acquire_bytes,
              " B, refill_interval=",
              refill_interval_ns,
              " ns, ticks/s=",
              static_cast<int64_t>(ticks_per_sec),
              ", shift=",
              sc.shift,
              ", backoff_cycles=",
              backoff_cycles,
              ", stats=",
              stats_on);

    return device_entry{ bucket,          dev_stats,
                         q.get_context(), static_cast<int64_t>(max_acquire_bytes),
                         (stats_on != 0), backoff_cycles };
}

} // namespace

bucket_handle get_handle(sycl::queue& q) {
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    std::lock_guard<std::mutex> lock(devices_mutex);
    auto it = devices_by_sycl_device.find(dev);
    if (it == devices_by_sycl_device.end()) {
        std::vector<device_entry> entries;
        entries.push_back(create_for_device(q));
        it = devices_by_sycl_device.emplace(dev, std::move(entries)).first;
    }

    std::vector<device_entry>& entries = it->second;
    auto entry_it = std::find_if(entries.begin(), entries.end(), [&](const device_entry& entry) {
        return entry.context == ctx;
    });
    if (entry_it == entries.end()) {
        entries.push_back(create_for_device(q));
        entry_it = std::prev(entries.end());
    }

    const device_entry& e = *entry_it;
    return bucket_handle{ e.bucket, e.max_acquire_bytes, e.stats_enabled, e.backoff_cycles };
}

rate_stats read_stats(sycl::queue& q, const bucket_handle& h) {
    rate_stats result{};
    if ((h.stats_enabled == 0) || (h.bucket == nullptr)) {
        return result;
    }
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    rate_stats* dev_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(devices_mutex);
        auto it = devices_by_sycl_device.find(dev);
        if (it == devices_by_sycl_device.end()) {
            return result;
        }
        auto entry_it = std::find_if(it->second.begin(), it->second.end(), [&](const device_entry& entry) {
            return entry.context == ctx;
        });
        if (entry_it == it->second.end() || !entry_it->dev_stats_ptr) {
            return result;
        }
        dev_ptr = entry_it->dev_stats_ptr;
    }
    q.memcpy(&result, dev_ptr, sizeof(rate_stats)).wait();
    return result;
}

void reset_stats(sycl::queue& q, const bucket_handle& h) {
    if ((h.stats_enabled == 0) || (h.bucket == nullptr)) {
        return;
    }
    sycl::device dev = q.get_device();
    sycl::context ctx = q.get_context();
    rate_stats* dev_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(devices_mutex);
        auto it = devices_by_sycl_device.find(dev);
        if (it == devices_by_sycl_device.end()) {
            return;
        }
        auto entry_it = std::find_if(it->second.begin(), it->second.end(), [&](const device_entry& entry) {
            return entry.context == ctx;
        });
        if (entry_it == it->second.end() || !entry_it->dev_stats_ptr) {
            return;
        }
        dev_ptr = entry_it->dev_stats_ptr;
    }
    rate_stats zeroed{};
    zeroed.min_acquire_ticks = UINT64_MAX;
    q.memcpy(dev_ptr, &zeroed, sizeof(rate_stats)).wait();
}

rate_stats print_stats(sycl::queue& q, const bucket_handle& h) {
    return print_stats(q, h, false);
}

rate_stats print_stats(sycl::queue& q, const bucket_handle& h, bool print_to_stdout) {
    auto emit_line = [&](const char* label, auto value) {
        if (print_to_stdout) {
            std::cout << label << value << std::endl;
        }
        else {
            LOG_INFO(label, value);
        }
    };

    if ((h.stats_enabled == 0) || (h.bucket == nullptr)) {
        if (print_to_stdout) {
            std::cout
                << "sycl_rate_limiter: print_stats: stats disabled or no bucket, returning zeros"
                << std::endl;
        }
        else {
            LOG_INFO(
                "sycl_rate_limiter: print_stats: stats disabled or no bucket, returning zeros");
        }
        return rate_stats{};
    }
    rate_stats s = read_stats(q, h);
    if (h.stats_enabled == 0) {
        return s;
    }
    if (print_to_stdout) {
        std::cout << "sycl_rate_limiter stats:" << std::endl;
    }
    else {
        LOG_INFO("sycl_rate_limiter stats:");
    }
    emit_line("  acquire_count:           ", s.acquire_count);
    emit_line("  acquire_total_bytes:     ", s.acquire_total_bytes);
    emit_line("  acquire_immediate_count: ", s.acquire_immediate_count);
    emit_line("  acquire_wait_count:      ", s.acquire_wait_count);
    emit_line("  acquire_cas_fail_count:  ", s.acquire_cas_fail_count);
    emit_line("  insufficient_count:      ", s.insufficient_tokens_count);
    emit_line("  min_acquire_ticks:       ",
              s.min_acquire_ticks == UINT64_MAX ? 0 : s.min_acquire_ticks);
    emit_line("  max_acquire_ticks:       ", s.max_acquire_ticks);
    emit_line("  total_acquire_ticks:     ", s.total_acquire_ticks);
    emit_line("  refill_count:            ", s.refill_count);
    emit_line("  refill_skip_count:       ", s.refill_skip_count);
    emit_line("  refill_cas_fail_count:   ", s.refill_cas_fail_count);
    emit_line("  refill_force_count:      ", s.refill_force_count);
    emit_line("  refill_total_earned:     ", s.refill_total_earned);
    emit_line("  max_acquire_bytes:       ", s.max_acquire_bytes);
    emit_line("  backoff_count:           ", s.backoff_count);
    if (s.acquire_count > 0) {
        uint64_t avg_ticks = s.total_acquire_ticks / s.acquire_count;
        double pct_immediate = 100.0 * static_cast<double>(s.acquire_immediate_count) /
                               static_cast<double>(s.acquire_count);
        uint64_t avg_bytes = s.acquire_total_bytes / s.acquire_count;
        if (print_to_stdout) {
            std::cout << "  -- derived --" << std::endl;
            std::cout << "  avg_acquire_ticks:       " << avg_ticks << std::endl;
            std::cout << "  avg_acquire_bytes:       " << avg_bytes << std::endl;
            std::cout << "  immediate_pct:           " << pct_immediate << "%" << std::endl;
        }
        else {
            LOG_INFO("  -- derived --");
            LOG_INFO("  avg_acquire_ticks:       ", avg_ticks);
            LOG_INFO("  avg_acquire_bytes:       ", avg_bytes);
            LOG_INFO("  immediate_pct:           ", pct_immediate, "%");
        }
    }
    return s;
}

// ---- Profiling (debug aid) -------------------------------------------------
namespace {

struct profile_state {
    profile_mode mode = profile_mode::none;
    // Per-call bookkeeping. host_t0/event_baseline are scratch fields written
    // by profile_host_start and consumed by the matching profile_host_stop.
    std::chrono::steady_clock::time_point host_t0{};
    size_t event_baseline = 0;
    // One entry per host_start/stop pair.
    std::vector<double> host_us_per_call;
    std::vector<size_t> events_per_call;
    // All events registered via profile_add_device_event, in order.
    std::vector<sycl::event> events;
};

profile_state& profile_state_instance() {
    static profile_state s;
    return s;
}

} // namespace

void set_profile_mode(profile_mode m) {
    profile_state_instance().mode = m;
}

profile_mode get_profile_mode() {
    return profile_state_instance().mode;
}

void profile_reset() {
    auto& s = profile_state_instance();
    s.host_t0 = {};
    s.event_baseline = 0;
    s.host_us_per_call.clear();
    s.events_per_call.clear();
    s.events.clear();
}

void profile_host_start() {
    auto& s = profile_state_instance();
    if (s.mode == profile_mode::none)
        return;
    s.host_t0 = std::chrono::steady_clock::now();
    s.event_baseline = s.events.size();
}

void profile_host_stop() {
    auto& s = profile_state_instance();
    if (s.mode == profile_mode::none)
        return;
    if (s.mode == profile_mode::host || s.mode == profile_mode::both) {
        auto t1 = std::chrono::steady_clock::now();
        s.host_us_per_call.push_back(
            std::chrono::duration<double, std::micro>(t1 - s.host_t0).count());
    }
    if (s.mode == profile_mode::device || s.mode == profile_mode::both) {
        s.events_per_call.push_back(s.events.size() - s.event_baseline);
    }
}

void profile_add_device_event(const sycl::event& ev) {
    auto& s = profile_state_instance();
    if (s.mode == profile_mode::device || s.mode == profile_mode::both) {
        s.events.push_back(ev);
    }
}

std::vector<double> profile_get_host_us_per_call() {
    return profile_state_instance().host_us_per_call;
}

std::vector<double> profile_get_device_us_per_call() {
    auto& s = profile_state_instance();
    std::vector<sycl::event> events;
    events.swap(s.events);
    const std::vector<size_t>& counts = s.events_per_call;
    std::vector<double> out;
    out.reserve(counts.size());
    size_t idx = 0;
    for (size_t n : counts) {
        double call_us = 0.0;
        for (size_t k = 0; k < n; ++k, ++idx) {
            if (idx >= events.size())
                break;
            try {
                events[idx].wait();
                uint64_t t0 =
                    events[idx].get_profiling_info<sycl::info::event_profiling::command_start>();
                uint64_t t1 =
                    events[idx].get_profiling_info<sycl::info::event_profiling::command_end>();
                call_us += static_cast<double>(t1 - t0) / 1000.0;
            }
            catch (const sycl::exception& e) {
                LOG_WARN("sycl_rate_limiter: profile_get_device_us_per_call: ",
                         e.what(),
                         " (queue likely lacks enable_profiling)");
                return {};
            }
        }
        out.push_back(call_us);
    }
    return out;
}

} // namespace ccl_rate_limiter
} // namespace ccl

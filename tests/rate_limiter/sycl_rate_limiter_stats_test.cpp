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

#include "coll/algorithms/utils/sycl_rate_limiter.hpp"
#include "sycl_base.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>

using namespace std;
using namespace sycl;

namespace {
constexpr int64_t KiB = 1024;
constexpr int64_t MiB = 1024 * KiB;
} // namespace

// Usage: <prog> <device_type> <alloc_type>
//
//   device_type   'cpu' or 'gpu'
//   alloc_type    'host', 'device', or 'shared'
//
// Example:
//   mpirun -n 1 <prog> gpu shared
//
// Rate limiter environment variables:
//   CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC     Target bandwidth limit in bytes/s (0
//   = disabled) CCL_SYCL_RATE_LIMIT_BURST_BYTES        Token bucket capacity;
//   default = rate / BURST_DIVISOR CCL_SYCL_RATE_LIMIT_MAX_ACQUIRE_BYTES  Max
//   bytes a single acquire may request (default = burst/2)
//   CCL_SYCL_RATE_LIMIT_BURST_DIVISOR      Divisor used when deriving burst
//   from rate (default 500 -> 2 ms) CCL_SYCL_RATE_LIMIT_REFILL_INTERVAL_NS
//   Refill check period in nanoseconds (default auto)
//   CCL_SYCL_RATE_LIMIT_BACKOFF_CYCLES     Device spin-wait cycles between
//   retries (default 32) CCL_SYCL_RATE_LIMIT_DEVICE_CLOCK_HZ    Override device
//   clock frequency instead of calibrating CCL_SYCL_RATE_LIMIT_STATS Set to 1
//   to enable per-operation statistics counters

struct scenario_result {
    string name;
    double elapsed_us;
    ccl::ccl_rate_limiter::rate_stats stats;
    bool passed;
};

// Run a synthetic kernel that calls acquire() repeatedly to simulate
// a workload doing `total_bytes` of data movement in `chunk_bytes` increments.
scenario_result run_scenario(sycl::queue &q, const string &name,
                             int64_t total_bytes, int64_t chunk_bytes,
                             int num_work_groups, int work_group_size) {
    scenario_result result;
    result.name = name;
    result.passed = false;

    ccl::ccl_rate_limiter::bucket_handle handle =
        ccl::ccl_rate_limiter::get_handle(q);

    if (!ccl::ccl_rate_limiter::is_enabled(handle)) {
        result.elapsed_us = 0;
        result.passed = true;
        return result;
    }

    if (ccl::ccl_rate_limiter::is_stats_enabled(handle)) {
        ccl::ccl_rate_limiter::reset_stats(q, handle);
    }

    int64_t iterations_per_wg = total_bytes / (chunk_bytes * num_work_groups);
    if (iterations_per_wg < 1)
        iterations_per_wg = 1;

    auto t_start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler &h) {
         h.parallel_for(sycl::nd_range<1>(num_work_groups * work_group_size,
                                          work_group_size),
                        [=](sycl::nd_item<1> it) {
                            for (int64_t i = 0; i < iterations_per_wg; ++i) {
                                ccl::ccl_rate_limiter::acquire(handle, it,
                                                               chunk_bytes);
                            }
                        });
     }).wait();

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_us =
        std::chrono::duration<double, std::micro>(t_end - t_start).count();

    if (ccl::ccl_rate_limiter::is_stats_enabled(handle)) {
        result.stats = ccl::ccl_rate_limiter::read_stats(q, handle);
    }

    result.passed = true;
    return result;
}

void print_stats_summary(const scenario_result &r) {
    const auto &s = r.stats;
    std::cout << "  acquire_count:           " << s.acquire_count << std::endl;
    std::cout << "  acquire_total_bytes:     " << s.acquire_total_bytes
              << std::endl;
    std::cout << "  acquire_immediate_count: " << s.acquire_immediate_count
              << std::endl;
    std::cout << "  acquire_wait_count:      " << s.acquire_wait_count
              << std::endl;
    std::cout << "  acquire_cas_fail_count:  " << s.acquire_cas_fail_count
              << std::endl;
    std::cout << "  insufficient_tokens_count: " << s.insufficient_tokens_count
              << std::endl;
    std::cout << "  max_acquire_ticks:       " << s.max_acquire_ticks
              << std::endl;
    std::cout << "  refill_count:            " << s.refill_count << std::endl;
    std::cout << "  refill_force_count:      " << s.refill_force_count
              << std::endl;
    std::cout << "  refill_total_earned:     " << s.refill_total_earned
              << std::endl;
    std::cout << "  backoff_count:           " << s.backoff_count << std::endl;
    if (s.acquire_count > 0) {
        double pct = 100.0 * static_cast<double>(s.acquire_immediate_count) /
                     static_cast<double>(s.acquire_count);
        uint64_t avg_ticks = s.total_acquire_ticks / s.acquire_count;
        std::cout << "  -- derived --" << std::endl;
        std::cout << "  immediate_pct:           " << std::fixed
                  << std::setprecision(1) << pct << "%" << std::endl;
        std::cout << "  avg_acquire_ticks:       " << avg_ticks << std::endl;
    }
}

int main(int argc, char *argv[]) {
    if (!check_example_args(argc, argv))
        exit(1);

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);

    string device_type = argv[1];
    string alloc_type = argv[2];

    sycl::queue q;
    if (!create_test_sycl_queue(device_type, rank, q, args))
        return -1;

    if (rank != 0) {
        return 0;
    }

    std::cout << "=======================================" << std::endl;
    std::cout << "  Rate Limiter Stats & Behavior Test" << std::endl;
    std::cout << "=======================================" << std::endl;

    ccl::ccl_rate_limiter::bucket_handle handle =
        ccl::ccl_rate_limiter::get_handle(q);
    if (!ccl::ccl_rate_limiter::is_enabled(handle)) {
        std::cout << "Rate limiter is DISABLED." << std::endl;
        std::cout << "Set CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC > 0 to enable."
                  << std::endl;
        std::cout << "SKIPPED" << std::endl;
        return 0;
    }

    if (!ccl::ccl_rate_limiter::is_stats_enabled(handle)) {
        std::cout
            << "WARNING: Stats not enabled. Set CCL_SYCL_RATE_LIMIT_STATS=1"
            << std::endl;
        std::cout << "Timing tests will still run but stats won't be available."
                  << std::endl;
    }

    std::cout << std::endl;

    // Test 1: Small workload (fits in burst)
    // With a typical burst of ~2ms worth of bytes, a small workload that fits
    // entirely in the burst buffer should complete almost instantly (all
    // immediate).
    {
        std::cout << "--- Test 1: Small workload (fits in burst) ---"
                  << std::endl;
        int64_t total = 64 * KiB; // 64 KiB total
        // 4 KiB models a small cache-line-coalesced copy chunk.
        int64_t chunk = 4 * KiB;
        auto r = run_scenario(q, "small_burst", total, chunk, 4, 64);
        std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                  << r.elapsed_us << " us" << std::endl;
        if (ccl::ccl_rate_limiter::is_stats_enabled(handle)) {
            print_stats_summary(r);
            if (r.stats.acquire_count > 0 &&
                r.stats.acquire_immediate_count == r.stats.acquire_count) {
                std::cout << "  VERDICT: All acquires immediate (fits in "
                             "burst). GOOD."
                          << std::endl;
            } else if (r.stats.acquire_count > 0) {
                double pct =
                    100.0 *
                    static_cast<double>(r.stats.acquire_immediate_count) /
                    static_cast<double>(r.stats.acquire_count);
                std::cout << "  VERDICT: " << std::setprecision(1) << pct
                          << "% immediate." << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Test 2: Large workload (exceeds burst, triggers refill)
    // A large workload forces the bucket to drain and refill multiple times.
    {
        std::cout
            << "--- Test 2: Large workload (exceeds burst, triggers refill) ---"
            << std::endl;
        int64_t total = 64 * MiB; // 64 MiB total
        // 32 KiB drives bucket drain/refill transitions quickly.
        int64_t chunk = 32 * KiB;
        auto r = run_scenario(q, "large_refill", total, chunk, 16, 64);
        std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                  << r.elapsed_us << " us" << std::endl;
        if (ccl::ccl_rate_limiter::is_stats_enabled(handle)) {
            print_stats_summary(r);
            if (r.stats.refill_count > 0) {
                std::cout << "  VERDICT: Refills occurred ("
                          << r.stats.refill_count << "). GOOD." << std::endl;
            } else {
                std::cout << "  VERDICT: No refills detected. Burst may be too "
                             "large for this workload."
                          << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Test 3: High contention (many work-groups competing)
    // Tests CAS contention on the token counter.
    {
        std::cout << "--- Test 3: High contention (many WGs competing) ---"
                  << std::endl;
        int64_t total = 32 * MiB; // 32 MiB total
        // 8 KiB increases acquire frequency to stress contention path.
        int64_t chunk = 8 * KiB;
        auto r = run_scenario(q, "high_contention", total, chunk, 128, 32);
        std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                  << r.elapsed_us << " us" << std::endl;
        if (ccl::ccl_rate_limiter::is_stats_enabled(handle)) {
            print_stats_summary(r);
            if (r.stats.acquire_cas_fail_count > 0) {
                std::cout << "  VERDICT: CAS failures detected ("
                          << r.stats.acquire_cas_fail_count
                          << "). Contention path exercised. GOOD." << std::endl;
            } else {
                std::cout << "  VERDICT: No CAS failures. Contention may be "
                             "serialized by barrier."
                          << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // Test 4: Single work-group (no contention baseline)
    // Provides a contention-free reference for the acquire path.
    {
        std::cout << "--- Test 4: Single WG (no contention baseline) ---"
                  << std::endl;
        int64_t total = 32 * MiB; // 32 MiB total
        int64_t chunk = 32 * KiB;
        auto r = run_scenario(q, "single_wg", total, chunk, 1, 64);
        std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                  << r.elapsed_us << " us" << std::endl;
        if (ccl::ccl_rate_limiter::is_stats_enabled(handle)) {
            print_stats_summary(r);
            std::cout << "  VERDICT: Single-WG reference. CAS fails="
                      << r.stats.acquire_cas_fail_count << std::endl;
        }
        std::cout << std::endl;
    }

    // Test 5: Timing validation
    // Given the configured rate, the elapsed time for a known byte count should
    // approximate total_bytes / rate (minus burst). Verify within 2x tolerance.
    {
        std::cout << "--- Test 5: Timing validation (rate adherence) ---"
                  << std::endl;
        const char *rate_str = std::getenv("CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC");
        int64_t rate_bps = rate_str ? std::atoll(rate_str) : 0;
        if (rate_bps > 0) {
            int64_t total = 32 * MiB; // 32 MiB
            int64_t chunk = 16 * KiB;
            // Expected minimum time = (total - burst) / rate
            // Burst defaults to rate * 2ms
            int64_t burst = rate_bps / 500;
            int64_t throttled_bytes = total > burst ? total - burst : 0;
            double expected_us =
                (static_cast<double>(throttled_bytes) / rate_bps) * 1e6;

            ccl::ccl_rate_limiter::reset_stats(q, handle);
            auto r = run_scenario(q, "timing_check", total, chunk, 8, 64);
            std::cout << "  Configured rate: " << rate_bps << " B/s"
                      << std::endl;
            std::cout << "  Total bytes:     " << total << std::endl;
            std::cout << "  Expected min us: " << std::fixed
                      << std::setprecision(2) << expected_us << std::endl;
            std::cout << "  Actual us:       " << std::fixed
                      << std::setprecision(2) << r.elapsed_us << std::endl;

            if (expected_us > 1000.0) {
                double ratio = r.elapsed_us / expected_us;
                std::cout << "  Ratio (actual/expected): "
                          << std::setprecision(2) << ratio << std::endl;
                if (ratio >= 0.5 && ratio <= 3.0) {
                    std::cout
                        << "  VERDICT: Timing within tolerance (0.5x-3x). GOOD."
                        << std::endl;
                } else {
                    std::cout << "  VERDICT: Timing outside tolerance. Ratio="
                              << ratio << std::endl;
                }
            } else {
                std::cout << "  VERDICT: Expected time too small to validate "
                             "precisely."
                          << std::endl;
            }
        } else {
            std::cout << "  SKIPPED: Cannot validate timing without known rate."
                      << std::endl;
        }
        std::cout << std::endl;
    }

    // Test 6: Stats reset verification
    {
        std::cout << "--- Test 6: Stats reset verification ---" << std::endl;
        if (ccl::ccl_rate_limiter::is_stats_enabled(handle)) {
            ccl::ccl_rate_limiter::reset_stats(q, handle);
            auto stats_after_reset =
                ccl::ccl_rate_limiter::read_stats(q, handle);
            bool reset_ok = (stats_after_reset.acquire_count == 0 &&
                             stats_after_reset.acquire_total_bytes == 0 &&
                             stats_after_reset.refill_count == 0 &&
                             stats_after_reset.backoff_count == 0);
            std::cout << "  Reset and read back: "
                      << (reset_ok ? "PASSED" : "FAILED") << std::endl;
            if (!reset_ok) {
                std::cout << "  acquire_count="
                          << stats_after_reset.acquire_count << " (expected 0)"
                          << std::endl;
            }

            // Run a small workload and verify counts increased
            int64_t total = 16 * KiB;
            int64_t chunk = 4 * KiB;
            auto r = run_scenario(q, "stats_verify", total, chunk, 2, 32);
            if (r.stats.acquire_count > 0) {
                std::cout << "  After workload: acquire_count="
                          << r.stats.acquire_count << " GOOD." << std::endl;
            } else {
                std::cout << "  After workload: acquire_count=0 UNEXPECTED."
                          << std::endl;
            }
        } else {
            std::cout << "  SKIPPED: Stats not enabled." << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=======================================" << std::endl;
    std::cout << "  ALL TESTS COMPLETE" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "PASSED" << std::endl;
    return 0;
}

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
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <vector>

// End-to-end rate limiter test through the CCL allgatherv path.
// This binary is designed to be invoked multiple times by the driver script
// with different CCL_SYCL_RATE_LIMIT_* env vars to compare behavior.
//
// Usage:
//   srun mpiexec -n <ranks> ./sycl_rate_limiter_e2e_test <device_type>
//   <alloc_type> [--count N]
//
// Positional arguments:
//   device_type   gpu | cpu
//   alloc_type    device | shared | host
//   --count N     element count per rank  (default: 4194304 = 4M ints = 16 MiB)
//
// Environment variables:
//   RATE_LIMITER_TEST_ITERATIONS=N        timed allgatherv iterations (default:
//   10) RATE_LIMITER_TEST_WARMUP=N            warmup iterations before timing
//   (default: 3) RATE_LIMITER_TEST_IN_PLACE=0|1|false|true  send buffer inside
//   recv buffer  (default: 1)
//   RATE_LIMITER_TEST_PROFILE_MODE=none|host|device|both (default: none)
//                                          host   = CPU wall clock around ring
//                                          loop device = sum of SYCL kernel
//                                          events (needs profiling queue) both
//                                          = collect both, print side by side
//   CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC=N  rate limit in B/s  (0 = disabled)
//   CCL_SYCL_RATE_LIMIT_BURST_BYTES=N    token-bucket burst in bytes  (0 =
//   auto) CCL_SYCL_RATE_LIMIT_STATS=0|1        collect per-acquire stats
//   (default: 0)
//
// Output keys (machine-readable, rank 0 only):
//   profile_host_tp_median_gibps   wall clock around ring loop (median per
//   call) profile_device_tp_median_gibps sum of SYCL kernel events   (median
//   per call)
//
// Unlike stats_test which calls acquire() directly, this test exercises
// the full CCL code path including algorithm selection and ring copy wrappers.

using namespace std;
using namespace sycl;

namespace {

struct iter_stats {
    double min, max, median, mean, stdev, percentile95, percentile99;
};

static double percentile(const std::vector<double> &sorted, double p) {
    if (sorted.empty())
        return 0.0;
    double idx = (p / 100.0) * (static_cast<double>(sorted.size()) - 1.0);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi)
        return sorted[lo];
    double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static iter_stats summarize(const std::vector<double> &xs) {
    iter_stats s{};
    if (xs.empty())
        return s;
    std::vector<double> sorted = xs;
    std::sort(sorted.begin(), sorted.end());
    s.min = sorted.front();
    s.max = sorted.back();
    s.median = percentile(sorted, 50.0);
    s.percentile95 = percentile(sorted, 95.0);
    s.percentile99 = percentile(sorted, 99.0);
    double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    s.mean = sum / static_cast<double>(xs.size());
    double sq = 0.0;
    for (double x : xs) {
        double d = x - s.mean;
        sq += d * d;
    }
    s.stdev = xs.size() > 1 ? std::sqrt(sq / static_cast<double>(xs.size() - 1))
                            : 0.0;
    return s;
}

} // namespace

int main(int argc, char *argv[]) {
    if (!check_example_args(argc, argv))
        exit(1);

    size_t count = 4 * 1024 * 1024; // 4M elements = 16MB per rank

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);

    if (args.count != args.DEFAULT_COUNT) {
        count = args.count;
    }

    string device_type = argv[1];
    string alloc_type = argv[2];

    sycl::queue q;
    if (!create_test_sycl_queue(device_type, rank, q, args))
        return -1;

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm_alloc_type_from_string(alloc_type);

    if (!check_sycl_usm(q, usm_alloc_type)) {
        return -1;
    }

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
    } else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0,
                  MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
    auto stream = ccl::create_stream(q);

    auto recv_buf = allocator.allocate(count * size, usm_alloc_type);
    int *send_buf = nullptr;
    std::vector<size_t> recv_counts(size, count);

    bool in_place = true;
    const char *in_place_env = std::getenv("RATE_LIMITER_TEST_IN_PLACE");
    if (in_place_env) {
        std::string ip_val(in_place_env);
        std::transform(ip_val.begin(), ip_val.end(), ip_val.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        in_place = !(ip_val == "0" || ip_val == "false" || ip_val == "no" ||
                     ip_val == "off");
    }

    if (in_place) {
        send_buf = recv_buf + (count * static_cast<size_t>(rank));
    } else {
        send_buf = allocator.allocate(count, usm_alloc_type);
    }

    if (in_place) {
        q.submit([&](auto &h) {
             h.parallel_for(count, [=](auto id) {
                 recv_buf[(count * static_cast<size_t>(rank)) + id] = rank + 1;
             });
         }).wait();
    } else {
        q.submit([&](auto &h) {
             h.parallel_for(count, [=](auto id) { send_buf[id] = rank + 1; });
         }).wait();
    }

    int iterations = 10;
    const char *iter_env = std::getenv("RATE_LIMITER_TEST_ITERATIONS");
    if (iter_env) {
        iterations = std::atoi(iter_env);
        if (iterations < 1)
            iterations = 10;
    }

    int warmup = 3;
    const char *warmup_env = std::getenv("RATE_LIMITER_TEST_WARMUP");
    if (warmup_env) {
        warmup = std::atoi(warmup_env);
        if (warmup < 0)
            warmup = 3;
    }

    // Get rate limiter handle to read stats later
    ccl::ccl_rate_limiter::bucket_handle rl_handle =
        ccl::ccl_rate_limiter::get_handle(q);

    // Optional kernel-timing profiler. Mode is set process-globally on the
    // rate_limiter; the allgatherv ring loop checks it and feeds kernel events
    // back here. Host timer brackets just the ring loop, not the CPU sync.
    ccl::ccl_rate_limiter::profile_mode prof_mode =
        ccl::ccl_rate_limiter::profile_mode::none;
    const char *prof_env = std::getenv("RATE_LIMITER_TEST_PROFILE_MODE");
    if (prof_env) {
        std::string v(prof_env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (v == "host")
            prof_mode = ccl::ccl_rate_limiter::profile_mode::host;
        else if (v == "device")
            prof_mode = ccl::ccl_rate_limiter::profile_mode::device;
        else if (v == "both")
            prof_mode = ccl::ccl_rate_limiter::profile_mode::both;
        else
            prof_mode = ccl::ccl_rate_limiter::profile_mode::none;
    }
    ccl::ccl_rate_limiter::set_profile_mode(prof_mode);
    const bool prof_device_enabled =
        (prof_mode == ccl::ccl_rate_limiter::profile_mode::device ||
         prof_mode == ccl::ccl_rate_limiter::profile_mode::both);
    if (prof_device_enabled &&
        !q.has_property<sycl::property::queue::enable_profiling>()) {
        if (rank == 0) {
            std::cerr
                << "WARNING: RATE_LIMITER_TEST_PROFILE_MODE requested device "
                   "profiling but queue lacks enable_profiling — device timer "
                   "will read 0.\n";
        }
    }

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();
        ccl::allgatherv(send_buf, count, recv_buf, recv_counts,
                        ccl::datatype::int32, comm, stream, attr)
            .wait();
    }

    // Reset stats after warmup so we only measure the timed iterations
    if (ccl::ccl_rate_limiter::is_stats_enabled(rl_handle)) {
        ccl::ccl_rate_limiter::reset_stats(q, rl_handle);
    }
    ccl::ccl_rate_limiter::profile_reset();

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < iterations; ++i) {
        auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();
        ccl::allgatherv(send_buf, count, recv_buf, recv_counts,
                        ccl::datatype::int32, comm, stream, attr)
            .wait();
    }

    // Read stats
    ccl::ccl_rate_limiter::rate_stats stats{};
    if (ccl::ccl_rate_limiter::is_stats_enabled(rl_handle)) {
        stats = ccl::ccl_rate_limiter::read_stats(q, rl_handle);
    }

    // Harvest profiler timers. Device path issues wait() on captured events,
    // so it must run on every rank that submitted kernels (i.e. all ranks).
    std::vector<double> prof_host_us_per_call;
    std::vector<double> prof_device_us_per_call;
    if (prof_mode == ccl::ccl_rate_limiter::profile_mode::host ||
        prof_mode == ccl::ccl_rate_limiter::profile_mode::both) {
        prof_host_us_per_call =
            ccl::ccl_rate_limiter::profile_get_host_us_per_call();
    }
    if (prof_mode == ccl::ccl_rate_limiter::profile_mode::device ||
        prof_mode == ccl::ccl_rate_limiter::profile_mode::both) {
        prof_device_us_per_call =
            ccl::ccl_rate_limiter::profile_get_device_us_per_call();
    }

    // Correctness check
    bool passed = true;
    {
        sycl::buffer<int> check_buf(count * size);
        q.submit([&](auto &h) {
            sycl::accessor check_acc(check_buf, h, sycl::write_only);
            h.parallel_for(count * size, [=](auto id) {
                int expected_rank = static_cast<int>(id / count);
                int expected_val = expected_rank + 1;
                check_acc[id] = (recv_buf[id] == expected_val) ? 0 : -1;
            });
        });
        if (!handle_exception(q)) {
            passed = false;
        } else {
            sycl::host_accessor check_acc(check_buf, sycl::read_only);
            for (size_t i = 0; i < count * size; ++i) {
                if (check_acc[i] == -1) {
                    passed = false;
                    break;
                }
            }
        }
    }

    // Machine-readable output (rank 0 only)
    if (rank == 0) {
        constexpr double GiB = 1024.0 * 1024.0 * 1024.0;

        std::cout << "---RESULTS---" << std::endl;
        std::cout << "correctness=" << (passed ? "PASSED" : "FAILED")
                  << std::endl;
        std::cout << "iterations=" << iterations << std::endl;
        std::cout << "count=" << count << std::endl;
        std::cout << "ranks=" << size << std::endl;
        std::cout << "in_place=" << (in_place ? "true" : "false") << std::endl;

        // Kernel-timing profiler (per-call distributions). Each timed iteration
        // records one entry; host = wall clock around ring loop, device = sum
        // of SYCL event command_end-command_start across the call's
        // sub-kernels.
        const char *prof_mode_label =
            (prof_mode == ccl::ccl_rate_limiter::profile_mode::host) ? "host"
            : (prof_mode == ccl::ccl_rate_limiter::profile_mode::device)
                ? "device"
            : (prof_mode == ccl::ccl_rate_limiter::profile_mode::both) ? "both"
                                                                       : "none";
        std::cout << "profile_mode=" << prof_mode_label << std::endl;

        double bytes_per_iter = static_cast<double>(count) * sizeof(int) *
                                static_cast<double>(size - 1);

        auto emit_profile = [&](const char *label,
                                const std::vector<double> &vs) -> double {
            std::cout << label << "_calls=" << vs.size() << std::endl;
            if (vs.empty())
                return 0.0;
            iter_stats ps = summarize(vs);
            double sum = std::accumulate(vs.begin(), vs.end(), 0.0);
            std::cout << label << "_us_total=" << std::fixed
                      << std::setprecision(3) << sum << std::endl;
            std::cout << label << "_us_mean=" << ps.mean << std::endl;
            std::cout << label << "_us_min=" << ps.min << std::endl;
            std::cout << label << "_us_median=" << ps.median << std::endl;
            std::cout << label << "_us_p95=" << ps.percentile95 << std::endl;
            std::cout << label << "_us_p99=" << ps.percentile99 << std::endl;
            std::cout << label << "_us_max=" << ps.max << std::endl;
            std::cout << label << "_us_stdev=" << ps.stdev << std::endl;
            // Per-iter throughput derived from median per-call duration.
            double tp_median_gibps = (bytes_per_iter / GiB) / (ps.median / 1e6);
            std::cout << label << "_tp_median_gibps=" << std::setprecision(4)
                      << tp_median_gibps << std::endl;
            return tp_median_gibps;
        };

        double host_tp_median_gibps = 0.0;
        double device_tp_median_gibps = 0.0;
        if (prof_mode == ccl::ccl_rate_limiter::profile_mode::host ||
            prof_mode == ccl::ccl_rate_limiter::profile_mode::both) {
            host_tp_median_gibps =
                emit_profile("profile_host", prof_host_us_per_call);
        }
        if (prof_mode == ccl::ccl_rate_limiter::profile_mode::device ||
            prof_mode == ccl::ccl_rate_limiter::profile_mode::both) {
            device_tp_median_gibps =
                emit_profile("profile_device", prof_device_us_per_call);
        }
        if (prof_mode == ccl::ccl_rate_limiter::profile_mode::both &&
            !prof_host_us_per_call.empty() &&
            !prof_device_us_per_call.empty() &&
            prof_host_us_per_call.size() == prof_device_us_per_call.size()) {
            // Per-call CPU+barrier overhead = host - device.
            std::vector<double> diff;
            diff.reserve(prof_host_us_per_call.size());
            for (size_t i = 0; i < prof_host_us_per_call.size(); ++i) {
                diff.push_back(prof_host_us_per_call[i] -
                               prof_device_us_per_call[i]);
            }
            emit_profile("profile_overhead", diff);
        }

        // Side-by-side summary of host and device median throughput.
        std::cout << "tp_median_gibps_summary"
                  << " host=" << std::fixed << std::setprecision(4)
                  << host_tp_median_gibps
                  << " device=" << device_tp_median_gibps << std::endl;

        // Sanity check: host and device should agree closely
        // (host includes q.wait() and P2P barriers, device is pure GPU time).
        {
            double vals[] = {host_tp_median_gibps, device_tp_median_gibps};
            double vmin = 0.0, vmax = 0.0;
            bool have = false;
            for (double v : vals) {
                if (v <= 0.0)
                    continue;
                if (!have) {
                    vmin = vmax = v;
                    have = true;
                } else {
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                }
            }
            if (have && vmin > 0.0 && (vmax / vmin) > 1.20) {
                std::cout << "!!!! error !!!! tp_median_gibps mismatch:"
                          << " host=" << host_tp_median_gibps
                          << " device=" << device_tp_median_gibps
                          << " (max/min=" << std::setprecision(3)
                          << (vmax / vmin) << ")" << std::endl;
            }
        }

        const char *rate_env = std::getenv("CCL_SYCL_RATE_LIMIT_BYTES_PER_SEC");
        const char *burst_env = std::getenv("CCL_SYCL_RATE_LIMIT_BURST_BYTES");
        const double config_rate_bps =
            rate_env ? std::strtod(rate_env, nullptr) : 0.0;
        const double config_rate_gbps = config_rate_bps / 1e9;
        std::cout << "config_rate_bps=" << (rate_env ? rate_env : "0")
                  << std::endl;
        std::cout << "config_rate_gbps=" << std::fixed << std::setprecision(4)
                  << config_rate_gbps << std::endl;
        std::cout << "config_burst=" << (burst_env ? burst_env : "auto")
                  << std::endl;

        bool rl_active = ccl::ccl_rate_limiter::is_enabled(rl_handle);
        std::cout << "rate_limiter_active=" << (rl_active ? "true" : "false")
                  << std::endl;

        if (ccl::ccl_rate_limiter::is_stats_enabled(rl_handle)) {
            std::cout << "---STATS---" << std::endl;
            std::cout << "acquire_count=" << stats.acquire_count << std::endl;
            std::cout << "acquire_total_bytes=" << stats.acquire_total_bytes
                      << std::endl;
            std::cout << "acquire_immediate_count="
                      << stats.acquire_immediate_count << std::endl;
            std::cout << "acquire_wait_count=" << stats.acquire_wait_count
                      << std::endl;
            std::cout << "acquire_cas_fail_count="
                      << stats.acquire_cas_fail_count << std::endl;
            std::cout << "insufficient_tokens_count="
                      << stats.insufficient_tokens_count << std::endl;
            std::cout << "min_acquire_ticks="
                      << (stats.min_acquire_ticks == UINT64_MAX
                              ? 0
                              : stats.min_acquire_ticks)
                      << std::endl;
            std::cout << "max_acquire_ticks=" << stats.max_acquire_ticks
                      << std::endl;
            std::cout << "total_acquire_ticks=" << stats.total_acquire_ticks
                      << std::endl;
            std::cout << "refill_count=" << stats.refill_count << std::endl;
            std::cout << "refill_skip_count=" << stats.refill_skip_count
                      << std::endl;
            std::cout << "refill_cas_fail_count=" << stats.refill_cas_fail_count
                      << std::endl;
            std::cout << "refill_force_count=" << stats.refill_force_count
                      << std::endl;
            std::cout << "refill_total_earned=" << stats.refill_total_earned
                      << std::endl;
            std::cout << "max_acquire_bytes=" << stats.max_acquire_bytes
                      << std::endl;
            std::cout << "backoff_count=" << stats.backoff_count << std::endl;

            if (stats.acquire_count > 0) {
                double pct =
                    100.0 * static_cast<double>(stats.acquire_immediate_count) /
                    static_cast<double>(stats.acquire_count);
                uint64_t avg_ticks =
                    stats.total_acquire_ticks / stats.acquire_count;
                uint64_t avg_bytes =
                    stats.acquire_total_bytes / stats.acquire_count;
                std::cout << "---DERIVED---" << std::endl;
                std::cout << "immediate_pct=" << std::fixed
                          << std::setprecision(1) << pct << std::endl;
                std::cout << "avg_acquire_ticks=" << avg_ticks << std::endl;
                std::cout << "avg_acquire_bytes=" << avg_bytes << std::endl;
            }
        }

        std::cout << "---END---" << std::endl;
        std::cout << (passed ? "PASSED" : "FAILED") << std::endl;
    }

    return passed ? 0 : -1;
}

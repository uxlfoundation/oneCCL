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

/*
 * Device-side AllReduce (sum) from the oneCCL v2 C API — LSA path.
 *
 * Each rank fills a symmetric send window with its own rank value; three kernel
 * variants then run the whole AllReduce on device, so each element comes out as
 * sum(0 .. nranks-1) = nranks*(nranks-1)/2.
 *
 *   - one_shot: remote reads only — every rank reduces the full buffer locally
 *     and writes its own recv window. One entry barrier suffices.
 *   - one_shot_optimized: one_shot + VS=4 vectorized reduction + (with
 *     LSA_ADAPTIVE) size-adaptive launch geometry.
 *   - two_shots: reduce-scatter style — each rank reduces its 1/N share, then
 *     remote-writes the sum to every peer. Needs entry and exit barriers.
 *
 * All three use a two-level barrier: one sub-group does the expensive cross-GPU
 * rendezvous, then a cheap intra-GPU grid barrier fans the all-clear out, so
 * the full thread count stays available for the reduction at any message size.
 *
 * Run:  mpirun -n <nranks> ./sycl_lsa_allreduce [min_count] [max_count]
 *       (sweeps element counts min..max, x2). Pin ranks to idle GPUs with
 *       ZE_AFFINITY_MASK; on Arc B, export NEOReadDebugKeys=1 and
 *       RenderCompressedBuffersEnabled=0.
 *
 * Timing defaults to pipelined host time: enqueue the batch, drain once, host
 * wall / iters, which is amortized throughput. Knobs (env):
 *
 *   LSA_NONPIPELINED=1  drain + MPI_Barrier every iter (per-op latency).
 *   LSA_KERNEL_TIME=1   also report device kernel time from event profiling.
 *   LSA_ADAPTIVE=1      one_shot_optimized: scale threads to the work.
 *   LSA_WG, LSA_WG_SIZE launch geometry (default 16 x 512).
 */
#include "oneapi/ccl.h"                 // v2 C API (host)
#include "oneapi/ccl/v2/device_api.hpp" // v2 device API (in-kernel)

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <vector>

/* Launch config, swept via LSA_WG (work-groups) and LSA_WG_SIZE (threads per
 * work-group). */
constexpr int default_n_work_groups = 16;
constexpr int default_work_group_size = 512;

/* Max LSA peers; bounds the per-thread arrays in the kernels. */
constexpr int MAX_PEERS = 16;

static int env_int(const char *name, int fallback) {
    const char *v = getenv(name);
    if (!v)
        return fallback;
    int n = atoi(v);
    return n > 0 ? n : fallback;
}

static sycl::queue create_queue(int local_rank) {
    for (const auto &p : sycl::platform::get_platforms()) {
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            return sycl::queue(
                p.get_devices()[local_rank % p.get_devices().size()],
                {sycl::property::queue::in_order{},
                 sycl::property::queue::enable_profiling{}});
        }
    }
    throw std::runtime_error("Level-Zero platform not found.");
}

/*
 * Intra-GPU (cross-work-group) grid barrier
 *   cnt   - arrivals this round; last arrival (cnt==n_wg) resets it to 0.
 *   epoch - monotonic round counter, bumped by the last arrival; it is the
 *           release signal (waiters spin until epoch != their snapshot).
 *           Separate from cnt so the reset-to-0 can't race the release check.
 *           cnt is reset BEFORE epoch is bumped, so a woken leader re-enters
 *           onto a clean cnt==0.
 */
static inline void grid_barrier(const sycl::nd_item<1> &it,
                                uint32_t *grid_count, uint32_t *grid_epoch,
                                int n_wg) {
    sycl::group_barrier(it.get_group());

    if (it.get_local_linear_id() == 0) {
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            cnt(*grid_count);
        sycl::atomic_ref<uint32_t, sycl::memory_order::acq_rel,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            epoch(*grid_epoch);
        uint32_t cur_epoch = epoch.load();
        uint32_t arrived = cnt.fetch_add(1u) + 1u;
        if (arrived == static_cast<uint32_t>(n_wg)) {
            cnt.store(0u);       /* reset for the next grid barrier / launch */
            epoch.fetch_add(1u); /* release the spinners */
        } else {
            while (epoch.load() == cur_epoch) { /* spin on device memory */
            }
        }
    }

    sycl::group_barrier(it.get_group());
}

/* AllReduce kernel variants — see the file header. */
enum class kernel_mode { one_shot, one_shot_optimized, two_shots };

static const char *kernel_mode_name(kernel_mode mode) {
    switch (mode) {
    case kernel_mode::one_shot:
        return "one_shot";
    case kernel_mode::one_shot_optimized:
        return "one_shot_optimized";
    case kernel_mode::two_shots:
        return "two_shots";
    default:
        return "unknown";
    }
}

int main(int argc, char *argv[]) {
    /* Sweep a range of element counts: [min_count] [max_count], stepping x2.
     * Defaults span 4 B .. 16 MB (float elements). */
    size_t min_count = (argc > 1) ? static_cast<size_t>(atoll(argv[1])) : 1;
    size_t max_count =
        (argc > 2) ? static_cast<size_t>(atoll(argv[2])) : (4 * 1024 * 1024);
    if (min_count == 0)
        min_count = 1;
    if (max_count < min_count)
        max_count = min_count;

    int rank = 0, local_rank = 0, world_size = 0;
    MPI_Comm local_comm = 0;
    onecclComm_t comm = nullptr;
    onecclUniqueId uid{};

    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    if (rank == 0)
        onecclGetUniqueId(&uid);
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

    sycl::queue q = create_queue(local_rank);

    if (onecclSetDevice(local_rank) != onecclSuccess ||
        onecclCommInitRank(&comm, world_size, uid, rank) != onecclSuccess) {
        std::cerr << "init failed\n";
        MPI_Finalize();
        return 1;
    }

    /* Launch geometry (env-overridable). Total threads = n_work_groups * wg. */
    const int n_work_groups = env_int("LSA_WG", default_n_work_groups);
    const int work_group_size = env_int("LSA_WG_SIZE", default_work_group_size);
    const size_t global_size =
        static_cast<size_t>(n_work_groups) * work_group_size;

    const size_t max_bytes = max_count * sizeof(float);

    /* Allocate symmetric device memory at the largest size; sub-ranges of these
     * buffers are reused for every count in the sweep. */
    float *d_sendbuf = sycl::malloc_device<float>(max_count, q);
    float *d_recvbuf = sycl::malloc_device<float>(max_count, q);

    /* Intra-GPU grid-barrier state (device memory, initialized once — the
     * sense-reversing barrier never needs a host-side reset between launches).
     */
    uint32_t *d_grid_count = sycl::malloc_device<uint32_t>(1, q);
    uint32_t *d_grid_epoch = sycl::malloc_device<uint32_t>(1, q);
    q.memset(d_grid_count, 0, sizeof(uint32_t)).wait();
    q.memset(d_grid_epoch, 0, sizeof(uint32_t)).wait();

    /* One slot is enough for any work-group count: the cross-GPU rendezvous
     * happens on a single sub-group and fans out via the grid barrier. */
    onecclDevCommRequirements_t reqs{};
    reqs.lsa_barrier_count = 1;
    onecclDevComm_t dev_comm{};
    onecclWindow_t send_win = nullptr, recv_win = nullptr;

    onecclResult_t setup = onecclDevCommCreate(comm, &reqs, &dev_comm);
    if (setup == onecclSuccess) {
        setup = onecclCommWindowRegister(comm, d_sendbuf, max_bytes, &send_win,
                                         ONECCL_WINDOW_COLL_SYMMETRIC);
    }
    if (setup == onecclSuccess) {
        setup = onecclCommWindowRegister(comm, d_recvbuf, max_bytes, &recv_win,
                                         ONECCL_WINDOW_COLL_SYMMETRIC);
    }
    /* Only some devices can share the peer memory the device API hands out. */
    if (setup == onecclNotImplemented) {
        std::cout << "SKIPPED: device API is not supported on this device\n";
        sycl::free(d_sendbuf, q);
        sycl::free(d_recvbuf, q);
        sycl::free(d_grid_count, q);
        sycl::free(d_grid_epoch, q);
        onecclCommDestroy(comm);
        MPI_Finalize();
        return 0;
    }
    if (setup != onecclSuccess) {
        std::cerr << "device-api host setup failed\n";
        MPI_Finalize();
        return 1;
    }

    /* Co-residency check: the intra-GPU grid barrier deadlocks if the whole
     * grid is not simultaneously resident. Bound conservatively by CU x max_wg.
     */
    {
        size_t max_wg =
            q.get_device().get_info<sycl::info::device::max_work_group_size>();
        size_t max_cu =
            q.get_device().get_info<sycl::info::device::max_compute_units>();
        size_t max_resident = max_cu * max_wg;
        if (global_size > max_resident) {
            if (rank == 0)
                std::cout << "error: grid " << global_size << " ("
                          << n_work_groups << " x " << work_group_size
                          << ") > max resident " << max_resident << " ("
                          << max_cu << " CU x " << max_wg
                          << "); the intra-GPU grid barrier would deadlock. "
                          << "Reduce LSA_WG / LSA_WG_SIZE.\n";
            onecclDevCommDestroy(comm, &dev_comm);
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        std::cout << "AllReduce (device API, LSA, two-level barrier): ranks="
                  << world_size << ", sweep count=" << min_count << ".."
                  << max_count << " (x2)\n";
        std::cout << "Launch geometry: " << n_work_groups << " work-groups x "
                  << work_group_size << " = " << global_size << " threads\n";
        std::cout << "Expected per-element result: sum(0.." << world_size - 1
                  << ") = " << (world_size * (world_size - 1)) / 2 << "\n";
    }

    const int grid_n_wg = n_work_groups;

    /*
     * one_shot: baseline — scalar reduction, fixed geometry, remote reads and a
     * local write. Sub-group 0 does the one cross-GPU barrier (acquire), the
     * grid barrier fans the all-clear out, then every thread acquire-fences at
     * system scope — so all readers, not just sub-group 0, see the peers'
     * released send data — and grid-strides the reduction.
     */
    auto launch_one_shot = [&](size_t count) {
        return q.submit([&](sycl::handler &h) {
            const int n_wg = grid_n_wg;
            uint32_t *gcount = d_grid_count;
            uint32_t *gepoch = d_grid_epoch;
            h.parallel_for(
                sycl::nd_range<1>(global_size, work_group_size),
                [=](sycl::nd_item<1> it) {
                    size_t gtid = it.get_global_linear_id();
                    const int sg_size = static_cast<int>(
                        it.get_sub_group().get_local_range()[0]);

                    /* Step 1: sub-group 0 only — cross-GPU entry rendezvous. */
                    if (gtid < static_cast<size_t>(sg_size)) {
                        onecclCoopSubGroup coop(it);
                        onecclLsaBarrierSession<onecclCoopSubGroup> bar(
                            coop, dev_comm, onecclTeamTagLsa{}, 0);
                        bar.sync(coop, sycl::memory_order::acquire);
                    }

                    /* Step 2: fan the all-clear out to the whole grid. */
                    grid_barrier(it, gcount, gepoch, n_wg);

                    /* Step 3: each reader acquires vs peers' remote stores. */
                    sycl::atomic_fence(sycl::memory_order::acquire,
                                       sycl::memory_scope::system);

                    int lsa_size = onecclLsaSize(dev_comm);
                    float *my_recv = static_cast<float *>(onecclGetLsaPointer(
                        recv_win, 0, onecclLsaRank(dev_comm)));

                    for (size_t off = gtid; off < count; off += global_size) {
                        float v = 0.0f;
                        for (int peer = 0; peer < lsa_size; peer++) {
                            float *src = static_cast<float *>(
                                onecclGetLsaPointer(send_win, 0, peer));
                            v += src[off];
                        }
                        my_recv[off] = v;
                    }
                });
        });
    };

    /* Size-adaptive launch (LSA_ADAPTIVE=1): scale threads to the work so tiny
     * messages pay a tiny launch. wg fixed at 16 (one sub-group),
     * n_wg = ceil(count/VS / 16) capped at n_work_groups. Derived together
     * because grid_barrier's n_wg must equal the launched WG count. */
    const bool adaptive = getenv("LSA_ADAPTIVE") != nullptr;

    /*
     * one_shot_optimized: one_shot's barrier structure, plus VS=4 vectorized
     * read-all-then-reduce and, with LSA_ADAPTIVE, size-adaptive geometry.
     */
    auto launch_one_shot_optimized = [&](size_t count) {
        constexpr int VS = 4;
        size_t g_size = global_size;
        int g_wg_size = work_group_size;
        int g_n_wg = grid_n_wg;
        if (adaptive) {
            size_t slots = (count + VS - 1) / VS;
            if (slots < 1)
                slots = 1;
            g_wg_size = 16; /* one sub-group */
            g_n_wg = static_cast<int>((slots + g_wg_size - 1) / g_wg_size);
            if (g_n_wg < 1)
                g_n_wg = 1;
            if (g_n_wg > grid_n_wg)
                g_n_wg = grid_n_wg; /* cap at registered geometry */
            g_size = static_cast<size_t>(g_n_wg) * g_wg_size;
        }
        return q.submit([&](sycl::handler &h) {
            const int n_wg = g_n_wg;
            const size_t gsz = g_size;
            uint32_t *gcount = d_grid_count;
            uint32_t *gepoch = d_grid_epoch;
            h.parallel_for(
                sycl::nd_range<1>(g_size, g_wg_size), [=](sycl::nd_item<1> it) {
                    size_t gtid = it.get_global_linear_id();
                    const int sg_size = static_cast<int>(
                        it.get_sub_group().get_local_range()[0]);

                    /* Step 1: sub-group 0 only — cross-GPU entry rendezvous. */
                    if (gtid < static_cast<size_t>(sg_size)) {
                        onecclCoopSubGroup coop(it);
                        onecclLsaBarrierSession<onecclCoopSubGroup> bar(
                            coop, dev_comm, onecclTeamTagLsa{}, 0);
                        bar.sync(coop, sycl::memory_order::acquire);
                    }

                    /* Step 2: fan the all-clear out to the whole grid. */
                    grid_barrier(it, gcount, gepoch, n_wg);

                    /* Step 3: each reader acquires vs peers' remote stores. */
                    sycl::atomic_fence(sycl::memory_order::acquire,
                                       sycl::memory_scope::system);

                    int lsa_size = onecclLsaSize(dev_comm);
                    float *my_recv = static_cast<float *>(onecclGetLsaPointer(
                        recv_win, 0, onecclLsaRank(dev_comm)));

                    /* Hoist peer send pointers, then read-all-then-reduce
                     * (float4 per peer -> local array -> sum), scalar tail. */
                    using vec_t = sycl::vec<float, VS>;
                    float *speer[MAX_PEERS];
                    for (int p = 0; p < lsa_size; p++)
                        speer[p] = static_cast<float *>(
                            onecclGetLsaPointer(send_win, 0, p));

                    size_t nvec = count / VS;
                    for (size_t iv = gtid; iv < nvec; iv += gsz) {
                        vec_t acc(0.0f);
                        vec_t tmp[MAX_PEERS];
                        for (int p = 0; p < lsa_size; p++)
                            tmp[p] =
                                *reinterpret_cast<vec_t *>(&speer[p][iv * VS]);
                        for (int p = 0; p < lsa_size; p++)
                            acc += tmp[p];
                        *reinterpret_cast<vec_t *>(&my_recv[iv * VS]) = acc;
                    }
                    size_t rem_start = nvec * VS;
                    for (size_t off = rem_start + gtid; off < count;
                         off += gsz) {
                        float v = 0.0f;
                        for (int peer = 0; peer < lsa_size; peer++)
                            v += speer[peer][off];
                        my_recv[off] = v;
                    }
                });
        });
    };

    /*
     * two_shots: reduce-scatter style. Entry is one_shot's. Work is split
     * across ranks in a flattened (rank, tid) space; each thread reduces its
     * share and remote-writes the sum to every peer's recv. Exit chains a
     * per-thread release fence -> grid barrier -> sub-group-0 barrier(release),
     * which is what guarantees peers see every remote write before the
     * collective returns.
     */
    auto launch_two_shots = [&](size_t count) {
        return q.submit([&](sycl::handler &h) {
            const int n_wg = grid_n_wg;
            uint32_t *gcount = d_grid_count;
            uint32_t *gepoch = d_grid_epoch;
            h.parallel_for(
                sycl::nd_range<1>(global_size, work_group_size),
                [=](sycl::nd_item<1> it) {
                    size_t gtid = it.get_global_linear_id();
                    const int sg_size = static_cast<int>(
                        it.get_sub_group().get_local_range()[0]);

                    /* Entry: sub-group 0 cross-GPU rendezvous + grid fan-out.
                     */
                    if (gtid < static_cast<size_t>(sg_size)) {
                        onecclCoopSubGroup coop(it);
                        onecclLsaBarrierSession<onecclCoopSubGroup> bar(
                            coop, dev_comm, onecclTeamTagLsa{}, 0);
                        bar.sync(coop, sycl::memory_order::acquire);
                    }
                    grid_barrier(it, gcount, gepoch, n_wg);
                    sycl::atomic_fence(sycl::memory_order::acquire,
                                       sycl::memory_scope::system);

                    int lsa_size = onecclLsaSize(dev_comm);

                    /* Spread work across all ranks: flattened (rank, tid). */
                    size_t global_tid =
                        gtid + global_size *
                                   static_cast<size_t>(onecclLsaRank(dev_comm));
                    size_t global_nthreads =
                        global_size * static_cast<size_t>(lsa_size);

                    for (size_t off = global_tid; off < count;
                         off += global_nthreads) {
                        float v = 0.0f;
                        for (int peer = 0; peer < lsa_size; peer++) {
                            float *src = static_cast<float *>(
                                onecclGetLsaPointer(send_win, 0, peer));
                            v += src[off];
                        }
                        for (int peer = 0; peer < lsa_size; peer++) {
                            float *dst = static_cast<float *>(
                                onecclGetLsaPointer(recv_win, 0, peer));
                            dst[off] = v;
                        }
                    }

                    /* Exit: publish this thread's remote writes, wait for all
                     * work-groups, then one cross-GPU release rendezvous. */
                    sycl::atomic_fence(sycl::memory_order::release,
                                       sycl::memory_scope::system);
                    grid_barrier(it, gcount, gepoch, n_wg);
                    if (gtid < static_cast<size_t>(sg_size)) {
                        onecclCoopSubGroup coop(it);
                        onecclLsaBarrierSession<onecclCoopSubGroup> bar(
                            coop, dev_comm, onecclTeamTagLsa{}, 0);
                        bar.sync(coop, sycl::memory_order::release);
                    }
                });
        });
    };

    auto launch_allreduce = [&](size_t count, kernel_mode mode) -> sycl::event {
        switch (mode) {
        case kernel_mode::one_shot_optimized:
            return launch_one_shot_optimized(count);
        case kernel_mode::two_shots:
            return launch_two_shots(count);
        case kernel_mode::one_shot:
        default:
            return launch_one_shot(count);
        }
    };

    constexpr int warmup_iters = 20;
    constexpr int timed_iters = 100;

    /* Fill send with this rank's value, recv with 0. The collective leaves
     * sendbuf alone, so the timed loop re-runs without re-initializing. */
    auto fill_send = [&](size_t count) {
        q.fill(d_sendbuf, static_cast<float>(rank), count).wait();
        q.fill(d_recvbuf, 0.0f, count).wait();
        MPI_Barrier(MPI_COMM_WORLD);
    };

    float expected = static_cast<float>((world_size * (world_size - 1)) / 2);
    std::vector<float> host(max_count);
    bool all_pass = true;

    /* Timing knobs (see header). */
    const bool kernel_time = getenv("LSA_KERNEL_TIME") != nullptr;
    const bool nonpipelined = getenv("LSA_NONPIPELINED") != nullptr;

    std::vector<kernel_mode> modes = {kernel_mode::one_shot,
                                      kernel_mode::one_shot_optimized,
                                      kernel_mode::two_shots};

    for (kernel_mode mode : modes) {
        if (rank == 0)
            std::cout << "--- mode=" << kernel_mode_name(mode)
                      << (nonpipelined ? " (non-pipelined)" : " (pipelined)")
                      << " ---\n";

        /* Sweep element counts from min_count to max_count, stepping x2. */
        for (size_t count = min_count; count <= max_count; count *= 2) {
            const size_t size_bytes = count * sizeof(float);

            fill_send(count);

            /* Warmup (batched; drained once). */
            for (int i = 0; i < warmup_iters; i++)
                launch_allreduce(count, mode);
            q.wait();

            /* Timed region; kernel_time keeps events for profiling after drain.
             */
            std::vector<sycl::event> evs;
            if (kernel_time)
                evs.reserve(timed_iters);
            double per_op_us = 0.0;
            if (nonpipelined) {
                double sum = 0.0;
                for (int i = 0; i < timed_iters; i++) {
                    MPI_Barrier(MPI_COMM_WORLD);
                    auto s = std::chrono::steady_clock::now();
                    sycl::event e = launch_allreduce(count, mode);
                    e.wait();
                    auto f = std::chrono::steady_clock::now();
                    sum += std::chrono::duration<double, std::micro>(f - s)
                               .count();
                    if (kernel_time)
                        evs.push_back(e);
                }
                per_op_us = sum / timed_iters;
            } else {
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < timed_iters; i++) {
                    sycl::event e = launch_allreduce(count, mode);
                    if (kernel_time)
                        evs.push_back(e);
                }
                q.wait();
                auto t1 = std::chrono::steady_clock::now();
                per_op_us =
                    std::chrono::duration<double, std::micro>(t1 - t0).count() /
                    timed_iters;
            }

            /* Average per-op time across ranks. */
            double avg_us = per_op_us;
            MPI_Allreduce(MPI_IN_PLACE, &avg_us, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD);
            avg_us /= world_size;

            /* Device kernel time from SYCL event profiling (after the drain).
             */
            double kavg_us = 0.0;
            if (kernel_time) {
                double sum = 0.0;
                for (auto &e : evs) {
                    uint64_t ks = e.get_profiling_info<
                        sycl::info::event_profiling::command_start>();
                    uint64_t ke = e.get_profiling_info<
                        sycl::info::event_profiling::command_end>();
                    sum += (ke - ks) / 1000.0; /* ns -> us */
                }
                kavg_us = sum / timed_iters;
                MPI_Allreduce(MPI_IN_PLACE, &kavg_us, 1, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD);
                kavg_us /= world_size;
            }

            /* Verify the last iteration's result. */
            bool pass = true;
            q.memcpy(host.data(), d_recvbuf, size_bytes).wait();
            for (size_t i = 0; i < count; i++) {
                if (host[i] != expected) {
                    std::cout << "FAILED: rank " << rank
                              << " mode=" << kernel_mode_name(mode)
                              << " count=" << count << " recv[" << i
                              << "] = " << host[i] << ", expected " << expected
                              << "\n";
                    pass = false;
                    all_pass = false;
                    break;
                }
            }

            /* Report algorithmic bandwidth (buffer / avg). */
            double gb = size_bytes / (1024.0 * 1024.0 * 1024.0);
            double bw_gbps = gb / (avg_us / 1e6);

            if (rank == 0) {
                std::cout << "mode=" << kernel_mode_name(mode)
                          << " count=" << count << " (" << size_bytes
                          << " B) per-op avg=" << avg_us << " us ("
                          << timed_iters << " iters, "
                          << (nonpipelined ? "non-pipelined" : "pipelined")
                          << "), ~" << bw_gbps << " GB/s (buffer/avg)";
                if (kernel_time)
                    std::cout << ", kernel avg=" << kavg_us
                              << " us (device event)";
                std::cout << (pass ? "  [PASSED]" : "  [FAILED]") << "\n";
            }
        }
    }

    onecclCommWindowDeregister(comm, send_win);
    onecclCommWindowDeregister(comm, recv_win);
    onecclDevCommDestroy(comm, &dev_comm);
    sycl::free(d_sendbuf, q);
    sycl::free(d_recvbuf, q);
    sycl::free(d_grid_count, q);
    sycl::free(d_grid_epoch, q);
    onecclCommDestroy(comm);
    MPI_Finalize();
    return all_pass ? 0 : 1;
}

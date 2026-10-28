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
 * Device-side cooperative reduce-sum from the oneCCL v2 C API.
 *
 * Each rank fills its window buffer with (rank + 1); after a barrier,
 * onecclLsaReduceSum reads every LSA peer through the window and sums locally,
 * so each element comes out as sum(1..nranks) = nranks*(nranks+1)/2.
 *
 * Cases: an aligned count; an unaligned count, which forces the scalar prefix
 * and tail; a short count; and bf16, whose accumulation is promoted to float.
 */
#include "oneapi/ccl.h"                 // v2 C API (host)
#include "oneapi/ccl/v2/device_api.hpp" // v2 device API (in-kernel)

#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

static sycl::queue create_queue(int local_rank) {
    for (const auto &p : sycl::platform::get_platforms()) {
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            return sycl::queue(
                p.get_devices()[local_rank % p.get_devices().size()],
                {sycl::property::queue::in_order{}});
        }
    }
    throw std::runtime_error("Level-Zero platform not found.");
}

/* One case: register a window over a freshly filled buffer, reduce every peer's
 * copy into out_buf on device, and verify on the host. */
template <typename T>
static bool run_case(sycl::queue &q, onecclComm_t comm,
                     const onecclDevComm_t &dev_comm, int rank, int world_size,
                     int work_group_size, int count, const char *label) {
    T *win_buf = sycl::malloc_device<T>(count, q);
    T *out_buf = sycl::malloc_device<T>(count, q);

    q.fill(win_buf, static_cast<T>(static_cast<float>(rank + 1)), count).wait();
    q.fill(out_buf, static_cast<T>(0.0f), count).wait();

    onecclWindow_t win = nullptr;
    if (onecclCommWindowRegister(comm, win_buf, count * sizeof(T), &win,
                                 ONECCL_WINDOW_COLL_SYMMETRIC) !=
        onecclSuccess) {
        std::cerr << "rank " << rank << ": window registration failed for "
                  << label << "\n";
        sycl::free(win_buf, q);
        sycl::free(out_buf, q);
        return false;
    }

    q.submit([&](sycl::handler &h) {
         h.parallel_for(sycl::nd_range<1>(work_group_size, work_group_size),
                        [=](sycl::nd_item<1> it) {
                            onecclCoopWorkGroup coop(it);

                            // Make every rank's fill visible before reading it.
                            onecclLsaBarrierSession<onecclCoopWorkGroup> bar(
                                coop, dev_comm, onecclTeamTagLsa{}, 0);
                            bar.sync(coop, sycl::memory_order::acq_rel);

                            onecclLsaReduceSum<T>(coop, win, /*offset=*/0,
                                                  out_buf, count, dev_comm);
                        });
     }).wait();

    std::vector<T> host(count);
    q.memcpy(host.data(), out_buf, count * sizeof(T)).wait();

    const float expected =
        static_cast<float>(world_size * (world_size + 1) / 2);
    bool ok = true;
    for (int i = 0; i < count; i++) {
        if (static_cast<float>(host[i]) != expected) {
            std::cout << "rank " << rank << ": " << label << " [" << i
                      << "] = " << static_cast<float>(host[i]) << ", expected "
                      << expected << "\n";
            ok = false;
            break;
        }
    }

    onecclCommWindowDeregister(comm, win);
    sycl::free(win_buf, q);
    sycl::free(out_buf, q);
    return ok;
}

int main() {
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

    constexpr int work_group_size = 32;

    onecclDevCommRequirements_t reqs{};
    reqs.lsa_barrier_count = 1;
    onecclDevComm_t dev_comm{};

    onecclResult_t setup = onecclDevCommCreate(comm, &reqs, &dev_comm);
    // Only some devices can share the peer memory the device API hands out.
    if (setup == onecclNotImplemented) {
        std::cout << "SKIPPED: device API is not supported on this device\n";
        onecclCommDestroy(comm);
        MPI_Finalize();
        return 0;
    }
    if (setup != onecclSuccess) {
        std::cerr << "device communicator creation failed\n";
        MPI_Finalize();
        return 1;
    }

    int checks = 0, passed = 0;
    auto run = [&](auto tag, int count, const char *label) {
        using T = decltype(tag);
        checks++;
        bool ok = run_case<T>(q, comm, dev_comm, rank, world_size,
                              work_group_size, count, label);
        if (ok)
            passed++;
        std::cout << "rank " << rank << ": " << label << " "
                  << (ok ? "PASS" : "FAIL") << "\n";
    };

    run(float{}, 1024, "float, aligned");
    run(float{}, 1023, "float, unaligned");
    run(float{}, 256, "float, short");
    run(bf16{}, 1024, "bf16, accumulation promoted to float");

    onecclDevCommDestroy(comm, &dev_comm);

    std::cout << "rank " << rank << ": " << passed << "/" << checks
              << " cases passed\n";

    onecclCommDestroy(comm);
    MPI_Finalize();
    return passed == checks ? 0 : 1;
}

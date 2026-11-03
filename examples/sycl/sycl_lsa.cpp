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
 * End-to-end device-API usage from the oneCCL v2 C API: host setup with
 * onecclDevCommCreate + onecclCommWindowRegister, then onecclGetLsaPointer and
 * onecclLsaBarrierSession inside a SYCL kernel on the handles those returned.
 *
 * Allreduce by direct peer reads: each rank fills send with its rank, sums
 * every LSA peer's send into recv, and verifies sum(0..n-1).
 */
#include "oneapi/ccl.h"                 // v2 C API (host)
#include "oneapi/ccl/v2/device_api.hpp" // v2 device API (onecclGetLsaPointer)

#include <cstdint>
#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>

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

    constexpr size_t count = 1024;
    constexpr size_t bytes = count * sizeof(float);

    float *sendbuf = sycl::malloc_device<float>(count, q);
    float *recvbuf = sycl::malloc_device<float>(count, q);
    q.fill(sendbuf, static_cast<float>(rank), count).wait();
    q.fill(recvbuf, 0.0f, count).wait();

    // ---- HOST device API via v2 C API ----
    onecclDevCommRequirements_t reqs{};
    reqs.lsa_barrier_count = 1;
    onecclDevComm_t dev_comm{};
    onecclWindow_t send_win = nullptr, recv_win = nullptr;

    onecclResult_t setup = onecclDevCommCreate(comm, &reqs, &dev_comm);
    if (setup == onecclSuccess) {
        setup = onecclCommWindowRegister(comm, sendbuf, bytes, &send_win,
                                         ONECCL_WINDOW_COLL_SYMMETRIC);
    }
    if (setup == onecclSuccess) {
        setup = onecclCommWindowRegister(comm, recvbuf, bytes, &recv_win,
                                         ONECCL_WINDOW_COLL_SYMMETRIC);
    }
    // Only some devices can share the peer memory the device API hands out.
    if (setup == onecclNotImplemented) {
        std::cout << "SKIPPED: device API is not supported on this device\n";
        MPI_Finalize();
        return 0;
    }
    if (setup != onecclSuccess) {
        std::cerr << "device-api host setup failed\n";
        MPI_Finalize();
        return 1;
    }

    // ---- HOST-side pointer resolution: no kernel needed ----
    void *host_local = nullptr, *host_lsa0 = nullptr, *host_peer0 = nullptr;
    if (onecclGetLocalDevicePointer(send_win, 0, &host_local) !=
            onecclSuccess ||
        onecclGetLsaDevicePointer(send_win, 0, 0, &host_lsa0) !=
            onecclSuccess ||
        onecclGetPeerDevicePointer(send_win, 0, 0, &host_peer0) !=
            onecclSuccess) {
        std::cerr << "host device-pointer resolve failed\n";
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ---- DEVICE device API inside a kernel ----
    // The v2 handles carry the device pointers, so the kernel captures and uses
    // them as-is.
    q.submit([&](sycl::handler &h) {
         h.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
             size_t gid = it.get_global_linear_id();
             onecclCoopWorkGroup coop(it);
             onecclLsaBarrierSession<onecclCoopWorkGroup> bar(
                 coop, dev_comm, onecclTeamTagLsa{}, 0);
             // Acquire every peer's send-buffer fill before reading it.
             bar.sync(coop, sycl::memory_order::acquire);

             int lsa_size = onecclLsaSize(dev_comm);
             float *my_recv = static_cast<float *>(
                 onecclGetLsaPointer(recv_win, 0, onecclLsaRank(dev_comm)));
             for (size_t off = gid; off < count; off += 256) {
                 float v = 0.0f;
                 for (int p = 0; p < lsa_size; p++)
                     v += static_cast<float *>(
                         onecclGetLsaPointer(send_win, 0, p))[off];
                 my_recv[off] = v;
             }

             // Release, so no peer reuses its send buffer while others read.
             bar.sync(coop, sycl::memory_order::release);
         });
     }).wait();

    // Verify sum(0..world_size-1).
    std::vector<float> host(count);
    q.memcpy(host.data(), recvbuf, bytes).wait();
    float expected = static_cast<float>(world_size * (world_size - 1) / 2);
    bool pass = true;
    for (size_t i = 0; i < count; i++)
        if (host[i] != expected) {
            pass = false;
            break;
        }

    // All three host resolves must succeed; the local one must be our sendbuf.
    bool ptr_ok = host_local != nullptr && host_lsa0 != nullptr &&
                  host_peer0 != nullptr && host_local == sendbuf;
    if (!ptr_ok)
        pass = false;

    std::cout << "rank " << rank << ": " << (pass ? "PASSED" : "FAILED")
              << " (expected " << expected << ", host_ptrs "
              << (ptr_ok ? "ok" : "BAD") << ")\n";

    onecclCommWindowDeregister(comm, send_win);
    onecclCommWindowDeregister(comm, recv_win);
    onecclDevCommDestroy(comm, &dev_comm);
    sycl::free(sendbuf, q);
    sycl::free(recvbuf, q);
    onecclCommDestroy(comm);
    MPI_Finalize();
    return pass ? 0 : 1;
}

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
 * Device-side peer writes through a registered window, from the oneCCL v2 C
 * API.
 *
 * Work-group g writes my_rank+1 into buf[g * nranks + my_rank] on every peer,
 * barriers, then checks the peers' writes landed. Each work-group gets its own
 * barrier slot; onecclCoopWorkGroup spreads the writes across its work-items
 * and onecclLsaBarrierSession does the cross-rank sync.
 *
 * So every rank ends with buf[g * nranks + r] == r + 1.
 */
#include "oneapi/ccl.h"                 // v2 C API (host)
#include "oneapi/ccl/v2/device_api.hpp" // v2 device API (in-kernel)

#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <vector>

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

    // One barrier slot per work-group.
    constexpr int n_work_groups = 1;
    constexpr int work_group_size = 16;
    constexpr size_t count = 64;

    int *buf = sycl::malloc_device<int>(count, q);
    q.memset(buf, 0, count * sizeof(int)).wait();

    onecclDevCommRequirements_t reqs{};
    reqs.lsa_barrier_count = n_work_groups;
    onecclDevComm_t dev_comm{};
    onecclWindow_t win = nullptr;

    onecclResult_t setup = onecclDevCommCreate(comm, &reqs, &dev_comm);
    if (setup == onecclSuccess) {
        setup = onecclCommWindowRegister(comm, buf, count * sizeof(int), &win,
                                         ONECCL_WINDOW_COLL_SYMMETRIC);
    }
    // Only some devices can share the peer memory the device API hands out.
    if (setup == onecclNotImplemented) {
        std::cout << "SKIPPED: device API is not supported on this device\n";
        sycl::free(buf, q);
        onecclCommDestroy(comm);
        MPI_Finalize();
        return 0;
    }
    if (setup != onecclSuccess) {
        std::cerr << "device-api host setup failed\n";
        MPI_Finalize();
        return 1;
    }

    const int global_size = n_work_groups * work_group_size;
    const int my_rank = rank;
    const int nranks = world_size;

    q.submit([&](sycl::handler &h) {
         h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                        [=](sycl::nd_item<1> it) {
                            int wg_id =
                                static_cast<int>(it.get_group_linear_id());

                            // One barrier slot per work-group.
                            onecclCoopWorkGroup coop(it);
                            onecclLsaBarrierSession<onecclCoopWorkGroup> bar(
                                coop, dev_comm, onecclTeamTagLsa{}, wg_id);

                            // Entry: every peer's buffer is ready before we
                            // write it.
                            bar.sync(coop, sycl::memory_order::acquire);

                            // Each work-item in the WG writes to a different
                            // peer.
                            size_t buf_idx = wg_id * nranks + my_rank;
                            for (int peer = coop.thread_rank(); peer < nranks;
                                 peer += coop.size()) {
                                int *peer_ptr = static_cast<int *>(
                                    onecclGetPeerPointer(win, 0, peer));
                                if (peer_ptr && buf_idx < count)
                                    peer_ptr[buf_idx] = my_rank + 1;
                            }

                            // Exit: all remote writes visible before anyone
                            // verifies.
                            bar.sync(coop, sycl::memory_order::release);
                        });
     }).wait();

    // Verify buf[g * nranks + r] == r + 1 for each work-group g and rank r.
    std::vector<int> host(count);
    q.memcpy(host.data(), buf, count * sizeof(int)).wait();

    bool pass = true;
    for (int g = 0; g < n_work_groups; g++) {
        for (int r = 0; r < world_size; r++) {
            size_t idx = g * world_size + r;
            if (idx < count && host[idx] != r + 1) {
                std::cout << "rank " << rank << ": buf[" << idx
                          << "] = " << host[idx] << ", expected " << r + 1
                          << "\n";
                pass = false;
            }
        }
    }

    std::cout << "rank " << rank << ": " << (pass ? "PASSED" : "FAILED")
              << " (device-side peer writes)\n";

    onecclCommWindowDeregister(comm, win);
    onecclDevCommDestroy(comm, &dev_comm);
    sycl::free(buf, q);
    onecclCommDestroy(comm);
    MPI_Finalize();
    return pass ? 0 : 1;
}

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
 * Host-side window pointer resolution from the oneCCL v2 C API: the peer, LSA
 * and local variants of onecclGet*DevicePointer, their offset arithmetic, their
 * rejection of out-of-range peers, and cross-rank writes driven entirely from
 * the host through the pointers they hand back.
 *
 * Unlike the device-side resolves, none of this needs a kernel.
 */
#include "oneapi/ccl.h" // v2 C API (host)

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

namespace {

int g_rank = 0;
bool g_pass = true;

void check(bool ok, const char *what) {
    if (!ok) {
        std::cout << "rank " << g_rank << ": FAILED " << what << "\n";
        g_pass = false;
    }
}

} // namespace

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
    g_rank = rank;

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

    constexpr size_t count = 64;
    int *buf = sycl::malloc_device<int>(count, q);
    q.memset(buf, 0, count * sizeof(int)).wait();

    onecclWindow_t win = nullptr;
    onecclResult_t setup = onecclCommWindowRegister(
        comm, buf, count * sizeof(int), &win, ONECCL_WINDOW_COLL_SYMMETRIC);
    // Only some devices can share the peer memory the device API hands out.
    if (setup == onecclNotImplemented) {
        std::cout << "SKIPPED: device API is not supported on this device\n";
        sycl::free(buf, q);
        onecclCommDestroy(comm);
        MPI_Finalize();
        return 0;
    }
    if (setup != onecclSuccess) {
        std::cerr << "window registration failed\n";
        MPI_Finalize();
        return 1;
    }

    char *const base = static_cast<char *>(static_cast<void *>(buf));

    /* 1: our own peer pointer is our local buffer. */
    {
        void *self = nullptr;
        check(onecclGetPeerDevicePointer(win, 0, rank, &self) ==
                      onecclSuccess &&
                  self == buf,
              "peer pointer for self does not match the local buffer");
    }

    /* 2: peer-pointer offset arithmetic. */
    {
        void *at16 = nullptr;
        check(onecclGetPeerDevicePointer(win, 16, rank, &at16) ==
                      onecclSuccess &&
                  at16 == base + 16,
              "peer pointer offset arithmetic");
    }

    /* 3, 4: out-of-range and negative peers are rejected, and the out pointer
     * is cleared rather than left as the caller had it. */
    {
        void *bad = reinterpret_cast<void *>(0xDEAD);
        check(onecclGetPeerDevicePointer(win, 0, world_size + 1, &bad) ==
                      onecclInvalidArgument &&
                  bad == nullptr,
              "out-of-range peer was not rejected");

        bad = reinterpret_cast<void *>(0xDEAD);
        check(onecclGetPeerDevicePointer(win, 0, -1, &bad) ==
                      onecclInvalidArgument &&
                  bad == nullptr,
              "negative peer was not rejected");
    }

    /* 5: cross-rank writes through host-resolved peer pointers. Each rank
     * writes (rank+1)*10 + peer into buf[rank] on every peer, so afterwards
     * buf[r] == (r+1)*10 + rank for every sender r. */
    {
        for (int peer = 0; peer < world_size; peer++) {
            void *peer_ptr = nullptr;
            if (onecclGetPeerDevicePointer(win, rank * sizeof(int), peer,
                                           &peer_ptr) == onecclSuccess) {
                int val = (rank + 1) * 10 + peer;
                q.memcpy(peer_ptr, &val, sizeof(int)).wait();
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> host(count);
        q.memcpy(host.data(), buf, count * sizeof(int)).wait();
        for (int r = 0; r < world_size; r++)
            check(host[r] == (r + 1) * 10 + rank,
                  "cross-rank write via the peer pointer");
    }

    /* 6, 7: the local resolve and its offset arithmetic. */
    {
        void *local = nullptr;
        check(onecclGetLocalDevicePointer(win, 0, &local) == onecclSuccess &&
                  local == buf,
              "local pointer does not match the local buffer");

        void *at32 = nullptr;
        check(onecclGetLocalDevicePointer(win, 32, &at32) == onecclSuccess &&
                  at32 == base + 32,
              "local pointer offset arithmetic");
    }

    /* 8: on a single node lsa_rank == world rank, so both resolves must agree
     * for the same index. */
    {
        for (int lsa_peer = 0; lsa_peer < world_size; lsa_peer++) {
            void *lsa_ptr = nullptr;
            void *peer_ptr = nullptr;
            onecclGetLsaDevicePointer(win, 0, lsa_peer, &lsa_ptr);
            onecclGetPeerDevicePointer(win, 0, lsa_peer, &peer_ptr);
            check(lsa_ptr == peer_ptr,
                  "the LSA and peer resolves disagree for the same index");
        }
    }

    /* 9: out-of-range LSA peers are rejected too. */
    {
        void *bad = reinterpret_cast<void *>(0xDEAD);
        check(onecclGetLsaDevicePointer(win, 0, world_size + 1, &bad) ==
                      onecclInvalidArgument &&
                  bad == nullptr,
              "out-of-range LSA peer was not rejected");
    }

    /* 10: the same cross-rank write as 5, but through the LSA resolve and into
     * the upper half of the buffer. */
    {
        q.memset(base + world_size * sizeof(int), 0, world_size * sizeof(int))
            .wait();
        MPI_Barrier(MPI_COMM_WORLD);

        for (int lsa_peer = 0; lsa_peer < world_size; lsa_peer++) {
            void *ptr = nullptr;
            if (onecclGetLsaDevicePointer(win,
                                          (rank + world_size) * sizeof(int),
                                          lsa_peer, &ptr) == onecclSuccess) {
                int val = (rank + 1) * 100 + lsa_peer;
                q.memcpy(ptr, &val, sizeof(int)).wait();
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<int> host(count);
        q.memcpy(host.data(), buf, count * sizeof(int)).wait();
        for (int r = 0; r < world_size; r++)
            check(host[r + world_size] == (r + 1) * 100 + rank,
                  "cross-rank write via the LSA pointer");
    }

    std::cout << "rank " << rank << ": " << (g_pass ? "PASSED" : "FAILED")
              << " (host-side pointer resolution)\n";

    onecclCommWindowDeregister(comm, win);
    sycl::free(buf, q);
    onecclCommDestroy(comm);
    MPI_Finalize();
    return g_pass ? 0 : 1;
}

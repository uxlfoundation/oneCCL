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

#include "oneapi/ccl.h"

#include <cstdint>
#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>

static sycl::queue create_queue(int local_rank) {
    auto platforms = sycl::platform::get_platforms();
    sycl::platform l0_platform;
    bool l0_found = false;

    for (const auto &platform : platforms) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            l0_platform = platform;
            l0_found = true;
            break;
        }
    }

    if (!l0_found) {
        throw std::runtime_error("Level-Zero platform not found.");
    }

    return sycl::queue(
        l0_platform
            .get_devices()[local_rank % l0_platform.get_devices().size()],
        {sycl::property::queue::in_order{},
         sycl::property::queue::enable_profiling{}});
}

int main() {
    int rank = 0;
    int local_rank = 0;
    int world_size = 0;

    MPI_Comm local_comm = 0;
    onecclComm_t comm = nullptr;
    onecclUniqueId uid{};

    constexpr size_t kBytes = 1ULL << 22;
    constexpr size_t kOffset = 1ULL << 20;
    constexpr size_t kCount = 1024;

    void *buffer = nullptr;
    void *sendbuff = nullptr;
    void *recvbuff = nullptr;
    void *reg_handle = nullptr;
    onecclWindow_t send_window = nullptr;
    onecclWindow_t recv_window = nullptr;

    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    if (rank == 0) {
        onecclGetUniqueId(&uid);
    }
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

    sycl::queue queue = create_queue(local_rank);

    onecclResult_t result = onecclSetDevice(local_rank);
    if (result != onecclSuccess) {
        std::cerr << "onecclSetDevice failed\n";
        MPI_Finalize();
        return 1;
    }

    result = onecclCommInitRank(&comm, world_size, uid, rank);
    if (result != onecclSuccess) {
        std::cerr << "onecclCommInitRank failed\n";
        MPI_Finalize();
        return 1;
    }

    result = onecclMemAlloc(&buffer, kBytes);
    if (result != onecclSuccess) {
        std::cerr << "onecclMemAlloc failed: " << onecclGetErrorString(result)
                  << '\n';
        onecclCommDestroy(comm);
        MPI_Finalize();
        return 1;
    }

    result = onecclCommRegister(comm, buffer, kBytes, &reg_handle);
    if (result != onecclSuccess) {
        std::cerr << "onecclCommRegister failed: "
                  << onecclGetErrorString(result) << '\n';
        onecclMemFree(buffer);
        onecclCommDestroy(comm);
        MPI_Finalize();
        return 1;
    }

    sendbuff = buffer;
    recvbuff = static_cast<void *>(static_cast<uint8_t *>(buffer) + kOffset);

    result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclFloat, onecclSum,
                             comm, &queue);
    if (result != onecclSuccess) {
        std::cerr << "onecclAllReduce failed\n";
    }

    result =
        onecclAllGather(sendbuff, recvbuff, kCount, onecclInt8, comm, &queue);
    if (result != onecclSuccess) {
        std::cerr << "onecclAllGather failed\n";
    }

    queue.wait();

    result = onecclCommWindowRegister(comm, sendbuff, kBytes / 2, &send_window,
                                      ONECCL_WINDOW_COLL_SYMMETRIC);
    if (result != onecclSuccess) {
        std::cerr << "onecclCommWindowRegister(send) failed: "
                  << onecclGetErrorString(result) << '\n';
    }

    result = onecclCommWindowRegister(comm, recvbuff, kBytes / 2, &recv_window,
                                      ONECCL_WINDOW_COLL_SYMMETRIC);
    if (result != onecclSuccess) {
        std::cerr << "onecclCommWindowRegister(recv) failed: "
                  << onecclGetErrorString(result) << '\n';
    }

    if (send_window != nullptr && recv_window != nullptr) {
        result = onecclAllGather(static_cast<uint8_t *>(sendbuff) + 0x1000,
                                 static_cast<uint8_t *>(recvbuff) + 0x2000, 1,
                                 onecclInt8, comm, &queue);
        if (result != onecclSuccess) {
            std::cerr << "window-backed onecclAllGather failed\n";
        }
    }

    queue.wait();

    if (send_window != nullptr) {
        onecclCommWindowDeregister(comm, send_window);
    }
    if (recv_window != nullptr) {
        onecclCommWindowDeregister(comm, recv_window);
    }

    if (reg_handle != nullptr) {
        onecclCommDeregister(comm, reg_handle);
    }

    if (buffer != nullptr) {
        onecclMemFree(buffer);
    }

    onecclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}

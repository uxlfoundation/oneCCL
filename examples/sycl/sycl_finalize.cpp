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

// examples/sycl/sycl_finalize.cpp
// Example demonstrating the use of onecclCommFinalize()

#include "oneapi/ccl.h"

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>

// Create SYCL queues by distributing them based on local rank
static sycl::queue create_queue(int local_rank) {
    auto platforms = sycl::platform::get_platforms();
    sycl::platform l0_platform;
    bool l0_found = false;

    // Find platform with Level-Zero
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
    int version = 0;
    constexpr int kCount = 1024;
    constexpr int kIterations = 10;

    MPI_Comm local_comm = 0;
    onecclComm_t comm = nullptr;
    onecclResult_t result = onecclSuccess;
    onecclUniqueId uid;

    // Initialize
    onecclGetVersion(&version);
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    if (rank == 0) {
        onecclGetUniqueId(&uid);
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    auto queue = create_queue(rank);

    result = onecclSetDevice(local_rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to set device.\n";
        return 1;
    }

    result = onecclCommInitRank(&comm, world_size, uid, rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to initialize communicator.\n";
        return 1;
    }

    // Allocate buffers
    int *sendbuff =
        static_cast<int *>(sycl::malloc_device(kCount * sizeof(int), queue));
    int *recvbuff =
        static_cast<int *>(sycl::malloc_device(kCount * sizeof(int), queue));

    if (rank == 0) {
        std::cout << "Running " << kIterations << " AllReduce operations...\n";
    }

    // Submit multiple AllReduce operations
    auto my_rank = rank;
    for (int i = 0; i < kIterations; i++) {
        queue.submit([&](sycl::handler &h) {
            h.parallel_for<class init_kernel>(
                sycl::range<1>(kCount),
                [=](sycl::id<1> idx) { sendbuff[idx] = my_rank + i; });
        });

        result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt,
                                 onecclSum, comm, &queue);
        if (result != onecclSuccess) {
            std::cerr << "AllReduce operation failed.\n";
            return 1;
        }
    }

    if (rank == 0) {
        std::cout << "Calling onecclCommFinalize()...\n";
    }

    // Finalize: Wait for all outstanding operations to complete
    // This is important before destroying the communicator to ensure
    // all collective operations have finished
    result = onecclCommFinalize(comm);
    if (result != onecclSuccess) {
        std::cerr << "Finalize communicator failed.\n";
        return 1;
    }

    if (rank == 0) {
        std::cout << "All operations completed successfully.\n";
    }

    // Verify results
    std::vector<int> recvbuff_host(kCount);
    queue.memcpy(recvbuff_host.data(), recvbuff, kCount * sizeof(int)).wait();

    // Expected sum: sum of (rank + (kIterations-1)) across all ranks
    int expected = 0;
    for (int r = 0; r < world_size; r++) {
        expected += r + (kIterations - 1);
    }

    bool success = true;
    for (int i = 0; i < kCount; i++) {
        if (recvbuff_host[i] != expected) {
            std::cerr << "Rank " << rank << ": Mismatch at index " << i
                      << ": expected " << expected << ", got "
                      << recvbuff_host[i] << "\n";
            success = false;
            break;
        }
    }

    if (rank == 0 && success) {
        std::cout << "Result verification passed!\n";
    }

    // Cleanup
    sycl::free(sendbuff, queue);
    sycl::free(recvbuff, queue);

    result = onecclCommDestroy(comm);
    if (result != onecclSuccess) {
        std::cerr << "Destroy communicator failed.\n";
        return 1;
    }

    MPI_Finalize();

    return 0;
}

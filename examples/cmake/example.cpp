/*
 Copyright 2016-2025 Intel Corporation
 
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

// examples/sycl/sycl.cpp
#include "oneapi/ccl.h"

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>

// Create SYCL queues by distributing them
// based on local rank of each process
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

    // Create a queue from the device on the Level-Zero platform
    return sycl::queue(l0_platform.get_devices()[local_rank],
                       {sycl::property::queue::in_order{},
                        sycl::property::queue::enable_profiling{}});
}

int main() {
    int rank;
    int world_size;
    constexpr int kCount = 8 * 1024 * 1024;
    constexpr int kIterations = 16;

    onecclComm_t comm;
    onecclUniqueId uid;
    onecclResult_t result;

    onecclInit(ONECCL_LEGACY); // Responsible for loading primary plugin, could
                               // be called on first API call to oneCCL

    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        // Create unique structure that can be used to communicate between
        // ranks. For legacy ccl backend it contains address of kvs, for nccl we
        // can use ncclUniqueId. Any other backend could use that provided they
        // can send the handle to other ranks
        onecclGetUniqueId(&uid);
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    auto compute_queue = create_queue(rank);
    auto ccl_queue = create_queue(rank);

    result = onecclSetDevice(rank);
    if (result != ONECCL_SUCCESS) {
        std::cerr << "Failed to set device.\n";
        return 1;
    }

    // Initialize communicator based on user provided id. In the future I think
    // we should use id's to select different backends, we could have equivalent
    // of MPI_COMM_WORLD allowing the user to quickly create MPI comms with our
    // own API.
    result = onecclCommInitRank(&comm, world_size, uid, rank);
    if (result != ONECCL_SUCCESS) {
        std::cerr << "Failed to initialize communicator.\n";
        return 1;
    }

    onecclStream_t ccl_stream;

    result = onecclStreamCreateXPU(&ccl_stream, &ccl_queue);
    if (result != ONECCL_SUCCESS) {
        std::cerr << "Create SYLC stream failed.\n";
        return 1;
    }

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), compute_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), compute_queue));

    for (int i = 0; i < kIterations; i++) {

        auto first_compute = compute_queue.submit([&](sycl::handler &h) {
            h.parallel_for<class compute_kernel_1>(
                sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                    for (int i = 0; i < 256; i++) {
                        sendbuff[idx] -= rank * i + 1;
                    }
                });
        });

        compute_queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel_2>(
                sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                    sendbuff[idx] += 7 * 1024;
                    for (int i = 0; i < 1024; i++) {
                        sendbuff[idx] -= 7;
                    }
                });
        });

        ccl_queue.ext_oneapi_set_external_event(first_compute);
        onecclResult_t result =
            onecclAllReduce(sendbuff, recvbuff, kCount, ONECCL_INT, ONECCL_SUM,
                            comm, ccl_stream);
        if (result != ONECCL_SUCCESS) {
            std::cerr << "AllReduce operation failed.\n";
            return 1;
        }
        auto allreduce_event = compute_queue.ext_oneapi_get_last_event();

        compute_queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(allreduce_event);
            cgh.parallel_for<class compute_kernel_3>(
                sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                    for (int i = 0; i < 1024; i++) {
                        sendbuff[idx] -= 7;
                    }
                    sendbuff[idx] += 7 * 1024;
                });
        });

        compute_queue.wait();
    }

    std::vector<int> recvbuff_host(kCount);
    compute_queue.memcpy(recvbuff_host.data(), recvbuff, kCount * sizeof(int))
        .wait();

    // Output the result
    for (int i = 0;
         i < std::min(recvbuff_host.size(), static_cast<unsigned long>(16));
         i++) {
        std::cout << recvbuff_host[i] << " ";
    }
    std::cout << std::endl;

    result = onecclStreamDestroy(ccl_stream);
    if (result != ONECCL_SUCCESS) {
        std::cerr << "Destroy SYCL stream failed.\n";
        return 1;
    }

    result = onecclCommDestroy(comm);
    if (result != ONECCL_SUCCESS) {
        std::cerr << "Destroy communicator failed.\n";
        return 1;
    }

    return 0;
}

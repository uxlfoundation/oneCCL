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
    constexpr int kCount = 128 * 1024 * 1024;
    constexpr int kIterations = 1000;

    MPI_Comm local_comm = 0;
    onecclComm_t comm = nullptr;
    onecclResult_t result = onecclSuccess;
    onecclUniqueId uid;

    // First call to oneCCL API will initialize the plugin.
    // It is not required, but in this case it will intialize
    // MPI based on oneCCL's requirements.
    onecclGetVersion(&version);

    // Initialize MPI
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Fetch local rank
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

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

    // Currently we provide the API to select device used
    // by oneCCL communicators, but in the future SYCL will
    // provide an API for this.
    result = onecclSetDevice(local_rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to set device.\n";
        return 1;
    }

    // Initialize communicator based on user provided id. In the future I think
    // we should use id's to select different backends, we could have equivalent
    // of MPI_COMM_WORLD allowing the user to quickly create MPI comms with our
    // own API.
    result = onecclCommInitRank(&comm, world_size, uid, rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to initialize communicator.\n";
        return 1;
    }

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), compute_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), compute_queue));
    int *computebuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), compute_queue));

    for (int i = 0; i < kIterations; i++) {
        // Submit first compute kernel to the GPU
        auto first_compute = compute_queue.submit([&](sycl::handler &h) {
            h.parallel_for<class compute_kernel_1>(
                sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                    for (int i = 0; i < 16; i++) {
                        sendbuff[idx] += (rank * i + 1) * 100000;
                    }
                });
        });

        // Submit second kernel to the same queue
        compute_queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel_2>(
                sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                    computebuff[idx] = 0;
                    for (int i = 0; i < 16; i++) {
                        computebuff[idx] += 1;
                    }
                });
        });

        // Record dependency on `ccl_queue` to the first compute kernel,
        // so AllReduce is running after the first kernel and in parallel
        // to the second one
        ccl_queue.ext_oneapi_set_external_event(first_compute);
        onecclResult_t const result = onecclAllReduce(
            sendbuff, recvbuff, kCount, onecclInt, onecclSum, comm, &ccl_queue);
        if (result != onecclSuccess) {
            std::cerr << "AllReduce operation failed.\n";
            return 1;
        }
        auto event_optional = ccl_queue.ext_oneapi_get_last_event();

#if defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER < 20250200)
        // Older Intel compiler returns event directly
        auto allreduce_event = event_optional;
#else
        // Newer Intel or open-source compilers return optional<event>
        sycl::event allreduce_event =
            event_optional.has_value() ? event_optional.value() : sycl::event();
#endif

        // Third compute kernel will run after the second kernel and Allreduce
        // The queue is in-order, so the only dependency we have to express
        // here, is the AllReduce running on `ccl_queue`
        compute_queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(allreduce_event);
            cgh.parallel_for<class compute_kernel_3>(
                sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                    sendbuff[idx] = sendbuff[idx] + computebuff[idx];
                });
        });

        compute_queue.wait();
    }

    std::vector<int> recvbuff_host(kCount);
    compute_queue.memcpy(recvbuff_host.data(), recvbuff, kCount * sizeof(int))
        .wait();

    // Output the result
    for (int i = 0; i < std::min<unsigned long>(recvbuff_host.size(),
                                                static_cast<unsigned long>(16));
         i++) {
        std::cout << recvbuff_host[i] << " ";
    }
    std::cout << '\n';

    result = onecclCommDestroy(comm);
    if (result != onecclSuccess) {
        std::cerr << "Destroy communicator failed.\n";
        return 1;
    }

    MPI_Finalize();
    return 0;
}
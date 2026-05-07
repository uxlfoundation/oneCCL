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
#include "oneapi/ccl/v2/types.h"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
sycl::queue create_queue(int local_rank) {
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

int oneccl_app(int rank, int world_size, void *shared_mem,
               void *ready_flag_ptr) {
    constexpr int kCount = 32;
    constexpr int kIterations = 2;
    constexpr int kStartFlag = 0xABCD;

    int expected_result = 0;

    onecclComm_t comm = nullptr;
    onecclResult_t result = onecclSuccess;
    onecclUniqueId uid;

    // Only executed on rank 0
    auto *ready_flag = new (ready_flag_ptr) std::atomic<int>();
    if (rank == 0) {
        onecclGetUniqueId(static_cast<onecclUniqueId *>(shared_mem));
        ready_flag->store(kStartFlag, std::memory_order_release);
    } else {
        while (ready_flag->load(std::memory_order_acquire) != kStartFlag) {
        }
    }

    memcpy(&uid, shared_mem, sizeof(uid));

    auto compute_queue = create_queue(rank);

    result = onecclSetDevice(rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to set device.\n";
        return 1;
    }

    result = onecclCommInitRank(&comm, world_size, uid, rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to initialize communicator.\n";
        return 1;
    }

    onecclComm_t new_comm = nullptr;
    onecclConfig_t config;
    result = onecclCommSplit(comm, rank / 2, 0, &new_comm, &config);
    if (result != onecclSuccess) {
        std::cerr << "Failed to split communicator.\n";
        return 1;
    }

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), compute_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), compute_queue));

    for (int i = 0; i < kIterations; i++) {
        if (i == 0) {
            compute_queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class setup_kernel>(
                    sycl::range<1>(kCount),
                    [=](sycl::id<1> idx) { sendbuff[idx] = 0; });
            });
        }

        // Submit first compute kernel to the GPU
        compute_queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class compute_kernel>(
                sycl::range<1>(kCount),
                [=](sycl::id<1> idx) { sendbuff[idx] += (rank + 1); });
        });

        onecclResult_t const result =
            onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt, onecclSum,
                            new_comm, &compute_queue);
        if (result != onecclSuccess) {
            std::cerr << "AllReduce operation failed.\n";
            return 1;
        }

        compute_queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class average_kernel>(
                sycl::range<1>(kCount),
                [=](sycl::id<1> idx) { sendbuff[idx] = recvbuff[idx]; });
        });

        int base_rank =
            rank - (rank % 2); // We split the comm in pairs, base_rank is the
                               // one with even rank in the pair
        expected_result = 2 * expected_result + (2 * base_rank + 3);

        compute_queue.wait();
    }

    std::vector<int> recvbuff_host(kCount);
    compute_queue.memcpy(recvbuff_host.data(), recvbuff, kCount * sizeof(int))
        .wait();

    // Wait for my turn
    while (ready_flag->load(std::memory_order_acquire) != kStartFlag + rank) {
    }

    // Validate the result
    std::cout << "Rank " << rank << " results:\n";
    for (int i = 0; i < std::min<unsigned long>(recvbuff_host.size(),
                                                static_cast<unsigned long>(16));
         i++) {
        std::cout << recvbuff_host[i] << " ";
        if (std::any_of(recvbuff_host.begin(), recvbuff_host.end(),
                        [=](int value) { return value != expected_result; })) {
            std::cout << "Incorrect result!";
            return 1;
        }
    }
    std::cout << '\n' << std::flush;
    sleep(1);

    // Let other rank start printing:
    ready_flag->fetch_add(1);

    result = onecclCommDestroy(comm);
    if (result != onecclSuccess) {
        std::cerr << "Destroy communicator failed.\n";
        return 1;
    }

    return 0;
}
} // namespace

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_processes>" << '\n';
        return 1;
    }

    int world_size = std::atoi(argv[1]);
    if (world_size <= 0) {
        std::cerr
            << "Please enter a positive number for the number of processes."
            << '\n';
        return 1;
    }

    void *unique_id_ptr =
        mmap(nullptr, sizeof(onecclUniqueId), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANON, -1, 0);
    if (unique_id_ptr == MAP_FAILED) {
        std::cerr << "mmap for onecclUniqueId failed" << '\n';
        return 1;
    }

    void *flag_ptr = mmap(nullptr, sizeof(onecclUniqueId),
                          PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, -1, 0);
    if (flag_ptr == MAP_FAILED) {
        std::cerr << "mmap for semaphore failed" << '\n';
        return 1;
    }
    auto *ready_flag = new (flag_ptr) std::atomic<int>(-1);

    std::vector<pid_t> children_pids(world_size);
    for (int i = 0; i < world_size; ++i) {
        pid_t pid = fork();
        if (pid < 0) {
            std::cerr << "Fork failed for child " << i << '\n';
            return 1;
        }

        if (pid == 0) {
            int return_value =
                oneccl_app(i, world_size, unique_id_ptr, flag_ptr);
            _exit(return_value);
        } else {
            children_pids[i] = pid;
        }
    }

    // The parent process waits for all child processes to terminate
    int return_value = 0;
    for (int i = 0; i < world_size; ++i) {
        int status = 1;
        std::cerr << "Waiting for " << children_pids[i] << "...\n";
        waitpid(children_pids[i], &status, 0);
        std::cerr << "Return from " << children_pids[i] << " => " << status
                  << '\n';
        if (status != 0) {
            return_value = 1;
        }
    }

    return return_value;
}

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

#include <thread>
#include <memory>
#include <vector>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <cstring>
#include <getopt.h>
#include <algorithm>

#include "sycl_base.hpp"

std::vector<sycl::queue> queues;
std::mutex mu;

using CommunicatorType = ccl::communicator;
using namespace sycl;
using namespace sycl::access;

struct thread_params {
    int global_thread_id;
    int count;
    int debug;
    // that form one CCL group.
    std::vector<std::vector<int>>* thread_groups;
    // Each element of kvss corresponds 1-to-1 with thread_groups.
    std::vector<ccl::shared_ptr_class<ccl::kvs>>* kvss;
};

void thread_func_alltoall(thread_params params) {
    int global_thread_id = params.global_thread_id;
    int count = params.count;
    int debug = params.debug;

    int rank_in_group = -1;
    int group_size = -1;
    ccl::shared_ptr_class<ccl::kvs> kvs;
    size_t group_index = 0;

    // Find which group this global_thread_id belongs to
    for (size_t i = 0; i < params.thread_groups->size(); ++i) {
        auto& group = (*params.thread_groups)[i];
        auto it = std::find(group.begin(), group.end(), global_thread_id);
        if (it != group.end()) {
            rank_in_group = static_cast<int>(std::distance(group.begin(), it));
            group_size = static_cast<int>(group.size());
            kvs = (*params.kvss)[i];
            group_index = i;
            break;
        }
    }

    if (rank_in_group == -1 || group_size == -1) {
        std::cerr << "Error: Rank not found in the group" << std::endl;
        return;
    }

    // Create SYCL objects for this thread
    auto q = queues[global_thread_id];
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto stream = ccl::create_stream(q);

    // Create communicator using rank_in_group, group_size, and kvs
    auto comm = ccl::create_communicatorExt(group_size, rank_in_group, dev, ctx, kvs);

    if (debug) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "[DEBUG] Created comm for global_thread_id " << global_thread_id
                  << " in group_index " << group_index << " with rank_in_group " << rank_in_group
                  << "/" << group_size << std::endl;
    }

    buf_allocator<int> allocator(q);

    // Buffers for the alltoall operation
    //   - send_buf: size = (count * group_size)
    //   - recv_buf: size = (count * group_size)
    auto send_buf = allocator.allocate(count * group_size, usm::alloc::device);
    auto recv_buf = allocator.allocate(count * group_size, usm::alloc::device);

    // Initialize send_buf and recv_buf on the device
    //   send_buf[id] = (id / count) + 1
    //   recv_buf[id] = -1  (marker)
    // Then we will need to check results after alltoall
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count * group_size, [=](auto id) {
            send_buf[id] = (id / count) + 1; // each block of "count" elements is rank+1
            recv_buf[id] = -1; // marker for "unset"
        });
    });
    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));

    if (debug) {
        buffer<int> check_buf_send(count * group_size);
        // Copy send_buf into host-visible buffer
        e = q.submit([&](auto& h) {
            accessor check_buf_acc_send(check_buf_send, h, write_only);
            h.parallel_for(count * group_size, [=](auto id) {
                check_buf_acc_send[id] = send_buf[id];
            });
        });
        e.wait(); // Wait to ensure data is available

        // Print out contents of send_buf
        host_accessor check_buf_acc_send(check_buf_send, read_only);
        {
            std::lock_guard<std::mutex> lock(mu);
            std::cout << "[DEBUG] Thread " << global_thread_id << " send_buf = [";
            for (int i = 0; i < count * group_size; i++) {
                std::cout << check_buf_acc_send[i];
                if (i < count * group_size - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        }
    }

    auto attr = ccl::create_operation_attr<ccl::alltoall_attr>();

    ccl::alltoall(send_buf, recv_buf, count, comm, stream, attr, deps).wait();

    if (debug) {
        buffer<int> debug_recv(count * group_size);
        q.submit([&](auto& h) {
             accessor debug_recv_acc(debug_recv, h, write_only);
             h.parallel_for(count * group_size, [=](auto i) {
                 debug_recv_acc[i] = recv_buf[i];
             });
         }).wait();

        host_accessor dbg(debug_recv, read_only);
        {
            std::lock_guard<std::mutex> lock(mu);
            std::cout << "[DEBUG] Thread " << global_thread_id << " alltoall recv_buf = [";
            for (int i = 0; i < count * group_size; i++) {
                std::cout << dbg[i];
                if (i < (count * group_size - 1))
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    // Now verify correctness:
    // We expect that each entry in recv_buf is (rank_in_group + 1).
    // We'll do this check *on device* to match your reference snippet.
    buffer<int> check_buf(count * group_size);
    q.submit([&](auto& h) {
         accessor check_buf_acc(check_buf, h, write_only);
         h.parallel_for(count * group_size, [=](auto id) {
             // if recv_buf[id] != (rank + 1), mark as -1, else 0
             if (recv_buf[id] != (rank_in_group + 1)) {
                 check_buf_acc[id] = -1;
             }
             else {
                 check_buf_acc[id] = 0;
             }
         });
     }).wait();

    // Finally, gather check results on the host side
    {
        host_accessor check_buf_acc(check_buf, read_only);
        bool pass = true;
        for (int i = 0; i < count * group_size; i++) {
            if (check_buf_acc[i] == -1) {
                pass = false;
                break;
            }
        }
        std::lock_guard<std::mutex> lock(mu);
        if (pass) {
            std::cout << "Thread " << global_thread_id << " PASSED alltoall check." << std::endl;
        }
        else {
            std::cout << "Thread " << global_thread_id << " FAILED alltoall check." << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    int count = 0, threads_num = 0, debug = 0;
    std::string task_name;

    // Define long options
    struct option long_options[] = { { "buffer-count", required_argument, 0, 'b' },
                                     { "threads", required_argument, 0, 't' },
                                     { "debug", required_argument, 0, 'd' },
                                     { "task", required_argument, 0, 'k' },
                                     { 0, 0, 0, 0 } };

    // Parse options using getopt
    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "b:t:d:k:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'b': {
                try {
                    count = std::stoi(optarg);
                }
                catch (const std::invalid_argument&) {
                    std::cerr << "Error: Buffer count must be an integer." << std::endl;
                    return 1;
                }
                break;
            }
            case 't': {
                try {
                    threads_num = std::stoi(optarg);
                }
                catch (const std::invalid_argument&) {
                    std::cerr << "Error: Threads per process must be an integer." << std::endl;
                    return 1;
                }
                break;
            }
            case 'd': {
                try {
                    debug = std::stoi(optarg);
                }
                catch (const std::invalid_argument&) {
                    std::cerr << "Error: Debug flag must be an integer." << std::endl;
                    return 1;
                }
                break;
            }
            case 'k': {
                task_name = optarg;
                break;
            }
            default: {
                std::cerr << "Usage: " << argv[0] << " --buffer-count <BUFFER_COUNT>"
                          << " --threads <THREADS_PER_PROCESS>"
                          << " --debug <DEBUG>"
                          << " --task <TASK_NAME>" << std::endl;
                return 1;
            }
        }
    }

    // Validate required arguments
    if (count <= 0 || threads_num <= 0) {
        std::cerr << "Error: Buffer count and threads per process must be positive integers."
                  << std::endl;
        return 1;
    }

    // Validate the user-provided task name
    // (the example re-uses "single_group", "two_group", "multi_group")
    if (task_name.empty() ||
        (task_name != "single_group" && task_name != "two_group" && task_name != "multi_group")) {
        std::cerr << "Error: Invalid task name. Use single_group, two_group, or multi_group."
                  << std::endl;
        return 1;
    }

    // Initialize MPI and CCL
    MPI_Init(NULL, NULL);
    atexit(mpi_finalize);
    ccl::init();

    // Create SYCL queues (example for Level-Zero GPUs)
    std::vector<sycl::device> devices;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        bool is_level_zero = (platform_name.find("Level-Zero") != std::string::npos);
        if (is_level_zero) {
            std::cout << "Platform_name is: " << platform_name << std::endl;
            auto device_list = platform.get_devices();
            for (const auto& device : device_list) {
                if (device.is_gpu()) {
                    devices.push_back(device);
                }
            }
        }
    }

    sycl::context context(devices);
    for (auto& device : devices) {
        queues.push_back(sycl::queue(context, device, { sycl::property::queue::in_order() }));
    }

    // Helper lambda to run the same pattern for the given groups & KVS
    auto execute_task = [&](auto thread_groups, auto kvss) {
        std::vector<std::thread> threads;
        for (int i = 0; i < threads_num; ++i) {
            thread_params params{ i, count, debug, &thread_groups, &kvss };
            threads.emplace_back(thread_func_alltoall, params);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    };

    // Depending on task_name, define thread_groups and matching kvss
    if (task_name == "single_group") {
        std::vector<std::vector<int>> thread_groups;
        // Create a single group containing all thread indices
        std::vector<int> group;
        for (int i = 0; i < threads_num; ++i) {
            group.push_back(i);
        }
        thread_groups.push_back(group);

        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss = { ccl::create_main_kvs() };
        execute_task(thread_groups, kvss);
    }
    else if (task_name == "two_group") {
        // Example: group1 = [0,2], group2 = [1,3] if threads_num=4
        std::vector<std::vector<int>> thread_groups{ { 0, 2 }, { 1, 3 } };
        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss{ ccl::create_main_kvs(),
                                                           ccl::create_main_kvs() };
        execute_task(thread_groups, kvss);
    }
    else if (task_name == "multi_group") {
        // Example: multiple pairs: [0,1], [2,3], [4,5], ...
        std::vector<std::vector<int>> thread_groups{ { 0, 1 }, { 2, 3 }, { 4, 5 },
                                                     { 6, 7 }, { 8, 9 }, { 10, 11 } };
        // Each group needs a separate KVS
        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss;
        for (size_t i = 0; i < thread_groups.size(); i++) {
            kvss.push_back(ccl::create_main_kvs());
        }
        execute_task(thread_groups, kvss);
    }

    return 0;
}

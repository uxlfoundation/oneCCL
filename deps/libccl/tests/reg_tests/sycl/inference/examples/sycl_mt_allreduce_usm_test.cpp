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

#include "sycl_base.hpp"

std::vector<sycl::queue> queues;
std::mutex mu; // Mutex to lock iostream for thread-safe logging

using CommunicatorType = ccl::communicator;
using namespace sycl;
using namespace sycl::access;

// Struct to hold parameters for thread functions
struct thread_params {
    int global_thread_id;
    int count;
    int debug;
    std::vector<std::vector<int>>* thread_groups;
    std::vector<ccl::shared_ptr_class<ccl::kvs>>* kvss;
};

// Thread function to perform allreduce
void thread_func(thread_params params) {
    int global_thread_id = params.global_thread_id;
    int count = params.count;
    int debug = params.debug;

    int rank_in_group = -1;
    int group_size = -1;
    ccl::shared_ptr_class<ccl::kvs> kvs;

    int group_sum = 0;
    // Find rank in the group and group size
    for (size_t i = 0; i < params.thread_groups->size(); ++i) {
        auto& group = (*params.thread_groups)[i];
        auto it = std::find(group.begin(), group.end(), global_thread_id);
        if (it != group.end()) {
            rank_in_group = std::distance(group.begin(), it);
            group_size = group.size();
            kvs = (*params.kvss)[i];
            for (int g = 0; g < group_size; g++)
                group_sum += (group[g] + 1);
            break;
        }
    }

    if (rank_in_group == -1 || group_size == -1) {
        std::cerr << "Error: Rank not found in the group" << std::endl;
        return;
    }

    auto q = queues[global_thread_id];
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto stream = ccl::create_stream(q);

    // Create communicator
    auto comm = ccl::create_communicatorExt(group_size, rank_in_group, dev, ctx, kvs);

    if (debug) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "Created comm for global_thread_id " << global_thread_id
                  << ", rank_in_group=" << rank_in_group << ", group_size=" << group_size
                  << std::endl;
    }

    buf_allocator<int> allocator(q);
    std::vector<ccl::event> deps;

    // Create buffers
    auto send_buf = allocator.allocate(count, usm::alloc::device);
    auto recv_buf = allocator.allocate(count, usm::alloc::device);

    // Initialize send buffer on the device side
    auto e = q.submit([&](auto& h) {
        h.parallel_for(count, [=](auto id) {
            send_buf[id] = 24 * (global_thread_id + 1);
        });
    });
    deps.push_back(ccl::create_event(e));

    // Verify the send buffer on the host side
    buffer<int> check_buf(count);
    e = q.submit([&](auto& h) {
        accessor check_buf_acc(check_buf, h, write_only);
        h.parallel_for(count, [=](auto id) {
            check_buf_acc[id] = send_buf[id];
        });
    });

    {
        host_accessor check_buf_acc(check_buf, read_only);
        if (debug) {
            std::lock_guard<std::mutex> lock(mu);
            std::cout << "Send_buf of global_thread_id " << global_thread_id << " is "
                      << check_buf_acc[0] << std::endl;
        }
    }

    deps.push_back(ccl::create_event(e));

    // Perform allreduce
    if (debug) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "Thread " << global_thread_id << " starting allreduce with count=" << count
                  << " (size=" << count * sizeof(int32_t) << " bytes)..." << std::endl;
    }

    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
    ccl::allreduce(send_buf,
                   recv_buf,
                   count,
                   ccl::datatype::int32,
                   ccl::reduction::sum,
                   comm,
                   stream,
                   attr,
                   deps)
        .wait();

    if (debug) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "Thread " << global_thread_id << " completed allreduce!" << std::endl;
    }

    // Verify the receive buffer
    buffer<int> check_buf1(count);
    q.submit([&](auto& h) {
         accessor check_buf_acc1(check_buf1, h, write_only);
         h.parallel_for(count, [=](auto id) {
             check_buf_acc1[id] = recv_buf[id];
         });
     }).wait();

    // Check if the received buffer is correct
    bool is_correct = true;
    {
        host_accessor check_buf_acc1(check_buf1, read_only);
        int expected_value = 24 * group_sum;
        for (int i = 0; i < count; ++i) {
            if (check_buf_acc1[i] != expected_value) {
                std::cerr << "Error: Recv_buf of global_thread_id " << global_thread_id
                          << ", element " << i << " has value " << check_buf_acc1[i]
                          << ", but expected " << expected_value << std::endl;
                is_correct = false;
                break;
            }
        }
    }

    if (is_correct) {
        std::cout << "PASSED" << std::endl;
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
            case 'b': try { count = std::stoi(optarg);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Buffer count must be an integer." << std::endl;
                    return 1;
                }
                break;
            case 't': try { threads_num = std::stoi(optarg);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Threads per process must be an integer." << std::endl;
                    return 1;
                }
                break;
            case 'd': try { debug = std::stoi(optarg);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Error: Debug flag must be an integer." << std::endl;
                    return 1;
                }
                break;
            case 'k': task_name = optarg; break;
            default:
                std::cerr
                    << "Usage: " << argv[0]
                    << " --buffer-count <BUFFER_COUNT> --threads <THREADS_PER_PROCESS> --debug <DEBUG> --task <TASK_NAME>"
                    << std::endl;
                return 1;
        }
    }

    // Validate required arguments
    if (count <= 0 || threads_num <= 0) {
        std::cerr << "Error: Buffer count and threads per process must be positive integers."
                  << std::endl;
        return 1;
    }

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

    // Create SYCL queues
    std::vector<sycl::device> devices;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto& platform : platform_list) {
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
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

    std::cout << "Found " << devices.size() << " GPU devices, creating " << threads_num
              << " threads" << std::endl;

    // Ensure we have enough devices for the number of threads
    if (devices.size() < threads_num) {
        std::cerr << "Error: Not enough GPU devices (" << devices.size()
                  << ") for the number of threads (" << threads_num << ")" << std::endl;
        return 1;
    }

    // Execute task based on task_name
    auto execute_task = [&](auto thread_groups, auto kvss) {
        std::vector<std::thread> threads;
        for (int i = 0; i < threads_num; ++i) {
            thread_params params{ i, count, debug, &thread_groups, &kvss };
            threads.emplace_back(thread_func, params);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    };

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
        std::vector<std::vector<int>> thread_groups{ std::vector{ 1, 3 }, std::vector{ 0, 2 } };
        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss{ ccl::create_main_kvs(),
                                                           ccl::create_main_kvs() };
        execute_task(thread_groups, kvss);
    }
    else if (task_name == "multi_group") {
        std::vector<std::vector<int>> thread_groups{ { 0, 1 }, { 2, 3 }, { 4, 5 },
                                                     { 6, 7 }, { 8, 9 }, { 10, 11 } };
        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss;
        for (size_t i = 0; i < 6; i++) {
            kvss.push_back(ccl::create_main_kvs());
        }

        execute_task(thread_groups, kvss);
    }

    return 0;
}

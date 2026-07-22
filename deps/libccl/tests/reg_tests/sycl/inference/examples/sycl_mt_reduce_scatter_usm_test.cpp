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
    std::vector<std::vector<int>>* thread_groups;
    std::vector<ccl::shared_ptr_class<ccl::kvs>>* kvss;
};

void thread_func_reduce_scatter(thread_params params) {
    int global_thread_id = params.global_thread_id;
    int count = params.count;
    int debug = params.debug;

    int rank_in_group = -1;
    int group_size = -1;
    ccl::shared_ptr_class<ccl::kvs> kvs;
    size_t group_index = 0;

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

    auto q = queues[global_thread_id];
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto stream = ccl::create_stream(q);

    auto comm = ccl::create_communicatorExt(group_size, rank_in_group, dev, ctx, kvs);

    if (debug) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "[DEBUG] Created comm for global_thread_id " << global_thread_id
                  << " in group_index " << group_index << " with rank_in_group " << rank_in_group
                  << "/" << group_size << std::endl;
    }

    buf_allocator<int> allocator(q);

    //   - send_buf: size = count * group_size
    //   - recv_buf: size = count
    auto send_buf = allocator.allocate(count * group_size, usm::alloc::device);
    auto recv_buf = allocator.allocate(count, usm::alloc::device);
    buffer<int> expected_buf(count);
    buffer<int> check_buf(count);

    // Fill data on the device and set up the expected results
    // Explanation for the "expected" formula:
    //   For each rank R, we do sum over i in [0..(group_size-1)] of (R + i)
    //   => This is group_size * (2*R + (group_size - 1)) / 2
    // We repeat that same sum in each element of 'recv_buf'.
    auto e = q.submit([&](auto& h) {
        sycl::accessor expected_buf_acc(expected_buf, h, sycl::write_only);
        h.parallel_for(range<1>(count), [=](id<1> idx) {
            int i = idx[0];

            // For verification, mark recv_buf invalid initially
            recv_buf[i] = -1;

            // Compute the sum of (rank_in_group + j) for j=0..group_size-1
            int sum_val = group_size * ((rank_in_group) + (rank_in_group + group_size - 1)) / 2;
            expected_buf_acc[i] = sum_val;

            // Now fill the entire "send_buf" portion for this rank with (rank_in_group + j)
            for (int j = 0; j < group_size; j++) {
                // each rank's "block" is size 'count'
                // send_buf layout: [0..count-1], [count..2*count-1], ...
                send_buf[j * count + i] = (rank_in_group + j);
            }
        });
    });

    std::vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));

    auto attr = ccl::create_operation_attr<ccl::reduce_scatter_attr>();

    ccl::reduce_scatter(send_buf,
                        recv_buf,
                        count,
                        ccl::datatype::int32,
                        ccl::reduction::sum,
                        comm,
                        stream,
                        attr,
                        deps)
        .wait();

    auto e2 = q.submit([&](auto& h) {
        accessor check_buf_acc(check_buf, h, write_only);
        h.parallel_for(range<1>(count), [=](id<1> idx) {
            check_buf_acc[idx] = recv_buf[idx];
        });
    });
    e2.wait();

    {
        host_accessor check_buf_acc(check_buf, read_only);
        host_accessor expected_buf_acc(expected_buf, read_only);

        bool is_correct = true;
        for (int i = 0; i < count; i++) {
            if (check_buf_acc[i] != expected_buf_acc[i]) {
                std::lock_guard<std::mutex> lock(mu);
                std::cerr << "[ERROR] Mismatch for global_thread_id " << global_thread_id
                          << " (comm rank " << rank_in_group << ")"
                          << " at element i=" << i << ", got " << check_buf_acc[i] << ", expected "
                          << expected_buf_acc[i] << std::endl;
                is_correct = false;
                break;
            }
        }
        if (debug) {
            std::lock_guard<std::mutex> lock(mu);
            std::cout << "[DEBUG] Thread " << global_thread_id << " reduce_scatter recv_buf[0..("
                      << count - 1 << ")]: ";
            for (int i = 0; i < count; i++) {
                std::cout << check_buf_acc[i];
                if (i < count - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }
        if (is_correct) {
            std::lock_guard<std::mutex> lock(mu);
            std::cout << "Thread " << global_thread_id << " PASSED reduce_scatter check."
                      << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    int count = 0, threads_num = 0, debug = 0;
    std::string task_name;

    struct option long_options[] = { { "buffer-count", required_argument, 0, 'b' },
                                     { "threads", required_argument, 0, 't' },
                                     { "debug", required_argument, 0, 'd' },
                                     { "task", required_argument, 0, 'k' },
                                     { 0, 0, 0, 0 } };

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

    MPI_Init(NULL, NULL);
    atexit(mpi_finalize);
    ccl::init();

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

    auto execute_task = [&](auto thread_groups, auto kvss) {
        std::vector<std::thread> threads;
        for (int i = 0; i < threads_num; ++i) {
            thread_params params{ i, count, debug, &thread_groups, &kvss };
            threads.emplace_back(thread_func_reduce_scatter, params);
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
        // Example: group1 = [1,3], group2 = [0,2]
        std::vector<std::vector<int>> thread_groups{ { 1, 3 }, { 0, 2 } };
        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss{ ccl::create_main_kvs(),
                                                           ccl::create_main_kvs() };
        execute_task(thread_groups, kvss);
    }
    else if (task_name == "multi_group") {
        // Example: multiple pairs (0,1), (2,3), (4,5), ...
        std::vector<std::vector<int>> thread_groups{ { 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 } };
        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss{ ccl::create_main_kvs(),
                                                           ccl::create_main_kvs(),
                                                           ccl::create_main_kvs(),
                                                           ccl::create_main_kvs() };
        execute_task(thread_groups, kvss);
    }

    return 0;
}

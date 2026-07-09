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

using namespace sycl;
using namespace sycl::access;

struct thread_params {
    int global_thread_id;
    int count;
    int debug;
    std::vector<std::vector<int>>* thread_groups;
    std::vector<ccl::shared_ptr_class<ccl::kvs>>* kvss;
};

void thread_func_allgather(thread_params params) {
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

    buf_allocator<int> allocator(q);
    std::vector<ccl::event> deps;

    auto send_buf = allocator.allocate(count, usm::alloc::device);
    auto recv_buf = allocator.allocate(count * group_size, usm::alloc::device);

    auto e = q.submit([&](auto& h) {
        h.parallel_for(count, [=](auto id) {
            send_buf[id] = 24 * (global_thread_id + 1);
        });
    });
    deps.push_back(ccl::create_event(e));

    if (debug) {
        buffer<int> check_buf(count);
        e = q.submit([&](auto& h) {
            accessor check_buf_acc(check_buf, h, write_only);
            h.parallel_for(count, [=](auto id) {
                check_buf_acc[id] = send_buf[id];
            });
        });
        e.wait();

        host_accessor check_buf_acc(check_buf, read_only);
        {
            std::lock_guard<std::mutex> lock(mu);
            std::cout << "Thread " << global_thread_id
                      << " initial send_buf[0] = " << check_buf_acc[0] << std::endl;
        }
    }

    auto attr = ccl::create_operation_attr<ccl::allgather_attr>();
    ccl::allgather(send_buf, recv_buf, count, ccl::datatype::int32, comm, stream, attr, deps)
        .wait();

    {
        buffer<int> check_buf_all(count * group_size);
        q.submit([&](auto& h) {
             accessor check_buf_acc_all(check_buf_all, h, write_only);
             h.parallel_for(count * group_size, [=](auto idx) {
                 check_buf_acc_all[idx] = recv_buf[idx];
             });
         }).wait();

        host_accessor check_buf_acc_all(check_buf_all, read_only);
        auto& my_group = (*params.thread_groups)[group_index];
        bool is_correct = true;
        for (int i = 0; i < group_size; i++) {
            int global_id_for_rank_i = my_group[i];
            int expected_val = 24 * (global_id_for_rank_i + 1);
            int start_idx = i * count;
            int end_idx = (i + 1) * count;
            for (int j = start_idx; j < end_idx; j++) {
                if (check_buf_acc_all[j] != expected_val) {
                    std::lock_guard<std::mutex> lock(mu);
                    std::cerr << "Error: Mismatch at global_thread_id " << global_thread_id
                              << ", index " << j << ", got " << check_buf_acc_all[j]
                              << ", expected " << expected_val << std::endl;
                    is_correct = false;
                    break;
                }
            }
            if (!is_correct) {
                break;
            }
        }

        if (is_correct) {
            std::lock_guard<std::mutex> lock(mu);
            std::cout << "Thread " << global_thread_id << " PASSED allgather check." << std::endl;
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
            case 'b': count = std::stoi(optarg); break;
            case 't': threads_num = std::stoi(optarg); break;
            case 'd': debug = std::stoi(optarg); break;
            case 'k': task_name = optarg; break;
            default:
                std::cerr << "Usage: " << argv[0] << " --buffer-count <BUFFER_COUNT>"
                          << " --threads <THREADS_PER_PROCESS>"
                          << " --debug <DEBUG>"
                          << " --task <TASK_NAME>" << std::endl;
                return 1;
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
        if (platform_name.find("Level-Zero") != std::string::npos) {
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
            threads.emplace_back(thread_func_allgather, params);
        }
        for (auto& thread : threads) {
            thread.join();
        }
    };

    if (task_name == "single_group") {
        std::vector<std::vector<int>> thread_groups;
        std::vector<int> single_group;
        for (int i = 0; i < threads_num; i++) {
            single_group.push_back(i);
        }
        thread_groups.push_back(single_group);
        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss{ ccl::create_main_kvs() };

        execute_task(thread_groups, kvss);
    }
    else if (task_name == "two_group") {
        std::vector<std::vector<int>> thread_groups{ { 1, 3 }, { 0, 2 } };
        std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss;
        for (size_t i = 0; i < thread_groups.size(); i++) {
            kvss.push_back(ccl::create_main_kvs());
        }
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

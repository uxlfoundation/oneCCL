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

// The per-thread broadcast function
void thread_func_bcast(thread_params params) {
    int global_thread_id = params.global_thread_id;
    int count = params.count;
    int debug = params.debug;

    int rank_in_group = -1;
    int group_size = -1;
    ccl::shared_ptr_class<ccl::kvs> kvs;
    size_t group_index = 0;

    // Identify which group this global_thread_id belongs to
    for (size_t i = 0; i < params.thread_groups->size(); i++) {
        auto& grp = (*params.thread_groups)[i];
        auto it = std::find(grp.begin(), grp.end(), global_thread_id);
        if (it != grp.end()) {
            rank_in_group = static_cast<int>(std::distance(grp.begin(), it));
            group_size = static_cast<int>(grp.size());
            kvs = (*params.kvss)[i];
            group_index = i;
            break;
        }
    }

    if (rank_in_group == -1 || group_size == -1) {
        std::lock_guard<std::mutex> lock(mu);
        std::cerr << "[ERROR] Thread " << global_thread_id << " not found in any group.\n";
        return;
    }

    // Create SYCL/Ccl objects
    auto q = queues[global_thread_id];
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto stream = ccl::create_stream(q);

    // Create communicator
    auto comm = ccl::create_communicatorExt(group_size, rank_in_group, dev, ctx, kvs);

    if (debug) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "[DEBUG] Thread " << global_thread_id << " => group_index=" << group_index
                  << ", rank_in_group=" << rank_in_group << ", group_size=" << group_size
                  << std::endl;
    }

    // We choose rank_in_group == 0 as the root
    int root_rank = 0;

    buf_allocator<int> allocator(q);

    auto send_buf = allocator.allocate(count, usm::alloc::device);
    auto recv_buf = allocator.allocate(count, usm::alloc::device);

    std::vector<ccl::event> deps;

    // If I'm the root, fill send_buf with a known value (say 10), and recv_buf with -1
    if (rank_in_group == root_rank) {
        auto e = q.submit([&](auto& h) {
            h.parallel_for(range<1>(count), [=](sycl::id<1> idx) {
                send_buf[idx] = 10;
                recv_buf[idx] = -1;
            });
        });
        deps.push_back(ccl::create_event(e));
    }

    // Perform broadcast (each rank uses the same count=... to broadcast from root_rank)
    auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();
    ccl::broadcast(send_buf, recv_buf, count, root_rank, comm, stream, attr, deps).wait();

    // We'll do a device-side check if the values are all 10, store 0 or -1 in check_buf
    sycl::buffer<int> check_buf(count);
    q.submit([&](auto& h) {
         accessor check_buf_acc(check_buf, h, sycl::write_only);
         h.parallel_for(range<1>(count), [=](sycl::id<1> idx) {
             check_buf_acc[idx] = (recv_buf[idx] == 10) ? 0 : -1;
         });
     }).wait();

    // Host read check_buf to see if any rank got a mismatch
    bool is_correct = true;
    {
        host_accessor host_acc(check_buf, sycl::read_only);
        for (int i = 0; i < count; i++) {
            if (host_acc[i] == -1) {
                is_correct = false;
                std::lock_guard<std::mutex> lock(mu);
                std::cerr << "[ERROR] Thread " << global_thread_id << " => mismatch at idx=" << i
                          << ", expected=10" << host_acc[i] << "\n";
                break;
            }
        }
    }
    if (is_correct) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "Thread " << global_thread_id << " => PASSED broadcast check.\n";
    }
}

int main(int argc, char* argv[]) {
    int count = 0, threads_num = 0, debug = 0;
    std::string task_name;

    // parse command line
    struct option long_opts[] = { { "buffer-count", required_argument, 0, 'b' },
                                  { "threads", required_argument, 0, 't' },
                                  { "debug", required_argument, 0, 'd' },
                                  { "task", required_argument, 0, 'k' },
                                  { 0, 0, 0, 0 } };

    int opt_idx = 0;
    int c;
    while ((c = getopt_long(argc, argv, "b:t:d:k:", long_opts, &opt_idx)) != -1) {
        switch (c) {
            case 'b': count = std::stoi(optarg); break;
            case 't': threads_num = std::stoi(optarg); break;
            case 'd': debug = std::stoi(optarg); break;
            case 'k': task_name = optarg; break;
            default: {
                std::cerr << "Usage: " << argv[0] << " --buffer-count <N>"
                          << " --threads <N>"
                          << " --debug <0|1>"
                          << " --task <single_group|two_group|multi_group>\n";
                return 1;
            }
        }
    }

    if (count <= 0 || threads_num <= 0) {
        std::cerr << "[ERROR] Buffer count and threads must be positive\n";
        return 1;
    }
    if (task_name.empty() ||
        (task_name != "single_group" && task_name != "two_group" && task_name != "multi_group")) {
        std::cerr << "[ERROR] Invalid task name\n";
        return 1;
    }

    // init
    MPI_Init(NULL, NULL);
    atexit(mpi_finalize);
    ccl::init();

    // create SYCL queues
    // (like your other tests)
    std::vector<sycl::device> devices;
    auto platform_list = sycl::platform::get_platforms();
    for (auto& p : platform_list) {
        auto pname = p.get_info<sycl::info::platform::name>();
        if (pname.find("Level-Zero") != std::string::npos) {
            auto devs = p.get_devices();
            for (auto& d : devs) {
                if (d.is_gpu()) {
                    devices.push_back(d);
                }
            }
        }
    }

    sycl::context ctx(devices);
    for (auto& d : devices) {
        queues.push_back(sycl::queue(ctx, d, sycl::property::queue::in_order()));
    }

    // define how we group threads
    auto execute_task = [&](auto thread_groups, auto kvss) {
        std::vector<std::thread> threads;
        threads.reserve(threads_num);
        for (int i = 0; i < threads_num; i++) {
            thread_params p{ i, count, debug, &thread_groups, &kvss };
            threads.emplace_back(thread_func_bcast, p);
        }
        for (auto& th : threads) {
            th.join();
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

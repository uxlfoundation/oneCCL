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
    int iters;
    std::vector<std::vector<int>>* thread_groups;
    std::vector<ccl::shared_ptr_class<ccl::kvs>>* kvss;
};

void thread_func_pt2pt(thread_params params) {
    int global_thread_id = params.global_thread_id;
    int count = params.count;
    int debug = params.debug;
    int iterations = params.iters;

    // Identify which group and rank we belong to
    int rank_in_group = -1;
    int group_size = -1;
    ccl::shared_ptr_class<ccl::kvs> kvs;
    size_t group_index = 0;

    for (size_t i = 0; i < params.thread_groups->size(); i++) {
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
        std::lock_guard<std::mutex> lock(mu);
        std::cerr << "[ERROR] Thread " << global_thread_id << " not found in any group"
                  << std::endl;
        return;
    }

    // Create SYCL objects for this thread
    auto q = queues[global_thread_id];
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto stream = ccl::create_stream(q);

    // Create communicator for the group
    auto comm = ccl::create_communicatorExt(group_size, rank_in_group, dev, ctx, kvs);

    if (debug) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "[DEBUG] Thread " << global_thread_id << " => group_index=" << group_index
                  << ", rank_in_group=" << rank_in_group << ", group_size=" << group_size
                  << std::endl;
    }

    // rank 0 => sends [0..count-1]
    // rank 1 => receives [0..count-1] and checks correctness.

    buf_allocator<int> allocator(q);
    auto buf_send = allocator.allocate(count, usm::alloc::device);
    auto buf_recv = allocator.allocate(count, usm::alloc::device);

    // We'll store the host copy for debug printing
    buffer<int> tmp_send(count);
    buffer<int> tmp_recv(count);

    bool is_sender = (rank_in_group == 0);
    bool is_receiver = (rank_in_group == 1);

    for (int iter_idx = 0; iter_idx < iterations; iter_idx++) {
        // Initialize
        auto e = q.submit([&](auto& h) {
            h.parallel_for(range<1>(count), [=](id<1> idx) {
                int i = idx[0];
                if (is_sender) {
                    // Fill send_buf with e.g. i + 100*(global_thread_id+1), so we can see who is the sender
                    buf_send[i] = i + 100 * (global_thread_id + 1);
                }
                else {
                    buf_send[i] = -1; // not used
                }
                buf_recv[i] = -1; // marker for "unset"
            });
        });
        e.wait();

        if (debug) {
            if (is_sender) {
                // Copy send_buf to host for debug printing
                e = q.submit([&](auto& h) {
                    accessor acc(tmp_send, h, write_only);
                    h.parallel_for(range<1>(count), [=](id<1> idx) {
                        acc[idx] = buf_send[idx];
                    });
                });
                e.wait();
                host_accessor host_acc(tmp_send, read_only);
                {
                    std::lock_guard<std::mutex> lock(mu);
                    std::cout << "[DEBUG] Thread " << global_thread_id << " => final send_buf = [";
                    for (int i = 0; i < count; i++) {
                        std::cout << host_acc[i];
                        if (i < count - 1)
                            std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
            }
        }

        // rank 0 => send
        // rank 1 => recv
        if (is_sender) {
            // We specify "dest=1" because in a 2-rank communicator, rank 1 is the other side
            auto send_event =
                ccl::send(buf_send, count, ccl::datatype::int32, /*dest*/ 1, comm, stream);
            send_event.wait();
        }
        else if (is_receiver) {
            // rank_in_group=1 => receive from rank 0
            auto recv_event =
                ccl::recv(buf_recv, count, ccl::datatype::int32, /*src*/ 0, comm, stream);
            recv_event.wait();
        }

        if (is_receiver) {
            // Now copy recv_buf to host and check correctness
            e = q.submit([&](auto& h) {
                accessor acc(tmp_recv, h, write_only);
                h.parallel_for(range<1>(count), [=](id<1> idx) {
                    acc[idx] = buf_recv[idx];
                });
            });
            e.wait();

            host_accessor host_acc(tmp_recv, read_only);

            // Debug printing
            if (debug) {
                std::lock_guard<std::mutex> lock(mu);
                std::cout << "[DEBUG] Thread " << global_thread_id << " => final recv_buf = [";
                for (int i = 0; i < count; i++) {
                    std::cout << host_acc[i];
                    if (i < count - 1)
                        std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }

            // Check correctness
            // We expect: i + 100*(sender_global_id+1)
            // The sender was the other thread in this group (with rank_in_group=0).
            // Let's find that "global_thread_id".
            int sender_global_id = -1;
            auto& group = (*params.thread_groups)[group_index];
            for (auto gtid : group) {
                // If gtid is rank_in_group=0 => that's the sender
                // But we only have 2 threads in the group: so if gtid != global_thread_id => that's the sender
                if (gtid != global_thread_id) {
                    sender_global_id = gtid;
                    break;
                }
            }

            if (sender_global_id < 0) {
                std::lock_guard<std::mutex> lock(mu);
                std::cerr << "[ERROR] Could not find sender thread in group" << std::endl;
                return;
            }

            bool pass = true;
            int expected_offset = 100 * (sender_global_id + 1);

            for (int i = 0; i < count; i++) {
                int expected_val = i + expected_offset;
                if (host_acc[i] != expected_val) {
                    pass = false;
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        std::cerr << "[ERROR] Thread " << global_thread_id << " => iter_idx "
                                  << iter_idx << " => mismatch at index=" << i
                                  << ", got=" << host_acc[i] << ", expected=" << expected_val
                                  << std::endl;
                    }
                    break;
                }
            }
            if (pass) {
                std::lock_guard<std::mutex> lock(mu);
                std::cout << "Thread " << global_thread_id << " => iter_idx " << iter_idx
                          << " => PASSED pt2pt check." << std::endl;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int count = 1024, threads_num = 2, debug = 0, iters = 10;

    // parse command line
    struct option long_options[] = { { "buffer-count", required_argument, 0, 'b' },
                                     { "threads", required_argument, 0, 't' },
                                     { "debug", required_argument, 0, 'd' },
                                     { "iters", required_argument, 0, 'i' },
                                     // { "task",         required_argument, 0, 'k' },
                                     { 0, 0, 0, 0 } };

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "b:t:d:i:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'b': {
                try {
                    count = std::stoi(optarg);
                }
                catch (...) {
                    std::cerr << "Error: --buffer-count must be an integer\n";
                    return 1;
                }
                break;
            }
            case 't': {
                try {
                    threads_num = std::stoi(optarg);
                }
                catch (...) {
                    std::cerr << "Error: --threads must be an integer\n";
                    return 1;
                }
                break;
            }
            case 'd': {
                try {
                    debug = std::stoi(optarg);
                }
                catch (...) {
                    std::cerr << "Error: --debug must be an integer\n";
                    return 1;
                }
                break;
            }
            case 'i': {
                try {
                    iters = std::stoi(optarg);
                }
                catch (...) {
                    std::cerr << "Error: --iters must be an integer\n";
                    return 1;
                }
                break;
            }
            default: {
                std::cerr << "Usage: " << argv[0]
                          << " --buffer-count=<N> --threads=<N> --debug=<0|1> --iters=<N> \n";
                return 1;
            }
        }
    }

    if (count <= 0 || threads_num <= 0 || iters <= 0) {
        std::cerr << "Error: buffer count and threads, iters must be positive\n";
        return 1;
    }

    // Initialize MPI & CCL
    MPI_Init(NULL, NULL);
    atexit(mpi_finalize);
    ccl::init();

    // Create SYCL queues (like in your other tests)
    std::vector<sycl::device> devices;
    auto platform_list = sycl::platform::get_platforms();
    for (auto& platform : platform_list) {
        auto pname = platform.get_info<sycl::info::platform::name>();
        if (pname.find("Level-Zero") != std::string::npos) {
            auto dlist = platform.get_devices();
            for (auto& dev : dlist) {
                if (dev.is_gpu()) {
                    devices.push_back(dev);
                }
            }
        }
    }

    sycl::context ctx(devices);
    for (auto& d : devices) {
        // in_order queue for determinism
        queues.push_back(sycl::queue(ctx, d, { sycl::property::queue::in_order() }));
    }

    // Define groups & matching KVS (like your other tests)
    auto execute_task = [&](auto thread_groups, auto kvss) {
        std::vector<std::thread> threads;
        threads.reserve(threads_num);
        for (int i = 0; i < threads_num; i++) {
            thread_params p{ i, count, debug, iters, &thread_groups, &kvss };
            threads.emplace_back(thread_func_pt2pt, p);
        }
        for (auto& t : threads) {
            t.join();
        }
    };

    // all threads in one group, e.g. [0..threads_num-1]
    // For pt2pt, only 2 threads do actual send/recv
    std::vector<int> group_vec;
    group_vec.reserve(threads_num);
    for (int i = 0; i < threads_num; i++) {
        group_vec.push_back(i);
    }
    std::vector<std::vector<int>> thread_groups{ group_vec };
    std::vector<ccl::shared_ptr_class<ccl::kvs>> kvss{ ccl::create_main_kvs() };

    execute_task(thread_groups, kvss);

    return 0;
}

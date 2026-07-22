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
#include <pthread.h>

std::vector<sycl::queue> queues;
std::mutex mu; // for thread-safe I/O
static pthread_barrier_t iter_barrier;

using namespace sycl;
using namespace sycl::access;

/**
 * Holds parameters for each thread.
 */
struct thread_params {
    int global_thread_id;
    int count; // number of elements to send/receive
    int debug; // enable debug printing
    int iters; // number of iterations
    std::vector<std::vector<int>>* thread_groups;
    std::vector<ccl::shared_ptr_class<ccl::kvs>>* kvss;
};

/**
 * Thread function performing a pt2pt send/recv within a 2-rank group.
 *
 * If the group_size != 2, we skip the send/recv logic (just do nothing).
 * If group_size=2:
 *   - rank_in_group=0 => Sender
 *   - rank_in_group=1 => Receiver
 */
void thread_func_pt2pt(thread_params params) {
    int global_thread_id = params.global_thread_id;
    int count = params.count;
    int debug = params.debug;
    int iterations = params.iters;

    // find rank_in_group, group_size, kvs, group_index
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
        std::cerr << "[ERROR] Thread " << global_thread_id << " not found in any group."
                  << std::endl;
        return;
    }

    // create SYCL objects for this thread
    auto q = queues[global_thread_id];
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto stream = ccl::create_stream(q);

    // create communicator for this group
    auto comm = ccl::create_communicatorExt(group_size, rank_in_group, dev, ctx, kvs);

    if (debug) {
        std::lock_guard<std::mutex> lock(mu);
        std::cout << "[DEBUG] Thread " << global_thread_id << " => group_index=" << group_index
                  << ", rank_in_group=" << rank_in_group << ", group_size=" << group_size
                  << std::endl;
    }

    // We'll do an allgatherv-like scenario with manual send/recv.
    // Each rank has 'count' elements, so the total is (count * group_size).
    // We'll define displs, so rank i writes into [displs[i] .. displs[i]+count-1].
    std::vector<int> recvcounts(group_size, count);
    std::vector<int> displs(group_size, 0);
    for (int i = 1; i < group_size; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    int total_recvcount = displs[group_size - 1] + recvcounts[group_size - 1];

    // allocate device buffers
    buf_allocator<int> allocator(q);
    auto send_buf = allocator.allocate(count, usm::alloc::device);
    auto recv_buf = allocator.allocate(total_recvcount, usm::alloc::device);

    for (int iter = 0; iter < iterations; iter++) {
        // 1. Fill send buffer with iteration-specific data
        q.submit([&](auto& h) {
            h.parallel_for(range<1>(count), [=](id<1> idx) {
                int j = idx[0];
                // Example pattern: add iter*1000 so each iteration is unique
                send_buf[j] = iter * 1000 + rank_in_group * group_size + j;
            });
        });

        // 2. Copy local portion into recv_buf
        q.memcpy(recv_buf + displs[rank_in_group], send_buf,
                 count * sizeof(int)).wait(); // ?

        // 3. Do pt2pt exchange with all peers
        std::vector<ccl::event> deps;
        auto send_attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
        auto recv_attr = ccl::create_operation_attr<ccl::pt2pt_attr>();

        ccl::group_start();
        for (int i = 0; i < group_size; i++) {
            if (i != rank_in_group) {
                ccl::send(send_buf,
                          count,
                          ccl::datatype::int32,
                          /*dest=*/i,
                          comm,
                          stream,
                          send_attr,
                          deps);

                ccl::recv(recv_buf + displs[i],
                          recvcounts[i],
                          ccl::datatype::int32,
                          /*src=*/i,
                          comm,
                          stream,
                          recv_attr,
                          deps);
            }
        }
        ccl::group_end();

        // this is needed to check async pt2pt correctness
        q.submit([&](handler& h) {
            h.parallel_for(count, [=](auto id) {
                send_buf[id] = -1;
            });
        });

        // 4. Copy results to host for validation
        std::vector<int> host_recv_buf(total_recvcount);
        q.memcpy(host_recv_buf.data(), recv_buf,
                 total_recvcount * sizeof(int)).wait(); // ?

        // 5. Verify results
        bool is_correct = true;
        for (int peer = 0; peer < group_size; peer++) {
            int start_idx = displs[peer];
            int end_idx = displs[peer] + recvcounts[peer];
            for (int idx = start_idx; idx < end_idx; idx++) {
                int j = idx - start_idx;
                int expected = iter * 1000 + peer * group_size + j;
                if (host_recv_buf[idx] != expected) {
                    std::lock_guard<std::mutex> lock(mu);
                    std::cerr << "[ERROR] Iter=" << iter << " Thread=" << global_thread_id
                              << " => mismatch at peer=" << peer << " idx=" << idx
                              << " got=" << host_recv_buf[idx] << " expected=" << expected
                              << std::endl;
                    is_correct = false;
                    break;
                }
            }
            if (!is_correct)
                break;
        }

        if (is_correct) {
            std::lock_guard<std::mutex> lock(mu);
            std::cout << "Iter " << iter << " Thread " << global_thread_id
                      << " => PASSED group_start/end pt2pt check.\n";
        }
    }
}

int main(int argc, char* argv[]) {
    int count = 4, threads_num = 4, debug = 0, iters = 10;

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

    // initialize the per-iteration barrier for all threads
    pthread_barrier_init(&iter_barrier, NULL, threads_num);

    execute_task(thread_groups, kvss);

    // destroy the barrier after threads have finished
    pthread_barrier_destroy(&iter_barrier);

    return 0;
}

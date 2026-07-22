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

#include "graph_integration.hpp"

#include <cstdint>
#include <iostream>
#include <thread>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include "oneapi/ccl.hpp"

struct MTInstanceArgs {
    size_t comm_rank;
    size_t comm_size;
    MTInstanceArgs(size_t comm_rank, size_t comm_size)
            : comm_rank(comm_rank),
              comm_size(comm_size) {}
};

std::shared_ptr<ccl::kvs> main_kvs = ccl::create_main_kvs();

static std::shared_ptr<ccl::kvs> create_threaded_kvs(size_t size, size_t rank) {
    return main_kvs;
}

void run_mt_thread(MTInstanceArgs instance_args) {
    size_t msg_size = 1;

    auto devices = select_devices();
    auto context = sycl::context(devices);

    TestInstanceAllreduce test_allreduce(msg_size,
                                         instance_args.comm_size,
                                         instance_args.comm_rank,
                                         devices,
                                         context,
                                         create_threaded_kvs,
                                         ccl::create_communicatorExt);
    TestInstanceReduceScatter test_reduce_scatter(msg_size,
                                                  instance_args.comm_size,
                                                  instance_args.comm_rank,
                                                  devices,
                                                  context,
                                                  create_threaded_kvs,
                                                  ccl::create_communicatorExt);
    // multithreaded allgather is not supported yet; add this test once the support is introduced
    // TestInstanceAllgather test_allgather(msg_size,
    //                                      instance_args.comm_size,
    //                                      instance_args.comm_rank,
    //                                      create_threaded_kvs,
    //                                      ccl::create_communicatorExt);

    for (size_t i = 0; i < 13; ++i) {
        test_allreduce.reinit(msg_size);
        run_test_scenario(test_allreduce, 10, 1);
        run_test_scenario(test_allreduce, 2, 10);

        test_reduce_scatter.reinit(msg_size);
        run_test_scenario(test_reduce_scatter, 10, 1);
        run_test_scenario(test_reduce_scatter, 2, 10);

        // multithreaded allgather is not supported yet; add this test once the support is introduced
        // test_allgather.reinit(msg_size);
        // run_test_scenario(test_allgather, 10, 1);
        // run_test_scenario(test_allgather, 2, 10);

        // test_alltoall.reinit(msg_size);
        // run_test_scenario(test_alltoall, 10, 1);
        // run_test_scenario(test_alltoall, 2, 10);
        //
        // test_broadcast.reinit(msg_size);
        // run_test_scenario(test_broadcast, 10, 1);
        // run_test_scenario(test_broadcast, 2, 10);

        msg_size *= 4;
    }
}

void run_mt_tests() {
    const size_t stress_repetitions = 5;
    const size_t nof_threads = 4;

    std::vector<std::thread> threads;
    threads.reserve(nof_threads);

    for (size_t rank = 0; rank < nof_threads; ++rank) {
        MTInstanceArgs instance_args(rank, nof_threads);
        threads.push_back(std::thread(run_mt_thread, instance_args));
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

int main(int argc, char* argv[]) {
    run_mt_tests();

    return 0;
}

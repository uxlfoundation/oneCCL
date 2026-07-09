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
#include <iomanip>
#include <ostream>
#include <sstream>
#include <thread>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include "oneapi/ccl.hpp"

static void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized) {
        MPI_Finalize();
    }
}

// no need to "finalize" - done automagically at exit
static void initialize_global(int& size, int& rank) {
    ccl::init();

    MPI_Init(nullptr, nullptr);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);
}

static std::shared_ptr<ccl::kvs> create_mpi_kvs(size_t size, size_t rank) {
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }
    return kvs;
}

static void run_benchmark(TestInstance& test_instance,
                          size_t size_to_bench,
                          size_t warmup_iterations,
                          size_t bench_iterations,
                          size_t iterations_in_recording,
                          std::ostream& out) {
    std::vector<uint64_t> timestamps;
    timestamps.reserve(4);

    size_t rank = test_instance.get_comm_rank();

    test_instance.reinit(size_to_bench);

    // warmup
    {
        std::vector<ccl::event> deps{};
        for (size_t i = 0; i < warmup_iterations; ++i) {
            test_instance.run_test(deps).wait();
        }
    }

    // actual run
    {
        timestamps.push_back(get_microseconds());
        for (size_t i = 0; i < bench_iterations; ++i) {
            std::vector<ccl::event> deps{};
            for (size_t i = 0; i < iterations_in_recording; ++i) {
                std::vector<ccl::event> tmp;
                tmp.push_back(test_instance.run_test(deps));
                deps = std::move(tmp);
            }
            assert(deps.size() > 0);
            deps.back().wait();
        }
        timestamps.push_back(get_microseconds());
    }
    sycl::ext::oneapi::experimental::command_graph graph(test_instance.get_queue().get_context(),
                                                         test_instance.get_queue().get_device());
    {
        // if (rank == 0) {
        //     std::cout << "record" << std::endl;
        // }
        graph.begin_recording(test_instance.get_queue());

        std::vector<ccl::event> deps{};
        for (size_t i = 0; i < iterations_in_recording; ++i) {
            std::vector<ccl::event> tmp;
            tmp.push_back(test_instance.run_test(deps));
            deps = std::move(tmp);
        }

        graph.end_recording();
    }
    auto executable_graph = graph.finalize();
    {
        timestamps.push_back(get_microseconds());
        for (size_t i = 0; i < bench_iterations; ++i)
            test_instance.get_queue().ext_oneapi_graph(executable_graph).wait();
        timestamps.push_back(get_microseconds());
    }

    uint64_t ptimestamp = timestamps[0];

    size_t run_idx_start = 0;
    size_t run_idx_end = 1;

    size_t replay_idx_start = 2;
    size_t replay_idx_end = 3;

    uint64_t run_time_us = timestamps[run_idx_end] - timestamps[run_idx_start];
    uint64_t replay_time_us = timestamps[replay_idx_end] - timestamps[replay_idx_start];

    double average_run_time_us =
        static_cast<double>(run_time_us) / bench_iterations / iterations_in_recording;
    double average_replay_time_us =
        static_cast<double>(replay_time_us) / bench_iterations / iterations_in_recording;

    double speedup = (average_run_time_us - average_replay_time_us) / average_run_time_us;
    double speedup_percent = speedup * 100;

    const size_t column_width = 26;
    const size_t number_width = column_width - 2; // column separator: " |"

    if (test_instance.get_comm_rank() == 0) {
        out << std::setw(number_width) << size_to_bench << " |" << std::setw(number_width)
            << std::setprecision(2) << std::fixed << average_run_time_us << " |"
            << std::setw(number_width) << std::setprecision(2) << std::fixed
            << average_replay_time_us << " |" << std::setw(number_width) << std::setprecision(2)
            << std::fixed << speedup_percent << " |" << std::endl;
    }
}

#include <sstream>

static void run_benchmark(TestInstance& test_instance,
                          const std::vector<size_t>& sizes_to_bench,
                          size_t warmup_iterations,
                          size_t bench_iterations,
                          size_t iterations_in_recording) {
    std::stringstream output;
    const size_t column_width = 26;
    if (test_instance.get_comm_rank() == 0) {
        output << std::endl;
        output << "Benchmarking operation: " << test_instance.get_name()
               << ", bench_iterations: " << bench_iterations
               << ", iterations_in_recording: " << iterations_in_recording << std::endl;
        output << std::setfill(' ') << std::setw(column_width) << " buffer count |"
               << std::setw(column_width) << " execution time [us] |" << std::setw(column_width)
               << " replay time [us] |" << std::setw(column_width) << " speedup [%] |" << std::endl;
    }
    for (const auto& size_to_bench : sizes_to_bench) {
        run_benchmark(test_instance,
                      size_to_bench,
                      warmup_iterations,
                      bench_iterations,
                      iterations_in_recording,
                      output);
    }
    if (test_instance.get_comm_rank() == 0) {
        output << std::endl << std::endl;
        std::cout << output.str();
    }
}

void run_benchmarks() {
    size_t size = 0, rank = 0;
    int t_size = 0, t_rank = 0;
    initialize_global(t_size, t_rank);

    if (t_size <= 0 || t_rank < 0) {
        std::cout << "incorrect mpi size/rank initialization" << std::endl
                  << " size:" << t_size << ", rank: " << t_rank << std::endl
                  << "FAILED" << std::endl;
        throw "Failed to initialize MPI variables";
    }

    size = static_cast<size_t>(t_size);
    rank = static_cast<size_t>(t_rank);

    const size_t min_msg_size = 1;
    const size_t nof_msg_sizes = 26;

    std::vector<size_t> sizes_to_bench;
    sizes_to_bench.reserve(nof_msg_sizes);

    size_t msg_size = min_msg_size;

    for (size_t i = 0; i < nof_msg_sizes; ++i) {
        sizes_to_bench.push_back(msg_size);
        msg_size *= 2;
    }

    auto devices = select_devices();
    auto context = sycl::context(devices);

    TestInstanceAllreduce test_allreduce(
        msg_size, size, rank, devices, context, create_mpi_kvs, ccl::create_communicator);
    TestInstanceReduceScatter test_reduce_scatter(
        msg_size, size, rank, devices, context, create_mpi_kvs, ccl::create_communicator);
    TestInstanceAllgather test_allgather(
        msg_size, size, rank, devices, context, create_mpi_kvs, ccl::create_communicator);
    TestInstanceAlltoall test_alltoall(
        msg_size, size, rank, devices, context, create_mpi_kvs, ccl::create_communicator);
    TestInstanceBroadcast test_broadcast(
        msg_size, size, rank, devices, context, create_mpi_kvs, ccl::create_communicator);

    run_benchmark(test_allreduce, sizes_to_bench, 10, 1000, 1);
    run_benchmark(test_allreduce, sizes_to_bench, 10, 10, 100);

    run_benchmark(test_reduce_scatter, sizes_to_bench, 10, 1000, 1);
    run_benchmark(test_reduce_scatter, sizes_to_bench, 10, 10, 100);

    test_allgather.reinit(msg_size);
    run_benchmark(test_allgather, sizes_to_bench, 10, 1000, 1);
    run_benchmark(test_allgather, sizes_to_bench, 10, 10, 100);

    test_alltoall.reinit(msg_size);
    run_benchmark(test_alltoall, sizes_to_bench, 10, 1000, 1);
    run_benchmark(test_alltoall, sizes_to_bench, 10, 10, 100);

    test_broadcast.reinit(msg_size);
    run_benchmark(test_broadcast, sizes_to_bench, 10, 1000, 1);
    run_benchmark(test_broadcast, sizes_to_bench, 10, 10, 100);
}

int main(int argc, char* argv[]) {
    try {
        run_benchmarks();
    }
    catch (const std::exception& e) {
        std::cout << "Exception occurred, aborting" << std::endl << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        throw;
    }
    catch (...) {
        std::cout << "Exception occurred, aborting" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        throw;
    }

    return 0;
}

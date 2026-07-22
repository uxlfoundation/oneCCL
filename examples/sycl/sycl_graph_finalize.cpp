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

// examples/sycl/sycl_graph_finalize.cpp
// Example demonstrating the interaction between SYCL graphs and
// onecclCommFinalize() Shows the warning when finalize is called after using a
// communicator in a graph

#include "oneapi/ccl.h"

#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>

#if !defined(SYCL_EXT_ONEAPI_GRAPH)
#error "This example requires SYCL graph extension support"
#endif

using namespace sycl::ext::oneapi::experimental;

// Create SYCL queues by distributing them based on local rank
static sycl::queue create_queue(int local_rank) {
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

    return sycl::queue(
        l0_platform
            .get_devices()[local_rank % l0_platform.get_devices().size()],
        {sycl::property::queue::in_order{},
         sycl::property::queue::enable_profiling{}});
}

int main() {
    int rank = 0;
    int local_rank = 0;
    int world_size = 0;
    int version = 0;
    constexpr int kCount = 1024;
    constexpr int kGraphLaunches = 3;
    constexpr int kNormalOps = 5;

    MPI_Comm local_comm = 0;
    onecclComm_t comm = nullptr;
    onecclResult_t result = onecclSuccess;
    onecclUniqueId uid;

    // Initialize
    onecclGetVersion(&version);
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);

    if (rank == 0) {
        onecclGetUniqueId(&uid);
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    auto queue = create_queue(rank);

    result = onecclSetDevice(local_rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to set device.\n";
        return 1;
    }

    result = onecclCommInitRank(&comm, world_size, uid, rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to initialize communicator.\n";
        return 1;
    }

    // Allocate buffers
    int *sendbuff =
        static_cast<int *>(sycl::malloc_device(kCount * sizeof(int), queue));
    int *recvbuff =
        static_cast<int *>(sycl::malloc_device(kCount * sizeof(int), queue));

    if (rank == 0) {
        std::cout << "\n=== Part 1: Normal operations ===\n";
        std::cout << "Running " << kNormalOps
                  << " normal AllReduce operations...\n";
    }

    auto my_rank = rank;
    for (int i = 0; i < kNormalOps; i++) {
        queue.submit([&](sycl::handler &h) {
            h.parallel_for<class normal_init>(
                sycl::range<1>(kCount),
                [=](sycl::id<1> idx) { sendbuff[idx] = my_rank + i; });
        });

        result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt,
                                 onecclSum, comm, &queue);
        if (result != onecclSuccess) {
            std::cerr << "AllReduce operation failed.\n";
            return 1;
        }
    }

    if (rank == 0) {
        std::cout
            << "Calling onecclCommFinalize() after normal operations...\n";
    }

    // Finalize works correctly here - all events are tracked
    result = onecclCommFinalize(comm);
    if (result != onecclSuccess) {
        std::cerr << "Finalize communicator failed.\n";
        return 1;
    }

    if (rank == 0) {
        std::cout << "Finalize completed successfully.\n";
    }

    if (rank == 0) {
        std::cout << "\n=== Part 2: Graph recording ===\n";
        std::cout << "Creating SYCL graph with AllReduce operation...\n";
    }

    // Create a SYCL graph
    command_graph graph(queue.get_context(), queue.get_device());

    // Record operations into the graph
    graph.begin_recording(queue);

    queue.submit([&](sycl::handler &h) {
        h.parallel_for<class graph_init>(
            sycl::range<1>(kCount),
            [=](sycl::id<1> idx) { sendbuff[idx] = my_rank + 100; });
    });

    // During graph recording, the communicator will be marked as used in graph
    // recording This happens during the first collective operation recorded
    // into the graph
    result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt, onecclSum,
                             comm, &queue);
    if (result != onecclSuccess) {
        std::cerr << "AllReduce during graph recording failed.\n";
        return 1;
    }

    graph.end_recording();

    if (rank == 0) {
        std::cout << "Finalizing graph...\n";
    }

    auto exec_graph = graph.finalize();

    if (rank == 0) {
        std::cout << "Launching graph " << kGraphLaunches << " times...\n";
    }

    // Launch the graph multiple times
    std::vector<sycl::event> graph_events;
    for (int i = 0; i < kGraphLaunches; i++) {
        auto event = queue.submit(
            [&](sycl::handler &h) { h.ext_oneapi_graph(exec_graph); });
        graph_events.push_back(event);
    }

    // This will trigger a WARNING and do nothing, because the communicator
    // was used during graph recording. Graph events are not tracked.
    result = onecclCommFinalize(comm);
    if (result != onecclSuccess) {
        std::cerr << "Finalize communicator failed.\n";
        return 1;
    }

    if (rank == 0) {
        std::cout << "Waiting for all graph launches to complete...\n";
    }

    // IMPORTANT: When using graphs, you MUST manually wait for graph events
    // before calling finalize, because finalize cannot track graph execution
    // events
    for (auto &event : graph_events) {
        event.wait();
    }

    if (rank == 0) {
        std::cout << "All graph launches completed.\n";
        std::cout
            << "\n=== Part 3: Attempting finalize after graph usage ===\n";
        std::cout << "Note: This will show a warning because the communicator "
                     "was used\n";
        std::cout << "      during graph recording. Finalize cannot track "
                     "graph events.\n";
    }

    if (rank == 0) {
        std::cout << "\n=== Summary ===\n";
        std::cout << "When using SYCL graphs with oneCCL:\n";
        std::cout
            << "1. Graph recording marks the communicator as 'used in graph'\n";
        std::cout
            << "2. onecclCommFinalize() will log a warning and do nothing\n";
        std::cout
            << "3. You MUST manually wait on all graph submission events\n";
        std::cout
            << "4. This is correct behavior - graph events cannot be tracked\n";
        std::cout
            << "   because the same graph can be launched multiple times\n";
    }

    // Verify results
    std::vector<int> recvbuff_host(kCount);
    queue.memcpy(recvbuff_host.data(), recvbuff, kCount * sizeof(int)).wait();

    int expected = 0;
    for (int r = 0; r < world_size; r++) {
        expected += r + 100;
    }

    bool success = true;
    for (int i = 0; i < kCount; i++) {
        if (recvbuff_host[i] != expected) {
            std::cerr << "Rank " << rank << ": Mismatch at index " << i
                      << ": expected " << expected << ", got "
                      << recvbuff_host[i] << "\n";
            success = false;
            break;
        }
    }

    if (rank == 0 && success) {
        std::cout << "\nResult verification passed!\n";
    }

    // Cleanup
    sycl::free(sendbuff, queue);
    sycl::free(recvbuff, queue);

    result = onecclCommDestroy(comm);
    if (result != onecclSuccess) {
        std::cerr << "Destroy communicator failed.\n";
        return 1;
    }

    MPI_Finalize();

    return 0;
}

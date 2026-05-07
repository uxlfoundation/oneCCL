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

#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <numeric>
#include <string>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

namespace {

enum class CollectiveType : std::uint8_t {
    ONECCL_ALLREDUCE,
    ONECCL_ALLGATHER,
    ONECCL_REDUCE_SCATTER
};

struct onecclTestParams {
    int threads_num;
    int count;
    std::vector<std::vector<int>> group_desc; // explicit groupings
    CollectiveType coll_type;

    friend std::ostream &operator<<(std::ostream &out_stream,
                                    const onecclTestParams &params) {
        out_stream << "threads=" << params.threads_num
                   << "_count=" << params.count
                   << "_groups=" << params.group_desc.size() << "_desc=";
        for (const auto &group : params.group_desc) {
            out_stream << "{";
            for (size_t i = 0; i < group.size(); ++i) {
                out_stream << group[i];
                if (i + 1 < group.size()) {
                    out_stream << ",";
                }
            }
            out_stream << "}";
        }
        out_stream << "_coll=";
        switch (params.coll_type) {
        case CollectiveType::ONECCL_ALLREDUCE:
            out_stream << "allreduce";
            break;
        case CollectiveType::ONECCL_ALLGATHER:
            out_stream << "allgather";
            break;
        case CollectiveType::ONECCL_REDUCE_SCATTER:
            out_stream << "reduce_scatter";
            break;
        }
        return out_stream;
    }
};

class SyclMtCollectivesUsmTest
    : public ::testing::TestWithParam<onecclTestParams> {
  public:
    static std::atomic<bool> print_results;

  protected:
    static std::mutex print_mutex;

    struct onecclThreadParams {
        int global_thread_id;
        int count;
        const std::vector<std::vector<int>> *thread_groups;
        const std::vector<onecclUniqueId> *group_ids;
        CollectiveType coll_type;
        bool multi_iteration = false; // Flag for multi-iteration tests
        int num_iterations = 1;       // Number of iterations for multi tests
        std::atomic<int> *sync_counter = nullptr; // For synchronization
    };

    static void set_up_test_suite() {
        int argc = 0;
        char **argv = nullptr; // NOLINT(misc-const-correctness)
        MPI_Init(&argc, &argv);
    }

    static void tear_down_test_suite() { MPI_Finalize(); }

    static sycl::queue create_sycl_queue(int device_idx) {
        auto platforms = sycl::platform::get_platforms();
        for (const auto &platform : platforms) {
            if (platform.get_backend() ==
                sycl::backend::ext_oneapi_level_zero) {
                auto devices = platform.get_devices();
                int gpu_count = 0;
                for (const auto &dev : devices) {
                    if (dev.is_gpu()) {
                        if (gpu_count == device_idx) {
                            return sycl::queue(
                                dev, {sycl::property::queue::in_order()});
                        }
                        ++gpu_count;
                    }
                }
            }
        }
        throw std::runtime_error("Not enough Level Zero GPU devices");
    }

    static bool
    find_thread_group_info(int global_thread_id,
                           const std::vector<std::vector<int>> &thread_groups,
                           int &rank_in_group, int &group_size,
                           int &group_index) {
        for (size_t i = 0; i < thread_groups.size(); ++i) {
            const auto &group = thread_groups[i];
            auto iter = std::find(group.begin(), group.end(), global_thread_id);
            if (iter != group.end()) {
                rank_in_group =
                    static_cast<int>(std::distance(group.begin(), iter));
                group_size = static_cast<int>(group.size());
                group_index = static_cast<int>(i);
                return true;
            }
        }
        return false;
    }

    static void
    run_allreduce_test(sycl::queue &queue, onecclComm_t comm, int count,
                       int global_thread_id, int group_index, int rank_in_group,
                       const std::vector<std::vector<int>> &thread_groups,
                       bool &success) {
        int *send_buf = nullptr;
        int *recv_buf = nullptr;
        send_buf = sycl::malloc_device<int>(count, queue);
        recv_buf = sycl::malloc_device<int>(count, queue);
        // Each thread sends a unique value: 24 * (thread_id + 1)
        // E.g., thread 0 -> 24, thread 1 -> 48, thread 2 -> 72
        // The factor 24 is arbitrary but makes verification easier
        std::vector<int> host_send(count, 24 * (global_thread_id + 1));
        queue.memcpy(send_buf, host_send.data(), count * sizeof(int)).wait();

        onecclResult_t const res = onecclAllReduce(
            send_buf, recv_buf, count, onecclInt32, onecclSum, comm, &queue);
        ASSERT_EQ(res, onecclSuccess);

        std::vector<int> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int)).wait();

        // Calculate expected result: sum of all thread contributions in the
        // group. For group {0, 1}: sum = (0+1) + (1+1) = 3, expected = 72
        int const group_sum = [&thread_groups, group_index]() {
            int const sum = std::accumulate(
                thread_groups[group_index].begin(),
                thread_groups[group_index].end(), 0,
                [](int acc, int gid) { return acc + (gid + 1); });
            return sum;
        }();
        int const expected = 24 * group_sum;
        success = std::all_of(host_recv.begin(), host_recv.end(),
                              [expected](int val) { return val == expected; });

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[Allreduce][thread " << global_thread_id << "] group "
                      << group_index << " rank_in_group " << rank_in_group
                      << " recv:";
            const int to_print = std::min(count, 8);
            for (int i = 0; i < to_print; ++i) {
                std::cout << " " << host_recv[i];
            }
            if (count > 8) {
                std::cout << " ...";
            }
            std::cout << '\n';
        }
    }

    // Multi-iteration allreduce test with synchronization between iterations
    static void run_allreduce_multi_test(
        sycl::queue &queue, onecclComm_t comm, int count, int global_thread_id,
        int group_index, int rank_in_group,
        const std::vector<std::vector<int>> &thread_groups, bool &success,
        std::atomic<int> &sync_counter, int group_size, int num_iterations) {
        int *send_buf = nullptr;
        int *recv_buf = nullptr;
        send_buf = sycl::malloc_device<int>(count, queue);
        recv_buf = sycl::malloc_device<int>(count, queue);

        success = true;

        for (int iter = 0; iter < num_iterations; ++iter) {
            // Synchronize all threads in the group before each iteration
            sync_counter.fetch_add(1, std::memory_order_release);
            while (sync_counter.load(std::memory_order_acquire) <
                   (iter + 1) * group_size) {
                std::this_thread::yield();
            }

            // Each thread sends a value based on iteration and thread ID
            const int alloc_value = (iter + 1) * 10 * (global_thread_id + 1);
            std::vector<int> host_send(count, alloc_value);
            queue.memcpy(send_buf, host_send.data(), count * sizeof(int))
                .wait();

            onecclResult_t const res =
                onecclAllReduce(send_buf, recv_buf, count, onecclInt32,
                                onecclSum, comm, &queue);
            if (res != onecclSuccess) {
                success = false;
                break;
            }

            std::vector<int> host_recv(count);
            queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int))
                .wait();

            // Calculate expected: sum of all thread contributions
            int const group_sum = [&thread_groups, group_index]() {
                int const sum = std::accumulate(
                    thread_groups[group_index].begin(),
                    thread_groups[group_index].end(), 0,
                    [](int acc, int gid) { return acc + (gid + 1); });
                return sum;
            }();
            int const expected = (iter + 1) * 10 * group_sum;

            if (!std::all_of(host_recv.begin(), host_recv.end(),
                             [expected](int val) { return val == expected; })) {
                success = false;
                if (print_results) {
                    const std::scoped_lock lock(print_mutex);
                    std::cout << "[AllreduceMulti][thread " << global_thread_id
                              << "][iter " << iter << "] FAILED - expected "
                              << expected << " but got different values\n";
                }
                break;
            }

            if (print_results && iter == 0) {
                const std::scoped_lock lock(print_mutex);
                std::cout << "[AllreduceMulti][thread " << global_thread_id
                          << "][iter " << iter << "] group " << group_index
                          << " rank_in_group " << rank_in_group
                          << " first recv value: " << host_recv[0] << '\n';
            }
        }

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
    }

    static void run_allgather_test(
        sycl::queue &queue, onecclComm_t comm, int count, int global_thread_id,
        int group_index, int rank_in_group, int group_size,
        const std::vector<std::vector<int>> &thread_groups, bool &success) {
        int *send_buf = nullptr;
        int *recv_buf = nullptr;
        send_buf = sycl::malloc_device<int>(count, queue);
        recv_buf = sycl::malloc_device<int>(
            static_cast<long>(count * group_size), queue);
        // Each thread sends its ID offset by 100 for easy identification
        // E.g., thread 0 -> 100, thread 1 -> 101, thread 2 -> 102
        std::vector<int> host_send(count, 100 + global_thread_id);
        queue.memcpy(send_buf, host_send.data(), count * sizeof(int)).wait();

        onecclResult_t const res = onecclAllGather(send_buf, recv_buf, count,
                                                   onecclInt32, comm, &queue);
        ASSERT_EQ(res, onecclSuccess);

        std::vector<int> host_recv(static_cast<size_t>(count * group_size));
        queue
            .memcpy(host_recv.data(), recv_buf,
                    static_cast<size_t>(count * group_size) * sizeof(int))
            .wait();

        // Verify each rank's data was gathered correctly
        success = true;
        for (int rank = 0; rank < group_size; ++rank) {
            const int expected = 100 + thread_groups[group_index][rank];
            for (int col = 0; col < count; ++col) {
                if (host_recv[(rank * count) + col] != expected) {
                    success = false;
                    break;
                }
            }
        }

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[Allgather][thread " << global_thread_id << "] group "
                      << group_index << " rank_in_group " << rank_in_group
                      << " recv:";
            const int to_print = std::min(count * group_size, 8);
            for (int i = 0; i < to_print; ++i) {
                std::cout << " " << host_recv[i];
            }
            if (count * group_size > 8) {
                std::cout << " ...";
            }
            std::cout << '\n';
        }
    }

    static void run_reduce_scatter_test(
        sycl::queue &queue, onecclComm_t comm, int count, int global_thread_id,
        int group_index, int rank_in_group, int group_size,
        const std::vector<std::vector<int>> &thread_groups, bool &success) {
        int *send_buf = nullptr;
        int *recv_buf = nullptr;
        send_buf = sycl::malloc_device<int>(
            static_cast<long>(count * group_size), queue);
        recv_buf = sycl::malloc_device<int>(count, queue);

        std::vector<int> host_send(static_cast<size_t>(count * group_size), 0);
        // Fill send buffer with unique values per rank position
        // Factor 7 creates distinct values: 7 * (thread_id + rank_position)
        for (int rank = 0; rank < group_size; ++rank) {
            const int val = 7 * (global_thread_id + rank);
            for (int col = 0; col < count; ++col) {
                host_send[(rank * count) + col] = val;
            }
        }
        queue
            .memcpy(send_buf, host_send.data(),
                    static_cast<size_t>(count * group_size) * sizeof(int))
            .wait();

        onecclResult_t const res = onecclReduceScatter(
            send_buf, recv_buf, count, onecclInt32, onecclSum, comm, &queue);
        ASSERT_EQ(res, onecclSuccess);

        std::vector<int> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int)).wait();

        // Expected: sum of contributions from all threads for this rank's block
        // Each thread contributes 7 * (thread_id + my_rank_in_group)
        int const expected = [&thread_groups, group_index, rank_in_group,
                              group_size]() {
            int exp = 0;
            for (int thread_idx = 0; thread_idx < group_size; ++thread_idx) {
                exp += 7 *
                       (thread_groups[group_index][thread_idx] + rank_in_group);
            }
            return exp;
        }();

        success = std::all_of(host_recv.begin(), host_recv.end(),
                              [expected](int val) { return val == expected; });

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[ReduceScatter][thread " << global_thread_id
                      << "] group " << group_index << " rank_in_group "
                      << rank_in_group << " recv:";
            const int to_print = std::min(count, 8);
            for (int i = 0; i < to_print; ++i) {
                std::cout << " " << host_recv[i];
            }
            if (count > 8) {
                std::cout << " ...";
            }
            std::cout << '\n';
        }
    }

    static void thread_func(onecclThreadParams params,
                            std::vector<bool> &thread_success, int idx) {
        const int global_thread_id = params.global_thread_id;
        const int count = params.count;

        int rank_in_group = -1;
        int group_size = -1;
        int group_index = -1;
        onecclUniqueId group_id{};

        // Find my group and group rank
        if (!find_thread_group_info(global_thread_id, *params.thread_groups,
                                    rank_in_group, group_size, group_index)) {
            const std::scoped_lock lock(print_mutex);
            thread_success[idx] = false;
            return;
        }

        group_id = (*params.group_ids)[group_index];

        sycl::queue queue = create_sycl_queue(global_thread_id);
        auto result = onecclSetDevice(global_thread_id);
        ASSERT_EQ(result, onecclSuccess);

        onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
        config.multiThreaded = 1; // Enable multi-threaded communicator

        onecclComm_t comm = nullptr;
        onecclResult_t const cres = onecclCommInitRankConfig(
            &comm, group_size, group_id, rank_in_group, &config);
        ASSERT_EQ(cres, onecclSuccess);

        bool success = false;
        if (params.multi_iteration &&
            params.coll_type == CollectiveType::ONECCL_ALLREDUCE) {
            // Multi-iteration test with synchronization
            run_allreduce_multi_test(
                queue, comm, count, global_thread_id, group_index,
                rank_in_group, *params.thread_groups, success,
                *params.sync_counter, group_size, params.num_iterations);
        } else if (params.coll_type == CollectiveType::ONECCL_ALLREDUCE) {
            run_allreduce_test(queue, comm, count, global_thread_id,
                               group_index, rank_in_group,
                               *params.thread_groups, success);
        } else if (params.coll_type == CollectiveType::ONECCL_ALLGATHER) {
            run_allgather_test(queue, comm, count, global_thread_id,
                               group_index, rank_in_group, group_size,
                               *params.thread_groups, success);
        } else if (params.coll_type == CollectiveType::ONECCL_REDUCE_SCATTER) {
            run_reduce_scatter_test(queue, comm, count, global_thread_id,
                                    group_index, rank_in_group, group_size,
                                    *params.thread_groups, success);
        } else {
            success = false;
        }

        thread_success[idx] = success;
        onecclCommDestroy(comm);
    }

    static void
    run_test_with_groups(const std::vector<std::vector<int>> &thread_groups,
                         int threads_num, int count, CollectiveType coll_type) {
        ASSERT_FALSE(thread_groups.empty());

        std::vector<onecclUniqueId> group_ids(thread_groups.size());
        for (size_t i = 0; i < thread_groups.size(); ++i) {
            onecclGetUniqueId(&group_ids[i]);
        }

        std::vector<std::thread> threads;
        std::vector<bool> thread_success(threads_num, false);
        for (int thread_id = 0; thread_id < threads_num; ++thread_id) {
            const onecclThreadParams params{thread_id, count, &thread_groups,
                                            &group_ids, coll_type};
            threads.emplace_back(thread_func, params, std::ref(thread_success),
                                 thread_id);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        for (const bool success_flag : thread_success) {
            ASSERT_TRUE(success_flag);
        }
    }

    // Multi-iteration test with overlap and synchronization
    static void run_multi_iteration_test_with_groups(
        const std::vector<std::vector<int>> &thread_groups, int threads_num,
        int count, CollectiveType coll_type, int num_iterations) {
        ASSERT_FALSE(thread_groups.empty());

        std::vector<onecclUniqueId> group_ids(thread_groups.size());
        for (size_t i = 0; i < thread_groups.size(); ++i) {
            onecclGetUniqueId(&group_ids[i]);
        }

        // Create atomic counters for each group to synchronize threads
        std::vector<std::atomic<int>> sync_counters(thread_groups.size());
        for (auto &counter : sync_counters) {
            counter.store(0);
        }

        std::vector<std::thread> threads;
        std::vector<bool> thread_success(threads_num, false);
        for (int thread_id = 0; thread_id < threads_num; ++thread_id) {
            // Find which group this thread belongs to
            int group_idx = -1;
            for (size_t group_index = 0; group_index < thread_groups.size();
                 ++group_index) {
                if (std::find(thread_groups[group_index].begin(),
                              thread_groups[group_index].end(),
                              thread_id) != thread_groups[group_index].end()) {
                    group_idx = static_cast<int>(group_index);
                    break;
                }
            }
            ASSERT_GE(group_idx, 0);

            // clang-format off
            const onecclThreadParams params{thread_id,
                                            count,
                                            &thread_groups,
                                            &group_ids,
                                            coll_type,
                                            true,
                                            num_iterations,
                                            &sync_counters[group_idx]};
            // clang-format on

            threads.emplace_back(thread_func, params, std::ref(thread_success),
                                 thread_id);
        }
        for (auto &thread : threads) {
            thread.join();
        }
        for (const bool success_flag : thread_success) {
            ASSERT_TRUE(success_flag);
        }
    }
};

std::mutex SyclMtCollectivesUsmTest::print_mutex;
std::atomic<bool> SyclMtCollectivesUsmTest::print_results = false;

// 2 threads groupings
const std::vector<std::vector<std::vector<int>>> kTwoThreadGroupings = {
    {{0, 1}}, {{1}, {0}}};
// 4 threads groupings
// const std::vector<std::vector<std::vector<int>>> kFourThreadGroupings = {
//     {{0, 1, 2, 3}}, {{1, 3}, {0, 2}}};
// // 8 threads groupings
// const std::vector<std::vector<std::vector<int>>> eight_thread_groupings = {
//     { {0, 1, 2, 3, 4, 5, 6, 7} },
//     { {0, 2, 4, 6}, {1, 3, 5, 7} }
// };

#define TESTPARAMS_ALL_COUNTS(threads, desc, coll)                             \
    onecclTestParams{threads, 1024, desc, coll}, onecclTestParams {            \
        threads, 33554432, desc, coll                                          \
    }

// NOLINTNEXTLINE(readability-identifier-naming)
INSTANTIATE_TEST_SUITE_P(
    MultiThreadedCollectivesCombinations, SyclMtCollectivesUsmTest,
    ::testing::Values(
        // 2 threads
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_ALLREDUCE),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_ALLREDUCE),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_ALLGATHER),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_ALLGATHER),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_REDUCE_SCATTER),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_REDUCE_SCATTER)

        // 4 threads
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
        //                       CollectiveType::ONECCL_ALLREDUCE),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
        //                       CollectiveType::ONECCL_ALLREDUCE),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
        //                       CollectiveType::ONECCL_ALLGATHER),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
        //                       CollectiveType::ONECCL_ALLGATHER),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
        //                       CollectiveType::ONECCL_REDUCE_SCATTER),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
        //                       CollectiveType::ONECCL_REDUCE_SCATTER)

        // // 8 threads
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
        //                      CollectiveType::ONECCL_ALLREDUCE),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
        //                      CollectiveType::ONECCL_ALLREDUCE),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
        //                      CollectiveType::ONECCL_ALLGATHER),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
        //                      CollectiveType::ONECCL_ALLGATHER),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
        //                      CollectiveType::ONECCL_REDUCE_SCATTER),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
        //                      CollectiveType::ONECCL_REDUCE_SCATTER)
        ));

// NOLINTNEXTLINE(readability-identifier-naming)
TEST_P(SyclMtCollectivesUsmTest, MultiThreadedCollectivesGroupVariants) {
    auto params = GetParam();
    // Check all threads are covered exactly once
    std::vector<bool> used(params.threads_num, false);
    for (const auto &group : params.group_desc) {
        for (const int thread_id : group) {
            ASSERT_LT(thread_id, params.threads_num);
            ASSERT_FALSE(used[thread_id]);
            used[thread_id] = true;
        }
    }
    for (const bool is_used : used) {
        ASSERT_TRUE(is_used);
    }

    run_test_with_groups(params.group_desc, params.threads_num, params.count,
                         params.coll_type);
}

// NOLINTNEXTLINE(readability-identifier-naming)
TEST_F(SyclMtCollectivesUsmTest, MultiIterationAllreduceWithSynchronization) {
    // Test multiple iterations of allreduce with atomic counter synchronization
    // This tests overlap and more complex scenarios as suggested in the review

    // Test with 2 threads, 5 iterations
    run_multi_iteration_test_with_groups(kTwoThreadGroupings[0], 2, 1024,
                                         CollectiveType::ONECCL_ALLREDUCE, 5);

    // Test with separate groups
    run_multi_iteration_test_with_groups(kTwoThreadGroupings[1], 2, 1024,
                                         CollectiveType::ONECCL_ALLREDUCE, 5);
}

// NOLINTNEXTLINE(readability-identifier-naming)
TEST_F(SyclMtCollectivesUsmTest,
       MultiIterationAllreduceWithSynchronizationLargeData) {
    // Test with larger data size to stress synchronization
    run_multi_iteration_test_with_groups(kTwoThreadGroupings[0], 2, 1048576,
                                         CollectiveType::ONECCL_ALLREDUCE, 3);
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Look for -print_res 1/0 in command line
    for (int i = 1; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "-print_res") == 0) {
            if (std::strcmp(argv[i + 1], "1") == 0) {
                SyclMtCollectivesUsmTest::print_results = true;
            } else {
                SyclMtCollectivesUsmTest::print_results = false;
            }
        }
    }
    return RUN_ALL_TESTS();
}

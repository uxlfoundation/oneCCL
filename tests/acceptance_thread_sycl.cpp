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
#include <array>
#include <atomic>
#include <cstring>
#include <iostream>
#include <map>
#include <mpi.h>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

namespace {

enum class CollectiveType : std::uint8_t {
    ONECCL_ALLREDUCE,
    ONECCL_ALLGATHER,
    ONECCL_REDUCE_SCATTER,
    ONECCL_SCATTER,
    ONECCL_REDUCE,
    ONECCL_BROADCAST,
    ONECCL_ALLTOALL,
    ONECCL_PT2PT,
    ONECCL_PT2PT_GROUP,
    ONECCL_PT2PT_NESTED_GROUP,
    ONECCL_LAST
};

// Names indexed by CollectiveType; keep in sync with the enum order above.
constexpr std::array<std::string_view,
                     static_cast<std::size_t>(CollectiveType::ONECCL_LAST)>
    kCollectiveNames = {"allreduce",         "allgather", "reduce_scatter",
                        "scatter",           "reduce",    "broadcast",
                        "alltoall",          "pt2pt",     "pt2pt_group",
                        "pt2pt_nested_group"};

constexpr std::string_view collective_name(CollectiveType coll_type) {
    const auto index = static_cast<std::size_t>(coll_type);
    return index < kCollectiveNames.size() ? kCollectiveNames[index]
                                           : "unknown";
}

// Everything a collective test runner may need. Every runner takes this one
// type, so they share a signature and can be stored as plain function pointers
// in the collective -> runner table, even though each uses a different subset.
struct CollectiveRunContext {
    sycl::queue &queue;
    onecclComm_t comm;
    int count;
    int global_thread_id;
    int group_index;
    int rank_in_group;
    int group_size;
    const std::vector<std::vector<int>> &thread_groups;
    // Only used by the multi-iteration runner.
    std::atomic<int> *sync_counter = nullptr;
    int num_iterations = 1;
};

using CollectiveRunner = bool (*)(const CollectiveRunContext &);

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
        out_stream << "_coll=" << collective_name(params.coll_type);
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

    static int get_gpu_count() {
        int gpu_count = 0;
        auto platforms = sycl::platform::get_platforms();
        for (const auto &platform : platforms) {
            if (platform.get_backend() ==
                sycl::backend::ext_oneapi_level_zero) {
                auto devices = platform.get_devices();
                gpu_count = devices.size();
            }
        }
        return gpu_count;
    }

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

    static bool run_allreduce_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const auto &thread_groups = ctx.thread_groups;

        bool result = true;
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
        if (res != onecclSuccess) {
            std::cout << "onecclAllReduce collective call failed with error "
                      << res << '\n';
            return false;
        }
        queue.wait();

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
        result = (res == onecclSuccess) &&
                 std::all_of(host_recv.begin(), host_recv.end(),
                             [expected](int val) { return val == expected; });

        if (print_results || !result) {
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

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        return result;
    }

    // Multi-iteration allreduce test with synchronization between iterations
    static bool run_allreduce_multi_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;
        std::atomic<int> &sync_counter = *ctx.sync_counter;
        const int num_iterations = ctx.num_iterations;

        int *send_buf = nullptr;
        int *recv_buf = nullptr;
        send_buf = sycl::malloc_device<int>(count, queue);
        recv_buf = sycl::malloc_device<int>(count, queue);

        bool success = true;

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
            queue.wait();

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
                // Always report the mismatch, even when print_results is off.
                const std::scoped_lock lock(print_mutex);
                std::cout << "[AllreduceMulti][thread " << global_thread_id
                          << "][iter " << iter << "] FAILED - expected "
                          << expected << " but got different values\n";
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
        return success;
    }

    static bool run_allgather_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;

        bool success = true;
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
        if (res != onecclSuccess) {
            std::cout << "onecclAllGather collective call failed with error "
                      << res << '\n';
            return false;
        }
        queue.wait();

        std::vector<int> host_recv(static_cast<size_t>(count * group_size));
        queue
            .memcpy(host_recv.data(), recv_buf,
                    static_cast<size_t>(count * group_size) * sizeof(int))
            .wait();

        // Verify each rank's data was gathered correctly
        success = res == onecclSuccess;
        for (int rank = 0; rank < group_size; ++rank) {
            const int expected = 100 + thread_groups[group_index][rank];
            for (int col = 0; col < count; ++col) {
                if (host_recv[(rank * count) + col] != expected) {
                    success = false;
                    break;
                }
            }
        }

        if (print_results || !success) {
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

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        return success;
    }

    static bool run_reduce_scatter_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;

        bool success = true;
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
        if (res != onecclSuccess) {
            std::cout
                << "onecclReduceScatter collective call failed with error "
                << res << '\n';
            return false;
        }
        queue.wait();
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

        success = res == onecclSuccess &&
                  std::all_of(host_recv.begin(), host_recv.end(),
                              [expected](int val) { return val == expected; });

        if (print_results || !success) {
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

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        return success;
    }

    static bool run_scatter_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;

        bool success = true;
        const int root = 0;

        // Only the root rank provides a send buffer; non-root ranks pass
        // nullptr to exercise scatter's root-only send_buf contract.
        const size_t total_elems =
            static_cast<size_t>(count) * static_cast<size_t>(group_size);
        int *send_buf = nullptr;
        if (rank_in_group == root) {
            send_buf = sycl::malloc_device<int>(total_elems, queue);
            std::vector<int> host_send(total_elems, 0);
            // Fill block dest with a value tagged by the destination rank's
            // global thread id, so each rank can verify it got its own block.
            for (int dest = 0; dest < group_size; ++dest) {
                const int val = 100 + thread_groups[group_index][dest];
                for (int col = 0; col < count; ++col) {
                    host_send[(static_cast<size_t>(dest) * count) + col] = val;
                }
            }
            queue.memcpy(send_buf, host_send.data(), total_elems * sizeof(int))
                .wait();
        }

        int *recv_buf = sycl::malloc_device<int>(count, queue);

        onecclResult_t const res = onecclScatter(
            send_buf, recv_buf, count, onecclInt32, root, comm, &queue);
        if (res != onecclSuccess) {
            std::cout << "onecclScatter collective call failed with error "
                      << res << '\n';
            return false;
        }
        queue.wait();

        std::vector<int> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int)).wait();

        // This rank sits at position rank_in_group in its group, so it should
        // receive the block tagged with its own global thread id.
        const int expected = 100 + thread_groups[group_index][rank_in_group];
        success = res == onecclSuccess &&
                  std::all_of(host_recv.begin(), host_recv.end(),
                              [expected](int val) { return val == expected; });

        if (print_results || !success) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[Scatter][thread " << global_thread_id << "] group "
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
        sycl::free(recv_buf, queue);
        if (send_buf != nullptr) {
            sycl::free(send_buf, queue);
        }
        return success;
    }

    static bool run_reduce_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;

        const int root = 0;
        bool success = true;
        int *send_buf = sycl::malloc_device<int>(count, queue);
        // Only the root rank needs a recv buffer; the reduced result lands
        // there. Non-root ranks still allocate one so onecclReduce has a valid
        // pointer to pass through.
        int *recv_buf = nullptr;
        if (rank_in_group == root) {
            recv_buf = sycl::malloc_device<int>(count, queue);
        }
        // Each thread sends a unique base value: 5 * (thread_id + 1), plus the
        // element index so no two elements share a value. A per-index value
        // catches a broadcast-of-element-0 style bug that a uniform buffer
        // would hide.
        // E.g., thread 0 -> 5, 6, 7, ...; thread 1 -> 10, 11, 12, ...
        std::vector<int> host_send(count);
        for (int i = 0; i < count; ++i) {
            host_send[i] = (5 * (global_thread_id + 1)) + i;
        }
        queue.memcpy(send_buf, host_send.data(), count * sizeof(int)).wait();

        onecclResult_t const res =
            onecclReduce(send_buf, recv_buf, count, onecclInt32, onecclSum,
                         root, comm, &queue);
        if (res != onecclSuccess) {
            std::cout << "onecclReduce collective call failed with error "
                      << res << '\n';
            return false;
        }
        queue.wait();
        success = res == onecclSuccess;

        // The result is only defined on the root rank; non-root ranks succeed
        // as long as the call returned onecclSuccess.
        if (rank_in_group != root) {
            queue.wait();
            sycl::free(send_buf, queue);
            sycl::free(recv_buf, queue);
            return success;
        } else {
            std::vector<int> host_recv(count);
            queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int))
                .wait();

            // Expected at element i: sum over the group of
            // (5 * (gid + 1) + i) == 5 * group_sum + group_size * i.
            // For group {0, 1}: sum = (0+1) + (1+1) = 3, base = 15
            int const group_sum = [&thread_groups, group_index]() {
                return std::accumulate(
                    thread_groups[group_index].begin(),
                    thread_groups[group_index].end(), 0,
                    [](int acc, int gid) { return acc + (gid + 1); });
            }();
            int const expected_base = 5 * group_sum;
            for (int i = 0; i < count; ++i) {
                int const expected = expected_base + (group_size * i);
                if (host_recv[i] != expected) {
                    success = false;
                    break;
                }
            }

            if (print_results) {
                const std::scoped_lock lock(print_mutex);
                std::cout << "[Reduce][thread " << global_thread_id
                          << "] group " << group_index << " rank_in_group "
                          << rank_in_group << " (root) recv:";
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

        queue.wait();

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        return success;
    }

    static bool run_alltoall_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;

        bool success = true;
        int *send_buf = sycl::malloc_device<int>(
            static_cast<long>(count * group_size), queue);
        int *recv_buf = sycl::malloc_device<int>(
            static_cast<long>(count * group_size), queue);

        // Fill the block destined for rank dest with a value tagged by both the
        // sender's global thread id and the destination rank, so the receiver
        // can verify it got exactly what the sender addressed to it.
        // Value sent to dest = (global_thread_id + 1) * 100 + dest
        std::vector<int> host_send(static_cast<size_t>(count * group_size), 0);
        for (int dest = 0; dest < group_size; ++dest) {
            const int val = ((global_thread_id + 1) * 100) + dest;
            for (int col = 0; col < count; ++col) {
                host_send[(static_cast<size_t>(dest) * count) + col] = val;
            }
        }
        queue
            .memcpy(send_buf, host_send.data(),
                    static_cast<size_t>(count * group_size) * sizeof(int))
            .wait();

        onecclResult_t const res = onecclAllToAll(send_buf, recv_buf, count,
                                                  onecclInt32, comm, &queue);
        if (res != onecclSuccess) {
            std::cout << "onecclAllToAll collective call failed with error "
                      << res << '\n';
            return false;
        }
        queue.wait();
        std::vector<int> host_recv(static_cast<size_t>(count * group_size));
        queue
            .memcpy(host_recv.data(), recv_buf,
                    static_cast<size_t>(count * group_size) * sizeof(int))
            .wait();

        // The block received from source rank src is the block that src
        // addressed to my rank_in_group:
        // (src_global_id + 1) * 100 + rank_in_group
        success = res == onecclSuccess;
        for (int src = 0; src < group_size; ++src) {
            const int expected =
                ((thread_groups[group_index][src] + 1) * 100) + rank_in_group;
            for (int col = 0; col < count; ++col) {
                if (host_recv[(src * count) + col] != expected) {
                    success = false;
                    break;
                }
            }
        }

        if (print_results || !success) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[AllToAll][thread " << global_thread_id << "] group "
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

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        return success;
    }

    static bool run_broadcast_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const auto &thread_groups = ctx.thread_groups;

        const int root = 0;
        bool success = true;
        // Only the root rank provides real send data; non-root ranks pass
        // nullptr to exercise broadcast's root-only send_buf contract.
        int *send_buf = nullptr;

        // The value every rank should end up with is tagged by the root's
        // global thread id so it is distinguishable per group.
        const int expected = 100 + thread_groups[group_index][root];

        if (rank_in_group == root) {
            send_buf = sycl::malloc_device<int>(count, queue);
            std::vector<int> host_send(count, expected);
            queue.memcpy(send_buf, host_send.data(), count * sizeof(int))
                .wait();
        }

        int *recv_buf = sycl::malloc_device<int>(count, queue);

        onecclResult_t const res = onecclBroadcast(
            send_buf, recv_buf, count, onecclInt32, root, comm, &queue);
        if (res != onecclSuccess) {
            std::cout << "onecclBroadcast collective call failed with error "
                      << res << '\n';
            return false;
        }
        queue.wait();

        std::vector<int> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int)).wait();

        success = res == onecclSuccess &&
                  std::all_of(host_recv.begin(), host_recv.end(),
                              [expected](int val) { return val == expected; });

        if (print_results || !success) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[Broadcast][thread " << global_thread_id << "] group "
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

        sycl::free(recv_buf, queue);
        if (send_buf != nullptr) {
            sycl::free(send_buf, queue);
        }
        return success;
    }

    static bool run_pt2pt_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;

        // A single-rank group has no peer to exchange with; nothing to do.
        bool success = true;
        if (group_size < 2) {
            if (print_results) {
                const std::scoped_lock lock(print_mutex);
                std::cout << "[Pt2Pt][thread " << global_thread_id << "] group "
                          << group_index << " rank_in_group " << rank_in_group
                          << " (single-rank group, skipped)\n";
            }
            return success;
        }

        // Paired exchange: even ranks (0, 2, 4, ...) send to their odd
        // right neighbor, and odd ranks (1, 3, 5, ...) receive from their
        // even left neighbor. The sender does not validate; only the
        // receiver checks recv_buf.
        const bool is_sender = (rank_in_group % 2 == 0);
        const int peer_rank = is_sender ? rank_in_group + 1 : rank_in_group - 1;

        // An even rank at the end of an odd-sized group has no odd peer to
        // send to; nothing to do for it.
        if (is_sender && peer_rank >= group_size) {
            success = true;
            if (print_results) {
                const std::scoped_lock lock(print_mutex);
                std::cout << "[Pt2Pt][thread " << global_thread_id << "] group "
                          << group_index << " rank_in_group " << rank_in_group
                          << " (unpaired sender, skipped)\n";
            }
            return success;
        }

        if (is_sender) {
            int *send_buf = sycl::malloc_device<int>(count, queue);
            // Each sender sends a unique value tagged by its global thread id:
            // (global_thread_id + 1) * 1000
            const int send_value = (global_thread_id + 1) * 1000;
            std::vector<int> host_send(count, send_value);
            queue.memcpy(send_buf, host_send.data(), count * sizeof(int))
                .wait();

            onecclResult_t res = onecclSend(send_buf, count, onecclInt32,
                                            peer_rank, comm, &queue);
            if (res != onecclSuccess) {
                std::cout << "onecclSend collective call failed with error "
                          << res << '\n';
                return false;
            }
            // The send copy is asynchronous; it must complete before send_buf
            // goes away, otherwise the device reads freed memory.
            queue.wait();

            // Senders do not validate the payload, only the API result.
            success = res == onecclSuccess;

            if (print_results) {
                const std::scoped_lock lock(print_mutex);
                std::cout << "[Pt2Pt][thread " << global_thread_id << "] group "
                          << group_index << " rank_in_group " << rank_in_group
                          << " send to " << peer_rank << " value " << send_value
                          << '\n';
            }
            sycl::free(send_buf, queue);
        } else {
            int *recv_buf = sycl::malloc_device<int>(count, queue);
            onecclResult_t res = onecclRecv(recv_buf, count, onecclInt32,
                                            peer_rank, comm, &queue);
            if (res != onecclSuccess) {
                std::cout << "onecclRecv collective call failed with error "
                          << res << '\n';
                return false;
            }
            queue.wait();
            std::vector<int> host_recv(count);
            queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int))
                .wait();

            // The data received comes from the even peer, so it carries that
            // peer's global-thread-id tag.
            const int expected =
                (thread_groups[group_index][peer_rank] + 1) * 1000;
            success =
                res == onecclSuccess &&
                std::all_of(host_recv.begin(), host_recv.end(),
                            [expected](int val) { return val == expected; });

            if (print_results || !success) {
                const std::scoped_lock lock(print_mutex);
                std::cout << "[Pt2Pt][thread " << global_thread_id << "] group "
                          << group_index << " rank_in_group " << rank_in_group
                          << " recv from " << peer_rank << ":";
                const int to_print = std::min(count, 8);
                for (int i = 0; i < to_print; ++i) {
                    std::cout << " " << host_recv[i];
                }
                if (count > 8) {
                    std::cout << " ...";
                }
                std::cout << '\n';
            }
            sycl::free(recv_buf, queue);
        }
        return success;
    }

    static bool run_pt2pt_group_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;

        bool success = true;
        if (group_size < 2) {
            if (print_results) {
                const std::scoped_lock lock(print_mutex);
                std::cout << "[Pt2PtGroup][thread " << global_thread_id
                          << "] group " << group_index << " rank_in_group "
                          << rank_in_group << " (single-rank group, skipped)\n";
            }
            return success;
        }
        std::uint32_t *send_buf =
            sycl::malloc_device<std::uint32_t>(count, queue);
        std::uint32_t *recv_buf =
            sycl::malloc_device<std::uint32_t>(count, queue);

        const int send_value = (global_thread_id + 1) * 1000;
        const bool is_even_rank = (rank_in_group % 2 == 0);
        const int peer_rank =
            is_even_rank ? rank_in_group + 1 : rank_in_group - 1;
        std::vector<int> host_send(count, send_value);
        queue.memcpy(send_buf, host_send.data(), count * sizeof(std::uint32_t))
            .wait();

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[Pt2PtGroup][thread " << global_thread_id
                      << "] group " << group_index << " rank_in_group "
                      << rank_in_group << " send to " << peer_rank << " value "
                      << send_value << '\n';
        }

        onecclGroupStart();
        onecclResult_t const send_res =
            onecclSend(send_buf, count, onecclUint32, peer_rank, comm, &queue);
        onecclResult_t const recv_res =
            onecclRecv(recv_buf, count, onecclUint32, peer_rank, comm, &queue);
        onecclResult_t const group_res = onecclGroupEnd();
        bool res = (send_res == onecclSuccess) && (recv_res == onecclSuccess) &&
                   (group_res == onecclSuccess);
        if (!res) {
            std::cout << "One of Group Point-to-Point collective call failed "
                         "with error "
                      << res << '\n';
            return false;
        }
        queue.wait();

        std::vector<std::uint32_t> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(std::uint32_t))
            .wait();

        // Each rank receives its peer's buffer, so the expected value carries
        // the peer's global-thread-id tag, not this thread's.
        const int expected = (thread_groups[group_index][peer_rank] + 1) * 1000;
        success =
            res &&
            std::all_of(host_recv.begin(), host_recv.end(),
                        [expected](std::uint32_t val) {
                            return val == static_cast<std::uint32_t>(expected);
                        });

        if (print_results || !success) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[Pt2PtGroup][thread " << global_thread_id
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

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        return success;
    }

    // Same paired send+recv exchange as run_pt2pt_group_test, but wrapped in
    // two levels of onecclGroupStart/End to exercise nested group semantics.
    static bool run_pt2pt_nested_group_test(const CollectiveRunContext &ctx) {
        sycl::queue &queue = ctx.queue;
        const onecclComm_t comm = ctx.comm;
        const int count = ctx.count;
        const int global_thread_id = ctx.global_thread_id;
        const int group_index = ctx.group_index;
        const int rank_in_group = ctx.rank_in_group;
        const int group_size = ctx.group_size;
        const auto &thread_groups = ctx.thread_groups;

        bool success = true;
        if (group_size < 2) {
            if (print_results) {
                const std::scoped_lock lock(print_mutex);
                std::cout << "[Pt2PtNestedGroup][thread " << global_thread_id
                          << "] group " << group_index << " rank_in_group "
                          << rank_in_group << " (single-rank group, skipped)\n";
            }
            return success;
        }
        std::uint32_t *send_buf =
            sycl::malloc_device<std::uint32_t>(count, queue);
        std::uint32_t *recv_buf =
            sycl::malloc_device<std::uint32_t>(count, queue);

        const int send_value = (global_thread_id + 1) * 1000;
        const bool is_even_rank = (rank_in_group % 2 == 0);
        const int peer_rank =
            is_even_rank ? rank_in_group + 1 : rank_in_group - 1;
        std::vector<int> host_send(count, send_value);
        queue.memcpy(send_buf, host_send.data(), count * sizeof(std::uint32_t))
            .wait();

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[Pt2PtNestedGroup][thread " << global_thread_id
                      << "] group " << group_index << " rank_in_group "
                      << rank_in_group << " send to " << peer_rank << " value "
                      << send_value << '\n';
        }

        onecclGroupStart();
        onecclGroupStart();
        onecclResult_t const send_res =
            onecclSend(send_buf, count, onecclUint32, peer_rank, comm, &queue);
        onecclResult_t const recv_res =
            onecclRecv(recv_buf, count, onecclUint32, peer_rank, comm, &queue);
        onecclResult_t const inner_group_res = onecclGroupEnd();
        onecclResult_t const outer_group_res = onecclGroupEnd();
        bool res = (send_res == onecclSuccess) && (recv_res == onecclSuccess) &&
                   (inner_group_res == onecclSuccess) &&
                   (outer_group_res == onecclSuccess);
        if (!res) {
            std::cout << "One of Nested Group Point-to-Point collective call "
                         "failed with error "
                      << res << '\n';
            return false;
        }
        queue.wait();

        std::vector<std::uint32_t> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(std::uint32_t))
            .wait();

        const int expected = (thread_groups[group_index][peer_rank] + 1) * 1000;
        success =
            res &&
            std::all_of(host_recv.begin(), host_recv.end(),
                        [expected](std::uint32_t val) {
                            return val == static_cast<std::uint32_t>(expected);
                        });

        if (print_results || !success) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[Pt2PtNestedGroup][thread " << global_thread_id
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

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        return success;
    }

    // Collective -> test runner. Add a collective by adding its enumerator and
    // its runner here; no dispatch code to touch.
    static const std::map<CollectiveType, CollectiveRunner> &
    collective_runners() {
        static const std::map<CollectiveType, CollectiveRunner> runners = {
            {CollectiveType::ONECCL_ALLREDUCE, &run_allreduce_test},
            {CollectiveType::ONECCL_ALLGATHER, &run_allgather_test},
            {CollectiveType::ONECCL_REDUCE_SCATTER, &run_reduce_scatter_test},
            {CollectiveType::ONECCL_SCATTER, &run_scatter_test},
            {CollectiveType::ONECCL_REDUCE, &run_reduce_test},
            {CollectiveType::ONECCL_BROADCAST, &run_broadcast_test},
            {CollectiveType::ONECCL_ALLTOALL, &run_alltoall_test},
            {CollectiveType::ONECCL_PT2PT, &run_pt2pt_test},
            {CollectiveType::ONECCL_PT2PT_GROUP, &run_pt2pt_group_test},
            {CollectiveType::ONECCL_PT2PT_NESTED_GROUP,
             &run_pt2pt_nested_group_test}};
        return runners;
    }

    static void thread_func(onecclThreadParams params,
                            std::vector<std::uint8_t> &thread_success,
                            int idx) {
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

        // Test onecclCommCount
        int comm_size = -1;
        onecclResult_t const count_res = onecclCommCount(comm, &comm_size);
        ASSERT_EQ(count_res, onecclSuccess);
        ASSERT_EQ(comm_size, group_size);

        const CollectiveRunContext ctx{queue,
                                       comm,
                                       count,
                                       global_thread_id,
                                       group_index,
                                       rank_in_group,
                                       group_size,
                                       *params.thread_groups,
                                       params.sync_counter,
                                       params.num_iterations};

        // The multi-iteration variant is a separate runner for the same
        // collective, so it is selected here rather than from the table.
        CollectiveRunner runner = nullptr;
        if (params.multi_iteration &&
            params.coll_type == CollectiveType::ONECCL_ALLREDUCE) {
            runner = &run_allreduce_multi_test;
        } else {
            const auto &runners = collective_runners();
            auto entry = runners.find(params.coll_type);
            if (entry != runners.end()) {
                runner = entry->second;
            }
        }

        const bool success = runner != nullptr && runner(ctx);

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
        std::vector<std::uint8_t> thread_success(threads_num, false);
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
        std::vector<std::uint8_t> thread_success(threads_num, false);
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

    // Thread function for CommSplit test
    static void
    comm_split_thread_func(int global_thread_id, int threads_num, int count,
                           const onecclUniqueId &group_id,
                           std::vector<std::uint8_t> &thread_success, int idx) {
        // Create sycl queue for this thread
        sycl::queue queue = create_sycl_queue(global_thread_id);
        auto result = onecclSetDevice(global_thread_id);
        ASSERT_EQ(result, onecclSuccess);

        // Create initial communicator with all threads
        onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
        config.multiThreaded = 1;

        onecclComm_t comm = nullptr;
        result = onecclCommInitRankConfig(&comm, threads_num, group_id,
                                          global_thread_id, &config);
        ASSERT_EQ(result, onecclSuccess);

        // Verify initial comm size
        int comm_size = -1;
        result = onecclCommCount(comm, &comm_size);
        ASSERT_EQ(result, onecclSuccess);
        ASSERT_EQ(comm_size, threads_num);

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[CommSplit][thread " << global_thread_id
                      << "] Initial comm size: " << comm_size << '\n';
        }

        // Split: even threads (0,2) vs odd threads (1,3)
        int const color = global_thread_id % 2; // 0 for even, 1 for odd
        int const key = global_thread_id;

        onecclComm_t split_comm = nullptr;
        onecclConfig_t split_config = ONECCL_CONFIG_INITIALIZER;
        split_config.multiThreaded = 1;

        result = onecclCommSplit(comm, color, key, &split_comm, &split_config);
        ASSERT_EQ(result, onecclSuccess);

        // Verify split comm size (should be 2)
        int split_comm_size = -1;
        result = onecclCommCount(split_comm, &split_comm_size);
        ASSERT_EQ(result, onecclSuccess);
        ASSERT_EQ(split_comm_size, 2);

        // Get rank in split comm
        int split_rank = -1;
        result = onecclCommUserRank(split_comm, &split_rank);
        ASSERT_EQ(result, onecclSuccess);

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[CommSplit][thread " << global_thread_id
                      << "] Split comm size: " << split_comm_size
                      << ", split rank: " << split_rank << ", color: " << color
                      << '\n';
        }

        // Verify rank is correct (even threads: 0→rank0, 2→rank1; odd: 1→rank0,
        // 3→rank1)
        int expected_split_rank = (color == 0) ? (global_thread_id / 2)
                                               : ((global_thread_id - 1) / 2);
        ASSERT_EQ(split_rank, expected_split_rank);

        // Run allreduce on split communicator
        int *send_buf = sycl::malloc_device<int>(count, queue);
        int *recv_buf = sycl::malloc_device<int>(count, queue);

        // Each thread sends: (global_thread_id + 1) * 100
        std::vector<int> host_send(count, (global_thread_id + 1) * 100);
        queue.memcpy(send_buf, host_send.data(), count * sizeof(int)).wait();

        result = onecclAllReduce(send_buf, recv_buf, count, onecclInt32,
                                 onecclSum, split_comm, &queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<int> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int)).wait();

        // Calculate expected result based on color (which group this thread is
        // in)
        int expected = 0;
        if (color == 0) {
            // Even group: threads 0 and 2
            expected = (0 + 1) * 100 + (2 + 1) * 100; // 100 + 300 = 400
        } else {
            // Odd group: threads 1 and 3
            expected = (1 + 1) * 100 + (3 + 1) * 100; // 200 + 400 = 600
        }

        bool success =
            std::all_of(host_recv.begin(), host_recv.end(),
                        [expected](int val) { return val == expected; });
        ASSERT_TRUE(success);

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        onecclCommDestroy(split_comm);
        onecclCommDestroy(comm);

        thread_success[idx] = true;
    }

    // Variant of comm_split_thread_func that exercises onecclAllGather on the
    // split sub-comm. Reproducer for MLSL-4790.
    static void
    comm_split_allgather_thread_func(int global_thread_id, int threads_num,
                                     int count, const onecclUniqueId &group_id,
                                     std::vector<std::uint8_t> &thread_success,
                                     int idx) {
        // Create sycl queue for this thread
        sycl::queue queue = create_sycl_queue(global_thread_id);
        auto result = onecclSetDevice(global_thread_id);
        ASSERT_EQ(result, onecclSuccess);

        // Create initial communicator with all threads
        onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
        config.multiThreaded = 1;

        onecclComm_t comm = nullptr;
        result = onecclCommInitRankConfig(&comm, threads_num, group_id,
                                          global_thread_id, &config);
        ASSERT_EQ(result, onecclSuccess);

        // Verify initial comm size
        int comm_size = -1;
        result = onecclCommCount(comm, &comm_size);
        ASSERT_EQ(result, onecclSuccess);
        ASSERT_EQ(comm_size, threads_num);

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[CommSplit][thread " << global_thread_id
                      << "] Initial comm size: " << comm_size << '\n';
        }

        // Split: even threads (0,2) vs odd threads (1,3)
        int const color = global_thread_id % 2; // 0 for even, 1 for odd
        int const key = global_thread_id;

        onecclComm_t split_comm = nullptr;
        onecclConfig_t split_config = ONECCL_CONFIG_INITIALIZER;
        split_config.multiThreaded = 1;

        result = onecclCommSplit(comm, color, key, &split_comm, &split_config);
        ASSERT_EQ(result, onecclSuccess);

        // Verify split comm size (should be 2)
        int split_comm_size = -1;
        result = onecclCommCount(split_comm, &split_comm_size);
        ASSERT_EQ(result, onecclSuccess);
        ASSERT_EQ(split_comm_size, 2);

        // Get rank in split comm
        int split_rank = -1;
        result = onecclCommUserRank(split_comm, &split_rank);
        ASSERT_EQ(result, onecclSuccess);

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[CommSplit][thread " << global_thread_id
                      << "] Split comm size: " << split_comm_size
                      << ", split rank: " << split_rank << ", color: " << color
                      << '\n';
        }

        // Verify rank is correct (even threads: 0→rank0, 2→rank1; odd: 1→rank0,
        // 3→rank1)
        int expected_split_rank = (color == 0) ? (global_thread_id / 2)
                                               : ((global_thread_id - 1) / 2);
        ASSERT_EQ(split_rank, expected_split_rank);

        // Run allgather on split communicator
        int *send_buf = sycl::malloc_device<int>(count, queue);
        int *recv_buf =
            sycl::malloc_device<int>(count * split_comm_size, queue);

        // Each thread sends: (global_thread_id + 1) * 100
        std::vector<int> host_send(count, (global_thread_id + 1) * 100);
        queue.memcpy(send_buf, host_send.data(), count * sizeof(int)).wait();

        result = onecclAllGather(send_buf, recv_buf, count, onecclInt32,
                                 split_comm, &queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<int> host_recv(count * split_comm_size);
        queue
            .memcpy(host_recv.data(), recv_buf,
                    count * split_comm_size * sizeof(int))
            .wait();

        // Calculate expected result based on color (which group this thread is
        // in)
        // int expected = 0;
        if (color == 0) {
            // Even group: threads 0 and 2
            // expected = (0 + 1) * 100 + (2 + 1) * 100; // 100 + 300 = 400
        } else {
            // Odd group: threads 1 and 3
            // expected = (1 + 1) * 100 + (3 + 1) * 100; // 200 + 400 = 600
        }

        // bool success =
        // std::all_of(host_recv.begin(), host_recv.end(),
        //[expected](int val) { return val == expected; });
        // ASSERT_TRUE(success);

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        onecclCommDestroy(split_comm);
        onecclCommDestroy(comm);

        thread_success[idx] = true;
    }

    // Thread function for CommSplit test: 8 ranks split into 3 comms
    static void
    comm_split_three_way_thread_func(int global_thread_id, int threads_num,
                                     int count, const onecclUniqueId &group_id,
                                     std::vector<std::uint8_t> &thread_success,
                                     int idx) {
        sycl::queue queue = create_sycl_queue(global_thread_id);
        auto result = onecclSetDevice(global_thread_id);
        ASSERT_EQ(result, onecclSuccess);

        onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
        config.multiThreaded = 1;

        onecclComm_t comm = nullptr;
        result = onecclCommInitRankConfig(&comm, threads_num, group_id,
                                          global_thread_id, &config);
        ASSERT_EQ(result, onecclSuccess);

        // Split into 3 groups: color = thread_id % 3
        // Group 0: threads {0, 3, 6} (size 3)
        // Group 1: threads {1, 4, 7} (size 3)
        // Group 2: threads {2, 5}    (size 2)
        int const color = global_thread_id % 3;
        int const key = global_thread_id;

        onecclComm_t split_comm = nullptr;
        onecclConfig_t split_config = ONECCL_CONFIG_INITIALIZER;
        split_config.multiThreaded = 1;

        result = onecclCommSplit(comm, color, key, &split_comm, &split_config);
        ASSERT_EQ(result, onecclSuccess);

        int split_comm_size = -1;
        result = onecclCommCount(split_comm, &split_comm_size);
        ASSERT_EQ(result, onecclSuccess);

        int expected_size = (color < 2) ? 3 : 2;
        ASSERT_EQ(split_comm_size, expected_size);

        int split_rank = -1;
        result = onecclCommUserRank(split_comm, &split_rank);
        ASSERT_EQ(result, onecclSuccess);

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[CommSplit3Way][thread " << global_thread_id
                      << "] color=" << color
                      << " split_size=" << split_comm_size
                      << " split_rank=" << split_rank << '\n';
        }

        // Run allreduce on split communicator
        int *send_buf = sycl::malloc_device<int>(count, queue);
        int *recv_buf = sycl::malloc_device<int>(count, queue);

        std::vector<int> host_send(count, (global_thread_id + 1) * 10);
        queue.memcpy(send_buf, host_send.data(), count * sizeof(int)).wait();

        result = onecclAllReduce(send_buf, recv_buf, count, onecclInt32,
                                 onecclSum, split_comm, &queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<int> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int)).wait();

        // Calculate expected: sum of (thread_id + 1) * 10 for all threads in
        // this color group
        int expected = 0;
        for (int t = 0; t < threads_num; ++t) {
            if (t % 3 == color) {
                expected += (t + 1) * 10;
            }
        }

        bool success =
            std::all_of(host_recv.begin(), host_recv.end(),
                        [expected](int val) { return val == expected; });
        ASSERT_TRUE(success);

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        onecclCommDestroy(split_comm);
        onecclCommDestroy(comm);

        thread_success[idx] = true;
    }

    // Thread function for concurrent split test: two parent comms split at the
    // same time
    static void comm_split_concurrent_thread_func(
        int global_thread_id, int count, const onecclUniqueId &group_id_a,
        const onecclUniqueId &group_id_b,
        std::vector<std::uint8_t> &thread_success, int idx) {
        // Threads 0-3 belong to comm A, threads 4-7 belong to comm B
        bool const is_group_a = (global_thread_id < 4);
        int const local_id =
            is_group_a ? global_thread_id : (global_thread_id - 4);
        int const group_size = 4;

        sycl::queue queue = create_sycl_queue(global_thread_id);
        auto result = onecclSetDevice(global_thread_id);
        ASSERT_EQ(result, onecclSuccess);

        onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
        config.multiThreaded = 1;

        // Create parent comm (A or B depending on thread)
        onecclComm_t comm = nullptr;
        const onecclUniqueId &my_group_id =
            is_group_a ? group_id_a : group_id_b;
        result = onecclCommInitRankConfig(&comm, group_size, my_group_id,
                                          local_id, &config);
        ASSERT_EQ(result, onecclSuccess);

        // Split: even local_ids vs odd local_ids within each parent comm
        int const color = local_id % 2;
        int const key = local_id;

        onecclComm_t split_comm = nullptr;
        onecclConfig_t split_config = ONECCL_CONFIG_INITIALIZER;
        split_config.multiThreaded = 1;

        result = onecclCommSplit(comm, color, key, &split_comm, &split_config);
        ASSERT_EQ(result, onecclSuccess);

        int split_comm_size = -1;
        result = onecclCommCount(split_comm, &split_comm_size);
        ASSERT_EQ(result, onecclSuccess);
        ASSERT_EQ(split_comm_size, 2);

        int split_rank = -1;
        result = onecclCommUserRank(split_comm, &split_rank);
        ASSERT_EQ(result, onecclSuccess);

        if (print_results) {
            const std::scoped_lock lock(print_mutex);
            std::cout << "[CommSplitConcurrent][thread " << global_thread_id
                      << "] group=" << (is_group_a ? "A" : "B")
                      << " local_id=" << local_id << " color=" << color
                      << " split_rank=" << split_rank << '\n';
        }

        // Run allreduce on split communicator
        int *send_buf = sycl::malloc_device<int>(count, queue);
        int *recv_buf = sycl::malloc_device<int>(count, queue);

        // Use global_thread_id to ensure values are unique across all threads
        std::vector<int> host_send(count, (global_thread_id + 1) * 50);
        queue.memcpy(send_buf, host_send.data(), count * sizeof(int)).wait();

        result = onecclAllReduce(send_buf, recv_buf, count, onecclInt32,
                                 onecclSum, split_comm, &queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<int> host_recv(count);
        queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int)).wait();

        // Expected: sum of contributions from the 2 threads sharing this split
        // comm Group A even: threads 0, 2 → (1)*50 + (3)*50 = 200 Group A odd:
        // threads 1, 3 → (2)*50 + (4)*50 = 300 Group B even: threads 4, 6 →
        // (5)*50 + (7)*50 = 600 Group B odd:  threads 5, 7 → (6)*50 + (8)*50 =
        // 700
        int expected = 0;
        int base = is_group_a ? 0 : 4;
        for (int i = 0; i < group_size; ++i) {
            if (i % 2 == color) {
                expected += (base + i + 1) * 50;
            }
        }

        bool success =
            std::all_of(host_recv.begin(), host_recv.end(),
                        [expected](int val) { return val == expected; });
        ASSERT_TRUE(success);

        sycl::free(send_buf, queue);
        sycl::free(recv_buf, queue);
        onecclCommDestroy(split_comm);
        onecclCommDestroy(comm);

        thread_success[idx] = true;
    }

    // Thread function for CommSplit NOCOLOR test
    // 4 threads: only thread 0 gets color=0, threads 1-3 get
    // ONECCL_SPLIT_NOCOLOR Only thread 0 gets a valid split comm (size 1);
    // others get nullptr
    static std::atomic<int> nocolor_done_count;

    static void
    comm_split_nocolor_thread_func(int global_thread_id, int threads_num,
                                   int count, const onecclUniqueId &group_id,
                                   std::vector<std::uint8_t> &thread_success,
                                   int idx) {
        sycl::queue queue = create_sycl_queue(global_thread_id);
        auto result = onecclSetDevice(global_thread_id);
        ASSERT_EQ(result, onecclSuccess);

        onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
        config.multiThreaded = 1;

        onecclComm_t comm = nullptr;
        result = onecclCommInitRankConfig(&comm, threads_num, group_id,
                                          global_thread_id, &config);
        ASSERT_EQ(result, onecclSuccess);

        // Only thread 0 gets color=0; all others get NOCOLOR
        int const color = (global_thread_id == 0) ? 0 : ONECCL_SPLIT_NOCOLOR;

        onecclComm_t split_comm = nullptr;
        onecclConfig_t split_config = ONECCL_CONFIG_INITIALIZER;
        split_config.multiThreaded = 1;

        result = onecclCommSplit(comm, color, 0, &split_comm, &split_config);
        ASSERT_EQ(result, onecclSuccess);

        if (global_thread_id == 0) {
            ASSERT_NE(split_comm, nullptr);

            int split_comm_size = -1;
            result = onecclCommCount(split_comm, &split_comm_size);
            ASSERT_EQ(result, onecclSuccess);
            ASSERT_EQ(split_comm_size, 1);

            // Allreduce on size-1 comm: output == input
            int const expected_value = 123;
            int *send_buf = sycl::malloc_device<int>(count, queue);
            int *recv_buf = sycl::malloc_device<int>(count, queue);

            std::vector<int> host_send(count, expected_value);
            queue.memcpy(send_buf, host_send.data(), count * sizeof(int))
                .wait();

            result = onecclAllReduce(send_buf, recv_buf, count, onecclInt32,
                                     onecclSum, split_comm, &queue);
            ASSERT_EQ(result, onecclSuccess);

            std::vector<int> host_recv(count);
            queue.memcpy(host_recv.data(), recv_buf, count * sizeof(int))
                .wait();

            for (int val : host_recv) {
                ASSERT_EQ(val, expected_value);
            }

            sycl::free(send_buf, queue);
            sycl::free(recv_buf, queue);
            onecclCommDestroy(split_comm);
        } else {
            ASSERT_EQ(split_comm, nullptr);
        }

        // Wait for all threads before destroying parent comm — NOCOLOR threads
        // finish early and must not free parent GPU resources while active
        // threads are still using them.
        nocolor_done_count.fetch_add(1);
        while (nocolor_done_count.load() < threads_num) {
            std::this_thread::yield();
        }

        onecclCommDestroy(comm);
        thread_success[idx] = true;
    }
};

std::mutex SyclMtCollectivesUsmTest::print_mutex;
std::atomic<bool> SyclMtCollectivesUsmTest::print_results = false;
std::atomic<int> SyclMtCollectivesUsmTest::nocolor_done_count{0};

// 2 threads groupings
const std::vector<std::vector<std::vector<int>>> kTwoThreadGroupings = {
    {{0, 1}}, {{1}, {0}}};
// 4 threads groupings
const std::vector<std::vector<std::vector<int>>> kFourThreadGroupings = {
    {{0, 1, 2, 3}}, {{1, 3}, {0, 2}}};
// 8 threads groupings
const std::vector<std::vector<std::vector<int>>> kEightThreadGroupings = {
    {{0, 1, 2, 3, 4, 5, 6, 7}}, {{0, 2, 4, 6}, {1, 3, 5, 7}}};

#define TESTPARAMS_ALL_COUNTS(threads, desc, coll)                             \
    onecclTestParams{threads, 1024, desc, coll}, onecclTestParams {            \
        threads, 33554432, desc, coll                                          \
    }

// Skips the current test when fewer GPUs are available than it needs. This is a
// macro rather than a helper function because GTEST_SKIP() expands to a return:
// from a helper it would only leave the helper and the test would keep running.
// Usable from both TEST_P and TEST_F bodies of SyclMtCollectivesUsmTest.
#define SKIP_IF_NOT_ENOUGH_GPUS(needed_threads)                                \
    do {                                                                       \
        const int gpu_count = get_gpu_count();                                 \
        if (gpu_count < (needed_threads)) {                                    \
            GTEST_SKIP() << "Not enough GPUs: need " << (needed_threads)       \
                         << " but only " << gpu_count << " available";         \
        }                                                                      \
    } while (0)

// Checks every thread's success flag, reporting each failing thread index.
// Wrapped in do/while so it behaves like a single statement at the call site.
#define VERIFY_THREADS_SUCCESS(threads, threads_success)                       \
    do {                                                                       \
        for (auto &thread : (threads)) {                                       \
            thread.join();                                                     \
        }                                                                      \
        bool res = true;                                                       \
        for (size_t i = 0; i < (threads_success).size(); ++i) {                \
            EXPECT_TRUE((threads_success)[i])                                  \
                << "Result for thread_success[" << i << "] is false";          \
            res &= (threads_success)[i];                                       \
        }                                                                      \
        ASSERT_TRUE(res);                                                      \
    } while (0)

// Names each parameterized case after the collective and count it runs, so
// the gtest output/filter shows e.g.
// TwoThreadSingleGroup/SyclMtCollectivesUsmTest.MultiThreadedCollectivesGroupVariants/alltoall_count1024
struct CollectiveParamName {
    std::string
    operator()(const ::testing::TestParamInfo<onecclTestParams> &info) const {
        return std::string(collective_name(info.param.coll_type)) + "_count_" +
               std::to_string(info.param.count);
    }
};

// 2 threads, all threads in a single group
INSTANTIATE_TEST_SUITE_P(
    TwoThreadSingleGroup, SyclMtCollectivesUsmTest,
    ::testing::Values(
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_ALLREDUCE),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_ALLGATHER),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_REDUCE_SCATTER),
        // TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
        //                       CollectiveType::ONECCL_SCATTER),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_REDUCE),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_BROADCAST),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_ALLTOALL),
        // TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
        //                       CollectiveType::ONECCL_PT2PT),
        // TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
        //                       CollectiveType::ONECCL_PT2PT),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_PT2PT_GROUP),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[0],
                              CollectiveType::ONECCL_PT2PT_NESTED_GROUP)),
    CollectiveParamName());

// 4 threads, all threads in a single group
INSTANTIATE_TEST_SUITE_P(
    FourThreadSingleGroup, SyclMtCollectivesUsmTest,
    ::testing::Values(
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
                              CollectiveType::ONECCL_ALLREDUCE),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
                              CollectiveType::ONECCL_ALLGATHER),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
                              CollectiveType::ONECCL_REDUCE_SCATTER),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
                              CollectiveType::ONECCL_REDUCE),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
                              CollectiveType::ONECCL_BROADCAST),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
                              CollectiveType::ONECCL_ALLTOALL),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
        //                       CollectiveType::ONECCL_PT2PT),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
        //                       CollectiveType::ONECCL_PT2PT),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
                              CollectiveType::ONECCL_PT2PT_GROUP),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[0],
                              CollectiveType::ONECCL_PT2PT_NESTED_GROUP)),
    CollectiveParamName());

// 8 threads, all threads in a single group
INSTANTIATE_TEST_SUITE_P(
    EightThreadSingleGroup, SyclMtCollectivesUsmTest,
    ::testing::Values(
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
                              CollectiveType::ONECCL_ALLREDUCE),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
                              CollectiveType::ONECCL_ALLGATHER),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
                              CollectiveType::ONECCL_REDUCE_SCATTER),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
                              CollectiveType::ONECCL_REDUCE),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
                              CollectiveType::ONECCL_BROADCAST),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
                              CollectiveType::ONECCL_ALLTOALL),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
        //                       CollectiveType::ONECCL_PT2PT),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
        //                       CollectiveType::ONECCL_PT2PT),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
                              CollectiveType::ONECCL_PT2PT_GROUP),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[0],
                              CollectiveType::ONECCL_PT2PT_NESTED_GROUP)),
    CollectiveParamName());

// 2 threads, threads split into separate groups
INSTANTIATE_TEST_SUITE_P(
    TwoThreadSeparateGroups, SyclMtCollectivesUsmTest,
    ::testing::Values(
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_ALLREDUCE),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_ALLGATHER),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_REDUCE_SCATTER),
        // TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
        //                       CollectiveType::ONECCL_SCATTER),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_REDUCE),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_BROADCAST),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_ALLTOALL),
        // TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
        //                       CollectiveType::ONECCL_PT2PT),
        // TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
        //                       CollectiveType::ONECCL_PT2PT),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_PT2PT_GROUP),
        TESTPARAMS_ALL_COUNTS(2, kTwoThreadGroupings[1],
                              CollectiveType::ONECCL_PT2PT_NESTED_GROUP)),
    CollectiveParamName());

// 4 threads, threads split into separate groups
INSTANTIATE_TEST_SUITE_P(
    FourThreadSeparateGroups, SyclMtCollectivesUsmTest,
    ::testing::Values(
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
                              CollectiveType::ONECCL_ALLREDUCE),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
                              CollectiveType::ONECCL_ALLGATHER),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
                              CollectiveType::ONECCL_REDUCE_SCATTER),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
                              CollectiveType::ONECCL_REDUCE),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
                              CollectiveType::ONECCL_BROADCAST),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
                              CollectiveType::ONECCL_ALLTOALL),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
        //                       CollectiveType::ONECCL_PT2PT),
        // TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
        //                       CollectiveType::ONECCL_PT2PT),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
                              CollectiveType::ONECCL_PT2PT_GROUP),
        TESTPARAMS_ALL_COUNTS(4, kFourThreadGroupings[1],
                              CollectiveType::ONECCL_PT2PT_NESTED_GROUP)),
    CollectiveParamName());

// 8 threads, threads split into separate groups
INSTANTIATE_TEST_SUITE_P(
    EightThreadSeparateGroups, SyclMtCollectivesUsmTest,
    ::testing::Values(
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
                              CollectiveType::ONECCL_ALLREDUCE),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
                              CollectiveType::ONECCL_ALLGATHER),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
                              CollectiveType::ONECCL_REDUCE_SCATTER),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
                              CollectiveType::ONECCL_REDUCE),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
                              CollectiveType::ONECCL_BROADCAST),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
                              CollectiveType::ONECCL_ALLTOALL),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
        //                       CollectiveType::ONECCL_PT2PT),
        // TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
        //                       CollectiveType::ONECCL_PT2PT),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
                              CollectiveType::ONECCL_PT2PT_GROUP),
        TESTPARAMS_ALL_COUNTS(8, kEightThreadGroupings[1],
                              CollectiveType::ONECCL_PT2PT_NESTED_GROUP)),
    CollectiveParamName());

TEST_P(SyclMtCollectivesUsmTest, MultiThreadedCollectivesGroupVariants) {
    // Verifying thread count
    auto params = GetParam();
    SKIP_IF_NOT_ENOUGH_GPUS(params.threads_num);

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

TEST_F(SyclMtCollectivesUsmTest, MultiIterationAllreduceWithSynchronization) {
    // Test multiple iterations of allreduce with atomic counter
    // synchronization
    // This tests overlap and more complex scenarios as suggested in the
    // review
    constexpr int threads_num = 2;

    // Verifying thread count
    SKIP_IF_NOT_ENOUGH_GPUS(threads_num);

    // Test with 2 threads, 5 iterations
    run_multi_iteration_test_with_groups(kTwoThreadGroupings[0], threads_num,
                                         1024, CollectiveType::ONECCL_ALLREDUCE,
                                         5);

    // Test with separate groups
    run_multi_iteration_test_with_groups(kTwoThreadGroupings[1], threads_num,
                                         1024, CollectiveType::ONECCL_ALLREDUCE,
                                         5);
}

TEST_F(SyclMtCollectivesUsmTest,
       MultiIterationAllreduceWithSynchronizationLargeData) {
    constexpr int threads_num = 2;

    // Verifying thread count
    SKIP_IF_NOT_ENOUGH_GPUS(threads_num);

    // Test with larger data size to stress synchronization
    run_multi_iteration_test_with_groups(kTwoThreadGroupings[0], threads_num,
                                         1048576,
                                         CollectiveType::ONECCL_ALLREDUCE, 3);
}

TEST_F(SyclMtCollectivesUsmTest, CommSplitTest) {
    // Test onecclCommSplit in MT context
    // Start with 4 threads in one communicator, then split into 2 sub-comms:
    // - Even threads (0, 2) → one sub-comm
    // - Odd threads (1, 3) → another sub-comm
    constexpr int threads_num = 4;
    constexpr int count = 1024;

    // Verifying thread count
    SKIP_IF_NOT_ENOUGH_GPUS(threads_num);

    // Get unique ID for the initial communicator
    onecclUniqueId group_id{};
    onecclGetUniqueId(&group_id);

    std::vector<std::thread> threads;
    std::vector<std::uint8_t> thread_success(threads_num, false);

    // Launch threads
    for (int thread_id = 0; thread_id < threads_num; ++thread_id) {
        threads.emplace_back(comm_split_thread_func, thread_id, threads_num,
                             count, std::cref(group_id),
                             std::ref(thread_success), thread_id);
    }

    // Verify all succeeded
    VERIFY_THREADS_SUCCESS(threads, thread_success);
}

TEST_F(SyclMtCollectivesUsmTest, CommSplitAllGatherTest) {
    // Regression for MLSL-4790: same 4-thread split shape as CommSplitTest, but
    // runs onecclAllGather on the split sub-comm instead of allreduce. The MT
    // split path left r2r_comm->is_uniform=false, which routed the SYCL
    // allgather dispatch into a scaleout path that dereferenced a null atl_comm
    // on the sub-comm.
    constexpr int threads_num = 4;
    constexpr int count = 1024;

    SKIP_IF_NOT_ENOUGH_GPUS(threads_num);

    onecclUniqueId group_id{};
    onecclGetUniqueId(&group_id);

    std::vector<std::thread> threads;
    std::vector<std::uint8_t> thread_success(threads_num, false);

    for (int thread_id = 0; thread_id < threads_num; ++thread_id) {
        threads.emplace_back(comm_split_allgather_thread_func, thread_id,
                             threads_num, count, std::cref(group_id),
                             std::ref(thread_success), thread_id);
    }

    VERIFY_THREADS_SUCCESS(threads, thread_success);
}

TEST_F(SyclMtCollectivesUsmTest, CommSplitThreeWayTest) {
    // Test onecclCommSplit: 8 ranks split into 3 sub-comms:
    // - Color 0: threads {0, 3, 6} (size 3)
    // - Color 1: threads {1, 4, 7} (size 3)
    // - Color 2: threads {2, 5}    (size 2)
    constexpr int threads_num = 8;
    constexpr int count = 1024;

    // Verifying thread count
    SKIP_IF_NOT_ENOUGH_GPUS(threads_num);

    onecclUniqueId group_id{};
    onecclGetUniqueId(&group_id);

    std::vector<std::thread> threads;
    std::vector<std::uint8_t> thread_success(threads_num, false);

    for (int thread_id = 0; thread_id < threads_num; ++thread_id) {
        threads.emplace_back(comm_split_three_way_thread_func, thread_id,
                             threads_num, count, std::cref(group_id),
                             std::ref(thread_success), thread_id);
    }

    VERIFY_THREADS_SUCCESS(threads, thread_success);
}

TEST_F(SyclMtCollectivesUsmTest, CommSplitConcurrentTest) {
    // Test two independent parent comms splitting at the same time.
    // Comm A (threads 0-3) and Comm B (threads 4-7) both split concurrently.
    // Each parent splits even/odd into sub-comms of size 2.
    // This validates that global_current_id allocation is collision-free
    // when multiple comms perform split simultaneously.
    constexpr int threads_num = 8;
    constexpr int count = 1024;

    // Verifying thread count
    SKIP_IF_NOT_ENOUGH_GPUS(threads_num);

    onecclUniqueId group_id_a{};
    onecclUniqueId group_id_b{};
    onecclGetUniqueId(&group_id_a);
    onecclGetUniqueId(&group_id_b);

    std::vector<std::thread> threads;
    std::vector<std::uint8_t> thread_success(threads_num, false);

    for (int thread_id = 0; thread_id < threads_num; ++thread_id) {
        threads.emplace_back(comm_split_concurrent_thread_func, thread_id,
                             count, std::cref(group_id_a),
                             std::cref(group_id_b), std::ref(thread_success),
                             thread_id);
    }

    VERIFY_THREADS_SUCCESS(threads, thread_success);
}

TEST_F(SyclMtCollectivesUsmTest, CommSplitNoColorTest) {
    // Test ONECCL_SPLIT_NOCOLOR: 4 threads, threads 0,1 get color=0,
    // threads 2,3 pass NOCOLOR and should receive nullptr back.
    // Only threads 0,1 run allreduce on the split comm.
    constexpr int threads_num = 4;
    constexpr int count = 1024;

    // Verifying thread count
    SKIP_IF_NOT_ENOUGH_GPUS(threads_num);

    onecclUniqueId group_id{};
    onecclGetUniqueId(&group_id);

    std::vector<std::thread> threads;
    std::vector<std::uint8_t> thread_success(threads_num, false);

    for (int thread_id = 0; thread_id < threads_num; ++thread_id) {
        threads.emplace_back(comm_split_nocolor_thread_func, thread_id,
                             threads_num, count, std::cref(group_id),
                             std::ref(thread_success), thread_id);
    }

    VERIFY_THREADS_SUCCESS(threads, thread_success);
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

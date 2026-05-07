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
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#define setenv(x, y, z) _putenv_s(x, y)
#endif

namespace {

class OneCCLTest : public ::testing::Test {
  protected:
    const size_t num_threads_ = 4;
    const size_t iter_count_ = 1;
    std::vector<std::thread> threads_;
    std::vector<int> results_;
    std::mutex events_lock_;
    std::mutex counter_mutex_;
    std::condition_variable counter_cv_;
    int allreduce_counter_ = 0;

    [[nodiscard]] int oneccl_thread_run() const {
        int const rank = 0;
        int const world_size = 1;
        onecclComm_t comm = nullptr;
        onecclResult_t result = onecclSuccess;
        onecclUniqueId uid;

        setenv("CCL_PLUGIN", "ONECCL_NULL", 0);

        onecclGetUniqueId(&uid);
        result = onecclCommInitRank(&comm, world_size, uid, rank);
        if (result != onecclSuccess) {
            std::cerr << "Failed to initialize communicator.\n";
            return -1;
        }

        std::vector<int> data = {1, 2, 3, 4};
        std::vector<int> recv_data(data.size());

        for (int iter = 0; iter < iter_count_; iter++) {
            result = onecclAllReduce(data.data(), recv_data.data(), data.size(),
                                     onecclInt, onecclSum, comm, nullptr);
            if (result != onecclSuccess) {
                std::cerr << "AllReduce operation failed.\n";
                return -1;
            }
        }

        result = onecclCommDestroy(comm);
        if (result != onecclSuccess) {
            std::cerr << "Failed to destroy communicator.\n";
            return -1;
        }

        return 0;
    }
};
} // namespace

TEST_F(OneCCLTest, ConcurrentAllreduces) {
    results_.resize(num_threads_, 0);

    for (size_t i = 0; i < num_threads_; ++i) {
        threads_.emplace_back(
            [this, i]() { results_[i] = oneccl_thread_run(); });
    }

    for (auto &thread : threads_) {
        thread.join();
    }

    for (const auto &result : results_) {
        EXPECT_EQ(result, 0) << "Error during AllReduce operation.";
    }
}

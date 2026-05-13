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

#include "null_hook.hpp"
#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"

#include <array>
#include <atomic>
#include <gtest/gtest.h>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define setenv(x, y, z) _putenv_s(x, y)
#endif

namespace {

// Test fixture for the oneCCL API
class NullPluginTest : public ::testing::Test {
  protected:
    void SetUp() override {}

    void TearDown() override {}
};

// Helper function to add a hook that throws an exception
onecclResult_t add_exception_throwing_hook(
    onecclHookPoint_t hookPoint,
    const std::string &exceptionMessage = "Test Exception") {
    ADD_HOOK(hookPoint, ONECCL_ONE_TIME, [exceptionMessage](onecclHookPoint_t) {
        throw std::runtime_error(exceptionMessage);
        return std::optional<onecclResult_t>();
    });
    return onecclSuccess;
}

void print_backtrace() {
    // TODO: add util function working on Windows
}

// Utility function to test exception handling
void test_with_exception(onecclHookPoint_t hookPoint,
                         const std::function<onecclResult_t()> &funcToTest,
                         onecclResult_t expectedExceptionResult,
                         onecclResult_t expectedSuccessResult) {
    // Expect the function to throw
    add_exception_throwing_hook(hookPoint);
    EXPECT_EQ(funcToTest(), expectedExceptionResult);

    // Expect the function to succeed
    EXPECT_EQ(funcToTest(), expectedSuccessResult);
    std::cout << "Done\n\n";
}

} // namespace

// Test for handling exceptions in onecclGetRank
TEST_F(NullPluginTest, CommExceptions) {
    int rank = -1;
    int size = -1;
    int device = -1;
    int const magic_number = 21123;
    constexpr size_t kCount = 1024;
    onecclResult_t const result = onecclSuccess;
    onecclComm_t comm_xpu = nullptr;
    onecclComm_t comm_cpu = nullptr;
    onecclUniqueId uid{};
    onecclRedOp_t custom_op{};
    int custom_scalar = 1;
    void *buffer = nullptr;
    void *reg_handle = nullptr;
    onecclWindow_t window_handle = nullptr;

    std::array<int, kCount> sendbuff{};
    std::array<int, kCount> recvbuff{};

    sendbuff.fill(magic_number);

    setenv("CCL_PLUGIN", "ONECCL_NULL", 0);
    test_with_exception(
        ONECCL_BEFORE_UNIQUE_ID, [&]() { return onecclGetUniqueId(&uid); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_SET_DEVICE, [&]() { return onecclSetDevice(0); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_MEM_ALLOC,
        [&]() { return onecclMemAlloc(&buffer, kCount * sizeof(int)); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_INIT_COMM,
        [&]() { return onecclCommInitRank(&comm_xpu, 1, uid, 0); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_GET_RANK,
        [&]() { return onecclCommUserRank(comm_xpu, &rank); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_GET_SIZE,
        [&]() { return onecclCommCount(comm_xpu, &size); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_COMM_REGISTER,
        [&]() {
            return onecclCommRegister(comm_xpu, buffer, kCount * sizeof(int),
                                      &reg_handle);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_COMM_WINDOW_REGISTER,
        [&]() {
            return onecclCommWindowRegister(
                comm_xpu, buffer, kCount * sizeof(int), &window_handle,
                ONECCL_WINDOW_COLL_SYMMETRIC);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_GET_DEVICE,
        [&]() { return onecclCommDevice(comm_xpu, &device); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_GET_SIZE,
        [&]() { return onecclCommCount(comm_xpu, &size); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_ALLREDUCE,
        [&]() {
            return onecclAllReduce(sendbuff.data(), recvbuff.data(), kCount,
                                   onecclInt, onecclSum, comm_xpu, nullptr);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_ALLGATHER,
        [&]() {
            return onecclAllGather(sendbuff.data(), recvbuff.data(), kCount,
                                   onecclInt, comm_xpu, nullptr);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_BROADCAST,
        [&]() {
            return onecclBroadcast(sendbuff.data(), recvbuff.data(), kCount,
                                   onecclInt, 0, comm_xpu, nullptr);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_REDUCE,
        [&]() {
            return onecclReduce(sendbuff.data(), recvbuff.data(), kCount,
                                onecclInt, onecclSum, 0, comm_xpu, nullptr);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_REDUCE_SCATTER,
        [&]() {
            return onecclReduceScatter(sendbuff.data(), recvbuff.data(), kCount,
                                       onecclInt, onecclSum, comm_xpu, nullptr);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_SEND,
        [&]() {
            return onecclSend(sendbuff.data(), kCount, onecclInt, 1, comm_xpu,
                              nullptr);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_RECV,
        [&]() {
            return onecclRecv(recvbuff.data(), kCount, onecclInt, 0, comm_xpu,
                              nullptr);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_CREATE_PRE_MUL_SUM,
        [&]() {
            return onecclRedOpCreatePreMulSum(
                &custom_op, &custom_scalar, onecclInt,
                onecclScalarResidence_t::onecclScalarHostImmediate, comm_xpu);
        },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_REDUCTION_DESTROY,
        [&]() { return onecclRedOpDestroy(custom_op, comm_xpu); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_COMM_WINDOW_DEREGISTER,
        [&]() { return onecclCommWindowDeregister(comm_xpu, window_handle); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_COMM_DEREGISTER,
        [&]() { return onecclCommDeregister(comm_xpu, reg_handle); },
        onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_MEM_FREE, [&]() { return onecclMemFree(buffer); },
        onecclPluginException, onecclSuccess);

    // test_with_exception(
    //     ONECCL_BEFORE_FINALIZE, [&]() { return onecclCommFinalize(comm_xpu);
    //     }, onecclPluginException, onecclSuccess);

    // test_with_exception(
    //     ONECCL_BEFORE_ABORT, [&]() { return onecclCommAbort(comm_xpu); },
    //     onecclPluginException, onecclSuccess);

    test_with_exception(
        ONECCL_BEFORE_DESTROY_COMM,
        [&]() { return onecclCommDestroy(comm_xpu); }, onecclPluginException,
        onecclSuccess);
}

TEST_F(NullPluginTest, AllGatherInvalidArgsLastError) {
    onecclComm_t comm = nullptr;

    // 1) Call onecclAllGather with invalid arguments
    onecclResult_t const result =
        onecclAllGather(nullptr, nullptr, 0, onecclInt, comm, nullptr);

    // 2) Expect onecclInvalidArgument
    EXPECT_EQ(result, onecclInvalidArgument);

    // 3) Retrieve the last error string
    const char *err_string = onecclGetLastError(comm);
    ASSERT_NE(err_string, nullptr); // ensure it's not null

    // 4) Check that the error string is not empty and includes a known
    // substring
    std::string const err_msg(err_string);
    EXPECT_FALSE(err_msg.empty());
    EXPECT_NE(err_msg.find("comm cannot be nullptr"), std::string::npos);
}

TEST_F(NullPluginTest, CommunicatorWithConfigInitializer) {
    onecclUniqueId uid{};
    onecclComm_t comm = nullptr;
    onecclConfig_t const config = ONECCL_CONFIG_INITIALIZER;

    auto result = onecclGetUniqueId(&uid);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclCommInitRankConfig(&comm, 1, uid, 0, &config);
    ASSERT_EQ(result, onecclSuccess);
}

// This test aims to validate thread-safety of onecclGetLastError.
// It uses two threads which should record different errors inside
// `onecclAllGather`. Then the `onecclGetLastError` string could
// be one of the two expected values in `check_error_string` lambda.
// If the function would not be thread-safe the received buffer
// could contain mixed messages.
TEST_F(NullPluginTest, MultiThreadedAllGatherInvalidArgsLastError) {
    onecclComm_t comm = nullptr;
    std::vector<std::thread> threads;
    std::mutex print_mutex;
    std::atomic<int> ready_count(0);

    // Function to check if error string contains a complete expected message
    auto check_error_string = [](const char *err_string) {
        ASSERT_NE(err_string, nullptr);
        std::string const err_msg(err_string);
        EXPECT_FALSE(err_msg.empty());
        bool const valid_error =
            (err_msg.find("comm cannot be nullptr") != std::string::npos);
        EXPECT_TRUE(valid_error);
    };

    auto send_buff_error_task = [&](onecclComm_t comm) {
        ready_count.fetch_add(1);

        // Wait until both threads are ready, so they both start together
        while (ready_count.load() < 2) {
            std::this_thread::yield();
        }

        // Do multiple invocations to increase chance of collision
        for (int i = 0; i < 100; ++i) {
            onecclResult_t const result =
                onecclAllGather(nullptr, nullptr, 0, onecclInt, comm, nullptr);

            EXPECT_EQ(result, onecclInvalidArgument);

            const char *err_string = onecclGetLastError(comm);

            std::scoped_lock const lock(print_mutex);
            check_error_string(err_string);
        }
    };

    auto recv_buff_error_task = [&](onecclComm_t comm) {
        ready_count.fetch_add(1);

        while (ready_count.load() < 2) {
            std::this_thread::yield();
        }

        for (int i = 0; i < 100; ++i) {
            char sendbuff[16] = {};
            onecclResult_t const result = onecclAllGather(
                &sendbuff, nullptr, 0, onecclInt, comm, nullptr);

            EXPECT_EQ(result, onecclInvalidArgument);

            const char *err_string = onecclGetLastError(comm);

            std::scoped_lock const lock(print_mutex);
            check_error_string(err_string);
        }
    };

    threads.emplace_back(send_buff_error_task, comm);
    threads.emplace_back(recv_buff_error_task, comm);

    for (auto &thread : threads) {
        thread.join();
    }
}

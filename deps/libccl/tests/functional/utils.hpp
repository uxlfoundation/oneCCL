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

#pragma once

#include <ctime>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdlib.h>
#include <string>

#include "gtest/gtest.h"

#include "mpi.h"

#include "conf.hpp"
#include "oneapi/ccl.hpp"

#define sizeofa(arr)   (sizeof(arr) / sizeof(*arr))
#define DTYPE          float
#define CACHELINE_SIZE 64

#define ITER_COUNT          2
#define ERR_MESSAGE_MAX_LEN 180

#define TIMEOUT     30
#define UNUSED_ATTR __attribute__((unused))

#define TEST_SUCCESS 0
#define TEST_FAILURE 1

#ifndef PRINT
#define PRINT(fmt, ...) \
    do { \
        fflush(stdout); \
        printf("\n(%ld): %s: " fmt "\n", GETTID(), __FUNCTION__, ##__VA_ARGS__); \
        fflush(stdout); \
    } while (0)
#endif // PRINT

#define OUTPUT_NAME_ARG "--gtest_output="
#define PATCH_OUTPUT_NAME_ARG(argc, argv, comm) \
    do { \
        if (comm.size() > 1) { \
            for (int idx = 1; idx < argc; idx++) { \
                if (strstr(argv[idx], OUTPUT_NAME_ARG)) { \
                    std::string patchedArg; \
                    std::string originArg = std::string(argv[idx]); \
                    size_t extPos = originArg.find(".xml"); \
                    size_t argLen = strlen(OUTPUT_NAME_ARG); \
                    patchedArg = originArg.substr(argLen, extPos - argLen) + "_" + \
                                 std::to_string(comm.rank()) + ".xml"; \
                    PRINT("originArg %s, extPos %zu, argLen %zu, patchedArg %s", \
                          originArg.c_str(), \
                          extPos, \
                          argLen, \
                          patchedArg.c_str()); \
                    argv[idx][0] = '\0'; \
                    if (comm.rank()) \
                        ::testing::GTEST_FLAG(output) = ""; \
                    else \
                        ::testing::GTEST_FLAG(output) = patchedArg.c_str(); \
                } \
            } \
        } \
    } while (0)

#define RUN_METHOD_DEFINITION(ClassName) \
    template <typename T> \
    int MainTest::run(test_param param) { \
        ClassName<T> className; \
        test_operation<T> op(param); \
        std::ostringstream output; \
        if (op.comm_rank == 0) \
            printf("%s", output.str().c_str()); \
        int result = className.run(op); \
        int result_final = 0; \
        static int glob_idx = 0; \
        auto& service_comm = transport_data::instance().get_service_comm(); \
        ccl::allreduce(&result, &result_final, 1, ccl::reduction::sum, service_comm).wait(); \
        if (result_final > 0) { \
            print_err_message(className.get_err_message(), output); \
            if (op.comm_rank == 0) { \
                printf("%s", output.str().c_str()); \
                if (glob_idx) { \
                    test_operation<T> prev_op(test_params[glob_idx - 1]); \
                    output << "Previous operation:\n"; \
                    prev_op.print(output); \
                } \
                output << "Current operation:\n"; \
                op.print(output); \
                EXPECT_TRUE(false) << output.str(); \
            } \
            output.str(""); \
            output.clear(); \
            glob_idx++; \
            return TEST_FAILURE; \
        } \
        glob_idx++; \
        return TEST_SUCCESS; \
    }

#define MAIN_FUNCTION_BODY() \
    init_test_params(); \
    auto& transport = transport_data::instance(); \
    auto& service_comm = transport.get_service_comm(); \
    PATCH_OUTPUT_NAME_ARG(argc, argv, service_comm); \
    testing::InitGoogleTest(&argc, argv); \
    int res = RUN_ALL_TESTS(); \
    transport.reset_comms(); \
    return res;

#define MAIN_FUNCTION() \
    int main(int argc, char** argv, char* envs[]) { \
        MAIN_FUNCTION_BODY() \
    }

/* Same main() as MAIN_FUNCTION, opened as a function-try-block so handlers can
 * follow. Object-like on purpose: it is meant to be written next to the handlers,
 * as MAIN_FUNCTION_WITH_TRY CATCH_EXCEPTION(type) CATCH_EXCEPTION(other_type);
 *
 * Setup and teardown run outside gtest, so an exception from ccl::init(), the
 * KVS/communicator creation or reset_comms() has nothing above it: without a
 * handler it reaches the top of main and terminates, and a bare CCL_THROW carries
 * its message only in the exception object. */
#define MAIN_FUNCTION_WITH_TRY \
    int main(int argc, char** argv, char* envs[]) try { MAIN_FUNCTION_BODY() }

/* One handler per invocation. Handlers compose by being written next to each
 * other, which is what C++ already means by consecutive catch blocks, so any
 * number of types can be listed without changing this macro.
 *
 * Naming the types instead of catching std::exception is deliberate: it keeps
 * testing::internal::GoogleTestFailureException propagating, which is what
 * --gtest_throw_on_failure relies on. */
#define CATCH_EXCEPTION(exception_type) \
    catch (const exception_type& e) { \
        fprintf(stderr, "TEST FAILED: " #exception_type ": %s\n", e.what()); \
        fflush(stderr); \
        return TEST_FAILURE; \
    }

/* sycl::exception only exists when the SYCL backend is built (transport.hpp pulls
 * in sycl_base.hpp under CCL_ENABLE_SYCL), so name a type that is never thrown
 * otherwise and keep the handler list in each test identical across
 * configurations. */
#ifdef CCL_ENABLE_SYCL
#define CCL_TEST_SYCL_EXCEPTION sycl::exception
#else // CCL_ENABLE_SYCL
namespace ccl_test {
struct no_sycl_exception : std::exception {};
} // namespace ccl_test
#define CCL_TEST_SYCL_EXCEPTION ccl_test::no_sycl_exception
#endif // CCL_ENABLE_SYCL

#define TEST_CASES_DEFINITION(FuncName) \
    TEST_P(MainTest, FuncName) { \
        test_param param = GetParam(); \
        EXPECT_EQ(TEST_SUCCESS, this->test(param)); \
    } \
    INSTANTIATE_TEST_CASE_P(test_params, MainTest, ::testing::ValuesIn(test_params));

class CustomPrinter : public ::testing::EmptyTestEventListener {
    virtual void OnTestCaseStart(const ::testing::TestCase& test_case) {
        printf("Overall %d tests from %s\n", test_case.test_to_run_count(), test_case.name());
        fflush(stdout);
    }

    virtual void OnTestCaseEnd(const ::testing::TestCase& test_case) {
        if (!::testing::GTEST_FLAG(print_time))
            return;

        printf("Overall %d tests from %s (%s ms total)\n\n",
               test_case.test_to_run_count(),
               test_case.name(),
               ::testing::internal::StreamableToString(test_case.elapsed_time()).c_str());
        fflush(stdout);
    }

    virtual void OnTestPartResult(const ::testing::TestPartResult& test_part_result) {
        if (test_part_result.failed()) {
            printf("%s in %s:%d\n%s\n",
                   "*** Failure",
                   test_part_result.file_name(),
                   test_part_result.line_number(),
                   test_part_result.summary());
        }
        else
            printf("*** Success");
    }

    virtual void OnTestEnd(const ::testing::TestInfo& test_info) {}

protected:
    testing::TestEventListener* listener;
};

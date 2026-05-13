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
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

class CollectiveTest
    : public ::testing::TestWithParam<
          std::tuple<onecclDataType_t, onecclRedOp_t, size_t>> {
  protected:
    static sycl::queue sycl_queue;
    static int rank;
    static int world_size;
    static int local_rank;
    static MPI_Comm local_comm;
    static onecclComm_t comm;

    static void SetUpTestSuite() {
        // Initialize oneCCL
        onecclUniqueId uid;

        setenv("CCL_PLUGIN", "ONECCL_LEGACY", 0);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &local_comm);
        MPI_Comm_rank(local_comm, &local_rank);

        if (rank == 0) {
            onecclGetUniqueId(&uid);
            MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
        }

        auto result = onecclSetDevice(local_rank);
        if (result != onecclSuccess) {
            std::cerr << "Failed to set device." << '\n';
            std::exit(1);
        }

        result = onecclCommInitRank(&comm, world_size, uid, rank);
        if (result != onecclSuccess) {
            std::cerr << "Failed to initialize communicator." << '\n';
            std::exit(1);
        }

        int device_index = -1;
        result = onecclCommDevice(comm, &device_index);
        if (result != onecclSuccess) {
            std::cerr << "Failed to get device index back." << '\n';
            std::exit(1);
        }

        sycl_queue = create_sycl_queue(local_rank);
    }

    static void TearDownTestSuite() { onecclCommDestroy(comm); }

    static sycl::queue create_sycl_queue(int local_rank) {
        auto platforms = sycl::platform::get_platforms();
        sycl::platform l0_platform;
        bool l0_found = false;

        for (const auto &platform : platforms) {
            if (platform.get_backend() ==
                sycl::backend::ext_oneapi_level_zero) {
                l0_platform = platform;
                l0_found = true;
                break;
            }
        }

        if (!l0_found) {
            throw std::runtime_error("Level-Zero platform not found.");
        }
        return sycl::queue(l0_platform.get_devices()[local_rank],
                           {sycl::property::queue::in_order{},
                            sycl::property::queue::enable_profiling{}});
    }

    template <typename T>
    void perform_reduce(onecclDataType_t datatype, onecclRedOp_t reduction_op,
                        size_t buf_size, int root) {
        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        T *sendbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        T *recvbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        sycl_queue.memcpy(sendbuff, host_data.data(), buf_size * sizeof(T))
            .wait();

        onecclResult_t const result =
            onecclReduce(sendbuff, recvbuff, buf_size, datatype, reduction_op,
                         root, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        sycl_queue.wait();

        if (rank == root) {
            std::vector<T> result_data(buf_size);
            sycl_queue
                .memcpy(result_data.data(), recvbuff, buf_size * sizeof(T))
                .wait();
            T expected_value = 0;
            for (const auto &reduce_scatter_result : result_data) {
                switch (reduction_op) {
                case onecclSum:
                    expected_value =
                        static_cast<int>(static_cast<float>(1 + world_size) /
                                         2 * static_cast<float>(world_size));
                    ASSERT_EQ(reduce_scatter_result, expected_value);
                    break;
                case onecclProd:
                    expected_value = 1;
                    for (int i = 1; i < world_size; ++i) {
                        int const num = i + 1;
                        expected_value *= num;
                    }
                    ASSERT_EQ(reduce_scatter_result, expected_value);
                    break;
                case onecclMax:
                    expected_value = world_size;
                    ASSERT_EQ(reduce_scatter_result, expected_value);
                    break;
                case onecclMin:
                    expected_value = 1;
                    ASSERT_EQ(reduce_scatter_result, expected_value);
                    break;
                default:
                    break;
                }
            }
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_broadcast(onecclDataType_t datatype, size_t buf_size) {
        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        T *sendbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        T *recvbuff = sycl::malloc_device<T>(buf_size, sycl_queue);

        sycl_queue.memcpy(sendbuff, host_data.data(), buf_size * sizeof(T))
            .wait();

        onecclResult_t const result = onecclBroadcast(
            sendbuff, recvbuff, buf_size, datatype, 0, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(buf_size);
        sycl_queue.memcpy(result_data.data(), recvbuff, buf_size * sizeof(T))
            .wait();
        for (const auto &bcast_result : result_data) {
            ASSERT_EQ(bcast_result, static_cast<T>(1));
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_all_gather(onecclDataType_t datatype, size_t sendcount) {
        std::vector<T> host_data(sendcount, static_cast<T>(rank + 1));
        T *sendbuff = sycl::malloc_device<T>(sendcount, sycl_queue);
        T *recvbuff =
            sycl::malloc_device<T>(sendcount * world_size, sycl_queue);
        sycl_queue.memcpy(sendbuff, host_data.data(), sendcount * sizeof(T))
            .wait();

        onecclResult_t const result = onecclAllGather(
            sendbuff, recvbuff, sendcount, datatype, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(sendcount * world_size);
        sycl_queue
            .memcpy(result_data.data(), recvbuff,
                    sendcount * world_size * sizeof(T))
            .wait();
        for (size_t i = 0; i < sendcount * world_size; ++i) {
            auto source_rank = i / sendcount;
            ASSERT_EQ(result_data[i], static_cast<T>(source_rank + 1));
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_all_to_all(onecclDataType_t datatype, size_t count) {
        size_t const total_count =
            count * world_size; // Total elements for all ranks
        std::vector<T> host_data_send(total_count);
        std::vector<T> host_data_recv(total_count);

        // Initialize send buffer with unique data for each rank
        for (size_t i = 0; i < total_count; ++i) {
            host_data_send[i] = static_cast<T>((rank * total_count) + i);
        }

        T *sendbuff = sycl::malloc_device<T>(total_count, sycl_queue);
        T *recvbuff = sycl::malloc_device<T>(total_count, sycl_queue);

        sycl_queue
            .memcpy(sendbuff, host_data_send.data(), total_count * sizeof(T))
            .wait();

        onecclResult_t const result = onecclAllToAll(
            sendbuff, recvbuff, count, datatype, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        sycl_queue
            .memcpy(host_data_recv.data(), recvbuff, total_count * sizeof(T))
            .wait();

        // Validate received data
        for (int i = 0; i < world_size; ++i) {
            for (size_t j = 0; j < count; ++j) {
                T expected_value =
                    static_cast<T>((i * total_count) + (rank * count) + j);
                ASSERT_EQ(host_data_recv[(i * count) + j], expected_value);
            }
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_reduce_scatter(onecclDataType_t datatype,
                                onecclRedOp_t reduction_op, size_t recvcount) {
        size_t const sendcount = recvcount * world_size;
        std::vector<T> host_data(sendcount, static_cast<T>(rank + 1));
        T *sendbuff = sycl::malloc_device<T>(sendcount, sycl_queue);
        T *recvbuff = sycl::malloc_device<T>(recvcount, sycl_queue);
        sycl_queue.memcpy(sendbuff, host_data.data(), sendcount * sizeof(T))
            .wait();

        onecclResult_t const result =
            onecclReduceScatter(sendbuff, recvbuff, recvcount, datatype,
                                reduction_op, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(recvcount);
        sycl_queue.memcpy(result_data.data(), recvbuff, recvcount * sizeof(T))
            .wait();

        T expected_value = 0;
        auto world_size_as_float = static_cast<float>(world_size);
        for (const auto &reduce_scatter_result : result_data) {
            switch (reduction_op) {
            case onecclSum:
                expected_value =
                    static_cast<int>(static_cast<float>(1 + world_size) / 2 *
                                     static_cast<float>(world_size));
                ASSERT_EQ(reduce_scatter_result, expected_value);
                break;
            case onecclProd:
                expected_value = 1;
                for (int i = 1; i < world_size; ++i) {
                    int const num = i + 1;
                    expected_value *= num;
                }
                ASSERT_EQ(reduce_scatter_result, expected_value);
                break;
            case onecclMax:
                expected_value = world_size;
                ASSERT_EQ(reduce_scatter_result, expected_value);
                break;
            case onecclMin:
                expected_value = 1;
                ASSERT_EQ(reduce_scatter_result, expected_value);
                break;
            case onecclAvg:
                expected_value =
                    static_cast<T>((static_cast<float>(1 + world_size) / 2) *
                                   world_size_as_float / world_size_as_float);
                ASSERT_EQ(reduce_scatter_result, expected_value);
                break;
            default:
                break;
            }
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_send_recv(onecclDataType_t datatype, size_t buf_size,
                           int peer) {
        T *sendbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        T *recvbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        if (rank == 0) {
            sycl_queue.memcpy(sendbuff, host_data.data(), buf_size * sizeof(T))
                .wait();
        }

        if (rank == 0) {
            onecclSend(sendbuff, buf_size, datatype, peer, comm, &sycl_queue);
        } else if (rank == peer) {
            onecclRecv(recvbuff, buf_size, datatype, 0, comm, &sycl_queue);
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_send_recv_group(onecclDataType_t datatype, size_t buf_size,
                                 int num_ranks) {
        T *sendbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        T *recvbuff = sycl::malloc_device<T>(buf_size * num_ranks, sycl_queue);

        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        sycl_queue.memcpy(sendbuff, host_data.data(), buf_size * sizeof(T));

        onecclGroupStart();

        // Send data from each rank to every other rank
        for (int i = 0; i < num_ranks; ++i) {
            onecclSend(sendbuff, buf_size, datatype, i, comm, &sycl_queue);
            onecclRecv(recvbuff + (i * buf_size), buf_size, datatype, i, comm,
                       &sycl_queue);
        }

        onecclGroupEnd();
        sycl_queue.wait();

        // TODO: Workaround for issues with legacy code!
        usleep(10);

        std::vector<T> received_full_data(buf_size * num_ranks);
        sycl_queue
            .memcpy(received_full_data.data(), recvbuff,
                    buf_size * num_ranks * sizeof(T))
            .wait();

        // Validate received data
        for (int i = 0; i < num_ranks; ++i) {
            T expected_value = i + 1;
            for (size_t j = 0; j < buf_size; ++j) {
                ASSERT_EQ(received_full_data[(i * buf_size) + j],
                          expected_value);
            }
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_all_reduce(onecclDataType_t datatype,
                            onecclRedOp_t reduction_op, size_t buf_size) {
        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        T *sendbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        T *recvbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        sycl_queue.memcpy(sendbuff, host_data.data(), buf_size * sizeof(T))
            .wait();

        onecclResult_t const result =
            onecclAllReduce(sendbuff, recvbuff, buf_size, datatype,
                            reduction_op, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(buf_size);
        sycl_queue.memcpy(result_data.data(), recvbuff, buf_size * sizeof(T))
            .wait();

        T expected_value = 0; // Depending on the operation
        for (const auto &allreduce_result : result_data) {
            switch (reduction_op) {
            case onecclSum:
                expected_value =
                    static_cast<int>(static_cast<float>(1 + world_size) / 2 *
                                     static_cast<float>(world_size));
                ASSERT_EQ(allreduce_result, expected_value);
                break;
            case onecclProd:
                expected_value = 1; // First rank value
                for (int i = 1; i < world_size; ++i) {
                    int const num = i + 1;
                    expected_value *= num;
                }
                ASSERT_EQ(allreduce_result, expected_value);
                break;
            case onecclMax:
                expected_value = world_size;
                ASSERT_EQ(allreduce_result, expected_value);
                break;
            case onecclMin:
                expected_value = 1;
                ASSERT_EQ(allreduce_result, expected_value);
                break;
            case onecclAvg:
                expected_value =
                    static_cast<T>((static_cast<float>(1 + world_size) / 2) *
                                   static_cast<float>(world_size) /
                                   static_cast<float>(world_size));
                ASSERT_EQ(allreduce_result, expected_value);
                break;
            default:
                break;
            }
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_custom_redop_all_reduce(onecclDataType_t datatype,
                                         size_t buf_size) {
        /* create testing buffer data */
        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        T *sendbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        T *recvbuff = sycl::malloc_device<T>(buf_size, sycl_queue);
        sycl_queue.memcpy(sendbuff, host_data.data(), buf_size * sizeof(T))
            .wait();

        /* create custom device side scalar multiplier */
        T scalar_value = static_cast<T>(rank + 1);
        T *custom_scalar = sycl::malloc_device<T>(1, sycl_queue);
        sycl_queue.memcpy(custom_scalar, &scalar_value, sizeof(T)).wait();

        /* create reference buffer data */
        /* multiply send data by the scalar localy */
        std::vector<T> host_data_ref(buf_size,
                                     static_cast<T>(rank + 1) * scalar_value);
        T *sendbuff_ref = sycl::malloc_device<T>(buf_size, sycl_queue);
        T *recvbuff_ref = sycl::malloc_device<T>(buf_size, sycl_queue);
        sycl_queue
            .memcpy(sendbuff_ref, host_data_ref.data(), buf_size * sizeof(T))
            .wait();

        /* create custom premul_sum reduction operation */
        onecclRedOp_t custom_op{};
        onecclRedOpCreatePreMulSum(&custom_op, custom_scalar, datatype,
                                   onecclScalarResidence_t::onecclScalarDevice,
                                   comm);

        onecclResult_t const result =
            onecclAllReduce(sendbuff, recvbuff, buf_size, datatype, custom_op,
                            comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(buf_size);
        sycl_queue.memcpy(result_data.data(), recvbuff, buf_size * sizeof(T))
            .wait();

        /* reference allreduce with just sum reduction */
        onecclResult_t const result_ref =
            onecclAllReduce(sendbuff_ref, recvbuff_ref, buf_size, datatype,
                            onecclRedOp_t::onecclSum, comm, &sycl_queue);
        ASSERT_EQ(result_ref, onecclSuccess);

        std::vector<T> result_data_ref(buf_size);
        sycl_queue
            .memcpy(result_data_ref.data(), recvbuff_ref, buf_size * sizeof(T))
            .wait();

        // Compare result_data (from custom op) with result_data_ref (reference)
        for (size_t i = 0; i < result_data.size(); ++i) {
            ASSERT_EQ(result_data[i], result_data_ref[i]);
        }

        /* destroy custom_op */
        onecclRedOpDestroy(custom_op, comm);

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
        // sycl::free(sendbuff_ref, sycl_queue);
        // sycl::free(recvbuff_ref, sycl_queue);
    }
};

// Derived class specifically for running a single test case with specific
// parameters
class SimpleCollectiveTest : public CollectiveTest {};

sycl::queue CollectiveTest::sycl_queue;
int CollectiveTest::rank;
int CollectiveTest::world_size;
int CollectiveTest::local_rank;
MPI_Comm CollectiveTest::local_comm;
onecclComm_t CollectiveTest::comm;

std::string to_string(onecclDataType_t datatype) {
    switch (datatype) {
    case onecclInt8:
        return "onecclInt8";
    case onecclInt32:
        return "onecclInt32";
    case onecclInt64:
        return "onecclInt64";
    case onecclUint8:
        return "onecclUint8";
    case onecclUint32:
        return "onecclUint32";
    case onecclUint64:
        return "onecclUint64";
    case onecclFloat32:
        return "onecclFloat32";
    case onecclFloat64:
        return "onecclFloat64";
    case onecclBfloat16:
        return "onecclBfloat16";
    case onecclFloat16:
        return "onecclFloat16";
    default:
        return "Unknown DataType";
    }
}

std::string to_string(onecclRedOp_t reduction_op) {
    switch (reduction_op) {
    case onecclSum:
        return "onecclSum";
    case onecclProd:
        return "onecclProd";
    case onecclMax:
        return "onecclMax";
    case onecclMin:
        return "onecclMin";
    case onecclAvg:
        return "onecclAvg";
    default:
        return "Unknown Reduction Op";
    }
}

bool is_avg_supported(onecclDataType_t data_type) {
    std::vector<onecclDataType_t> unsupported_types = {
        onecclInt8,   onecclInt64,  onecclUint8,
        onecclUint32, onecclUint64, onecclFloat64};

    // Check if the data_type is in the list of unsupported types
    return std::find(unsupported_types.begin(), unsupported_types.end(),
                     data_type) == unsupported_types.end();
}

} // namespace

// NOLINTBEGIN(readability-identifier-naming, misc-use-internal-linkage)
void PrintTo(onecclDataType_t datatype, ::std::ostream *out_stream) {
    *out_stream << to_string(datatype);
}

void PrintTo(onecclRedOp_t reduction_op, ::std::ostream *out_stream) {
    *out_stream << to_string(reduction_op);
}
// NOLINTEND(readability-identifier-naming, misc-use-internal-linkage)

// Instantiate test suite for wide scope testing
INSTANTIATE_TEST_SUITE_P(
    SyclAcceptanceTests, CollectiveTest,
    ::testing::Combine(::testing::Values(onecclInt8, onecclInt32, onecclInt64,
                                         onecclUint8, onecclUint32,
                                         onecclUint64, onecclFloat32,
                                         onecclFloat64, onecclBfloat16,
                                         onecclFloat16),
                       ::testing::Values(onecclSum, onecclProd, onecclMax,
                                         onecclMin, onecclAvg),
                       ::testing::Values(16UL, 2048UL, 8UL * 1024 * 1024)));

// Instantiate test suite for a simple tests
INSTANTIATE_TEST_SUITE_P(SpecificParamsTest, SimpleCollectiveTest,
                         ::testing::Combine(::testing::Values(onecclInt32),
                                            ::testing::Values(onecclSum),
                                            ::testing::Values(1024UL)));

TEST_P(CollectiveTest, AllReduceOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    onecclRedOp_t const reduction_op = std::get<1>(params);
    size_t const buf_size = std::get<2>(params);

    std::cout << "Test Params - DataType: " << to_string(datatype)
              << ", ReductionOp: " << to_string(reduction_op)
              << ", Buffer Size: " << buf_size << '\n';

    if (reduction_op == onecclAvg && !is_avg_supported(datatype)) {
        GTEST_SKIP_("Average reduction is not supported yet in libccl for the "
                    "datatype");
    }

    switch (datatype) {
    case onecclInt8:
        perform_all_reduce<char>(datatype, reduction_op, buf_size);
        break;
    case onecclInt32:
        perform_all_reduce<int>(datatype, reduction_op, buf_size);
        break;
    case onecclInt64:
        perform_all_reduce<long>(datatype, reduction_op, buf_size);
        break;
    case onecclUint8:
        perform_all_reduce<unsigned char>(datatype, reduction_op, buf_size);
        break;
    case onecclUint32:
        perform_all_reduce<uint>(datatype, reduction_op, buf_size);
        break;
    case onecclUint64:
        perform_all_reduce<ulong>(datatype, reduction_op, buf_size);
        break;
    case onecclFloat32:
        perform_all_reduce<float>(datatype, reduction_op, buf_size);
        break;
    case onecclFloat64:
        perform_all_reduce<double>(datatype, reduction_op, buf_size);
        break;
    case onecclBfloat16:
        perform_all_reduce<sycl::ext::oneapi::bfloat16>(datatype, reduction_op,
                                                        buf_size);
        break;
    case onecclFloat16:
        perform_all_reduce<sycl::half>(datatype, reduction_op, buf_size);
        break;
    default:
        FAIL() << "Unsupported type!";
    }
}

TEST_P(CollectiveTest, ReduceOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    onecclRedOp_t const reduction_op = std::get<1>(params);
    size_t const buf_size = std::get<2>(params);

    if (reduction_op == onecclAvg) {
        GTEST_SKIP_("Average reduction is not supported yet in libccl for "
                    "onecclReduce");
    }

    std::cout << "Test Params - DataType: " << to_string(datatype)
              << ", Buffer Size: " << buf_size
              << ", Reduction: " << to_string(reduction_op) << "\n";

    int const root = 0;
    switch (datatype) {
    case onecclInt8:
        perform_reduce<char>(datatype, reduction_op, buf_size, root);
        break;
    case onecclInt32:
        perform_reduce<int>(datatype, reduction_op, buf_size, root);
        break;
    case onecclInt64:
        perform_reduce<long>(datatype, reduction_op, buf_size, root);
        break;
    case onecclUint8:
        perform_reduce<unsigned char>(datatype, reduction_op, buf_size, root);
        break;
    case onecclUint32:
        perform_reduce<uint>(datatype, reduction_op, buf_size, root);
        break;
    case onecclUint64:
        perform_reduce<ulong>(datatype, reduction_op, buf_size, root);
        break;
    case onecclFloat32:
        perform_reduce<float>(datatype, reduction_op, buf_size, root);
        break;
    case onecclFloat64:
        perform_reduce<double>(datatype, reduction_op, buf_size, root);
        break;
    case onecclBfloat16:
        perform_reduce<sycl::ext::oneapi::bfloat16>(datatype, reduction_op,
                                                    buf_size, root);
        break;
    case onecclFloat16:
        perform_reduce<sycl::half>(datatype, reduction_op, buf_size, root);
        break;
    default:
        FAIL() << "Unsupported type!";
    }
}

TEST_P(CollectiveTest, BroadcastOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    size_t const buf_size = std::get<2>(params);

    switch (datatype) {
    case onecclInt8:
        perform_broadcast<char>(datatype, buf_size);
        break;
    case onecclInt32:
        perform_broadcast<int>(datatype, buf_size);
        break;
    case onecclInt64:
        perform_broadcast<long>(datatype, buf_size);
        break;
    case onecclUint8:
        perform_broadcast<unsigned char>(datatype, buf_size);
        break;
    case onecclUint32:
        perform_broadcast<uint>(datatype, buf_size);
        break;
    case onecclUint64:
        perform_broadcast<ulong>(datatype, buf_size);
        break;
    case onecclFloat32:
        perform_broadcast<float>(datatype, buf_size);
        break;
    case onecclFloat64:
        perform_broadcast<double>(datatype, buf_size);
        break;
    case onecclBfloat16:
        perform_broadcast<sycl::ext::oneapi::bfloat16>(datatype, buf_size);
        break;
    case onecclFloat16:
        perform_broadcast<sycl::half>(datatype, buf_size);
        break;
    default:
        FAIL() << "Unsupported type!";
    }
}

TEST_P(CollectiveTest, AllGatherOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    size_t const send_count = std::get<2>(params);

    switch (datatype) {
    case onecclInt8:
        perform_all_gather<char>(datatype, send_count);
        break;
    case onecclInt32:
        perform_all_gather<int>(datatype, send_count);
        break;
    case onecclInt64:
        perform_all_gather<long>(datatype, send_count);
        break;
    case onecclUint8:
        perform_all_gather<unsigned char>(datatype, send_count);
        break;
    case onecclUint32:
        perform_all_gather<uint>(datatype, send_count);
        break;
    case onecclUint64:
        perform_all_gather<ulong>(datatype, send_count);
        break;
    case onecclFloat32:
        perform_all_gather<float>(datatype, send_count);
        break;
    case onecclFloat64:
        perform_all_gather<double>(datatype, send_count);
        break;
    case onecclBfloat16:
        perform_all_gather<sycl::ext::oneapi::bfloat16>(datatype, send_count);
        break;
    case onecclFloat16:
        perform_all_gather<sycl::half>(datatype, send_count);
        break;
    default:
        FAIL() << "Unsupported type!";
    }
}

TEST_P(CollectiveTest, AllToAllOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    size_t const count = std::get<2>(params);

    switch (datatype) {
    case onecclInt8:
        perform_all_to_all<char>(datatype, count);
        break;
    case onecclInt32:
        perform_all_to_all<int>(datatype, count);
        break;
    case onecclInt64:
        perform_all_to_all<long>(datatype, count);
        break;
    case onecclUint8:
        perform_all_to_all<unsigned char>(datatype, count);
        break;
    case onecclUint32:
        perform_all_to_all<uint>(datatype, count);
        break;
    case onecclUint64:
        perform_all_to_all<ulong>(datatype, count);
        break;
    case onecclFloat32:
        perform_all_to_all<float>(datatype, count);
        break;
    case onecclFloat64:
        perform_all_to_all<double>(datatype, count);
        break;
    case onecclBfloat16:
        perform_all_to_all<sycl::ext::oneapi::bfloat16>(datatype, count);
        break;
    case onecclFloat16:
        perform_all_to_all<sycl::half>(datatype, count);
        break;
    default:
        FAIL() << "Unsupported type!";
    }
}

TEST_P(CollectiveTest, ReduceScatterOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    onecclRedOp_t const reduction_op = std::get<1>(params);
    size_t const recv_count = std::get<2>(params);

    if (reduction_op == onecclAvg && !is_avg_supported(datatype)) {
        GTEST_SKIP_("Average reduction is not supported yet in libccl for the "
                    "datatype");
    }

    switch (datatype) {
    case onecclInt8:
        perform_reduce_scatter<char>(datatype, reduction_op, recv_count);
        break;
    case onecclInt32:
        perform_reduce_scatter<int>(datatype, reduction_op, recv_count);
        break;
    case onecclInt64:
        perform_reduce_scatter<long>(datatype, reduction_op, recv_count);
        break;
    case onecclUint8:
        perform_reduce_scatter<unsigned char>(datatype, reduction_op,
                                              recv_count);
        break;
    case onecclUint32:
        perform_reduce_scatter<uint>(datatype, reduction_op, recv_count);
        break;
    case onecclUint64:
        perform_reduce_scatter<ulong>(datatype, reduction_op, recv_count);
        break;
    case onecclFloat32:
        perform_reduce_scatter<float>(datatype, reduction_op, recv_count);
        break;
    case onecclFloat64:
        perform_reduce_scatter<double>(datatype, reduction_op, recv_count);
        break;
    case onecclBfloat16:
        perform_reduce_scatter<sycl::ext::oneapi::bfloat16>(
            datatype, reduction_op, recv_count);
        break;
    case onecclFloat16:
        perform_reduce_scatter<sycl::half>(datatype, reduction_op, recv_count);
        break;
    default:
        FAIL() << "Unsupported type!";
    }
}

TEST_P(CollectiveTest, SendRecvOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    size_t const buf_size = std::get<2>(params);

    int const peer = 1; // example peer
    if (world_size > 1) {
        switch (datatype) {
        case onecclInt8:
            perform_send_recv<char>(datatype, buf_size, peer);
            break;
        case onecclInt32:
            perform_send_recv<int>(datatype, buf_size, peer);
            break;
        case onecclInt64:
            perform_send_recv<long>(datatype, buf_size, peer);
            break;
        case onecclUint8:
            perform_send_recv<unsigned char>(datatype, buf_size, peer);
            break;
        case onecclUint32:
            perform_send_recv<uint>(datatype, buf_size, peer);
            break;
        case onecclUint64:
            perform_send_recv<ulong>(datatype, buf_size, peer);
            break;
        case onecclFloat32:
            perform_send_recv<float>(datatype, buf_size, peer);
            break;
        case onecclFloat64:
            perform_send_recv<double>(datatype, buf_size, peer);
            break;
        case onecclBfloat16:
            perform_send_recv<sycl::ext::oneapi::bfloat16>(datatype, buf_size,
                                                           peer);
            break;
        case onecclFloat16:
            perform_send_recv<sycl::half>(datatype, buf_size, peer);
            break;
        default:
            FAIL() << "Unsupported type!";
        }
    }
}

TEST_P(CollectiveTest, SendRecvGroupOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    size_t const buf_size = std::get<2>(params);

    switch (datatype) {
    case onecclInt8:
        perform_send_recv_group<char>(datatype, buf_size, world_size);
        break;
    case onecclInt32:
        perform_send_recv_group<int>(datatype, buf_size, world_size);
        break;
    case onecclInt64:
        perform_send_recv_group<long>(datatype, buf_size, world_size);
        break;
    case onecclUint8:
        perform_send_recv_group<unsigned char>(datatype, buf_size, world_size);
        break;
    case onecclUint32:
        perform_send_recv_group<uint>(datatype, buf_size, world_size);
        break;
    case onecclUint64:
        perform_send_recv_group<ulong>(datatype, buf_size, world_size);
        break;
    case onecclFloat32:
        perform_send_recv_group<float>(datatype, buf_size, world_size);
        break;
    case onecclFloat64:
        perform_send_recv_group<double>(datatype, buf_size, world_size);
        break;
    case onecclBfloat16:
        perform_send_recv_group<sycl::ext::oneapi::bfloat16>(datatype, buf_size,
                                                             world_size);
        break;
    case onecclFloat16:
        perform_send_recv_group<sycl::half>(datatype, buf_size, world_size);
        break;
    default:
        FAIL() << "Unsupported type!";
    }
}

TEST_P(SimpleCollectiveTest, CustomRedopAllReduceOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    size_t const buf_size = std::get<2>(params);

    std::cout << "Test Params - DataType: " << to_string(datatype)
              << ", Buffer Size: " << buf_size << '\n';

    perform_custom_redop_all_reduce<int>(datatype, buf_size);
}

TEST_P(SimpleCollectiveTest, SplitOperation) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    onecclRedOp_t const reduction_op = std::get<1>(params);
    size_t const buf_size = std::get<2>(params);

    std::cout << "Test Params - DataType: " << to_string(datatype)
              << ", ReductionOp: " << to_string(reduction_op)
              << ", Buffer Size: " << buf_size << '\n';

    onecclComm_t new_comm = nullptr;
    onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;

    // Create comm with just the root rank
    int const color = rank == 0 ? 0 : ONECCL_SPLIT_NOCOLOR;
    auto result = onecclCommSplit(comm, color, 0, &new_comm, &config);
    ASSERT_EQ(result, onecclSuccess);

    if (rank == 0) {
        int const expected_value = 123;
        std::vector<int> host_data(buf_size, expected_value);
        int *sendbuff = sycl::malloc_device<int>(buf_size, sycl_queue);
        int *recvbuff = sycl::malloc_device<int>(buf_size, sycl_queue);
        sycl_queue.memcpy(sendbuff, host_data.data(), buf_size * sizeof(int))
            .wait();

        result = onecclAllReduce(sendbuff, recvbuff, buf_size, datatype,
                                 reduction_op, new_comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<int> result_data(buf_size);
        sycl_queue.memcpy(result_data.data(), recvbuff, buf_size * sizeof(int))
            .wait();

        for (const auto &allreduce_result : result_data) {
            ASSERT_EQ(allreduce_result, expected_value);
        }
    }
}

TEST_P(SimpleCollectiveTest, BlockingComm) {
    auto params = GetParam();
    onecclDataType_t const datatype = std::get<0>(params);
    onecclRedOp_t const reduction_op = std::get<1>(params);
    size_t const buf_size = std::get<2>(params);

    std::cout << "Test Params - DataType: " << to_string(datatype)
              << ", ReductionOp: " << to_string(reduction_op)
              << ", Buffer Size: " << buf_size << '\n';

    onecclComm_t new_comm = nullptr;
    onecclConfig_t config = ONECCL_CONFIG_INITIALIZER;
    config.blocking = 1;

    // Create comm with the same ranks but blocking
    int const color = 0;
    auto result = onecclCommSplit(comm, color, 0, &new_comm, &config);
    ASSERT_EQ(result, onecclSuccess);

    int const send_value = 123;
    int const expected_value = send_value * world_size;
    int const iter_count = 32;
    std::vector<int> host_data(buf_size, send_value);
    int *sendbuff = sycl::malloc_device<int>(buf_size, sycl_queue);
    int *recvbuff = sycl::malloc_device<int>(buf_size, sycl_queue);

    for (int iter = 0; iter < iter_count; iter++) {
        sycl_queue.memcpy(sendbuff, host_data.data(), buf_size * sizeof(int))
            .wait();

        result = onecclAllReduce(sendbuff, recvbuff, buf_size, datatype,
                                 reduction_op, new_comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        // NOLINTBEGIN
        sycl_queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class validation_kernel>(
                sycl::range<1>(buf_size), [=](sycl::id<1> idx) {
                    if (recvbuff[idx] != expected_value) {
                        recvbuff[idx] = -1;
                    }

                    // Reset sendbuff for the next iteration
                    sendbuff[idx] = 0;
                });
        });
        // NOLINTEND

        std::vector<int> validation_result(buf_size);
        sycl_queue
            .memcpy(validation_result.data(), recvbuff, buf_size * sizeof(int))
            .wait();
        std::for_each(validation_result.begin(), validation_result.end(),
                      [=](int result) { EXPECT_EQ(result, expected_value); });
    }
}

TEST_P(SimpleCollectiveTest, BufferRegistrationApi) {
    constexpr size_t kBytes = 1ULL << 22;
    constexpr size_t kOffset = 1ULL << 20;
    constexpr size_t kCount = 1024;

    void *buffer = nullptr;
    void *sendbuff = nullptr;
    void *recvbuff = nullptr;
    void *reg_handle = nullptr;

    auto result = onecclMemAlloc(&buffer, kBytes);
    if (result == onecclNotImplemented) {
        GTEST_SKIP_("onecclMemAlloc is not implemented for current backend");
    }
    ASSERT_EQ(result, onecclSuccess);
    ASSERT_NE(buffer, nullptr);

    result = onecclCommRegister(comm, buffer, kBytes, &reg_handle);
    if (result == onecclNotImplemented) {
        onecclMemFree(buffer);
        GTEST_SKIP_(
            "onecclCommRegister is not implemented for current backend");
    }
    ASSERT_EQ(result, onecclSuccess);

    sendbuff = buffer;
    recvbuff = static_cast<void *>(static_cast<uint8_t *>(buffer) + kOffset);

    result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclFloat, onecclSum,
                             comm, &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclAllGather(sendbuff, recvbuff, kCount, onecclInt8, comm,
                             &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    sycl_queue.wait();

    result = onecclCommDeregister(comm, reg_handle);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclMemFree(buffer);
    ASSERT_EQ(result, onecclSuccess);
}

TEST_P(SimpleCollectiveTest, WindowRegistrationApi) {
    constexpr size_t kBytes = 1ULL << 22;

    void *src = nullptr;
    void *dst = nullptr;
    onecclWindow_t src_win = nullptr;
    onecclWindow_t dst_win = nullptr;

    auto result = onecclMemAlloc(&src, kBytes);
    if (result == onecclNotImplemented) {
        GTEST_SKIP_("onecclMemAlloc is not implemented for current backend");
    }
    ASSERT_EQ(result, onecclSuccess);

    result = onecclMemAlloc(&dst, kBytes);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclCommWindowRegister(comm, src, kBytes, &src_win,
                                      ONECCL_WINDOW_COLL_SYMMETRIC);
    if (result == onecclNotImplemented) {
        onecclMemFree(src);
        onecclMemFree(dst);
        GTEST_SKIP_("onecclCommWindowRegister is not implemented for current "
                    "backend");
    }
    ASSERT_EQ(result, onecclSuccess);

    result = onecclCommWindowRegister(comm, dst, kBytes, &dst_win,
                                      ONECCL_WINDOW_COLL_SYMMETRIC);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclAllGather(static_cast<uint8_t *>(src) + 0x1000,
                             static_cast<uint8_t *>(dst) + 0x2000, 1,
                             onecclInt8, comm, &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    sycl_queue.wait();

    result = onecclCommWindowDeregister(comm, src_win);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclCommWindowDeregister(comm, dst_win);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclMemFree(src);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclMemFree(dst);
    ASSERT_EQ(result, onecclSuccess);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    int const result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}

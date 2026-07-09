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
#include <chrono>
#include <cmath> // std::abs
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits> // std::numeric_limits
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <vector>

namespace {

// =============================================================================
// TriangleUtility<T>
//
// Generates deterministic, rank-unique test data for AllReduce validation.
//
// The idea: each rank fills its send buffer with a "triangle wave" pattern --
// values that ramp up to a peak and back down -- where the peak position is
// derived from (rank, cycle_index) using a golden-ratio step. This guarantees:
//   - No two ranks hold the same buffer at the same time (uniqueness).
//   - Values never overflow the type's safe range after reduction across all
//     ranks.
//   - The pattern is reproduced identically by get_expected_val(), so the
//     validator can compute the expected result without any MPI communication.
//
// Design decisions:
//   - is_fp_like: sycl::half and bfloat16 are custom SYCL types not recognized
//     by std::is_floating_point_v, so we must explicitly include them.
//   - range_limit: for integers, capped at Lim::max()/world_size so the sum
//     across all ranks can never overflow T.
//   - MANTISSA_MAX (2^53): double can only represent integers exactly up to
//     2^53; values above that lose uniqueness when stored in double.
//   - cycle_len: for integers, set to effective_max so consecutive triangle
//     values differ by at least 1 after truncation to T (e.g. int8 has only
//     31 usable values, so the triangle must span exactly 31 steps). For
//     floating-point types, fixed at 1024 (arbitrary; values stay in [0,1]).
// =============================================================================
template <typename T> class TriangleUtility {
  public:
    // sycl::half and bfloat16 are NOT detected by std::is_floating_point_v
    // (they are custom SYCL types). We must treat them as floating-point for
    // range/cycle purposes to keep values in [0, 1.0], preventing them from
    // filling up to Lim::max()/world_size (~32752 for half), which would push
    // sums near 65504 where 1 ULP of float16 is 32 -- far beyond any fixed
    // absolute tolerance.
    static constexpr bool is_fp_like =
        std::is_floating_point_v<T> || std::is_same_v<T, sycl::half> ||
        std::is_same_v<T, sycl::ext::oneapi::bfloat16>;

    // get_expected_val(index, rank, world_size, buf_size)
    //
    // Returns the raw double value that fill_buffer would store at position
    // [index] for a given rank. This is the single source of truth used by
    // both:
    //   - fill_buffer()        -- to populate the send buffer before AllReduce.
    //   - validate_allreduce() -- to recompute the expected result per element.
    //
    // CRITICAL: Both fill_buffer and the validator must cast the returned
    // double to T identically. Casting loses precision (e.g. 31.7 to int8(31),
    // 0.123456789 to float(0.1234568)), so if the validator didn't cast through
    // T, it would compare the GPU's T-precision result against a
    // double-precision expectation and fail. The contract ensures fill_buffer
    // and validator see identical values.
    //
    // Parameters:
    //   index      - element position within the buffer [0, buf_size)
    //   rank       - this MPI rank's index [0, world_size)
    //   world_size - total number of MPI ranks in the AllReduce (i.e. the
    //                number of values that will be summed/reduced together).
    //                Used to bound range_limit so no overflow occurs.
    //   buf_size   - total buffer length. Used to cap cycle_len so the
    //                triangle peak always falls inside the buffer.
    static double get_expected_val(size_t index, int rank, int world_size,
                                   size_t buf_size) {
        using Lim = std::numeric_limits<T>;

        // Step #1 - Calculate safe range to prevent Allreduce overflow.
        //
        // range_limit = the maximum value a single rank may contribute, such
        // that summing all world_size ranks cannot overflow type T:
        //   world_size * range_limit ≤ Lim::max()
        // For fp types, values are kept in [0, 1.0] so overflow is impossible.
        double range_limit;
        if constexpr (is_fp_like) {
            range_limit = 1.0;
        } else {
            range_limit = static_cast<double>(Lim::max()) /
                          static_cast<double>(world_size);
        }

        // Step #2 - Mantissa Guard: Cap at 2^53 to ensure unique neighbor
        // representation in double.
        //
        // effective_max = tighter of two constraints:
        //   - range_limit: overflow safety (world_size × val ≤ Lim::max())
        //   - MANTISSA_MAX = 2^53: double precision safety. Integers above
        //     2^53 lose uniqueness in double (e.g. 2^53 and 2^53+1 are both
        //     stored as 2^53), so consecutive triangle values would collide.
        const double MANTISSA_MAX = 9007199254740992.0;
        double effective_max = std::min(range_limit, MANTISSA_MAX);

        // Step #3 - Define Cycle Length: ensure Slope >= 1.0 for integers.
        //
        // cycle_len is then capped to buf_size so the triangle peak always
        // falls within the buffer. Without this, small buffers (e.g. 16
        // elements) with a large cycle_len (e.g. 1024 for fp, or billions for
        // int32) would only ever see the start of the rising slope - the peak
        // would be unreachable, making the pattern look like a plain ramp
        // rather than a triangle.
        size_t cycle_len;
        if constexpr (is_fp_like) {
            cycle_len = 1024;
        } else {
            cycle_len = static_cast<size_t>(effective_max);
            // Guard against cycle_len = 1 (can happen for int8 with world_size
            // >= 64: effective_max = 127/64 = 1.98 → size_t(1)). With
            // cycle_len=1, local_i and h are always 0, making all buffer values
            // identical-defeats the test.
            if (cycle_len < 2)
                cycle_len = 2;
        }
        // Cap to buf_size so the peak is always reachable within the buffer.
        if (cycle_len > buf_size)
            cycle_len = buf_size;

        // Step #4 - Triangle Calculation Logic
        size_t local_i = index % cycle_len;
        size_t cycle_idx = index / cycle_len;
        // Peak position: golden-ratio spacing (≈ 5/8) spreads peaks across
        // the buffer without clustering near 0.
        size_t golden_step = std::max(size_t(1), cycle_len * 5 / 8);
        size_t peak_index =
            ((static_cast<size_t>(rank) + 1) * golden_step + cycle_idx) %
            cycle_len;

        return calculate_val(local_i, peak_index, cycle_len, effective_max);
    }

    // fill_buffer(buf, buf_size, rank, world_size)
    //
    // Fills buf[0..buf_size-1] with values from get_expected_val(), cast to T.
    // The cast is intentional: it simulates the precision loss that will happen
    // on the GPU (e.g. int8 truncation, float16 rounding). validate_allreduce()
    // applies the same cast so both sides agree on what was actually sent.
    static void fill_buffer(T *buf, size_t buf_size, int rank, int world_size) {
        for (size_t i = 0; i < buf_size; ++i) {
            buf[i] =
                static_cast<T>(get_expected_val(i, rank, world_size, buf_size));
        }
    }

  private:
    static double calculate_val(size_t local_i, size_t peak_index,
                                size_t cycle_len, double effective_max) {
        double val;
        if (local_i <= peak_index) {
            val = (peak_index == 0)
                      ? effective_max
                      : (static_cast<double>(local_i) / peak_index) *
                            effective_max;
        } else {
            val = (static_cast<double>(cycle_len - 1 - local_i) /
                   (cycle_len - 1 - peak_index)) *
                  effective_max;
        }
        // For integers, clamp to 1.0: tail values of 0 would make every Prod
        // result 0 at those positions, hiding real data corruption bugs.
        if constexpr (!is_fp_like) {
            val = std::max(val, 1.0);
        }
        return val;
    }
};

// =============================================================================
// validate_allreduce<T>(result_data, reduction_op, world_size)
//
// Recomputes the expected AllReduce result per element and compares against
// result_data[]. Two code paths:
//
//   Integer (!is_fp_like): accumulates in T to match hardware wrapping;
//     uses ASSERT_EQ.
//   FP (is_fp_like): accumulates in double; uses ASSERT_NEAR with
//     type-sized tolerance (sizeof==2: 0.01, sizeof==4: 1e-4, sizeof==8: 1e-9).
// =============================================================================
template <typename T>
void validate_allreduce(const std::vector<T> &result_data,
                        onecclRedOp_t reduction_op, int world_size) {

    const size_t buf_size = result_data.size();

    for (size_t i = 0; i < buf_size; ++i) {

        // Using same logic as TriangleUtility::fill_buffer to compute the
        // expected value per rank
        if constexpr (!TriangleUtility<T>::is_fp_like) {
            // Integer path: accumulate in T (not double) to match hardware
            // semantics. double loses precision for 64-bit values near 2^53,
            // and Prod in double overflows; T gives natural wrapping like GPU.
            T expected;
            switch (reduction_op) {
            case onecclSum:
            case onecclAvg:
                expected = static_cast<T>(0);
                break;
            case onecclMax:
                expected = std::numeric_limits<T>::min();
                break;
            case onecclMin:
                expected = std::numeric_limits<T>::max();
                break;
            case onecclProd:
                expected = static_cast<T>(1);
                break;
            default:
                expected = static_cast<T>(0);
                break;
            }

            for (int r = 0; r < world_size; ++r) {
                // Cast through T to simulate the truncation fill_buffer applied
                // when storing.
                T rank_val =
                    static_cast<T>(TriangleUtility<T>::get_expected_val(
                        i, r, world_size, buf_size));
                switch (reduction_op) {
                case onecclSum:
                case onecclAvg: {
                    // Unsigned cast avoids C++ signed-overflow UB.
                    // effective_max guarantees no actual overflow for Sum,
                    // so the result is mathematically identical to signed.
                    using U = std::make_unsigned_t<T>;
                    expected = static_cast<T>(static_cast<U>(expected) +
                                              static_cast<U>(rank_val));
                    break;
                }
                case onecclProd: {
                    // Prod CAN overflow (effective_max only bounds per-rank
                    // values, not their product). Unsigned multiplication
                    // gives well-defined modular wraparound matching GPU.
                    using U = std::make_unsigned_t<T>;
                    expected = static_cast<T>(static_cast<U>(expected) *
                                              static_cast<U>(rank_val));
                    break;
                }
                case onecclMax:
                    expected = std::max(expected, rank_val);
                    break;
                case onecclMin:
                    expected = std::min(expected, rank_val);
                    break;
                default:
                    break;
                }
            }

            if (reduction_op == onecclAvg) {
                expected /= static_cast<T>(world_size);
            }

            if (result_data[i] != expected) {
                ASSERT_EQ(result_data[i], expected)
                    << "Data Corruption at index [" << i
                    << "] for Allreduce Size " << buf_size
                    << " (Type size: " << sizeof(T) << " bytes)";
            }
        } else {
            // FP path: accumulate in double for precision.
            double expected = 0.0;
            switch (reduction_op) {
            case onecclSum:
            case onecclAvg:
                expected = 0.0;
                break;
            case onecclMax:
                expected = -1e18;
                break;
            case onecclMin:
                expected = 1e18;
                break;
            case onecclProd:
                expected = 1.0;
                break;
            default:
                break;
            }

            for (int r = 0; r < world_size; ++r) {
                // Cast through T to simulate the rounding fill_buffer applied
                // when storing.
                double rank_val = static_cast<double>(
                    static_cast<T>(TriangleUtility<T>::get_expected_val(
                        i, r, world_size, buf_size)));
                switch (reduction_op) {
                case onecclSum:
                case onecclAvg:
                    expected += rank_val;
                    break;
                case onecclProd:
                    expected *= rank_val;
                    break;
                case onecclMax:
                    expected = std::max(expected, rank_val);
                    break;
                case onecclMin:
                    expected = std::min(expected, rank_val);
                    break;
                default:
                    break;
                }
            }

            if (reduction_op == onecclAvg) {
                expected /= static_cast<double>(world_size);
            }

            // Fixed absolute tolerance per type. Safe because TriangleUtility
            // caps fp values to [0,1], so accumulated sums stay small.
            //   sizeof==2 (float16/bfloat16): 0.01
            //   sizeof==4 (float):            1e-4
            //   sizeof==8 (double):           1e-9
            const double tolerance = (sizeof(T) == 2)   ? 0.01
                                     : (sizeof(T) == 4) ? 1e-4
                                                        : 1e-9;
            if (std::abs(static_cast<double>(result_data[i]) - expected) >
                tolerance) {
                ASSERT_NEAR(static_cast<double>(result_data[i]), expected,
                            tolerance)
                    << "Precision Mismatch at index [" << i
                    << "] for Allreduce Size " << buf_size;
            }
        }
    }
}

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
    static void *send_buff;
    static void *recv_buff;
    static size_t send_buff_size;
    static size_t recv_buff_size;

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
        size_t buff_size = (static_cast<long>(32 * 1024) * 1024);
        send_buff_size = buff_size * sizeof(onecclInt64);
        recv_buff_size = buff_size * world_size * sizeof(onecclInt64);
        // Allocate send and recv buffers for the test run
        send_buff = sycl::malloc_device(send_buff_size, sycl_queue);
        ASSERT_NE(send_buff, nullptr)
            << "sycl::malloc_device failed for send_buff (count="
            << send_buff_size << ")";
        recv_buff =
            sycl::malloc_device(static_cast<long>(recv_buff_size), sycl_queue);
        ASSERT_NE(recv_buff, nullptr)
            << "sycl::malloc_device failed for recv_buff (count="
            << recv_buff_size << ")";
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
        ASSERT_TRUE(buf_size * sizeof(T) <= send_buff_size);
        sycl_queue.memcpy(send_buff, host_data.data(), buf_size * sizeof(T))
            .wait();

        onecclResult_t const result =
            onecclReduce(send_buff, recv_buff, buf_size, datatype, reduction_op,
                         root, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        sycl_queue.wait();

        if (rank == root) {
            std::vector<T> result_data(buf_size);
            sycl_queue
                .memcpy(result_data.data(), recv_buff, buf_size * sizeof(T))
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
        ASSERT_TRUE(buf_size * sizeof(T) <= send_buff_size);
        ASSERT_TRUE(buf_size * sizeof(T) <= recv_buff_size);
        sycl_queue.memcpy(send_buff, host_data.data(), buf_size * sizeof(T))
            .wait();

        onecclResult_t const result = onecclBroadcast(
            send_buff, recv_buff, buf_size, datatype, 0, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(buf_size);
        sycl_queue.memcpy(result_data.data(), recv_buff, buf_size * sizeof(T))
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

        // Check if send\recv buffers are big enough
        ASSERT_TRUE(sendcount * sizeof(T) <= send_buff_size);
        ASSERT_TRUE((sendcount * world_size) * sizeof(T) <= recv_buff_size);

        sycl_queue.memcpy(send_buff, host_data.data(), sendcount * sizeof(T))
            .wait();

        onecclResult_t const result = onecclAllGather(
            send_buff, recv_buff, sendcount, datatype, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(sendcount * world_size);
        sycl_queue
            .memcpy(result_data.data(), recv_buff,
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

        // Check if send\recv buffers are big enough
        ASSERT_TRUE(total_count * sizeof(T) <= send_buff_size);
        ASSERT_TRUE(total_count * sizeof(T) <= recv_buff_size);

        sycl_queue
            .memcpy(send_buff, host_data_send.data(), total_count * sizeof(T))
            .wait();

        onecclResult_t const result = onecclAllToAll(
            send_buff, recv_buff, count, datatype, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        sycl_queue
            .memcpy(host_data_recv.data(), recv_buff, total_count * sizeof(T))
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

        // Check if send\recv buffers are big enough
        ASSERT_TRUE(sendcount * sizeof(T) <= send_buff_size);
        ASSERT_TRUE(recvcount * sizeof(T) <= recv_buff_size);

        sycl_queue.memcpy(send_buff, host_data.data(), sendcount * sizeof(T))
            .wait();

        onecclResult_t const result =
            onecclReduceScatter(send_buff, recv_buff, recvcount, datatype,
                                reduction_op, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(recvcount);
        sycl_queue.memcpy(result_data.data(), recv_buff, recvcount * sizeof(T))
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
        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        if (rank == 0) {
            ASSERT_TRUE(buf_size * sizeof(T) <= send_buff_size);
            sycl_queue.memcpy(send_buff, host_data.data(), buf_size * sizeof(T))
                .wait();
        }

        if (rank == 0) {
            onecclSend(send_buff, buf_size, datatype, peer, comm, &sycl_queue);
        } else if (rank == peer) {
            onecclRecv(recv_buff, buf_size, datatype, 0, comm, &sycl_queue);
        }

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_send_recv_group(onecclDataType_t datatype, size_t buf_size,
                                 int num_ranks) {

        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        // Check if send\recv buffers are big enough
        ASSERT_TRUE(buf_size * sizeof(T) <= send_buff_size);
        ASSERT_TRUE(buf_size * num_ranks * sizeof(T) <= recv_buff_size);
        sycl_queue.memcpy(send_buff, host_data.data(), buf_size * sizeof(T));

        onecclGroupStart();

        // Send data from each rank to every other rank
        for (int i = 0; i < num_ranks; ++i) {
            onecclSend(send_buff, buf_size, datatype, i, comm, &sycl_queue);
            onecclRecv(static_cast<T *>(recv_buff) + (i * buf_size), buf_size,
                       datatype, i, comm, &sycl_queue);
        }

        onecclGroupEnd();
        sycl_queue.wait();

        // TODO: Workaround for issues with legacy code!
        usleep(10);

        std::vector<T> received_full_data(buf_size * num_ranks);
        sycl_queue
            .memcpy(received_full_data.data(), recv_buff,
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

    // perform_all_reduce<T>(datatype, reduction_op, buf_size)
    //
    // Runs a complete AllReduce test cycle for a single (type, op, size)
    // combination:
    //
    //   Step 1 - Fill:     TriangleUtility<T>::fill_buffer() populates
    //   host_send_buf with
    //                       deterministic, rank-unique values safe from
    //                       overflow.
    //   Step 2 - Upload:   host_send_buf is copied to a SYCL device buffer
    //   (sendbuff). Step 3 - Reduce:   onecclAllReduce() is called; the
    //   hardware reduces sendbuff
    //                       across all MPI ranks into recvbuff.
    //   Step 4 - Download: recvbuff is copied back to host into result_data.
    //   Step 5 - Validate: validate_allreduce<T>() recomputes the expected
    //   result
    //                       mathematically and asserts element-by-element
    //                       equality.
    template <typename T>
    void perform_all_reduce(onecclDataType_t datatype,
                            onecclRedOp_t reduction_op, size_t buf_size) {
        // Step 1: Fill host_send_buf with unique triangle-pattern values
        std::vector<T> host_send_buf(buf_size);
        TriangleUtility<T>::fill_buffer(host_send_buf.data(), buf_size, rank,
                                        world_size);

        // Check if send\recv buffers are big enough
        ASSERT_TRUE(buf_size * sizeof(T) <= send_buff_size);
        ASSERT_TRUE(buf_size * sizeof(T) <= recv_buff_size);

        // Step 2: Copy to GPU send buffer
        sycl_queue.memcpy(send_buff, host_send_buf.data(), buf_size * sizeof(T))
            .wait();

        // Step 3: AllReduce
        onecclResult_t const result =
            onecclAllReduce(send_buff, recv_buff, buf_size, datatype,
                            reduction_op, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        // Step 4: Copy result back to host
        std::vector<T> result_data(buf_size);
        sycl_queue.memcpy(result_data.data(), recv_buff, buf_size * sizeof(T))
            .wait();

        // Step 5: Validate
        validate_allreduce<T>(result_data, reduction_op, world_size);

        // sycl::free(sendbuff, sycl_queue);
        // sycl::free(recvbuff, sycl_queue);
    }

    template <typename T>
    void perform_custom_redop_all_reduce(onecclDataType_t datatype,
                                         size_t buf_size) {
        /* create testing buffer data */
        std::vector<T> host_data(buf_size, static_cast<T>(rank + 1));
        ASSERT_TRUE(buf_size * sizeof(T) <= send_buff_size);
        ASSERT_TRUE(buf_size * sizeof(T) <= recv_buff_size);
        sycl_queue.memcpy(send_buff, host_data.data(), buf_size * sizeof(T))
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
            onecclAllReduce(send_buff, recv_buff, buf_size, datatype, custom_op,
                            comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);

        std::vector<T> result_data(buf_size);
        sycl_queue.memcpy(result_data.data(), recv_buff, buf_size * sizeof(T))
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
void *CollectiveTest::send_buff = nullptr;
void *CollectiveTest::recv_buff = nullptr;
size_t CollectiveTest::recv_buff_size = 0;
size_t CollectiveTest::send_buff_size = 0;

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

// =============================================================================
// Finalize Tests
// =============================================================================

TEST_P(SimpleCollectiveTest, FinalizeAfterCollectives) {
    constexpr size_t kCount = 1024;
    constexpr int kIterations = 5;

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));

    ASSERT_NE(sendbuff, nullptr);
    ASSERT_NE(recvbuff, nullptr);

    // Submit multiple collectives
    auto my_rank = rank;
    for (int i = 0; i < kIterations; i++) {
        sycl_queue.submit([&](sycl::handler &h) {
            h.parallel_for<class finalize_init>(
                sycl::range<1>(kCount),
                [=](sycl::id<1> idx) { sendbuff[idx] = my_rank + i; });
        });

        auto result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt,
                                      onecclSum, comm, &sycl_queue);
        ASSERT_EQ(result, onecclSuccess);
    }

    // Finalize should wait for all operations
    auto result = onecclCommFinalize(comm);
    ASSERT_EQ(result, onecclSuccess);

    // Verify results after finalize
    std::vector<int> host_recvbuff(kCount);
    sycl_queue.memcpy(host_recvbuff.data(), recvbuff, kCount * sizeof(int))
        .wait();

    int expected = 0;
    for (int r = 0; r < world_size; r++) {
        expected += r + (kIterations - 1);
    }

    for (size_t i = 0; i < kCount; i++) {
        ASSERT_EQ(host_recvbuff[i], expected);
    }

    sycl::free(sendbuff, sycl_queue);
    sycl::free(recvbuff, sycl_queue);
}

TEST_P(SimpleCollectiveTest, FinalizeWithGroupAPI) {
    constexpr size_t kCount = 1024;

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));

    ASSERT_NE(sendbuff, nullptr);
    ASSERT_NE(recvbuff, nullptr);

    // Initialize buffer
    auto my_rank = rank;
    sycl_queue.submit([&](sycl::handler &h) {
        h.parallel_for<class group_finalize_init>(
            sycl::range<1>(kCount),
            [=](sycl::id<1> idx) { sendbuff[idx] = my_rank; });
    });

    // Use group API
    auto result = onecclGroupStart();
    ASSERT_EQ(result, onecclSuccess);

    result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt, onecclSum,
                             comm, &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclGroupEnd();
    ASSERT_EQ(result, onecclSuccess);

    // Finalize after group operations should work
    result = onecclCommFinalize(comm);
    ASSERT_EQ(result, onecclSuccess);

    // Verify results
    std::vector<int> host_recvbuff(kCount);
    sycl_queue.memcpy(host_recvbuff.data(), recvbuff, kCount * sizeof(int))
        .wait();

    int expected = 0;
    for (int r = 0; r < world_size; r++) {
        expected += r;
    }

    for (size_t i = 0; i < kCount; i++) {
        ASSERT_EQ(host_recvbuff[i], expected);
    }

    sycl::free(sendbuff, sycl_queue);
    sycl::free(recvbuff, sycl_queue);
}

TEST_P(SimpleCollectiveTest, FinalizeWithGroupSendRecv) {
    if (world_size < 2) {
        GTEST_SKIP_("This test requires at least 2 ranks");
    }

    constexpr size_t kCount = 1024;

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));

    ASSERT_NE(sendbuff, nullptr);
    ASSERT_NE(recvbuff, nullptr);

    // Initialize buffer
    auto my_rank = rank;
    sycl_queue.submit([&](sycl::handler &h) {
        h.parallel_for<class group_sendrecv_init>(
            sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                sendbuff[idx] = my_rank * 1000 + static_cast<int>(idx);
            });
    });

    // Use group API with send/recv
    auto result = onecclGroupStart();
    ASSERT_EQ(result, onecclSuccess);

    // Ring pattern: rank i sends to (i+1)%world_size, receives from
    // (i-1+world_size)%world_size
    int send_to = (rank + 1) % world_size;
    int recv_from = (rank - 1 + world_size) % world_size;

    result =
        onecclSend(sendbuff, kCount, onecclInt, send_to, comm, &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    result =
        onecclRecv(recvbuff, kCount, onecclInt, recv_from, comm, &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    result = onecclGroupEnd();
    ASSERT_EQ(result, onecclSuccess);

    // Finalize after group send/recv
    result = onecclCommFinalize(comm);
    ASSERT_EQ(result, onecclSuccess);

    // Verify results - should have received from (rank-1)
    std::vector<int> host_recvbuff(kCount);
    sycl_queue.memcpy(host_recvbuff.data(), recvbuff, kCount * sizeof(int))
        .wait();

    for (size_t i = 0; i < kCount; i++) {
        int expected = recv_from * 1000 + static_cast<int>(i);
        ASSERT_EQ(host_recvbuff[i], expected)
            << "Rank " << rank << " received wrong data at index " << i;
    }

    sycl::free(sendbuff, sycl_queue);
    sycl::free(recvbuff, sycl_queue);
}

#if defined(SYCL_EXT_ONEAPI_GRAPH)
TEST_P(SimpleCollectiveTest, GraphWithManualSync) {
    using namespace sycl::ext::oneapi::experimental;

    constexpr size_t kCount = 1024;
    constexpr int kGraphLaunches = 3;

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));

    ASSERT_NE(sendbuff, nullptr);
    ASSERT_NE(recvbuff, nullptr);

    // Create and record graph
    command_graph graph(sycl_queue.get_context(), sycl_queue.get_device());
    graph.begin_recording(sycl_queue);

    auto my_rank = rank;
    sycl_queue.submit([&](sycl::handler &h) {
        h.parallel_for<class graph_test_init>(
            sycl::range<1>(kCount),
            [=](sycl::id<1> idx) { sendbuff[idx] = my_rank; });
    });

    auto result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt,
                                  onecclSum, comm, &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    graph.end_recording();

    auto exec_graph = graph.finalize();

    // Launch graph multiple times
    std::vector<sycl::event> events;
    for (int i = 0; i < kGraphLaunches; i++) {
        auto event = sycl_queue.submit(
            [&](sycl::handler &h) { h.ext_oneapi_graph(exec_graph); });
        events.push_back(event);
    }

    // Manual synchronization required for graphs
    for (auto &event : events) {
        event.wait();
    }

    // Verify results
    std::vector<int> host_recvbuff(kCount);
    sycl_queue.memcpy(host_recvbuff.data(), recvbuff, kCount * sizeof(int))
        .wait();

    int expected = 0;
    for (int r = 0; r < world_size; r++) {
        expected += r;
    }

    for (size_t i = 0; i < kCount; i++) {
        ASSERT_EQ(host_recvbuff[i], expected);
    }

    sycl::free(sendbuff, sycl_queue);
    sycl::free(recvbuff, sycl_queue);
}

TEST_P(SimpleCollectiveTest, FinalizeAfterGraphRecording) {
    using namespace sycl::ext::oneapi::experimental;

    constexpr size_t kCount = 1024;
    constexpr int kGraphLaunches = 2;

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));

    ASSERT_NE(sendbuff, nullptr);
    ASSERT_NE(recvbuff, nullptr);

    // Create and record graph
    command_graph graph(sycl_queue.get_context(), sycl_queue.get_device());
    graph.begin_recording(sycl_queue);

    auto my_rank = rank;
    sycl_queue.submit([&](sycl::handler &h) {
        h.parallel_for<class graph_finalize_init>(
            sycl::range<1>(kCount),
            [=](sycl::id<1> idx) { sendbuff[idx] = my_rank + 100; });
    });

    // This marks the comm as used in graph recording
    auto result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt,
                                  onecclSum, comm, &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    graph.end_recording();

    auto exec_graph = graph.finalize();

    // Launch graph multiple times
    std::vector<sycl::event> events;
    for (int i = 0; i < kGraphLaunches; i++) {
        auto event = sycl_queue.submit(
            [&](sycl::handler &h) { h.ext_oneapi_graph(exec_graph); });
        events.push_back(event);
    }

    // IMPORTANT: Manual sync required before finalize
    for (auto &event : events) {
        event.wait();
    }

    // Finalize will log a warning but should not fail
    // because the communicator was used during graph recording
    result = onecclCommFinalize(comm);
    ASSERT_EQ(result, onecclSuccess);

    // Verify results
    std::vector<int> host_recvbuff(kCount);
    sycl_queue.memcpy(host_recvbuff.data(), recvbuff, kCount * sizeof(int))
        .wait();

    int expected = 0;
    for (int r = 0; r < world_size; r++) {
        expected += r + 100;
    }

    for (size_t i = 0; i < kCount; i++) {
        ASSERT_EQ(host_recvbuff[i], expected);
    }

    sycl::free(sendbuff, sycl_queue);
    sycl::free(recvbuff, sycl_queue);
}
#endif // SYCL_EXT_ONEAPI_GRAPH

TEST_P(SimpleCollectiveTest, FinalizeDuringGroupThrows) {
    constexpr size_t kCount = 1024;

    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));

    ASSERT_NE(sendbuff, nullptr);
    ASSERT_NE(recvbuff, nullptr);

    auto my_rank = rank; // Copy rank for sycl kernels
    sycl_queue.submit([&](sycl::handler &h) {
        h.parallel_for<class finalize_group_throw_init>(
            sycl::range<1>(kCount),
            [=](sycl::id<1> idx) { sendbuff[idx] = my_rank; });
    });

    auto result = onecclGroupStart();
    ASSERT_EQ(result, onecclSuccess);

    result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt, onecclSum,
                             comm, &sycl_queue);
    ASSERT_EQ(result, onecclSuccess);

    // Finalize during active group should fail
    // Note: The C++ implementation throws, but the C API should return an error
    result = onecclCommFinalize(comm);
    // Implementation should either throw or return error
    // For C API, we expect it to handle the exception and return an error code

    // Clean up by ending the group
    result = onecclGroupEnd();
    ASSERT_EQ(result, onecclSuccess);

    sycl::free(sendbuff, sycl_queue);
    sycl::free(recvbuff, sycl_queue);
}

TEST_P(SimpleCollectiveTest, FinalizeMultipleQueuesEventLoop) {
    // This test simulates an event loop scenario where:
    // 1. Each iteration creates a new SYCL queue (simulating event handling)
    // 2. Submits compute work and oneCCL collectives to that queue
    // 3. All queues execute asynchronously (showing overlap)
    // 4. Finalize waits for all queues at the end
    //
    // This pattern is useful for async inference pipelines where each request
    // gets its own queue for independent execution.

    constexpr size_t kCount = 256 * 1024;      // 256K elements
    constexpr int kNumIterations = 5;          // Number of "events" to process
    constexpr int kSlowLoopIterations = 50000; // Volatile loop for slowdown

    // Allocate buffers (shared across all queues)
    int *sendbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));
    int *recvbuff = static_cast<int *>(
        sycl::malloc_device(kCount * sizeof(int), sycl_queue));

    ASSERT_NE(sendbuff, nullptr);
    ASSERT_NE(recvbuff, nullptr);

    auto my_rank = rank;

    if (rank == 0) {
        std::cout << "\n=== Multi-Queue Event Loop Test ===\n";
        std::cout << "Simulating " << kNumIterations
                  << " async events with separate queues\n";
    }

    // Event loop: each iteration creates a new queue and submits work
    for (int iter = 0; iter < kNumIterations; iter++) {
        // Create a new queue for this "event" (simulating event handling)
        // Don't store it - just use it and let it go out of scope
        sycl::queue new_queue(sycl_queue.get_context(), sycl_queue.get_device(),
                              {sycl::property::queue::in_order{}});

        if (rank == 0) {
            std::cout << "Event " << iter
                      << ": Creating queue and submitting work...\n";
        }

        // Phase 1: Pre-processing kernel (slow but simple)
        new_queue.submit([&](sycl::handler &h) {
            h.parallel_for<class event_loop_preprocess>(
                sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                    // Simple computation: rank * 1000 + iter * 100 + idx
                    int value =
                        my_rank * 1000 + iter * 100 + static_cast<int>(idx);

                    // Volatile loop to slow down execution for visible overlap
                    volatile int dummy = 0;
                    for (int i = 0; i < kSlowLoopIterations; i++) {
                        dummy++;
                    }

                    sendbuff[idx] = value;
                });
        });

        // Phase 2: AllReduce collective
        auto result = onecclAllReduce(sendbuff, recvbuff, kCount, onecclInt,
                                      onecclSum, comm, &new_queue);
        ASSERT_EQ(result, onecclSuccess);

        // Phase 3: Post-processing kernel (simple verification work)
        new_queue.submit([&](sycl::handler &h) {
            h.parallel_for<class event_loop_postprocess>(
                sycl::range<1>(kCount), [=](sycl::id<1> idx) {
                    // Simple post-processing: add iteration number
                    int value = recvbuff[idx] + iter;

                    // Volatile loop to slow down
                    volatile int dummy = 0;
                    for (int i = 0; i < kSlowLoopIterations; i++) {
                        dummy++;
                    }

                    sendbuff[idx] = value;
                });
        });

        if (rank == 0) {
            std::cout << "Event " << iter
                      << ": Work submitted (executing asynchronously)...\n";
        }
    }

    if (rank == 0) {
        std::cout
            << "\nAll events submitted. Queues are executing in parallel...\n";
        std::cout
            << "Calling onecclCommFinalize() to wait for all operations...\n";
    }

    // Wait for all operations across all queues
    auto start_time = std::chrono::high_resolution_clock::now();
    auto result = onecclCommFinalize(comm);
    auto end_time = std::chrono::high_resolution_clock::now();
    ASSERT_EQ(result, onecclSuccess);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

    if (rank == 0) {
        std::cout << "Finalize completed in " << duration << " ms\n";
        std::cout << "All " << kNumIterations
                  << " events completed successfully!\n";
    }

    // Verify results - use the LAST iteration's computation
    // Expected: sum across all ranks of (rank * 1000 + (kNumIterations-1) * 100
    // + idx) + (kNumIterations-1)
    std::vector<int> host_sendbuff(kCount);
    sycl_queue.memcpy(host_sendbuff.data(), sendbuff, kCount * sizeof(int))
        .wait();

    // Verify a few values
    for (size_t i = 0; i < std::min<size_t>(kCount, 10); i++) {
        int expected = 0;
        for (int r = 0; r < world_size; r++) {
            expected +=
                r * 1000 + (kNumIterations - 1) * 100 + static_cast<int>(i);
        }
        expected += (kNumIterations - 1); // Post-processing adds last iter

        ASSERT_EQ(host_sendbuff[i], expected)
            << "Rank " << rank << " mismatch at index " << i << ": expected "
            << expected << ", got " << host_sendbuff[i];
    }

    if (rank == 0) {
        std::cout << "Result verification passed!\n";
        std::cout << "=== Multi-Queue Event Loop Test Complete ===\n\n";
    }

    // Cleanup
    sycl::free(sendbuff, sycl_queue);
    sycl::free(recvbuff, sycl_queue);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    int const result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}

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

enum device_id : uint32_t {
    unknown = 0x0,
    id1 = 0x200,
    id2 = 0xbd0,
    id3 = 0xb60,
    id4 = 0x56a0, /* ARC A-series */
    id5 = 0xe200, /* ARC B-series */
    id6 = 0xe210, /* ARC B-series DT6 */
    id7 = 0xe220, /* ARC B-series DT1, DT2, DT3 */
};

// inline: avoid conflicts with oneCCL symbols
inline bool is_arc_card(ze_device_handle_t device) {
#if defined(CCL_SYCL_ENABLE_ARCA) || defined(CCL_SYCL_ENABLE_ARCB)
    return true;
#else
    return false;
#endif
}

enum class TestScope {
    LimitedNoFallback,
    PVC, // no pt2pt tests
    Full,
};

void run_regular_tests() {
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

    sycl::queue temp;
    sycl::device temp_device = temp.get_device();

    // TestScope ts = is_arc_card(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(temp_device))
    //                    ? TestScope::LimitedNoFallback
    //                    : TestScope::Full;
    TestScope ts = TestScope::PVC; // arc cards should be fully supported

    if (ts == TestScope::LimitedNoFallback) {
        std::cout << "Running on ARC card, limited scope of SYCL GRAPH testing !" << std::endl;
    }

    size_t msg_size = 1;
    size_t iterations = 13;

    if (ts == TestScope::LimitedNoFallback) {
        msg_size = 8;
        iterations = 11;
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
    TestInstancePt2Pt test_pt2pt(msg_size, size, rank, devices, context, create_mpi_kvs, ccl::create_communicator);

    for (size_t i = 0; i < iterations; ++i) {
        test_allreduce.reinit(msg_size);
        run_test_scenario(test_allreduce, 10, 1);
        run_test_scenario(test_allreduce, 2, 10);

        test_reduce_scatter.reinit(msg_size);
        run_test_scenario(test_reduce_scatter, 10, 1);
        run_test_scenario(test_reduce_scatter, 2, 10);

        test_allgather.reinit(msg_size);
        run_test_scenario(test_allgather, 10, 1);
        run_test_scenario(test_allgather, 2, 10);

        if (ts != TestScope::LimitedNoFallback) {
            test_alltoall.reinit(msg_size);
            run_test_scenario(test_alltoall, 10, 1);
            run_test_scenario(test_alltoall, 2, 10);

            // test_broadcast.reinit(msg_size);
            // run_test_scenario(test_broadcast, 10, 1);
            // run_test_scenario(test_broadcast, 2, 10);

            if (ts != TestScope::PVC) {
                test_pt2pt.reinit(msg_size);
                run_test_scenario(test_pt2pt, 10, 1);
                run_test_scenario(test_pt2pt, 2, 10);
            }
        }

        // check if internal state is properly managed
        // internal state: offset buf indices
        run_test_scenario(test_allreduce, 1, 1);
        run_test_scenario(test_reduce_scatter, 1, 1);
        // run_test_scenario(test_allgather, 1, 1);
        if (ts != TestScope::LimitedNoFallback) {
            run_test_scenario(test_alltoall, 1, 1);
            // run_test_scenario(test_broadcast, 1, 1);
            if (ts != TestScope::PVC) {
                run_test_scenario(test_pt2pt, 1, 1);
            }
        }
        run_test_scenario(test_allreduce, 2, 2);
        run_test_scenario(test_reduce_scatter, 2, 2);
        run_test_scenario(test_allgather, 2, 2);
        if (ts != TestScope::LimitedNoFallback) {
            run_test_scenario(test_alltoall, 2, 2);
            // run_test_scenario(test_broadcast, 2, 2);
            if (ts != TestScope::PVC) {
                run_test_scenario(test_pt2pt, 2, 2);
            }
        }
        run_test_scenario(test_allreduce, 1, 1);
        run_test_scenario(test_reduce_scatter, 1, 1);
        // run_test_scenario(test_allgather, 1, 1);
        if (ts != TestScope::LimitedNoFallback) {
            run_test_scenario(test_alltoall, 1, 1);
            // run_test_scenario(test_broadcast, 1, 1);
            if (ts != TestScope::PVC) {
                run_test_scenario(test_pt2pt, 1, 1);
            }
        }
        msg_size *= 4;
    }
    if (ts != TestScope::LimitedNoFallback) {
        msg_size = 1;
        for (size_t i = 0; i < 7; ++i) {
            test_allreduce.reinit(msg_size);
            run_test_scenario(test_allreduce, 10, 1);
            run_test_scenario(test_allreduce, 2, 10);

            test_reduce_scatter.reinit(msg_size);
            run_test_scenario(test_reduce_scatter, 10, 1);
            run_test_scenario(test_reduce_scatter, 2, 10);

            test_allgather.reinit(msg_size);
            run_test_scenario(test_allgather, 10, 1);
            run_test_scenario(test_allgather, 2, 10);

            test_alltoall.reinit(msg_size);
            run_test_scenario(test_alltoall, 10, 1);
            run_test_scenario(test_alltoall, 2, 10);

            // test_broadcast.reinit(msg_size); // introduced
            // run_test_scenario(test_broadcast, 10, 1);
            // run_test_scenario(test_broadcast, 2, 10);

            if (ts != TestScope::PVC) {
                test_pt2pt.reinit(msg_size);
                run_test_scenario(test_pt2pt, 10, 1);
                run_test_scenario(test_pt2pt, 2, 10);
            }

            // check if internal state is properly managed
            // internal state: offset buf indices
            run_test_scenario(test_allreduce, 1, 1);
            run_test_scenario(test_reduce_scatter, 1, 1);
            run_test_scenario(test_allgather, 1, 1);
            run_test_scenario(test_alltoall, 1, 1);
            // run_test_scenario(test_broadcast, 1, 1);
            if (ts != TestScope::PVC) {
                run_test_scenario(test_pt2pt, 1, 1);
            }
            run_test_scenario(test_allreduce, 2, 2);
            run_test_scenario(test_reduce_scatter, 2, 2);
            run_test_scenario(test_allgather, 2, 2);
            run_test_scenario(test_alltoall, 2, 2);
            // run_test_scenario(test_broadcast, 2, 2);
            if (ts != TestScope::PVC) {
                run_test_scenario(test_pt2pt, 2, 2);
            }
            run_test_scenario(test_allreduce, 1, 1);
            run_test_scenario(test_reduce_scatter, 1, 1);
            run_test_scenario(test_allgather, 1, 1);
            run_test_scenario(test_alltoall, 1, 1);
            // run_test_scenario(test_broadcast, 1, 1);
            if (ts != TestScope::PVC) {
                run_test_scenario(test_pt2pt, 1, 1);
            }

            msg_size *= 10;
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        run_regular_tests();
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

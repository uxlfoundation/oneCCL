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

// Tests invoke_p2p_barrier_pair() — GPU barrier between any 2 arbitrary ranks.
//
// Runs under GTest. Launched under mpiexec; MPI_Init / MPI_Finalize bracket the
// gtest main, and the shared comm/stream/coll_init state is set up once by a
// process-wide ::testing::Environment (CclEnv). Each TEST walks every unordered
// rank pair in one sub-comm and exercises invoke_p2p_barrier_pair on just those
// two ranks while the rest of the world synchronizes via MPI_Barrier.
//
// This binary links against the ccl target (shared) and uses internal oneCCL
// headers, so it can call invoke_p2p_barrier_pair() and reach the underlying
// sub-comms directly — no public API shim needed.

#include <iostream>
#include <memory>

#include <gtest/gtest.h>
#include <mpi.h>
#include <sycl/sycl.hpp>

#include "oneapi/ccl.hpp"

#include "comm/comm.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

namespace ccl {
namespace v1 {
// Same shape as the impl_dispatch defined in src/comm/comm.cpp. ccl::v1::impl_dispatch
// is declared as friend of ccl::communicator, so it can access the protected
// get_impl() method. Redeclaring it here gives the test the same privilege.
struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t &operator()(const Object &obj) {
        return obj.get_impl();
    }
};
} // namespace v1
} // namespace ccl

namespace {

// Process-wide state: comm/stream are created once and reused across all tests.
struct CclEnv : public ::testing::Environment {
    static CclEnv *instance;

    int world_size = 0;
    int world_rank = 0;

    std::unique_ptr<sycl::queue> q;
    ccl::shared_ptr_class<ccl::kvs> kvs;
    std::unique_ptr<ccl::communicator> comm;
    std::unique_ptr<ccl::stream> stream;

    ccl_comm *cc = nullptr;
    std::shared_ptr<ccl_comm> node_comm;

    CclEnv() {
        instance = this;
    }

    void SetUp() override {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        ASSERT_GE(world_size, 2) << "p2p_barrier_pair requires >= 2 ranks";

        // Matches the other SYCL reg_tests: in_order queue on the GPU selector
        // (mpiexec rank-to-device affinity puts each rank on a distinct GPU).
        sycl::property_list props{ sycl::property::queue::in_order{} };
        q = std::make_unique<sycl::queue>(sycl::gpu_selector_v, props);

        ccl::kvs::address_type main_addr;
        if (world_rank == 0) {
            kvs = ccl::create_main_kvs();
            main_addr = kvs->get_address();
            MPI_Bcast(main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        }
        else {
            MPI_Bcast(main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            kvs = ccl::create_kvs(main_addr);
        }

        auto dev = ccl::create_device(q->get_device());
        auto ctx = ccl::create_context(q->get_context());
        comm = std::make_unique<ccl::communicator>(
            ccl::create_communicator(world_size, world_rank, dev, ctx, kvs));
        stream = std::make_unique<ccl::stream>(ccl::create_stream(*q));

        ccl::v1::impl_dispatch disp;
        cc = static_cast<ccl_comm *>(disp(*comm).get());
        ccl_stream *cclstream = get_stream_ptr(disp(*stream));

        // Trigger coll_init — allocates the pair_flag IPC buffers and sets up
        // the flag/barrier pointers that p2p_barrier_pair needs for every sub-comm.
        coll_init(cc, cclstream);

        node_comm = cc->get_node_comm();
    }
};

CclEnv *CclEnv::instance = nullptr;

// Helper: run invoke_p2p_barrier_pair for every unordered pair (a, b) in the
// given sub-comm. Only the two ranks in the pair call the barrier (num_iters
// back-to-back invocations); everyone else — in the sub-comm and outside it —
// waits at MPI_COMM_WORLD's barrier so the pairs never overlap and every rank
// advances in lockstep through the schedule. A completed MPI_Barrier after
// each pair is the proof-of-liveness: a hang or desync would stall it.
//
// Rank 0 prints each scheduled (a, b) pair so the test log shows the full
// coverage that was exercised (rank ids are sub-comm-local).
void exercise_all_pairs(const std::shared_ptr<ccl_comm> &sub_comm,
                        const char *label,
                        sycl::queue &q,
                        int num_iters) {
    const int size = sub_comm->size();
    const int rank = sub_comm->rank();
    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    for (int a = 0; a < size; a++) {
        for (int b = a + 1; b < size; b++) {
            if (world_rank == 0) {
                std::cout << "  [" << label << "] testing pair (" << a << ", " << b << ")"
                          << std::endl;
            }
            if (rank == a || rank == b) {
                int peer = (rank == a) ? b : a;
                for (int i = 0; i < num_iters; i++) {
                    invoke_p2p_barrier_pair(sub_comm, peer, q, {}).wait();
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
}

// Number of back-to-back invocations per pair. Any value >= 2 is enough to
// catch a broken counter or stuck-flag bug; kept at 16 so a handful of pairs
// still produce visible work without bloating test time.
constexpr int num_iters = 16;

// Walk every unordered pair (a, b) with a < b in node_comm. This stresses the
// per-peer counter independence: between pairs that share a rank (e.g. (0,1)
// then (0,2)), rank 0's counter for peer 1 must not leak into its counter for
// peer 2.
TEST(PairBarrier, AllPairsNodeComm) {
    auto &env = *CclEnv::instance;
    exercise_all_pairs(env.node_comm, "node_comm", *env.q, num_iters);
    SUCCEED();
}

// Same all-pairs walk, but on even_comm. coll_init wires pair-flag state on
// every sub_comm (node_comm, numa_comm, even_comm, pair_comm) inside the
// shared sub_comms loop, so invoke_p2p_barrier_pair works uniformly on any
// of them. Rank ids passed to the helper are even_comm-local.
TEST(PairBarrier, AllPairsEvenComm) {
    auto &env = *CclEnv::instance;
    auto even_comm = env.cc->get_even_comm();
    if (even_comm->size() < 2) {
        if (env.world_rank == 0) {
            std::cout << "[  SKIPPED ] even_comm size < 2\n";
        }
        return;
    }
    exercise_all_pairs(even_comm, "even_comm", *env.q, num_iters);
    SUCCEED();
}

// Same all-pairs walk, on pair_comm. On most topologies pair_comm has size 2
// (one tile pair), so this degenerates to a single (0, 1) loop — but that's
// still a useful smoke test that the pair-flag state is wired on pair_comm.
TEST(PairBarrier, AllPairsPairComm) {
    auto &env = *CclEnv::instance;
    auto pair_comm = env.cc->get_pair_comm();
    if (pair_comm->size() < 2) {
        if (env.world_rank == 0) {
            std::cout << "[  SKIPPED ] pair_comm size < 2\n";
        }
        return;
    }
    exercise_all_pairs(pair_comm, "pair_comm", *env.q, num_iters);
    SUCCEED();
}

} // namespace

int main(int argc, char *argv[]) {
    ccl::init();
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CclEnv());
    int rc = RUN_ALL_TESTS();
    MPI_Finalize();
    return rc;
}

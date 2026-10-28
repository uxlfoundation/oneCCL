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

// Tests the SYCL-side simple pt2pt protocols — both the fused
// single-kernel variant (send_simple/recv_simple) and the q.memcpy +
// barrier-kernels variant (send_memcpy/recv_memcpy).
//
// Calls the algorithm functions directly through internal headers so the
// is_arc_card() dispatch gate in send_sycl.cpp/recv_sycl.cpp is bypassed
// and the kernels run on any SYCL GPU. Mirrors p2p_barrier_pair.cpp for
// process-wide gtest setup (CclEnv, MPI_Init bracket, impl_dispatch trick).
//
// The substantive coverage is the iteration loop: each protocol's chunk
// loop emits two p2p_barrier_pair calls — the second one is what makes
// staging-slot reuse safe across iterations. A regression there hangs
// or corrupts iter i+1.

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>
#include <sycl/sycl.hpp>

#include "oneapi/ccl.hpp"

#include "comm/comm.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/send/sycl/send_simple.hpp"
#include "coll/algorithms/recv/sycl/recv_simple.hpp"
#include "coll/algorithms/send/sycl/send_memcpy.hpp"
#include "coll/algorithms/recv/sycl/recv_memcpy.hpp"

namespace ccl {
namespace v1 {
// Same shape as the impl_dispatch defined in src/comm/comm.cpp. Re-declared
// here so this test can reach the protected get_impl() on ccl::communicator
// and ccl::stream — same trick as p2p_barrier_pair.cpp.
struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t &operator()(const Object &obj) {
        return obj.get_impl();
    }
};
} // namespace v1
} // namespace ccl

namespace {

struct CclEnv : public ::testing::Environment {
    static CclEnv *instance;

    int world_size = 0;
    int world_rank = 0;

    std::unique_ptr<sycl::queue> q;
    ccl::shared_ptr_class<ccl::kvs> kvs;
    std::unique_ptr<ccl::communicator> comm;
    std::unique_ptr<ccl::stream> stream;

    ccl_comm *cc = nullptr;
    ccl_stream *cclstream = nullptr;
    std::shared_ptr<ccl_comm> node_comm;

    CclEnv() {
        instance = this;
    }

    void SetUp() override {
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        ASSERT_GE(world_size, 2) << "p2p_simple_pair requires >= 2 ranks";

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
        cclstream = get_stream_ptr(disp(*stream));

        // Sets up pair_flag IPC buffers and the tmp staging slots that
        // send_simple/recv_simple read via node_comm->pair_flag_data() and
        // node_comm->get_all_tmp_bufs(true).
        coll_init(cc, cclstream);

        node_comm = cc->get_node_comm();
    }
};

CclEnv *CclEnv::instance = nullptr;

// Iteration count. Anything >= 2 covers the staging-slot reuse barrier; 16
// gives a few cycles of headroom without bloating runtime.
constexpr int num_iters = 16;
// Sized to force a multi-work-group kernel launch (default work_group_size=64,
// vec_size=16 for fp32 → one WG covers 1024 floats). 4096 spans 4 WGs, which
// is what send_ll handles routinely; send_simple must match the same range.
constexpr size_t count = 4096;

// Function-pointer signatures for send/recv: same shape across the simple
// (kernel) and memcpy variants, so both can be plugged into the same
// round-trip body.
using send_fn_t = ccl::event (*)(const void *,
                                 size_t,
                                 ccl::datatype,
                                 int,
                                 ccl_comm *,
                                 ccl_stream *,
                                 const ccl::vector_class<ccl::event> &,
                                 bool &);
using recv_fn_t = ccl::event (*)(void *,
                                 size_t,
                                 ccl::datatype,
                                 int,
                                 ccl_comm *,
                                 ccl_stream *,
                                 const ccl::vector_class<ccl::event> &,
                                 bool &);

void run_roundtrip_iters(send_fn_t send_fn, recv_fn_t recv_fn) {
    auto &env = *CclEnv::instance;

    // Only ranks 0 and 1 participate. Other ranks just keep MPI_COMM_WORLD
    // alive so the inter-iter MPI_Barrier doesn't deadlock.
    const int sender = 0;
    const int receiver = 1;
    if (env.world_rank != sender && env.world_rank != receiver) {
        for (int i = 0; i < num_iters; i++) {
            MPI_Barrier(MPI_COMM_WORLD);
        }
        SUCCEED();
        return;
    }

    const int peer = (env.world_rank == sender) ? receiver : sender;
    ASSERT_NE(peer, env.world_rank);

    using T = float;
    const size_t bytes = count * sizeof(T);
    T *dev_buf = sycl::malloc_device<T>(count, *env.q);
    ASSERT_NE(dev_buf, nullptr);

    std::vector<T> host(count);

    for (int iter = 0; iter < num_iters; iter++) {
        if (env.world_rank == sender) {
            for (size_t i = 0; i < count; i++) {
                host[i] = static_cast<T>(iter * 1000 + static_cast<int>(i));
            }
            env.q->memcpy(dev_buf, host.data(), bytes).wait();

            bool done = false;
            ccl::event ev = send_fn(dev_buf,
                                    count,
                                    ccl::datatype::float32,
                                    peer,
                                    env.cc,
                                    env.cclstream,
                                    {},
                                    done);
            ev.wait();
            ASSERT_TRUE(done);
        }
        else {
            bool done = false;
            ccl::event ev = recv_fn(dev_buf,
                                    count,
                                    ccl::datatype::float32,
                                    peer,
                                    env.cc,
                                    env.cclstream,
                                    {},
                                    done);
            ev.wait();
            ASSERT_TRUE(done);

            std::fill(host.begin(), host.end(), T{ 0 });
            env.q->memcpy(host.data(), dev_buf, bytes).wait();
            // EXPECT_EQ (not ASSERT_EQ) so a mismatch logs the failure but
            // still lets us reach the trailing MPI_Barrier — otherwise the
            // sender deadlocks. Cap the per-iter log volume at 4 lines.
            int reported = 0;
            for (size_t i = 0; i < count; i++) {
                T expected = static_cast<T>(iter * 1000 + static_cast<int>(i));
                if (host[i] != expected && reported < 4) {
                    EXPECT_EQ(host[i], expected) << "iter=" << iter << " i=" << i;
                    ++reported;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    sycl::free(dev_buf, *env.q);
}

TEST(SimplePt2pt, RoundtripIters) {
    run_roundtrip_iters(&send_simple, &recv_simple);
}

TEST(MemcpyPt2pt, RoundtripIters) {
    run_roundtrip_iters(&send_memcpy, &recv_memcpy);
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

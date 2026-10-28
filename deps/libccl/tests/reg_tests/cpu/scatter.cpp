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

#include "base.hpp"

// CPU scatter reg-test — exercises host ccl::scatter across warmup / persistent / regular
// attr-caching modes. NCCL semantics: root's send_buf holds size*msg_count elements, block r
// at offset r*msg_count; each rank's recv_buf receives its own count-sized block.
//
// In-place (argv[1]==1): at root, recv_buf aliases send_buf + rank*msg_count; non-root ranks
// pass nullptr as send_buf.

void run_collective(const char* cmd_name,
                    const std::vector<float>& send_buf,
                    float* recv_buf,
                    const size_t recv_count,
                    const ccl::communicator& comm,
                    const ccl::scatter_attr& attr,
                    const bool in_place,
                    const int root) {
    std::chrono::system_clock::duration exec_time{ 0 };

    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        const float* send_ptr =
            (comm.rank() == root) ? send_buf.data() : nullptr;
        ccl::scatter(send_ptr, recv_buf, recv_count, root, comm, attr).wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    // Each rank expects its own rank id as the fill value for its block
    // (send buffer is initialized so block r contains value r).
    const float expected = static_cast<float>(comm.rank());

    // In-place at root: recv_buf aliases send_buf's root block, which was seeded with
    // root's rank value from init — same expected value. Verify for all ranks.
    for (size_t idx = 0; idx < recv_count; idx++) {
        if (recv_buf[idx] != expected) {
            fprintf(stderr,
                    "idx %zu, expected %4.4f, got %4.4f\n",
                    idx,
                    expected,
                    recv_buf[idx]);
            std::cout << "FAILED" << std::endl;
            std::terminate();
        }
    }

    ccl::barrier(comm);

    std::cout << "avg time of " << cmd_name << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(exec_time).count() / ITERS
              << ", us" << std::endl;
}

int main(int argc, char* argv[]) {
    ccl::init();

    bool in_place = false;

    if (argc > 1) {
        in_place = atoi(argv[1]);
    }

    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

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

    auto comm = ccl::create_communicator(size, rank, kvs);
    auto attr = ccl::create_operation_attr<ccl::scatter_attr>();

    const int root = 0;

    MSG_LOOP(
        comm,
        // Root builds size*msg_count send buffer with block r filled with value r; non-root
        // ranks only allocate a placeholder (nullptr passed to ccl::scatter).
        std::vector<float> send_buf;
        if (comm.rank() == root) {
            send_buf.resize(msg_count * comm.size());
            for (int r = 0; r < comm.size(); ++r) {
                std::fill(send_buf.begin() + r * msg_count,
                          send_buf.begin() + (r + 1) * msg_count,
                          static_cast<float>(r));
            }
        }
        std::vector<float> recv_buf_separate(msg_count, 0);
        float* recv_buf = nullptr;

        if (in_place && comm.rank() == root) {
            recv_buf = &(send_buf[comm.rank() * msg_count]);
        }
        else {
            recv_buf = recv_buf_separate.data();
        }

        attr.set<ccl::operation_attr_id::to_cache>(false);
        run_collective("warmup scatter", send_buf, recv_buf, msg_count, comm, attr, in_place, root);
        attr.set<ccl::operation_attr_id::to_cache>(true);
        run_collective("persistent scatter", send_buf, recv_buf, msg_count, comm, attr, in_place, root);
        attr.set<ccl::operation_attr_id::to_cache>(false);
        run_collective("regular scatter", send_buf, recv_buf, msg_count, comm, attr, in_place, root););

    return 0;
}

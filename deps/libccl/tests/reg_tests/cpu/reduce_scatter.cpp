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

void run_collective(const char* cmd_name,
                    const std::vector<float>& send_buf,
                    float* recv_buf,
                    const size_t recv_count,
                    const ccl::communicator& comm,
                    const ccl::reduce_scatter_attr& attr,
                    const bool in_place) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float expected = (comm.size() - 1) * (static_cast<float>(comm.size()) / 2);

    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        ccl::reduce_scatter(send_buf.data(), recv_buf, recv_count, ccl::reduction::sum, comm, attr)
            .wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    // Note: in in-place case, send buffer is updated with the reduction result on each iteration.
    // Therefore, each iteration accumulates the new reduction result plus previous reduction.
    // It is possible to update the formula with something like this:
    // "expected += (ITERS - 1) * (expected - comm.rank())". However, we have no control how many times
    // the function is executed, including the warmup. Therefore, decided to skip correctness check
    // for in-place case.
    if (!in_place) {
        for (size_t idx = 0; idx < recv_count; idx++) {
            if (recv_buf[idx] != expected) {
                fprintf(
                    stderr, "idx %zu, expected %4.4f, got %4.4f\n", idx, expected, recv_buf[idx]);

                std::cout << "FAILED" << std::endl;
                std::terminate();
            }
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
    auto attr = ccl::create_operation_attr<ccl::reduce_scatter_attr>();

    MSG_LOOP(
        comm, std::vector<float> send_buf(msg_count * comm.size(), static_cast<float>(comm.rank()));
        std::vector<float> recv_buf_separate(msg_count, 0);
        float* recv_buf = nullptr;

        if (in_place) { recv_buf = &(send_buf[comm.rank() * msg_count]); } else {
            recv_buf = recv_buf_separate.data();
        }

        attr.set<ccl::operation_attr_id::to_cache>(false);
        run_collective(
            "warmup reduce_scatter", send_buf, recv_buf, msg_count, comm, attr, in_place);
        attr.set<ccl::operation_attr_id::to_cache>(true);
        run_collective(
            "persistent reduce_scatter", send_buf, recv_buf, msg_count, comm, attr, in_place);
        attr.set<ccl::operation_attr_id::to_cache>(false);
        run_collective(
            "regular reduce_scatter", send_buf, recv_buf, msg_count, comm, attr, in_place););

    return 0;
}

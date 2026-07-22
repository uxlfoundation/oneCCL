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
                    float* send_buf,
                    size_t send_count,
                    std::vector<float>& recv_buf,
                    std::vector<size_t>& recv_counts,
                    const ccl::communicator& comm,
                    const ccl::allgatherv_attr& attr) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float expected = send_count;
    float received;

    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        ccl::allgatherv(send_buf, send_count, recv_buf.data(), recv_counts, comm, attr).wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    for (size_t idx = 0; idx < recv_buf.size(); idx++) {
        received = recv_buf[idx];
        if (received != expected) {
            fprintf(stderr, "idx %zu, expected %4.4f, got %4.4f\n", idx, expected, received);

            std::cout << "FAILED" << std::endl;
            std::terminate();
        }
    }

    ccl::barrier(comm);

    std::cout << "avg time of " << cmd_name << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(exec_time).count() / ITERS
              << ", us" << std::endl;
}

void run_collective_vector(const char* cmd_name,
                           float* send_buf,
                           size_t send_count,
                           std::vector<float*>& recv_bufs,
                           std::vector<size_t>& recv_counts,
                           const ccl::communicator& comm,
                           const ccl::allgatherv_attr& attr) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float expected = send_count;
    float received;

    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        ccl::allgatherv(send_buf, send_count, recv_bufs, recv_counts, comm, attr).wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    for (size_t buf_idx = 0; buf_idx < recv_bufs.size(); buf_idx++) {
        float* recv_buf = recv_bufs[buf_idx];
        for (size_t idx = 0; idx < recv_counts[buf_idx]; idx++) {
            received = recv_buf[idx];
            if (received != expected) {
                fprintf(stderr, "idx %zu, expected %4.4f, got %4.4f\n", idx, expected, received);
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
    auto kvs_attr = ccl::create_kvs_attr();
    if (rank == 0) {
        kvs = ccl::create_main_kvs(kvs_attr);
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr, kvs_attr);
    }

    auto dev = ccl::create_device();
    auto ctx = ccl::create_context();
    auto comm_attr = ccl::create_comm_attr();
    // an example on how to set blocking
    comm_attr.set<ccl::comm_attr_id::blocking>(-1);
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs, comm_attr);
    auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();

    MSG_LOOP(
        comm,

        std::vector<float> send_buf_separate(msg_count, static_cast<float>(msg_count));
        std::vector<float> recv_buf(comm.size() * msg_count, static_cast<float>(msg_count));
        std::vector<float*> recv_bufs(comm.size(), nullptr);
        std::vector<size_t> recv_counts(comm.size(), msg_count);

        for (int rank_idx = 0; rank_idx < comm.size(); rank_idx++) recv_bufs[rank_idx] =
            new float[msg_count];

        float* send_buf = nullptr;
        if (in_place) { send_buf = &(recv_buf[comm.rank() * msg_count]); } else {
            send_buf = send_buf_separate.data();
        }

        attr.set<ccl::operation_attr_id::to_cache>(false);
        run_collective("warmup_allgatherv", send_buf, msg_count, recv_buf, recv_counts, comm, attr);
        run_collective_vector(
            "warmup_allgatherv_vector", send_buf, msg_count, recv_bufs, recv_counts, comm, attr);

        ccl::string_class regular_match_id = std::to_string(msg_count);
        ccl::string_class vector_match_id = regular_match_id + std::string("_vector");
        attr.set<ccl::operation_attr_id::match_id>(regular_match_id);
        attr.set<ccl::operation_attr_id::to_cache>(true);
        run_collective(
            "persistent_allgatherv", send_buf, msg_count, recv_buf, recv_counts, comm, attr);
        attr.set<ccl::operation_attr_id::match_id>(vector_match_id);
        run_collective_vector("persistent_allgatherv_vector",
                              send_buf,
                              msg_count,
                              recv_bufs,
                              recv_counts,
                              comm,
                              attr);

        attr.set<ccl::operation_attr_id::to_cache>(false);
        run_collective(
            "regular_allgatherv", send_buf, msg_count, recv_buf, recv_counts, comm, attr);
        run_collective_vector(
            "regular_allgatherv_vector", send_buf, msg_count, recv_bufs, recv_counts, comm, attr););

    return 0;
}

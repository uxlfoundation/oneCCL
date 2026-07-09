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
                    std::vector<float>& recv_buf,
                    size_t count,
                    const ccl::communicator& comm,
                    const ccl::allgather_attr& attr) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float expected = count;
    float received;

    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        ccl::allgather(send_buf, recv_buf.data(), count, comm, attr).wait();
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
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs, comm_attr);
    auto attr = ccl::create_operation_attr<ccl::allgather_attr>();

    MSG_LOOP(
        comm,

        std::vector<float> send_buf_separate(msg_count, static_cast<float>(msg_count));
        std::vector<float> recv_buf(comm.size() * msg_count, static_cast<float>(msg_count));

        float* send_buf = nullptr;
        if (in_place) { send_buf = &(recv_buf[comm.rank() * msg_count]); } else {
            send_buf = send_buf_separate.data();
        }

        attr.set<ccl::operation_attr_id::to_cache>(false);
        run_collective("warmup_allgather", send_buf, recv_buf, msg_count, comm, attr);
        ccl::string_class regular_match_id = std::to_string(msg_count);
        ccl::string_class vector_match_id = regular_match_id + std::string("_vector");
        attr.set<ccl::operation_attr_id::match_id>(regular_match_id);
        attr.set<ccl::operation_attr_id::to_cache>(true);
        run_collective("persistent_allgather", send_buf, recv_buf, msg_count, comm, attr);
        attr.set<ccl::operation_attr_id::to_cache>(false);
        run_collective("regular_allgather", send_buf, recv_buf, msg_count, comm, attr););

    return 0;
}

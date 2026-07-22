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

#define EVEN_RANK_SEND_COUNT 100
#define ODD_RANK_SEND_COUNT  200

void run_collective(const char* cmd_name,
                    std::vector<int>& send_buf,
                    std::vector<int>& recv_buf,
                    std::vector<size_t>& send_counts,
                    std::vector<size_t>& recv_counts,
                    const ccl::communicator& comm,
                    const ccl::alltoallv_attr& attr) {
    std::chrono::system_clock::duration exec_time{ 0 };

    ccl::barrier(comm);

    size_t iter_idx;
    for (iter_idx = 0; iter_idx < ITERS; iter_idx++) {
        std::fill(recv_buf.begin(), recv_buf.end(), 0);
        size_t elem_idx = 0;

        for (int rank_idx = 0; rank_idx < comm.size(); rank_idx++) {
            for (size_t idx = 0; idx < send_counts[rank_idx]; idx++) {
                send_buf[elem_idx] = comm.rank();
                elem_idx++;
            }
        }

        auto start = std::chrono::system_clock::now();
        ccl::alltoallv(send_buf.data(), send_counts, recv_buf.data(), recv_counts, comm, attr)
            .wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    ccl::barrier(comm);

    size_t elem_idx = 0;
    for (int rank_idx = 0; rank_idx < comm.size(); rank_idx++) {
        int expected = rank_idx;
        for (size_t idx = 0; idx < recv_counts[rank_idx]; idx++) {
            if (recv_buf[elem_idx] != expected) {
                printf("iter %zu, idx %zu, expected %d, got %d\n",
                       iter_idx,
                       elem_idx,
                       expected,
                       recv_buf[elem_idx]);
                ASSERT(0, "unexpected value");
            }
            elem_idx++;
        }
    }

    ccl::barrier(comm);

    std::cout << "avg time of " << cmd_name << ": "
              << std::chrono::duration_cast<std::chrono::microseconds>(exec_time).count() / ITERS
              << ", us" << std::endl;
}

int main() {
    ccl::init();

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
    auto attr = ccl::create_operation_attr<ccl::alltoallv_attr>();

    int is_even = (rank % 2 == 0) ? 1 : 0;
    size_t send_count = (is_even) ? EVEN_RANK_SEND_COUNT : ODD_RANK_SEND_COUNT;

    size_t total_send_count = send_count * size;
    size_t total_recv_count = (EVEN_RANK_SEND_COUNT + ODD_RANK_SEND_COUNT) * (size / 2);

    std::vector<int> send_buf(total_send_count);
    std::vector<int> recv_buf(total_recv_count);

    std::vector<size_t> send_counts(comm.size());
    std::vector<size_t> recv_counts(comm.size());

    for (int idx = 0; idx < comm.size(); idx++) {
        int is_even_peer = (idx % 2 == 0) ? 1 : 0;
        send_counts[idx] = send_count;
        recv_counts[idx] = (is_even_peer) ? EVEN_RANK_SEND_COUNT : ODD_RANK_SEND_COUNT;
    }

    MSG_LOOP(comm, attr.set<ccl::operation_attr_id::to_cache>(false); run_collective(
                 "warmup alltoallv", send_buf, recv_buf, send_counts, recv_counts, comm, attr);
             attr.set<ccl::operation_attr_id::to_cache>(true);
             run_collective(
                 "persistent alltoallv", send_buf, recv_buf, send_counts, recv_counts, comm, attr);
             attr.set<ccl::operation_attr_id::to_cache>(false);
             run_collective(
                 "regular alltoallv", send_buf, recv_buf, send_counts, recv_counts, comm, attr););

    return 0;
}

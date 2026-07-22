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

// Compile command:
// icpx -fsycl -lmpi -lccl  alltoallv_zero_split_gpu.cpp -o alltoallv_zero_split_gpu
// Run command:
// mpirun -np num ./alltoallv_zero_split_gpu

#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <set>
#include <string>
#include <numeric>

#include "sycl_base.hpp"

int main(int argc, char *argv[]) {
    std::string device_name = "gpu";

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);

    sycl::queue q;
    if (!create_test_sycl_queue("gpu", rank, q, args))
        return -1;

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* Ranks of [0, size/2) are active ranks, others are silent ranks([size/2, size)).*/
    int border = size / 2;

    auto is_activate = [border](int r) {
        return r >= 0 && r < border;
    };

    std::vector<size_t> send_counts(size, 0);
    std::vector<size_t> recv_counts(size, 0);
    size_t total_send_counts = 0;
    size_t send_count = 4096;
    for (size_t i = 0; i < send_counts.size(); i++) {
        if (is_activate(rank) && is_activate(i)) {
            send_counts[i] = send_count;
            total_send_counts += send_count;
        }
    }

    // get recv_size
    MPI_Alltoall(send_counts.data(), 1, MPI_LONG, recv_counts.data(), 1, MPI_LONG, MPI_COMM_WORLD);

    size_t total_receive_counts = 0;
    for (size_t i = 0; i < recv_counts.size(); i++) {
        total_receive_counts += recv_counts[i];
    }

    std::vector<int> send_buf_host(total_send_counts, -1);
    for (size_t i = 0; i < total_send_counts; ++i) {
        int rankp1 = rank + 1;
        send_buf_host[i] = rankp1 * rankp1 + i;
    }
    std::vector<int> recv_buf_host(total_receive_counts, 0);
    size_t send_bytes = send_buf_host.size() * sizeof(int);
    size_t recv_bytes = recv_buf_host.size() * sizeof(int);

    void *send_buf_device = sycl::aligned_alloc_device(4096, send_bytes, q);
    void *recv_buf_device = sycl::aligned_alloc_device(4096, recv_bytes, q);
    q.wait();

    q.memcpy(send_buf_device, send_buf_host.data(), send_bytes).wait();

    utils::dump_vec(send_counts, "sendcounts");
    utils::dump_vec(recv_counts, "recvcounts");

    if (send_buf_device == nullptr) {
        std::cout << "send buf device is nullptr\n";
    }
    else {
        std::cout << "send buf device is not nullptr\n";
    }

    if (recv_buf_device == nullptr) {
        std::cout << "recv buf device is nullptr\n";
    }
    else {
        std::cout << "recv buf device is not nullptr\n";
    }

    /* invoke alltoallv */
    ccl::alltoallv(send_buf_device,
                   send_counts,
                   recv_buf_device,
                   recv_counts,
                   ccl::datatype::int32,
                   comm,
                   stream)
        .wait();

    q.memcpy(const_cast<int *>(recv_buf_host.data()), recv_buf_device, recv_bytes).wait();

    /* print out the result of the test on the host side */
    utils::dump_vec(send_buf_host, "send buf");
    utils::dump_vec(recv_buf_host, "recv buf");
    size_t i;

    if (is_activate(rank)) {
        for (i = 0; i < total_receive_counts; i++) {
            int foreign_rankp1 = i / send_count + 1;
            int foreign_indexp1 = i % send_count + rank * send_count;
            int value = foreign_rankp1 * foreign_rankp1 + foreign_indexp1;
            if (recv_buf_host[i] != value) {
                std::cout << "FAILED\n";
                break;
            }
        }
        if (i == total_receive_counts) {
            std::cout << "PASSED\n";
        }
    }

    return 0;
}

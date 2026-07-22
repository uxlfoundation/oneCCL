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

#include <numeric>
#include <vector>
#include <iostream>

#include "sycl_base.hpp"

int main(int argc, char *argv[]) {
    int size;
    int rank;

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

    /* create buffers */
    // rank 0 sends nothing
    size_t count = rank;
    std::vector<int> send_buf_host(count * size);
    size_t recv_size = (size * (size - 1) / 2);
    std::vector<int> recv_buf_host(recv_size);

    std::vector<size_t> send_counts(size, count);
    std::vector<size_t> recv_counts(size);
    for (int i = 0; i < size; i++) {
        recv_counts[i] = i;
    }

    for (int i = 0; i < size; i++) {
        for (size_t j = 0; j < count; j++) {
            send_buf_host[(i * count) + j] = i + 1;
        }
    }

    size_t send_bytes = count * size * sizeof(int);
    size_t recv_bytes = recv_size * sizeof(int);
    void *send_buf_device = sycl::aligned_alloc_device(4096, send_bytes, q);
    void *recv_buf_device = sycl::aligned_alloc_device(4096, recv_bytes, q);
    q.wait();

    q.memcpy(send_buf_device, send_buf_host.data(), send_bytes).wait();

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
    size_t i;
    utils::dump_vec(send_buf_host, "send buf");
    utils::dump_vec(recv_buf_host, "recv buf");

    for (i = 0; i < recv_size; i++) {
        if (recv_buf_host[i] != rank + 1) {
            std::cout << "FAILED\n";
            break;
        }
    }
    if (i == recv_size) {
        std::cout << "PASSED\n";
    }

    return 0;
}

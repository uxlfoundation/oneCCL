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

#include "sycl_base.hpp"

#include <vector>
#include <iostream>

// Reproducer for issue:
// When sending/receiving to self (rank i to rank i),
// the operation hangs or produces incorrect results

bool test_self_alltoall_pt2pt(int size,
                              int rank,
                              sycl::queue& q,
                              ccl::communicator& comm,
                              ccl::stream& stream) {
    buf_allocator<int> allocator(q);
    auto usm_alloc_type = sycl::usm::alloc::device;

    // Simple reproducer: chunk_size elements per rank
    const int chunk_size = 1024;
    const int offset = chunk_size;
    const int total_elems = size * chunk_size;

    auto send_buf = allocator.allocate(total_elems, usm_alloc_type);
    auto recv_buf = allocator.allocate(total_elems, usm_alloc_type);

    // Initialize send buffer
    q.submit([&](sycl::handler& h) {
         h.parallel_for(total_elems, [=](auto id) {
             // Each rank prepares a unique block of data for every destination rank (including itself)
             // The block for destination <dst> starts at offset <dst>*chunk_size in the send buffer
             send_buf[id] = rank * total_elems + id;
         });
     }).wait();

    // Initialize recv buffer to detect unwritten data
    q.submit([&](sycl::handler& h) {
         h.parallel_for(total_elems, [=](auto id) {
             recv_buf[id] = -1;
         });
     }).wait();

    // alltoall pattern: each rank sends a unique chunk to every other rank (including itself)
    // This tests self-send/recv (rank i to rank i) along with regular sends/recvs
    ccl::group_start();
    for (int i = 0; i < size; ++i) {
        // Send to rank i
        ccl::send(send_buf + i * offset, chunk_size, ccl::datatype::int32, i, comm, stream);
        // Receive from rank i
        ccl::recv(recv_buf + i * offset, chunk_size, ccl::datatype::int32, i, comm, stream);
    }
    ccl::group_end();

    // Verify results
    std::vector<int> host_recv_buf(total_elems);
    q.memcpy(host_recv_buf.data(), recv_buf, total_elems * sizeof(int)).wait();

    bool is_correct = true;
    for (int src = 0; src < size; ++src) {
        for (int j = 0; j < chunk_size; ++j) {
            const int index = src * offset + j;
            // We receive from <src> the block that <src> prepared for destination <rank>.
            // That block starts at (rank * chunk_size) in <src>'s send buffer.
            const int expected_value = src * total_elems + rank * chunk_size + j;
            if (host_recv_buf[index] != expected_value) {
                std::cout << "Rank " << rank << " - FAILED at index " << index << " (src=" << src
                          << ", j=" << j << ")"
                          << ": Expected " << expected_value << ", got " << host_recv_buf[index]
                          << std::endl;
                is_correct = false;
                break;
            }
        }
        if (!is_correct) {
            break;
        }
    }
    return is_correct;
}

// Reproducer for issue:
// When posting recv before send for self-communication (rank i to rank i)
// Note: for general recv posted before send case with many ranks
// the ipc exchange will hang, not supported right now
bool test_reverse_self_pt2pt(int size,
                             int rank,
                             sycl::queue& q,
                             ccl::communicator& comm,
                             ccl::stream& stream) {
    buf_allocator<int> allocator(q);
    auto usm_alloc_type = sycl::usm::alloc::device;

    // Simple reproducer: chunk_size elements per rank
    const int chunk_size = 1024;

    auto send_buf = allocator.allocate(chunk_size, usm_alloc_type);
    auto recv_buf = allocator.allocate(chunk_size, usm_alloc_type);

    // Initialize send buffer
    q.submit([&](sycl::handler& h) {
         h.parallel_for(chunk_size, [=](auto id) {
             send_buf[id] = rank * chunk_size + id;
         });
     }).wait();

    // Initialize recv buffer to detect unwritten data
    q.submit([&](sycl::handler& h) {
         h.parallel_for(chunk_size, [=](auto id) {
             recv_buf[id] = -1;
         });
     }).wait();

    // Reverse order: post recv before send for self-communication
    ccl::group_start();

    // Receive from rank i
    ccl::recv(recv_buf, chunk_size, ccl::datatype::int32, rank, comm, stream);
    // Send to rank i
    ccl::send(send_buf, chunk_size, ccl::datatype::int32, rank, comm, stream);

    ccl::group_end();

    // Verify results
    std::vector<int> host_recv_buf(chunk_size);
    q.memcpy(host_recv_buf.data(), recv_buf, chunk_size * sizeof(int)).wait();

    bool is_correct = true;
    for (int i = 0; i < chunk_size; ++i) {
        const int expected_value = rank * chunk_size + i;
        if (host_recv_buf[i] != expected_value) {
            std::cout << "Rank " << rank << " - FAILED at index " << i << ": Expected "
                      << expected_value << ", got " << host_recv_buf[i] << std::endl;
            is_correct = false;
            break;
        }
    }
    return is_correct;
}

int main(int argc, char* argv[]) {
    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    sycl::queue q;
    sycl::property_list props = { sycl::property::queue::in_order{} };

    if (!create_sycl_queue(argc, argv, rank, q, props)) {
        return -1;
    }

    /* create kvs */
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

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* Test self-send/recv */
    if (test_self_alltoall_pt2pt(size, rank, q, comm, stream)) {
        std::cout << "Rank " << rank << " - test_self_alltoall_pt2pt PASSED\n";
    }
    else {
        std::cout << "Rank " << rank << " - test_self_alltoall_pt2pt FAILED\n";
    }

    if (test_reverse_self_pt2pt(size, rank, q, comm, stream)) {
        std::cout << "Rank " << rank << " - test_reverse_self_pt2pt PASSED\n";
    }
    else {
        std::cout << "Rank " << rank << " - test_reverse_self_pt2pt FAILED\n";
    }
    return 0;
}

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
#include <algorithm>
#include <iostream>

using namespace sycl;
using namespace std;

// Point-to-point Alltoall Pattern:
// Example with 4 ranks (0,1,2,3), sendcount_per_peer = N:
// - Each rank prepares N elements for each peer, laid out contiguously.
// - Rank 0 send_buf layout: [0→0, 0→1, 0→2, 0→3]
//   (for peer 0: [0..N), peer 1: [N..2N), etc.)
// - Each rank:
//     - memcpy own block locally.
//     - sends N elements to every other rank using ccl::send
//     - receives N elements from every other rank using ccl::recv
// - Result: recv_buf holds N elements from each peer at offset peer * N.

int main(int argc, char* argv[]) {
    ccl::init();

    MPI_Init(&argc, &argv);
    int size = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    sycl::property_list props = { sycl::property::queue::in_order{} };
    queue q;
    if (!create_sycl_queue(argc, argv, rank, q, props)) {
        return -1;
    }

    buf_allocator<int> allocator(q);
    auto usm_alloc_type = sycl::usm::alloc::device;

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast(main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast(main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
    auto stream = ccl::create_stream(q);

    const int sendcount_per_peer = 33554432;
    const int total_elems = sendcount_per_peer * size;

    int* send_buf = allocator.allocate(total_elems, usm_alloc_type);
    int* recv_buf = allocator.allocate(total_elems, usm_alloc_type);

    // initialise send buffe
    // Value pattern: value = rank * total_elems + idx
    q.submit([&](handler& h) {
         h.parallel_for(total_elems, [=](auto idx) {
             send_buf[idx] = rank * total_elems + idx;
         });
     }).wait();

    // local copy of self‑segment
    q.memcpy(recv_buf + rank * sendcount_per_peer,
             send_buf + rank * sendcount_per_peer,
             sendcount_per_peer * sizeof(int))
        .wait();

    std::vector<ccl::event> deps = {};
    auto attr = ccl::create_operation_attr<ccl::pt2pt_attr>();

    ccl::group_start();
    for (int peer = 0; peer < size; ++peer) {
        if (peer == rank)
            continue;

        // Send the peer‑th segment of our send buffer to <peer>
        ccl::send(send_buf + peer * sendcount_per_peer,
                  sendcount_per_peer,
                  ccl::datatype::int32,
                  peer,
                  comm,
                  stream,
                  attr,
                  deps);

        // Receive the segment that <peer> sends to us
        ccl::recv(recv_buf + peer * sendcount_per_peer,
                  sendcount_per_peer,
                  ccl::datatype::int32,
                  peer,
                  comm,
                  stream,
                  attr,
                  deps);
    }
    ccl::group_end();

    // touch send buffer to test true async behaviour
    q.submit([&](handler& h) {
        h.parallel_for(total_elems, [=](auto idx) {
            send_buf[idx] = 0xDEADBEEF;
        });
    });

    std::vector<int> host_recv_buf(static_cast<size_t>(total_elems));
    q.memcpy(host_recv_buf.data(), recv_buf, total_elems * sizeof(int)).wait();

    bool is_correct = true;

    for (int src = 0; src < size && is_correct; ++src) {
        for (int i = 0; i < sendcount_per_peer; ++i) {
            size_t idx = static_cast<size_t>(src) * sendcount_per_peer + i;
            int expected = src * total_elems + rank * sendcount_per_peer + i;
            if (host_recv_buf[idx] != expected) {
                std::cout << "Rank " << rank << " – FAILED at index " << idx << ": expected "
                          << expected << ", got " << host_recv_buf[idx] << std::endl;
                is_correct = false;
                break;
            }
        }
    }

    if (is_correct) {
        std::cout << "PASSED" << std::endl;
    }
    else {
        std::cout << "FAILED" << std::endl;
    }

    return 0;
}

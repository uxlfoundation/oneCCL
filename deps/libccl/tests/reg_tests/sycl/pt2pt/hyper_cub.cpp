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

using namespace std;
using namespace sycl;

// Hypercube Allgather Pattern:
// Example with 4 ranks (0,1,2,3), sendcount=block:
// Step 1 (mask = 1):
//   - Pairs: (0 <-> 1) and (2 <-> 3) each exchange block elements.
//   - After: ranks 0 and 1 each hold data [0,1], ranks 2 and 3 hold [2,3].
// Step 2 (mask = 2):
//   - Groups: (0 <-> 2) and (1 <-> 3) each exchange 2 * block elements.
//   - After: all ranks hold [0,1,2,3].
// General: log2(size) steps, doubling segment size each time. Partner = rank ^ mask.

int main(int argc, char* argv[]) {
    int size = 0, rank = 0;

    ccl::init();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    atexit(mpi_finalize);

    queue q;
    sycl::property_list props = { sycl::property::queue::in_order{} };
    if (!create_sycl_queue(argc, argv, rank, q, props))
        return -1;

    buf_allocator<int> allocator(q);
    auto usm_alloc_type = usm::alloc::device;

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

    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
    auto stream = ccl::create_stream(q);

    int sendcount = 33554432;
    auto send_buf = allocator.allocate(sendcount, usm_alloc_type);

    // Assume uniform recvcounts: all ranks send sendcount
    std::vector<int> recvcounts(size, sendcount);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        displs[i] = i * sendcount;
    }
    int total_recvcount = sendcount * size;
    auto recv_buf = allocator.allocate(total_recvcount, usm_alloc_type);

    q.submit([&](handler& h) {
         h.parallel_for(sendcount, [=](auto idx) {
             send_buf[idx] = rank * size + idx;
         });
     }).wait();

    // Copy local block into receive buffer
    q.memcpy(recv_buf + displs[rank], send_buf, sendcount * sizeof(int)).wait();

    // Hypercube Allgather pattern
    int block = sendcount;
    for (int mask = 1; mask < size; mask <<= 1) {
        int s = rank & ~(mask - 1);
        int r = s ^ mask;
        int count = block * mask;
        int partner = rank ^ mask;

        ccl::group_start();
        ccl::send(recv_buf + s * block, count, ccl::datatype::int32, partner, comm, stream);
        ccl::recv(recv_buf + r * block, count, ccl::datatype::int32, partner, comm, stream);
        ccl::group_end();
    }

    // Copy result back to host and verify
    std::vector<int> host_buf(total_recvcount);
    q.memcpy(host_buf.data(), recv_buf, total_recvcount * sizeof(int)).wait();

    bool passed = true;
    for (int rnk = 0; rnk < size; ++rnk) {
        for (int i = 0; i < sendcount; ++i) {
            int idx = displs[rnk] + i;
            int expected = rnk * size + i;
            if (host_buf[idx] != expected) {
                std::cerr << "Rank " << rank << " FAILED at idx " << idx << ": expected "
                          << expected << ", got " << host_buf[idx] << std::endl;
                passed = false;
                break;
            }
        }
        if (!passed)
            break;
    }

    std::cout << (passed ? "PASSED\n" : "FAILED\n");

    return 0;
}

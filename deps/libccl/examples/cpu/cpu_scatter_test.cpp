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

#include <iostream>
#include <mpi.h>
#include <vector>

#include "base.hpp"
#include "oneapi/ccl.hpp"

int main() {
    const size_t count = 128;
    const int root = COLL_ROOT;

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

    rank = comm.rank();
    size = comm.size();

    /* root owns nranks * count elements; each rank receives count elements */
    std::vector<int> send_buf(size * count, 0);
    std::vector<int> recv_buf(count, 0);

    /* root fills one big contiguous buffer of size*count elements:
     * [0, 1, 2, ..., 127, 128, 129, ..., 255, 256, ..., 383, 384, ..., 511]
     *  ^--- rank 0 block ---^  ^--- rank 1 block ---^  ...
     * scatter splits it so each rank receives its 128-element slice. */
    if (rank == root) {
        for (int i = 0; i < size * static_cast<int>(count); i++) {
            send_buf[i] = i;
        }
    }

    /* every rank prints its recv_buf before scatter, one at a time */
    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank != r)
            continue;
        std::cout << "rank " << rank << " recv_buf BEFORE scatter (first 8): ";
        for (size_t i = 0; i < std::min(count, static_cast<size_t>(8)); i++)
            std::cout << recv_buf[i] << " ";
        std::cout << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* root prints the full send_buf layout so the partition is visible */
    if (rank == root) {
        std::cout << "root send_buf (" << size << " blocks x " << count << " elems):\n";
        for (int r = 0; r < size; r++) {
            std::cout << "  block[" << r << "] (-> rank " << r << "): " << send_buf[r * count]
                      << " .. " << send_buf[(r + 1) * count - 1] << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* invoke scatter: non-root passes nullptr for send_buf */
    ccl::scatter((rank == root) ? send_buf.data() : nullptr,
                 recv_buf.data(),
                 count,
                 ccl::datatype::int32,
                 root,
                 comm)
        .wait();

    /* print one rank at a time to avoid interleaved output */
    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank != r)
            continue;

        std::cout << "rank " << rank << " recv_buf (first 8): ";
        for (size_t i = 0; i < std::min(count, static_cast<size_t>(8)); i++)
            std::cout << recv_buf[i] << " ";
        std::cout << " -- expected: ";
        for (size_t i = 0; i < std::min(count, static_cast<size_t>(8)); i++)
            std::cout << rank * static_cast<int>(count) + static_cast<int>(i) << " ";
        std::cout << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* each rank checks it received send_buf[rank*count .. (rank+1)*count) */
    bool passed = true;
    for (size_t i = 0; i < count; i++) {
        if (recv_buf[i] != rank * static_cast<int>(count) + static_cast<int>(i)) {
            passed = false;
            break;
        }
    }

    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank != r)
            continue;
        if (passed)
            std::cout << "rank " << rank << ": PASSED\n";
        else
            std::cout << "rank " << rank << ": FAILED"
                      << " (expected " << rank * static_cast<int>(count) << ", got " << recv_buf[0]
                      << " at elem 0)\n";
    }

    return 0;
}

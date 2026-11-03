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

    /* every rank contributes count elements; root receives size * count elements */
    std::vector<int> send_buf(count);
    std::vector<int> recv_buf((rank == root) ? static_cast<size_t>(size) * count : 0, 0);

    /* each rank fills its send buffer with rank * count + i */
    for (size_t i = 0; i < count; i++) {
        send_buf[i] = rank * static_cast<int>(count) + static_cast<int>(i);
    }

    /* print what each rank is sending */
    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank != r)
            continue;
        std::cout << "rank " << rank << " send_buf (first 8): ";
        for (size_t i = 0; i < std::min(count, static_cast<size_t>(8)); i++)
            std::cout << send_buf[i] << " ";
        std::cout << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* invoke gather: non-root passes nullptr for recv_buf */
    ccl::gather(send_buf.data(),
                (rank == root) ? recv_buf.data() : nullptr,
                count,
                ccl::datatype::int32,
                root,
                comm)
        .wait();

    /* root prints its recv_buf */
    if (rank == root) {
        std::cout << "root recv_buf (" << size << " blocks x " << count << " elems):\n";
        for (int r = 0; r < size; r++) {
            std::cout << "  block[" << r << "] (from rank " << r << "): "
                      << recv_buf[static_cast<size_t>(r) * count] << " .. "
                      << recv_buf[static_cast<size_t>(r + 1) * count - 1] << "\n";
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* root verifies: slot r should contain rank r's send data */
    bool passed = true;
    if (rank == root) {
        for (int r = 0; r < size; r++) {
            for (size_t i = 0; i < count; i++) {
                int expected = r * static_cast<int>(count) + static_cast<int>(i);
                if (recv_buf[static_cast<size_t>(r) * count + i] != expected) {
                    passed = false;
                    break;
                }
            }
            if (!passed)
                break;
        }
    }

    if (rank == root) {
        if (passed)
            std::cout << "root: PASSED\n";
        else
            std::cout << "root: FAILED\n";
    }

    return 0;
}

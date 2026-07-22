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

using namespace std;

int main() {
    const size_t count = 128;

    size_t i = 0;
    std::vector<int> send_buf;
    std::vector<int> recv_buf;
    std::vector<size_t> recv_counts;

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

    send_buf.resize(count, rank);
    recv_buf.resize(size * count);
    recv_counts.resize(size, count);

    /* modify send_buf */
    for (i = 0; i < count; i++) {
        send_buf[i] = 10;
    }

    /* invoke broadcast */
    ccl::broadcast(send_buf.data(), recv_buf.data(), count, 0, comm).wait();

    /* check correctness of recv_buf */
    for (i = 0; i < count; i++) {
        if (recv_buf[i] != 10) {
            recv_buf[i] = -1;
        }
    }

    /* print out the result of the test */
    for (i = 0; i < count; i++) {
        if (recv_buf[i] == -1) {
            std::cout << "FAILED\n";
            break;
        }
    }
    if (i == count) {
        std::cout << "PASSED\n";
    }

    return 0;
}

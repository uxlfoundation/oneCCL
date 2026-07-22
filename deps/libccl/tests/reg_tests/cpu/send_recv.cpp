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

#include <string.h>
#include "base.hpp"

using namespace std;

int main() {
    ccl::init();

    int size, rank;
    int iters = 5;
    size_t count = 1024 * 1024;
    float received = -1;
    std::vector<float> buf(count);

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
    auto attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
    string match_id = "_len_" + std::to_string(count);
    attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(match_id));
    attr.set<ccl::operation_attr_id::to_cache>(true);

    if (size != 2) {
        std::cout << "This test requires excatly two ranks\n";
        exit(0);
    }

    for (size_t j = 0; j < count; j++) {
        buf[j] = rank + 1;
    }

    ccl::barrier(comm);

    for (int i = 0; i < iters; i++) {
        if (rank == 0) {
            ccl::send(buf.data(), count, ccl::datatype::float32, 1, comm, attr).wait();
        }
        else if (rank == 1) {
            ccl::recv(buf.data(), count, ccl::datatype::float32, 0, comm, attr).wait();
        }
    }

    {
        for (size_t idx = 0; idx < count; idx++) {
            received = buf[idx];
            if (received != 1) {
                fprintf(stderr,
                        "idx %zu, expected %4.4f, got %4.4f\n",
                        idx,
                        static_cast<float>(idx),
                        received);

                std::cout << "FAILED\n";
                terminate();
            }
        }
    }

    std::cout << "PASSED\n";

    return 0;
}

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

void run_collective(const char* cmd_name,
                    std::vector<float>& buf,
                    const ccl::communicator& comm,
                    const ccl::broadcast_attr& attr) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float received;

    if (comm.rank() == COLL_ROOT) {
        for (size_t idx = 0; idx < buf.size(); idx++) {
            buf[idx] = static_cast<float>(idx);
        }
    }
    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        auto start = std::chrono::system_clock::now();
        ccl::broadcast(buf.data(), buf.size(), COLL_ROOT, comm, attr).wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    for (size_t idx = 0; idx < buf.size(); idx++) {
        received = buf[idx];
        if (received != idx) {
            fprintf(stderr,
                    "idx %zu, expected %4.4f, got %4.4f\n",
                    idx,
                    static_cast<float>(idx),
                    received);

            std::cout << "FAILED" << std::endl;
            std::terminate();
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
    auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();

    MSG_LOOP(comm, std::vector<float> buf(msg_count);
             attr.set<ccl::operation_attr_id::to_cache>(false);
             run_collective("warmup_bcast", buf, comm, attr);
             attr.set<ccl::operation_attr_id::to_cache>(true);
             run_collective("persistent_bcast", buf, comm, attr);
             attr.set<ccl::operation_attr_id::to_cache>(false);
             run_collective("regular_bcast", buf, comm, attr););

    return 0;
}

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

using namespace std;
using namespace sycl;

void run_collective(const char* cmd_name,
                    const std::vector<float>& send_buf,
                    std::vector<float>& recv_buf,
                    const ccl::communicator& comm,
                    const ccl::allreduce_attr& attr) {
    float expected = (static_cast<float>(comm.size()) + 1) / 2 * static_cast<float>(comm.size());

    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITERS; ++idx) {
        ccl::allreduce(
            send_buf.data(), recv_buf.data(), recv_buf.size(), ccl::reduction::sum, comm, attr)
            .wait();
    }

    for (size_t idx = 0; idx < recv_buf.size(); idx++) {
        if (recv_buf[idx] != expected) {
            fprintf(stderr, "idx %zu, expected %4.4f, got %4.4f\n", idx, expected, recv_buf[idx]);
            std::cout << "FAILED" << std::endl;
            std::terminate();
        }
    }

    ccl::barrier(comm);
}

int main(int argc, char* argv[]) {
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
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);

        kvs = ccl::create_kvs(main_addr);
        main_addr = kvs->get_address();
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());

    try {
        ccl::create_communicator(size, rank, dev, ctx, kvs);
        std::cout << "FAILED\n";
        return -1;
    }
    catch (...) {
        std::cout << "can not create device communicator\n";
    }

    auto comm = ccl::create_communicator(size, rank, kvs);
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
    std::cout << "host communicator is created\n";

    MSG_LOOP(comm, std::vector<float> send_buf(msg_count, static_cast<float>(comm.rank() + 1));
             std::vector<float> recv_buf(msg_count);
             run_collective("regular allreduce", send_buf, recv_buf, comm, attr););
    std::cout << "PASSED\n";

    return 0;
}

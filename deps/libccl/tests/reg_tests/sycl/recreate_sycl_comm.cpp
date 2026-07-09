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

int main(int argc, char *argv[]) {
    size_t count = 10 * 1024;

    ccl::init();

    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "problem occurred during MPI init" };
    }

    int size = 0;
    int rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);
    test_args args(argc, argv, rank);

    if (args.count != args.DEFAULT_COUNT) {
        count = args.count;
    }

    sycl::queue q;
    if (!create_test_sycl_queue("gpu", rank, q, args))
        return -1;

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

    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto stream = ccl::create_stream(q);

    for (int i = 0; i < 20; i++) {
        if (rank == 0) {
            std::cout << "start iter " << i << "\n";
        }

        auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

        auto send_buf = sycl::malloc_device<float>(count, q);
        auto recv_buf = sycl::malloc_device<float>(count, q);
        std::vector<float> host_buf(count, static_cast<float>(comm.rank() + 1));

        q.memcpy(send_buf, host_buf.data(), count * sizeof(float)).wait();

        ccl::allreduce(
            send_buf, recv_buf, count, ccl::datatype::float32, ccl::reduction::sum, comm, stream)
            .wait();

        q.memcpy(host_buf.data(), recv_buf, count * sizeof(float)).wait();

        sycl::free(send_buf, q);
        sycl::free(recv_buf, q);

        float expected =
            (static_cast<float>(comm.size()) + 1) / 2 * static_cast<float>(comm.size());
        if (std::any_of(host_buf.begin(), host_buf.end(), [expected](float value) {
                return value != expected;
            })) {
            std::cout << "FAILED, unexpected allreduce result" << std::endl;
            return -1;
        }
    }

    if (rank == 0) {
        std::cout << "PASSED" << std::endl;
    }

    return 0;
}

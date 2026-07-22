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

#include <numeric>
#include <sstream>

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
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

    buf_allocator<unsigned char> allocator(q);

    /* create kvs */
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

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    // Buffer size 10 Gb
    size_t buffer_size =
        (unsigned long)10 * (unsigned long)1024 * (unsigned long)1024 * (unsigned long)1024;

    // allreduce byte count 8 Gb. Previously hanged up, that's the reason for this test
    size_t byte_count =
        (unsigned long)8 * (unsigned long)1024 * (unsigned long)1024 * (unsigned long)1024;

    unsigned char *recv_buf = allocator.allocate(buffer_size, usm::alloc::device);
    unsigned char *send_buf = allocator.allocate(buffer_size, usm::alloc::device);

    size_t check_range = (unsigned long)1024 * (unsigned long)1024 * (unsigned long)1024;
    size_t factor = 32;
    auto sycl_event = q.submit([&](auto &h) {
        h.parallel_for(check_range, [=](auto i) {
            send_buf[i] = i % factor;
            recv_buf[i] = 0;
        });
    });

    auto deps_event = ccl::create_event(sycl_event);
    std::vector<ccl::event> deps;
    deps.emplace_back(std::move(deps_event));
    auto attrs = ccl::create_operation_attr<ccl::allreduce_attr>();
    auto allreduce_event = ccl::allreduce(
        send_buf, recv_buf, byte_count, ccl::reduction::sum, comm, stream, attrs, std::move(deps));

    sycl::buffer<char> check_buf{ check_range };
    q.submit([&](auto &h) {
        auto check_buf_acc = check_buf.get_access<sycl::access_mode::write>(h);
        h.depends_on(allreduce_event.get_native());
        h.parallel_for(check_range, [=](auto i) {
            check_buf_acc[i] = (recv_buf[i] == size * (i % factor));
        });
    });

    auto check_buf_acc = check_buf.get_host_access(sycl::read_only);
    for (size_t i = 0; i < check_range; i++) {
        if (check_buf_acc[i] == 0) {
            std::cout << "FAILED\n";
            return -1;
        }
    }

    std::cout << "PASSED\n";

    return 0;
}

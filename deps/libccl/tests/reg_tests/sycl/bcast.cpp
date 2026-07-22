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
#include "oneapi/ccl.hpp"
#include "sycl_base.hpp"

#include <numeric>
#include <sstream>

#include <getopt.h>

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

    // Buffer size 5 Gb
    size_t buffer_size = (unsigned long)5 * 1024 * 1024 * 1024;

    // broadcast byte count, start from 1 GB
    size_t byte_count = (unsigned long)1024 * 1024 * 1024;
    // Each time increase 2 MB
    size_t increase_cout = (unsigned long)2 * 1024 * 1024;

    unsigned char *send_buf = allocator.allocate(buffer_size, usm::alloc::device);

    int i = 1;
    std::vector<ccl::event> events;
    auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();

    if (args.group_api && args.queue == queue_type::in_order) {
        ccl::group_start();
    }

    while (i < 100) {
        if (args.queue == queue_type::in_order) {
            if (args.group_api) {
                ccl::broadcast(send_buf, byte_count, ccl::datatype::int8, 0, comm, stream);
            }
            else {
                ccl::broadcast(send_buf, byte_count, ccl::datatype::int8, 0, comm, stream).wait();
            }
        }
        else {
            events.push_back(ccl::broadcast(
                send_buf, byte_count, ccl::datatype::int8, 0, comm, stream, attr, events));
            events.back().wait();
        }
        byte_count += increase_cout;
        ++i;
    }

    if (args.group_api && args.queue == queue_type::in_order) {
        ccl::group_end();
    }

    std::cout << "PASSED\n" << std::endl;

    return 0;
}

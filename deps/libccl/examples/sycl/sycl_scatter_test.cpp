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

// Out-of-place scatter via the sycl::buffer<int> public API overload
// (api_functions.hpp:2304), not the raw-USM-pointer overload exercised by
// sycl_scatter_usm_test.cpp. Same kernel paths as the USM out-of-place test;
// this test exists to cover the buffer/accessor binding.

#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "usage ./sycl_scatter_test [device]\n";
        cout << "device could be 'cpu' or 'gpu'\n";
        cout << "example: ./sycl_scatter_test cpu\n";
        exit(1);
    }

    string device_type = argv[1];

    if (device_type != "cpu" && device_type != "gpu") {
        cout << "error: Invalid device '" << device_type << "'.\n";
        cout << "device must be either 'cpu' or 'gpu'.\n";
        exit(1);
    }

    size_t count = 10 * 1024 * 1024;
    const size_t root_rank = 0;

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);

    if (args.count != args.DEFAULT_COUNT) {
        count = args.count;
    }

    sycl::queue q;
    if (!create_test_sycl_queue(device_type, rank, q, args))
        return -1;

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

    /* create buffers — root supplies size*count, every rank receives count */
    sycl::buffer<int> send_buf(count * size);
    sycl::buffer<int> recv_buf(count);

    {
        /* open buffers and initialize them on the host side */
        sycl::host_accessor send_buf_acc(send_buf, sycl::write_only);
        sycl::host_accessor recv_buf_acc(recv_buf, sycl::write_only);

        if (rank == root_rank) {
            for (int i = 0; i < size; i++) {
                for (size_t j = 0; j < count; j++) {
                    send_buf_acc[(i * count) + j] = i + 1;
                }
            }
        }
        for (size_t i = 0; i < count; i++) {
            recv_buf_acc[i] = -1;
        }
    }

    /* invoke scatter */
    ccl::scatter(send_buf, recv_buf, count, root_rank, comm, stream).wait();

    /* open recv_buf and check its correctness on the device side */
    sycl::buffer<int> check_buf(count);
    const int expected = rank + 1;
    q.submit([&](auto &h) {
        sycl::accessor recv_buf_acc(recv_buf, h, sycl::read_only);
        sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            if (recv_buf_acc[id] != expected) {
                check_buf_acc[id] = -1;
            }
            else {
                check_buf_acc[id] = 0;
            }
        });
    });

    if (!handle_exception(q))
        return -1;

    /* print out the result of the test on the host side */
    {
        sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
        size_t i;
        for (i = 0; i < count; i++) {
            if (check_buf_acc[i] == -1) {
                std::cout << "FAILED\n";
                break;
            }
        }
        if (i == count) {
            std::cout << "PASSED\n";
        }
    }

    return 0;
}

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

// Compile command:
// icpx -fsycl -lmpi -lccl allreduce_mixed_bufs.cpp -o allreduce_mixed_bufs
// Run command:
// mpirun -np num ./allreduce_mixed_bufs

#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <set>
#include <string>
#include <numeric>

#include <sycl/sycl.hpp>
#include "oneapi/ccl.hpp"
#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "usage ./allreduce_mixed_bufs [device]\n";
        cout << "device could be 'cpu' or 'gpu'\n";
        cout << "example: ./allreduce_mixed_bufs cpu\n";
        exit(1);
    }

    string device_name = argv[1];

    if (device_name != "cpu" && device_name != "gpu") {
        cout << "error: Invalid device '" << device_name << "'.\n";
        cout << "device must be either 'cpu' or 'gpu'.\n";
        exit(1);
    }

    size_t count = 4;

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    auto gpu_devices = create_sycl_gpu_devices(false);

    test_args args(argc, argv, rank);
    if (args.count != args.DEFAULT_COUNT) {
        count = args.count;
    }

    sycl::queue q;
    if (!create_test_sycl_queue(device_name, rank, q, args))
        return -1;

    auto sycl_ctx = q.get_context();
    auto sycl_dev = q.get_device();

    std::cout << "Current rank is " << rank << "\n";
    std::cout << "Selected device: " << q.get_device().get_info<info::device::name>() << "\n";

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
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    std::vector<int> send_buf_host(count, 1);

    // only rank 0 has value of 0.
    if (rank == 0) {
        for (auto& num : send_buf_host)
            num = 0;
    }

    const int nBytes = count * sizeof(int);
    auto* send_buf_device = sycl::aligned_alloc_device(64, nBytes, q);

    std::string send_str = "[";
    for (size_t i = 0; i < count; i++) {
        send_str += std::to_string(send_buf_host[i]) + ",";
    }
    send_str.pop_back();
    send_str += "]";

    std::cout << "send_buf: " << send_str << "\n";

    q.memcpy(send_buf_device, send_buf_host.data(), nBytes);
    q.wait();

    // Rank 0 uses replaced buffer.
    auto* recv_buf_device = send_buf_device;
    // Others rank has receive buffer.
    if (rank != 0) {
        recv_buf_device = sycl::aligned_alloc_device(64, nBytes, q);
    }

    /* invoke allreduce */
    ccl::allreduce(send_buf_device,
                   recv_buf_device,
                   count,
                   ccl::datatype::int32,
                   ccl::reduction::sum,
                   comm,
                   stream)
        .wait();

    q.memcpy(const_cast<int*>(send_buf_host.data()), recv_buf_device, nBytes);
    q.wait();

    int expect_value = 1 * (size - 1);

    std::string actual_str = "[";
    std::string expect_str = "[";

    for (size_t i = 0; i < count; i++) {
        auto actual_value = send_buf_host[i];
        if (actual_value != expect_value) {
            std::cout << "FAILED\n";
            return -1;
        }
        actual_str += std::to_string(actual_value) + ",";
        expect_str += std::to_string(expect_value) + ",";
    }

    actual_str.pop_back();
    actual_str += "]";
    expect_str.pop_back();
    expect_str += "]";

    std::cout << "actual: " << actual_str << "\n";
    std::cout << "expect: " << expect_str << "\n";

    std::cout << "PASSED\n";
    return 0;
}

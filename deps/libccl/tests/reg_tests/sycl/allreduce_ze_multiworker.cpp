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

#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "base.hpp"
#include "bf16.hpp"
#include "oneapi/ccl.hpp"
#include "sycl_base.hpp"
#include "utils.hpp"

#define COUNT (1048576 / 256)

using namespace std;

int main(int argc, char* argv[]) {
    size_t count = COUNT;

    int iter = 5;

    ccl::init();

    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);

    if (args.count != args.DEFAULT_COUNT) {
        count = args.count;
    }

    vector<float> send_buf(count);
    vector<float> recv_buf(count);

    const size_t buf_size = sizeof(float) * count;

    sycl::queue q;
    if (!create_test_sycl_queue("gpu", rank, q, args))
        return -1;

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

    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);
    /* create stream */
    auto stream = ccl::create_stream(q);

    void* send_buf_fp32_device = malloc_device(buf_size, q);
    void* recv_buf_fp32_device = malloc_device(buf_size, q);

    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
    string match_id = "_len_" + std::to_string(count);

    attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(match_id));
    attr.set<ccl::operation_attr_id::to_cache>(true);

    float sum = 0.0;
    for (int j = 1; j < size + 1; j++) {
        sum += j;
    }
    for (int j = 0; j < iter; j++) {
        for (size_t idx = 0; idx < count; idx++) {
            send_buf[idx] = rank + 1;
            recv_buf[idx] = 0.0;
        }

        // copy our buffer to device memory
        q.memcpy(send_buf_fp32_device, send_buf.data(), buf_size).wait();
        q.memcpy(recv_buf_fp32_device, recv_buf.data(), buf_size).wait();

        ccl::allreduce(send_buf_fp32_device,
                       recv_buf_fp32_device,
                       count,
                       ccl::datatype::float32,
                       ccl::reduction::sum,
                       comm,
                       stream,
                       attr)
            .wait();
        q.memcpy(recv_buf.data(), recv_buf_fp32_device, buf_size).wait();

        if (rank == 0) {
            for (size_t idx = 0; idx < count; idx++) {
                if (recv_buf[idx] != sum) {
                    std::cout << "Error at " << idx << ", val = " << recv_buf[idx]
                              << ", iter = " << j << ", waiting for " << sum << "\n";
                    std::cout << "FAILED\n";
                    return -1;
                }
            }
        }
    }

    std::cout << "PASSED\n";
    return 0;
}

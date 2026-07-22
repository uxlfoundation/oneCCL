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

/*
 * Example demonstrating the use of communicator.finalize() API
 *
 * This test:
 * 1. Submits multiple consecutive allreduce operations without waiting
 * 2. Calls comm.finalize() to wait for all operations across all streams
 * 3. Verifies that all operations completed successfully
 *
 * Usage:
 *   mpirun -n 2 ./sycl_allreduce_finalize_usm_test gpu device
 *   mpirun -n 4 ./sycl_allreduce_finalize_usm_test cpu host
 *
 * The finalize() API ensures all outstanding collective operations on the
 * communicator complete before returning, regardless of which stream they
 * were submitted to.
 */

#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    if (!check_example_args(argc, argv))
        exit(1);

    size_t count = 10 * 1024 * 1024;
    const int num_iters = 5; // Number of consecutive allreduce operations

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

    string device_type = argv[1];
    string alloc_type = argv[2];

    sycl::queue q;
    if (!create_test_sycl_queue(device_type, rank, q, args))
        return -1;

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm_alloc_type_from_string(alloc_type);

    if (!check_sycl_usm(q, usm_alloc_type)) {
        return -1;
    }

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

    /* create USM buffers for multiple iterations */
    vector<int*> send_bufs(num_iters);
    vector<int*> recv_bufs(num_iters);

    for (int iter = 0; iter < num_iters; iter++) {
        send_bufs[iter] = allocator.allocate(count, usm_alloc_type);
        recv_bufs[iter] = allocator.allocate(count, usm_alloc_type);
    }

    /* Launch multiple allreduce operations without waiting */
    if (rank == 0) {
        cout << "Launching " << num_iters << " consecutive allreduce operations..." << std::endl;
    }

    for (int iter = 0; iter < num_iters; iter++) {
        /* Initialize send buffer on device */
        int* send_buf = send_bufs[iter];  // Local copy for kernel capture
        q.submit([=](auto &h) {
            h.parallel_for(count, [=](auto id) {
                send_buf[id] = rank + 1 + iter;
            });
        });

        auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
        vector<ccl::event> deps;  // Empty deps, rely on in-order queue
        auto allreduce_event = ccl::allreduce(send_bufs[iter],
                                              recv_bufs[iter],
                                              count,
                                              ccl::datatype::int32,
                                              ccl::reduction::sum,
                                              comm,
                                              stream,
                                              attr,
                                              deps);

        if (rank == 0) {
            cout << "  Iteration " << iter << " submitted" << std::endl;
        }
    }

    /* Call finalize to wait for all operations */
    if (rank == 0) {
        cout << "Calling comm.finalize() to wait for all operations..." << std::endl;
    }

    comm.finalize();

    if (rank == 0) {
        cout << "All operations completed via finalize()" << std::endl;
    }

    /* Verify results on device */
    bool all_passed = true;
    for (int iter = 0; iter < num_iters; iter++) {
        int expected = size * (size + 1) / 2 + iter * size;

        sycl::buffer<int> check_buf(count);
        int* recv_buf = recv_bufs[iter];  // Local copy for kernel capture
        q.submit([=, &check_buf](auto &h) {
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                if (recv_buf[id] != expected) {
                    check_buf_acc[id] = -1;
                }
                else {
                    check_buf_acc[id] = 0;
                }
            });
        });
        q.wait();

        /* Check results on host */
        {
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            for (size_t i = 0; i < count; i++) {
                if (check_buf_acc[i] == -1) {
                    if (rank == 0) {
                        cout << "FAILED at iteration " << iter << ", element " << i << std::endl;
                    }
                    all_passed = false;
                    break;
                }
            }
        }

        if (!all_passed) break;
    }

    if (!handle_exception(q))
        return -1;

    if (all_passed && rank == 0) {
        cout << "PASSED - All " << num_iters << " allreduce operations completed correctly" << std::endl;
    }

    /* Cleanup */
    for (int iter = 0; iter < num_iters; iter++) {
        allocator.deallocate(send_bufs[iter]);
        allocator.deallocate(recv_bufs[iter]);
    }

    return all_passed ? 0 : -1;
}

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

// In-place USM scatter: at root, recv_buf aliases send_buf + root*count*dsize so
// the root reads its slice in place. Exercises the inplace branch —
// is_scatter_inplace() returns true on root, scatter_large_impl_ipc skips
// q.memcpy on root, scatter_small_sycl_impl skips phase 3 read-out on root via
// skip_recv_root.

#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    if (!check_example_args(argc, argv))
        exit(1);

    const size_t root_rank = 0;

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);
    const size_t count = args.count;

    string device_type = argv[1];

    sycl::queue q;
    if (!create_test_sycl_queue(device_type, rank, q, args))
        return -1;

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm_alloc_type_from_string(argv[2]);

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

    /* in-place: root allocates size*count and aliases recv_buf into its own
     * root_rank slot. Non-root ranks have a separate recv_buf and pass
     * nullptr send_buf (per spec, only root supplies send_buf). */
    int* send_buf = nullptr;
    int* recv_buf = nullptr;
    if (rank == root_rank) {
        send_buf = allocator.allocate(count * size, usm_alloc_type);
        recv_buf = send_buf + root_rank * count;
    }
    else {
        recv_buf = allocator.allocate(count, usm_alloc_type);
    }

    /* do not wait completion of kernel and provide it as dependency for operation */
    vector<ccl::event> deps;

    if (rank == root_rank) {
        /* fill send_buf so block i is filled with (i + 1) */
        auto e = q.submit([&](auto &h) {
            h.parallel_for(count * size, [=](auto id) {
                size_t block = id / count;
                send_buf[id] = static_cast<int>(block) + 1;
            });
        });
        deps.push_back(ccl::create_event(e));
    }
    else {
        /* zero recv_buf on non-root to catch under-writes */
        auto e = q.submit([&](auto &h) {
            h.parallel_for(count, [=](auto id) { recv_buf[id] = -1; });
        });
        deps.push_back(ccl::create_event(e));
    }

    /* invoke scatter (in-place on root) */
    auto attr = ccl::create_operation_attr<ccl::scatter_attr>();
    ccl::scatter(send_buf, recv_buf, count, root_rank, comm, stream, attr, deps).wait();

    /* check rank received block (rank+1) */
    sycl::buffer<int> check_buf(count);
    const int expected = rank + 1;
    q.submit([&](auto &h) {
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

    if (!handle_exception(q))
        return -1;

    /* print out the result of the test on the host side */
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

    return 0;
}

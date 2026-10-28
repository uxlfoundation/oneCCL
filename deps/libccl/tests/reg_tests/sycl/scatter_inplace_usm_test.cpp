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

// SYCL scatter reg-test in-place variant. NCCL in-place semantics:
//   - at root: recv_buf points into send_buf at root's block (send_buf + root*count).
//   - non-root ranks own an independent recv_buf and pass nullptr for send_buf.
// Exercises the same scatter_small / scatter_large kernels as scatter_usm_test but through
// the in-place layout. The .sh driver sweeps CCL_SYCL_SCATTER_LARGE_TMP_BUF to cover both
// IPC-direct and tmp-buf staged paths.

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "usage ./scatter_inplace_usm_test [device]\n";
        cout << "device could be 'cpu' or 'gpu'\n";
        cout << "example: ./scatter_inplace_usm_test gpu device\n";
        exit(1);
    }

    string device_name = argv[1];

    if (device_name != "cpu" && device_name != "gpu") {
        cout << "error: Invalid device '" << device_name << "'.\n";
        cout << "device must be either 'cpu' or 'gpu'.\n";
        exit(1);
    }

    size_t count = 10 * 1024 * 1024;

    const int root = 0;
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
    if (!create_test_sycl_queue(device_name, rank, q, args))
        return -1;

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm::alloc::shared;
    if (argc > 2) {
        usm_alloc_type = usm_alloc_type_from_string(argv[2]);
    }

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

    /* NCCL in-place layout: root owns full size*count send buf; recv_buf aliases root's block.
     * Non-root ranks own an independent recv_buf and pass nullptr as send_buf. */
    int *send_buf = nullptr;
    int *recv_buf = nullptr;
    if (rank == root) {
        send_buf = allocator.allocate(count * size, usm_alloc_type);
        recv_buf = send_buf + root * count;
    }
    else {
        recv_buf = allocator.allocate(count, usm_alloc_type);
    }

    sycl::buffer<int> expected_buf(count);
    sycl::buffer<int> check_buf(count);

    /* Root tags block i element id with (i*7 + id%256); each rank expects the same tag. */
    auto e = q.submit([&](auto &h) {
        sycl::accessor expected_buf_acc(expected_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            expected_buf_acc[id] = static_cast<int>((rank * 7 + (id % 256)) % 100000);
            if (send_buf != nullptr) {
                for (int i = 0; i < size; i++) {
                    send_buf[i * count + id] =
                        static_cast<int>((i * 7 + (id % 256)) % 100000);
                }
            }
        });
    });

    vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));

    auto attr = ccl::create_operation_attr<ccl::scatter_attr>();
    ccl::scatter(send_buf, recv_buf, count, ccl::datatype::int32, root, comm, stream, attr, deps)
        .wait();

    q.submit([&](auto &h) {
        sycl::accessor expected_buf_acc(expected_buf, h, sycl::read_only);
        sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            if (recv_buf[id] != expected_buf_acc[id]) {
                check_buf_acc[id] = -1;
            }
            else {
                check_buf_acc[id] = 0;
            }
        });
    });

    if (!handle_exception(q))
        return -1;

    {
        sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
        size_t i;
        for (i = 0; i < count; i++) {
            if (check_buf_acc[i] == -1) {
                std::cout << "FAILED at index " << i << "\n";
                break;
            }
        }
        if (i == count) {
            std::cout << "PASSED\n";
        }
    }

    return 0;
}

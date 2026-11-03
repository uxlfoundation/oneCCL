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

/*
    Same scenario layout as sycl_reduce_scatter_usm_comm_split.cpp. Operates
    with four ranks (0, 1, 2, 3). Each scenario assigns ranks to specific
    communicators based on `color`, with `key` controlling ordering within
    each communicator. `key` values are set as `rank % 3`.

    The test invokes ccl::scatter on the resulting sub-communicator with
    sub-rank 0 as root. Root supplies a send_buf where block i is filled with
    (i + 1); every sub-rank receives a recv_buf of `count` elements expected
    to match (sub_rank + 1). Scenarios 1 and 4 exercise non-zero original
    ranks acting as sub-comm root.

    | Scenario     | Ranks   | Color | Keys       | Group Members  | Sub-comm root (orig rank) |
    |--------------|---------|-------|------------|----------------|---------------------------|
    | Scenario 1   | 0, 1, 2 | 1     | 0, 1, 2    | 0, 1, 2        | 0                         |
    |              | 3       | 0     | 0          | 3              | 3                         |
    | Scenario 2   | 0-3     | 1     | 0, 1, 2, 0 | 0, 3, 1, 2     | 0                         |
    | Scenario 3   | 0-3     | 0     | 0, 1, 2, 0 | 0, 3, 1, 2     | 0                         |
    | Scenario 4   | 0, 1    | 2     | 0, 1       | 0, 1           | 0                         |
    |              | 2, 3    | 1     | 0, 1       | 2, 3           | 2                         |
    | Default      | 0, 2    | 0     | 0, 2       | 0, 2           | 0                         |
    |              | 1, 3    | 1     | 1, 0       | 3, 1           | 3                         |
*/

int main(int argc, char* argv[]) {
    size_t count = 1024;

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    if (size != 4) {
        if (rank == 0) {
            std::cout << "FAILED: This test requires exactly 4 ranks.\n";
        }
        return -1;
    }

    queue q;
    sycl::property_list props;
    if (argc > 3) {
        if (strcmp("in_order", argv[3]) == 0) {
            props = { sycl::property::queue::in_order{} };
        }
    }

    if (argc > 4) {
        count = (size_t)std::atoi(argv[4]);
    }

    if (!create_sycl_queue(argc, argv, rank, q, props)) {
        return -1;
    }

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm::alloc::shared;
    if (argc > 2) {
        usm_alloc_type = usm_alloc_type_from_string(argv[2]);
    }

    if (!check_sycl_usm(q, usm_alloc_type)) {
        return -1;
    }

    int scenario = (argc > 5) ? std::atoi(argv[5]) : 0;

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

    auto stream = ccl::create_stream(q);

    int color = 0;
    int key = rank % 3;

    switch (scenario) {
        case 1: color = (rank < 3) ? 1 : 0; break;
        case 2: color = 1; break;
        case 3: color = 0; break;
        case 4: color = (rank < 2) ? 2 : 1; break;
        default: color = (rank % 2 == 0) ? 0 : 1; break;
    }

    auto new_comm = ccl::split_communicator(comm, color, key);

    const int sub_size = new_comm.size();
    const int sub_rank = new_comm.rank();
    const int sub_root = 0;

    /* Root allocates sub_size*count, every sub-rank receives count. */
    auto send_buf =
        (sub_rank == sub_root) ? allocator.allocate(count * sub_size, usm_alloc_type) : nullptr;
    auto recv_buf = allocator.allocate(count, usm_alloc_type);

    vector<ccl::event> deps;

    if (sub_rank == sub_root) {
        /* fill send_buf so block i is filled with (i + 1) */
        auto e = q.submit([&](auto& h) {
            h.parallel_for(count * sub_size, [=](auto id) {
                size_t block = id / count;
                send_buf[id] = static_cast<int>(block) + 1;
            });
        });
        deps.push_back(ccl::create_event(e));
    }

    {
        auto e = q.submit([&](auto& h) {
            h.parallel_for(count, [=](auto id) { recv_buf[id] = -1; });
        });
        deps.push_back(ccl::create_event(e));
    }

    auto attr = ccl::create_operation_attr<ccl::scatter_attr>();
    ccl::scatter(send_buf, recv_buf, count, sub_root, new_comm, stream, attr, deps).wait();

    sycl::buffer<int> check_buf(count);
    const int expected = sub_rank + 1;
    q.submit([&](auto& h) {
        sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            check_buf_acc[id] = (recv_buf[id] != expected) ? -1 : 0;
        });
    });

    if (!handle_exception(q))
        return -1;

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

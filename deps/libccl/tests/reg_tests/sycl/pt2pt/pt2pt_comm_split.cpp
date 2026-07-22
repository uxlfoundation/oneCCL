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
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace sycl;

int main(int argc, char* argv[]) {
    size_t count = 1024;

    int size = 0;
    int rank = 0;

    ccl::init();
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    atexit(mpi_finalize);

    queue q;
    sycl::property_list props;
    if (argc > 3 && strcmp("in_order", argv[3]) == 0) {
        props = { sycl::property::queue::in_order{} };
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
    auto stream = ccl::create_stream(q);

    // Assign color based on rank, even ranks get color 0, odd ranks get color 1
    int color = (rank % 2 == 0) ? 0 : 1;
    auto new_comm = ccl::split_communicator(comm, color, 0);

    // Increase tensor size to stress-test data transfer
    int tensor_size = 1024;
    auto tensor = allocator.allocate(tensor_size, usm_alloc_type);
    int new_comm_rank = new_comm.rank();

    auto e = q.submit([&](auto& h) {
        h.parallel_for(tensor_size, [=](auto i) {
            tensor[i] = new_comm_rank * 1000 +
                        i; // Fill with values based on local rank in sub-communicator
        });
    });
    e.wait();

    bool success = true;
    sycl::buffer<int> check_buf(tensor_size); // Buffer to store verification results

    for (int iteration = 0; iteration < 1; ++iteration) {
        for (int peer = 0; peer < new_comm.size(); ++peer) {
            auto p2pTargetRank = (new_comm.rank() + 1) % new_comm.size();

            if (new_comm.rank() == 0) {
                // Send the tensor with correct data type
                auto send_attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
                ccl::send(tensor,
                          tensor_size,
                          ccl::datatype::int32,
                          p2pTargetRank,
                          new_comm,
                          stream,
                          send_attr)
                    .wait();
            }
            else {
                // Allocate receive buffer and set up receive attributes
                auto recv_buf = allocator.allocate(tensor_size, usm_alloc_type);
                auto recv_attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
                ccl::recv(recv_buf,
                          tensor_size,
                          ccl::datatype::int32,
                          p2pTargetRank,
                          new_comm,
                          stream,
                          recv_attr)
                    .wait();

                q.submit([&](auto& h) {
                     sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
                     h.parallel_for(tensor_size, [=](auto id) {
                         int expected_value = p2pTargetRank * 1000 + id;
                         check_buf_acc[id] = (recv_buf[id] != expected_value) ? -1 : 0;
                     });
                 }).wait();

                // Host-side check of the verification results
                sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
                for (size_t i = 0; i < tensor_size; i++) {
                    if (check_buf_acc[i] == -1) {
                        std::cerr << "Iteration " << iteration << " Error: Rank " << rank
                                  << " received incorrect value at index " << i
                                  << " from p2pTargetRank " << p2pTargetRank
                                  << " in sub-communicator (received " << recv_buf[i]
                                  << ", expected " << (p2pTargetRank * 1000 + i) << ")\n";
                        success = false;
                        break;
                    }
                }
                allocator.deallocate(recv_buf);
            }
        }
    }

    if (success) {
        std::cout << "PASSED\n";
    }
    else {
        std::cout << "FAILED\n";
    }
    return 0;
}

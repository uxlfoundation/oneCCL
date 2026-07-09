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

using namespace std;
using namespace sycl;

// Calculation of displs:
// Each process has a send count (sendcount) equal to its rank plus one. Therefore, the send counts are as follows:

// Process 0: sendcount = 1
// Process 1: sendcount = 2
// Process 2: sendcount = 3
// Process 3: sendcount = 4
// The recvcounts array, which is the result of an MPI_Allgather of the send counts, will look like this:

// recvcounts: [1, 2, 3, 4]
// The displs array is calculated based on the recvcounts:

// displs[0] = 0
// displs[1] = displs[0] + recvcounts[0] = 0 + 1 = 1
// displs[2] = displs[1] + recvcounts[1] = 1 + 2 = 3
// displs[3] = displs[2] + recvcounts[2] = 3 + 3 = 6
// So the final displs array is:

// displs: [0, 1, 3, 6]

// Manual allgatherv Implementation:
// Each process sends its data to all other processes and receives data from all other processes.
// The data is placed in the receive buffer at the offsets specified by displs. Here's how the data is distributed:

// Process 0 sends: [0]
// Process 1 sends: [1, 1]
// Process 2 sends: [2, 2, 2]
// Process 3 sends: [3, 3, 3, 3]
// After the send and recv operations, the receive buffer for each process looks like this:

// Recv buffer (all processes): [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
// The displacements ensure that the data from each process is placed at the correct position in the receive buffer.
// For example, Process 2 data starts at index 3 (displs[2]), and there are 3 elements (recvcounts[2]) from Process 2 in the buffer.

// Final results gathered data in the receive buffer for each process:

// Recv buffer indices:  0 1 2 3 4 5 6 7 8 9
// Process 0 gathered: [0, 1, 1, 2, 2, 2, 3, 3, 3, 3] // 0
// Process 1 gathered: [0, 1, 1, 2, 2, 2, 3, 3, 3, 3] // 1 1
// Process 2 gathered: [0, 1, 1, 2, 2, 2, 3, 3, 3, 3] // 2 2 2
// Process 3 gathered: [0, 1, 1, 2, 2, 2, 3, 3, 3, 3] // 3 3 3 3

int main(int argc, char* argv[]) {
    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    queue q;
    sycl::property_list props = { sycl::property::queue::in_order{} };

    if (!create_sycl_queue(argc, argv, rank, q, props)) {
        return -1;
    }

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm::alloc::device;

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

    // Define the local send buffer size
    int sendcount = 1025;
    auto send_buf = allocator.allocate(sendcount, usm_alloc_type);

    // Calculate total receive buffer size and allocate it
    std::vector<int> recvcounts(size);
    MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(size);
    displs[0] = 0;
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    int total_recvcount = displs[size - 1] + recvcounts[size - 1];
    auto recv_buf = allocator.allocate(total_recvcount, usm_alloc_type);

    // Initialize send buffer with rank
    q.submit([&](handler& h) {
         h.parallel_for(sendcount, [=](auto id) {
             send_buf[id] = rank * size + id;
         });
     }).wait();

    // Copy local data to the correct position in recv_buf
    q.memcpy(recv_buf + displs[rank], send_buf, sendcount * sizeof(int)).wait();

    // Perform allgatherv
    std::vector<ccl::event> deps = {};

    auto send_attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
    auto recv_attr = ccl::create_operation_attr<ccl::pt2pt_attr>();

    ccl::group_start();
    for (int i = 0; i < size; ++i) {
        if (i != rank) {
            // Send local data to process i
            ccl::send(send_buf, sendcount, ccl::datatype::int32, i, comm, stream, send_attr, deps);

            // Receive data from process i
            ccl::recv(recv_buf + displs[i],
                      recvcounts[i],
                      ccl::datatype::int32,
                      i,
                      comm,
                      stream,
                      recv_attr,
                      deps);
        }
    }
    ccl::group_end();

    // this is needed to check async pt2pt correctness
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sendcount, [=](auto id) {
            send_buf[id] = rank * size + id + 1;
        });
    });

    // Copy data back to host for verification
    std::vector<int> host_recv_buf(total_recvcount);
    q.memcpy(host_recv_buf.data(), recv_buf, total_recvcount * sizeof(int)).wait();

    bool is_correct = true;
    for (int rank_to_check = 0; rank_to_check < size; ++rank_to_check) {
        for (int i = 0; i < sendcount; ++i) {
            int index = displs[rank_to_check] + i;
            int expected_value = rank_to_check * size + i;
            if (host_recv_buf[index] != expected_value) {
                std::cout << "Process " << rank << " - FAILED at index " << index << ": Expected "
                          << expected_value << ", but got " << host_recv_buf[index] << std::endl;
                is_correct = false;
                break;
            }
        }
        if (!is_correct) {
            break;
        }
    }

    if (is_correct) {
        std::cout << "PASSED\n";
    }
    else {
        std::cout << "FAILED\n";
    }
    return 0;
}

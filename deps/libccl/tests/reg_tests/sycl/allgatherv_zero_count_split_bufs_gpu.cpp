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

#include <iostream>
#include <map>
#include <mpi.h>
#include <numeric>
#include <string>

#include "sycl_base.hpp"

int main(int argc, char *argv[]) {
    size_t failure_count = 0;
    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);

    // setting the queue type as out_or_order manually until the hangup problem
    // with in_order queue is fixed. see:
    // https://jira.devtools.intel.com/projects/MLSL/issues/MLSL-3351
    args.queue = queue_type::out_of_order;

    sycl::queue q;
    if (!create_test_sycl_queue("gpu", rank, q, args))
        return -1;

    // create kvs
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

    // create communicator
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    // create stream
    auto stream = ccl::create_stream(q);

    // Ranks of [0, size/2) are active ranks, others are silent ranks([size/2, size)).*/
    int border = size / 2;

    auto is_activate = [border](int r) {
        return r >= 0 && r < border;
    };

    std::vector<int> send_counts = { 0, 1, 2, 16, 17, 1024, 1025 };

    // perform the test case for each send count
    for (auto activate_send_count : send_counts) {
        // split all ranks into active and inactive ones
        // active ones should send a vector of required size
        // others should not send anything
        int send_count = is_activate(rank) ? activate_send_count : 0;

        std::vector<size_t> recv_counts(size, 0);

        // prepare the receive vectors
        size_t total_receive_counts = 0;
        for (size_t i = 0; i < recv_counts.size(); i++) {
            recv_counts[i] = is_activate(i) ? activate_send_count : 0;
            total_receive_counts += recv_counts[i];
        }

        // prepare the send buffer
        std::vector<int> send_buf_host(send_count, -1);
        for (int i = 0; i < send_count; ++i) {
            // fill the buffer with values that:
            //  - are unique inside the send buffer,
            //  - rarely repeat between different buffers
            // Fill pattern:
            // 1 2 3 4 5 ....
            // 4 5 6 7
            // 9 10 ...
            // (rank+1)^2 + i is used to avoid number repetitions for smaller matrices
            int rank_plus_1 = rank + 1;
            send_buf_host[i] = rank_plus_1 * rank_plus_1 + i;
        }
        std::vector<int> recv_buf_host(total_receive_counts, 0);
        size_t send_bytes = send_buf_host.size() * sizeof(int);

        // buffer size needs to be > 0 - level zero requirement
        int *send_buf_device = sycl::malloc_device<int>(send_count > 0 ? send_count : 1, q);
        // create buffers
        std::vector<int *> recv_bufs;
        for (int i = 0; i < size; ++i) {
            // buffer size needs to be > 0 - level zero requirement
            recv_bufs.push_back(sycl::malloc_device<int>(activate_send_count, q));
        }
        q.wait();

        if (send_count > 0) {
            q.memcpy(send_buf_device, send_buf_host.data(), send_bytes).wait();
        }

        utils::dump_vec(recv_counts, "recvcounts");

        if (send_buf_device == nullptr) {
            std::cout << "send buf device is nullptr, send count is " << send_count << "\n";
        }
        else {
            std::cout << "send buf device is not nullptr, send count is " << send_count << "\n";
        }
        // invoke allgatherv
        ccl::allgatherv(send_buf_device, send_count, recv_bufs, recv_counts, comm, stream).wait();

        size_t offset = 0;
        for (size_t i = 0; i < recv_counts.size(); ++i) {
            size_t recv_bytes = recv_counts[i] * sizeof(int);
            q.memcpy(const_cast<int *>(recv_buf_host.data() + offset), recv_bufs[i], recv_bytes)
                .wait();
            offset += recv_counts[i];
        }

        // print out the result of the test on the host side
        if (activate_send_count < 30) { // don't flood stdout with big vectors
            utils::dump_vec(send_buf_host, "send buf");
            utils::dump_vec(recv_buf_host, "recv buf");
        }
        else {
            std::cout << "send count too high, skipping buffer dump\n";
        }
        size_t total_received_count = 0;

        offset = 0;
        // iterate over receive buffers, by receive rank, to check contents
        for (size_t rank_idx = 0; rank_idx < (size_t)size; rank_idx++) {
            if (is_activate(rank_idx)) {
                for (size_t i = 0; i < (size_t)activate_send_count; i++, total_received_count++) {
                    int rank_idxp1 = rank_idx + 1;
                    int value = rank_idxp1 * rank_idxp1 + i;
                    if (recv_buf_host[offset++] != value) {
                        ++failure_count;
                        std::cout << "FAILED: expected: " << value
                                  << ", found: " << recv_buf_host[i] << std::endl;
                        break;
                    }
                }
            }
            else if (recv_counts[rank_idx] != 0) {
                ++failure_count;
                std::cout
                    << "FAILED: receive counts of a buffer that should be empty are not zero\n";
                break;
            }
        }
        if (total_received_count != (size_t)(activate_send_count * (size / 2))) {
            ++failure_count;
            std::cout << "FAILED: incorrect total" << total_received_count << " vs "
                      << (size_t)(activate_send_count * (size / 2)) << "\n";
            break;
        }
    }

    if (failure_count == 0) {
        std::cout << "PASSED\n";
    }

    return 0;
}

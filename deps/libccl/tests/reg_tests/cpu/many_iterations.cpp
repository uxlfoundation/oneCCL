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

#include "base.hpp"

int main() {
    const size_t iter_count = 128;
    const size_t check_buffer_period = 8;

    const std::vector<size_t> elem_counts = {
        2048000, 1048576, 2359296, 1048576, 1048576, 2359296, 1048576, 1048576, 2359296,
        524288,  2097152, 262144,  58982,   262144,  262144,  589824,  262144,  262144,
        589824,  262144,  262144,  589824,  262144,  262144,  589824,  262144,  262144,
        589824,  131072,  524288,  65536,   147456,  65536,   65536,   147456,  65536,
        65536,   147456,  65536,   65536,   147456,  32768,   131072,  16384,   36864,
        16384,   16384,   36864,   16384,   16384,   36864,   4096,    16384,   9408
    };

    ccl::init();

    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

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

    auto comm = ccl::create_communicator(size, rank, kvs);

    std::vector<std::vector<float>> send_buffers;
    std::vector<std::vector<float>> recv_buffers;
    for (const auto elem_count : elem_counts) {
        send_buffers.push_back(std::vector<float>(elem_count));
        recv_buffers.push_back(std::vector<float>(elem_count));
    }

    const size_t buf_count = elem_counts.size();

    float expected_send_value = static_cast<float>(comm.rank() + 1);
    float expected_recv_value =
        (static_cast<float>(comm.size()) + 1) / 2 * static_cast<float>(comm.size());

    for (size_t iter_idx = 0; iter_idx < iter_count; iter_idx++) {
        std::vector<ccl::event> events;
        std::chrono::system_clock::duration iter_time{ 0 };

        if (rank == 0) {
            std::cout << "started iter " << iter_idx << "\n";
        }

        ccl::barrier(comm);

        auto start = std::chrono::system_clock::now();

        for (size_t buf_idx = 0; buf_idx < buf_count; buf_idx++) {
            auto& send_buf = send_buffers[buf_idx];
            auto& recv_buf = recv_buffers[buf_idx];

            if (iter_idx % check_buffer_period == 0) {
                std::fill(send_buf.begin(), send_buf.end(), expected_send_value);
                std::fill(recv_buf.begin(), recv_buf.end(), 0);
            }

            events.push_back(ccl::allreduce(send_buf.data(),
                                            recv_buf.data(),
                                            send_buf.size(),
                                            ccl::datatype::float32,
                                            ccl::reduction::sum,
                                            comm));
        }

        for (size_t buf_idx = 0; buf_idx < buf_count; buf_idx++) {
            events[buf_idx].wait();
        }

        iter_time = std::chrono::system_clock::now() - start;

        if (iter_idx % check_buffer_period == 0) {
            for (size_t buf_idx = 0; buf_idx < buf_count; buf_idx++) {
                auto& send_buf = send_buffers[buf_idx];
                auto& recv_buf = recv_buffers[buf_idx];

                auto check_buf = [&](const std::vector<float>& buf,
                                     const float expected_value,
                                     const std::string& name) {
                    if (std::any_of(buf.begin(), buf.end(), [expected_value](float value) {
                            return value != expected_value;
                        })) {
                        std::cout << "FAILED, unexpected " << name << ": iter_idx: " << iter_idx
                                  << ", buf_idx: " << buf_idx << "\n";
                        return -1;
                    }
                    return 0;
                };

                if (check_buf(send_buf, expected_send_value, "send_buf")) {
                    return -1;
                }

                if (check_buf(recv_buf, expected_recv_value, "recv_buf")) {
                    return -1;
                }
            }
        }

        if (rank == 0) {
            std::cout << "completed iter " << iter_idx << ", time "
                      << std::chrono::duration_cast<std::chrono::microseconds>(iter_time).count()
                      << " usec\n";
        }
    }

    if (rank == 0) {
        std::cout << "PASSED\n";
    }

    return 0;
}

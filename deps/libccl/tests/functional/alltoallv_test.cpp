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

#define ALGO_SELECTION_ENV "CCL_ALLTOALLV"

#include "test_impl.hpp"

template <typename T>
class alltoallv_test : public base_test<T> {
public:
    std::vector<size_t> send_counts;
    std::vector<size_t> recv_counts;

    int check(test_operation<T>& op) {
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            size_t global_elem_idx_base = 0;
            for (int rank = 0; rank < op.comm_size; rank++) {
                T expected = static_cast<T>(rank + buf_idx);
                for (size_t idx = 0; idx < recv_counts[rank]; idx += op.get_check_step(idx)) {
                    if (base_test<T>::check_error(
                            op, expected, buf_idx, global_elem_idx_base + idx))
                        return TEST_FAILURE;
                }
                global_elem_idx_base += recv_counts[rank];
            }
        }
        return TEST_SUCCESS;
    }

    void alloc_buffers(test_operation<T>& op) {
        send_counts.resize(op.comm_size);
        recv_counts.resize(op.comm_size);
        if (op.param.place_type == PLACE_IN) {
            /*
               Specifying the in-place option indicates that
               the same amount and type of data is sent and received
               between any two processes in the group of the communicator.
               Different pairs of processes can exchange different amounts of data.
               https://docs.microsoft.com/en-us/message-passing-interface/mpi-alltoallv-function#remarks
             */
            for (int rank = 0; rank < op.comm_size; rank++) {
                size_t common_size = (op.comm_rank + rank) * (op.elem_count / 4);
                recv_counts[rank] = ((common_size > op.elem_count) || (common_size == 0))
                                        ? op.elem_count
                                        : common_size;
                send_counts[rank] = recv_counts[rank];
            }
        }
        else {
            bool is_even_rank = (op.comm_rank % 2 == 0) ? true : false;
            size_t send_count = (is_even_rank) ? (op.elem_count / 2) : op.elem_count;
            for (int rank = 0; rank < op.comm_size; rank++) {
                int is_even_peer = (rank % 2 == 0) ? true : false;
                send_counts[rank] = send_count;
                recv_counts[rank] = (is_even_peer) ? (op.elem_count / 2) : op.elem_count;
            }
        }
    }

    void fill_send_buffers(test_operation<T>& op) {
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < op.comm_size * op.elem_count; elem_idx++) {
                op.send_bufs[buf_idx][elem_idx] = op.comm_rank + buf_idx;
            }
        }
    }

    void run_derived(test_operation<T>& op) {
        void* send_buf;
        void* recv_buf;

        auto param = op.get_param();
        auto attr = ccl::create_operation_attr<ccl::alltoallv_attr>();

        for (auto buf_idx : op.buf_indexes) {
            op.prepare_attr(attr, buf_idx);
            send_buf = op.get_send_buf(buf_idx);
            recv_buf = op.get_recv_buf(buf_idx);

            op.events.push_back(ccl::alltoallv((param.place_type == PLACE_IN) ? recv_buf : send_buf,
                                               send_counts,
                                               recv_buf,
                                               recv_counts,
                                               op.datatype,
                                               transport_data::instance().get_comm(),
                                               transport_data::instance().get_stream(),
                                               attr));
        }
    }
};

RUN_METHOD_DEFINITION(alltoallv_test);
TEST_CASES_DEFINITION(alltoallv_test);
MAIN_FUNCTION();

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

#define ALGO_SELECTION_ENV "CCL_ALLGATHERV"

#include "test_impl.hpp"

template <typename T>
class allgatherv_test : public base_test<T> {
public:
    std::vector<size_t> recv_counts;
    std::vector<size_t> offset_counts;

    int check(test_operation<T>& op) {
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (int rank = 0; rank < op.comm_size; rank++) {
                for (size_t elem_idx = 0; elem_idx < recv_counts[rank];
                     elem_idx += op.get_check_step(elem_idx)) {
                    size_t idx = offset_counts[rank] + elem_idx;
                    T expected = static_cast<T>(rank + buf_idx);
                    if (base_test<T>::check_error(op, expected, buf_idx, idx)) {
                        return TEST_FAILURE;
                    }
                }
            }
        }
        return TEST_SUCCESS;
    }

    void alloc_buffers(test_operation<T>& op) {
        recv_counts.resize(op.comm_size);
        offset_counts.resize(op.comm_size);
        recv_counts[0] = op.elem_count;
        offset_counts[0] = 0;

        for (int rank = 1; rank < op.comm_size; rank++) {
            recv_counts[rank] = ((int)op.elem_count > rank) ? op.elem_count - rank : op.elem_count;
            offset_counts[rank] = recv_counts[rank - 1] + offset_counts[rank - 1];
        }

        // if (op.param.place_type == PLACE_OUT) {
        //     size_t total_recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
        //     for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        //         op.recv_bufs[buf_idx].resize(total_recv_count);
        //     }
        // }
    }

    void fill_send_buffers(test_operation<T>& op) {
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < recv_counts[op.comm_rank]; elem_idx++) {
                op.send_bufs[buf_idx][elem_idx] = op.comm_rank + buf_idx;
            }
        }
    }

    void fill_recv_buffers(test_operation<T>& op) {
        if (op.param.place_type != PLACE_IN)
            return;

        /* in case of in-place i-th rank already has result in i-th block of send buffer */
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < recv_counts[op.comm_rank]; elem_idx++) {
                op.recv_bufs[buf_idx][offset_counts[op.comm_rank] + elem_idx] =
                    op.comm_rank + buf_idx;
            }
        }
    }

    void run_derived(test_operation<T>& op) {
        void* send_buf;
        void* recv_buf;

        auto param = op.get_param();
        auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();

        for (auto buf_idx : op.buf_indexes) {
            op.prepare_attr(attr, buf_idx);
            send_buf = op.get_send_buf(buf_idx);
            recv_buf = op.get_recv_buf(buf_idx);
            auto recv_buf_char_ptr = (char*)recv_buf;
            auto recv_buf_with_offset =
                recv_buf_char_ptr + (offset_counts[op.comm_rank] * op.datatype_size);

            op.events.push_back(
                ccl::allgatherv((param.place_type == PLACE_IN) ? recv_buf_with_offset : send_buf,
                                recv_counts[op.comm_rank],
                                recv_buf,
                                recv_counts,
                                op.datatype,
                                transport_data::instance().get_comm(),
                                transport_data::instance().get_stream(),
                                attr));
        }
    }
};

RUN_METHOD_DEFINITION(allgatherv_test);
TEST_CASES_DEFINITION(allgatherv_test);
MAIN_FUNCTION();

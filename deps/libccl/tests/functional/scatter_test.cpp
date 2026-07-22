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

#define ALGO_SELECTION_ENV "CCL_SCATTER"

#include "test_impl.hpp"

template <typename T>
class scatter_test : public base_test<T> {
public:
    /*
     * Scatter semantic verification (NCCL spec):
     * - Root owns send_buf of nranks * count elements.
     * - Block for rank i is at send_buf + i * count.
     * - Each rank receives exactly its own block in recv_buf.
     *
     * In-place (PLACE_IN) condition:
     * - recv_buf == send_buf + root * count * dtype_size
     * - For ROOT_RANK=0: recv_buf == send_buf (offset 0), copy is skipped.
     */
    int check(test_operation<T>& op) {
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (size_t elem_idx = 0; elem_idx < op.elem_count;
                 elem_idx += op.get_check_step(elem_idx)) {
                T expected = static_cast<T>((op.comm_rank * op.elem_count + elem_idx) % 256);
                if (base_test<T>::check_error(op, expected, buf_idx, elem_idx))
                    return TEST_FAILURE;
            }
        }
        return TEST_SUCCESS;
    }

    /* Root fills send buffer: block for rank r = r*elem_count + elem_idx (% 256) */
    void fill_send_buffers(test_operation<T>& op) {
        if (op.comm_rank != ROOT_RANK)
            return;

        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            for (int r = 0; r < op.comm_size; r++) {
                for (size_t elem_idx = 0; elem_idx < op.elem_count; elem_idx++) {
                    size_t flat_idx = static_cast<size_t>(r) * op.elem_count + elem_idx;
                    op.send_bufs[buf_idx][flat_idx] =
                        static_cast<T>((r * op.elem_count + elem_idx) % 256);
                }
            }
        }
    }

    void run_derived(test_operation<T>& op) {
        void* send_buf;
        void* recv_buf;
        auto param = op.get_param();
        auto attr = ccl::create_operation_attr<ccl::scatter_attr>();

        for (auto buf_idx : op.buf_indexes) {
            op.prepare_attr(attr, buf_idx);

            if (op.comm_rank == ROOT_RANK) {
                if (param.place_type == PLACE_IN) {
                    /* For PLACE_IN the test framework loads fill data into device_recv_bufs
                     * (not device_send_bufs). Use recv_buf as send_buf so root reads from
                     * the correct device buffer. Since ROOT_RANK=0, recv_buf == send_buf,
                     * satisfying the NCCL in-place condition:
                     *   recv_buf == send_buf + root * count * dtype_size  (offset = 0) */
                    send_buf = op.get_recv_buf(buf_idx);
                    recv_buf = op.get_recv_buf(buf_idx);
                }
                else {
                    send_buf = op.get_send_buf(buf_idx);
                    recv_buf = op.get_recv_buf(buf_idx);
                }
            }
            else {
                /* Non-root: send_buf is not used; recv_buf is always independent. */
                send_buf = nullptr;
                recv_buf = op.get_recv_buf(buf_idx);
            }

            op.events.push_back(ccl::scatter(send_buf,
                                             recv_buf,
                                             op.elem_count,
                                             op.datatype,
                                             ROOT_RANK,
                                             transport_data::instance().get_comm(),
                                             transport_data::instance().get_stream(),
                                             attr));
        }
    }
};

RUN_METHOD_DEFINITION(scatter_test);
TEST_CASES_DEFINITION(scatter_test);
MAIN_FUNCTION();

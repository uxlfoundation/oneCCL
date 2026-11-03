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

#pragma once

#include "cpu_coll.hpp"
#include "scatter_strategy.hpp"

template <class Dtype>
struct cpu_scatter_coll : cpu_base_coll<Dtype, scatter_strategy_impl> {
    using coll_base = cpu_base_coll<Dtype, scatter_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;

    cpu_scatter_coll(bench_init_attr init_attr) : coll_base(init_attr) {}

    virtual void prepare_internal(size_t elem_count,
                                  ccl::communicator& comm,
                                  ccl::stream& stream,
                                  size_t rank_idx) override {
        size_t send_count = elem_count * comm.size();
        size_t recv_count = elem_count;
        size_t send_bytes = send_count * base_coll::get_dtype_size();
        size_t recv_bytes = recv_count * base_coll::get_dtype_size();

        bool is_root = comm.rank() == COLL_ROOT;
        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            // Only the root holds a send buffer; non-root send_bufs are nullptr.
            if (is_root) {
                auto fill_vector = base_coll::template get_initial_values<Dtype>(
                    send_count, static_cast<int>(b_idx));
                memcpy(send_bufs[b_idx][rank_idx], fill_vector.data(), send_bytes);
            }
            // In-place root has no standalone recv buffer; its receive region
            // lives inside send_bufs at root*elem_count and is already filled.
            if (recv_bufs[b_idx][rank_idx] != nullptr) {
                memset(recv_bufs[b_idx][rank_idx], 0, recv_bytes);
            }
        }
    }

    virtual void finalize_internal(size_t elem_count,
                                   ccl::communicator& comm,
                                   ccl::stream& stream,
                                   size_t rank_idx) override {
        // Each rank receives the block at index [rank * elem_count] from root's send buffer,
        // which was filled uniformly with b_idx, so every element should equal b_idx.
        bool is_root = comm.rank() == COLL_ROOT;
        Dtype value;
        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            // In-place root reads its result from inside send_bufs at root*elem_count.
            Dtype* recv = static_cast<Dtype*>(recv_bufs[b_idx][rank_idx]);
            if (recv == nullptr && is_root) {
                recv = static_cast<Dtype*>(send_bufs[b_idx][rank_idx]) + elem_count * COLL_ROOT;
            }
            Dtype expected = get_val<Dtype>(static_cast<float>(b_idx));
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
                value = recv[e_idx];
                if (base_coll::check_error<Dtype>(value, expected, comm)) {
                    std::cout << this->name() << " recv_bufs: buf_idx " << b_idx << ", rank_idx "
                              << rank_idx << ", elem_idx " << e_idx << ", expected " << expected
                              << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};

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
#include "reduce_scatter_strategy.hpp"

template <class Dtype>
struct cpu_reduce_scatter_coll : cpu_base_coll<Dtype, reduce_scatter_strategy_impl> {
    using coll_base = cpu_base_coll<Dtype, reduce_scatter_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;

    cpu_reduce_scatter_coll(bench_init_attr init_attr) : coll_base(init_attr) {}

    virtual void finalize_internal(size_t elem_count,
                                   ccl::communicator& comm,
                                   ccl::stream& stream,
                                   size_t rank_idx) override {
        Dtype sbuf_expected = get_val<Dtype>(static_cast<float>(comm.rank()));
        Dtype rbuf_expected = get_val<Dtype>((comm.size() - 1) * ((float)comm.size() / 2));
        Dtype value;

        size_t recv_elem_count = elem_count / comm.size();

        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
                if (!base_coll::get_inplace()) {
                    value = ((Dtype*)send_bufs[b_idx][rank_idx])[e_idx];
                    if (value != sbuf_expected) {
                        std::cout << this->name() << " send_bufs: buf_idx " << b_idx
                                  << ", rank_idx " << rank_idx << ", elem_idx " << e_idx
                                  << ", expected " << sbuf_expected << ", got " << value
                                  << std::endl;
                        ASSERT(0, "unexpected value");
                    }
                }
            }

            for (size_t e_idx = 0; e_idx < recv_elem_count; e_idx++) {
                auto recv_buf_adjusted = (Dtype*)recv_bufs[b_idx][rank_idx];
                if (base_coll::get_inplace()) {
                    size_t offset_count = recv_elem_count * comm.rank();
                    auto send_buf = (Dtype*)send_bufs[b_idx][rank_idx];
                    recv_buf_adjusted = send_buf + offset_count;
                }
                value = recv_buf_adjusted[e_idx];
                if (base_coll::check_error<Dtype>(value, rbuf_expected, comm)) {
                    std::cout << this->name() << " recv_bufs: buf_idx " << b_idx << ", rank_idx "
                              << rank_idx << ", elem_idx " << e_idx << ", expected "
                              << rbuf_expected << ", got " << value << std::endl;
                    ASSERT(0, "unexpected value");
                }
            }
        }
    }
};

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
#include "allgatherv_strategy.hpp"

template <class Dtype>
struct cpu_allgatherv_coll : cpu_base_coll<Dtype, allgatherv_strategy_impl> {
    using coll_base = cpu_base_coll<Dtype, allgatherv_strategy_impl>;
    using coll_base::send_bufs;
    using coll_base::recv_bufs;

    cpu_allgatherv_coll(bench_init_attr init_attr) : coll_base(init_attr) {}

    virtual void finalize_internal(size_t elem_count,
                                   ccl::communicator& comm,
                                   ccl::stream& stream,
                                   size_t rank_idx) override {
        Dtype sbuf_expected = get_val<Dtype>(static_cast<float>(comm.rank()));
        Dtype value;
        for (size_t b_idx = 0; b_idx < base_coll::get_buf_count(); b_idx++) {
            if (!base_coll::get_inplace()) {
                for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
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

            for (int idx = 0; idx < comm.size(); idx++) {
                Dtype rbuf_expected = get_val<Dtype>(static_cast<float>(idx));
                for (size_t e_idx = 0; e_idx < elem_count; e_idx++) {
                    value = ((Dtype*)recv_bufs[b_idx][rank_idx])[idx * elem_count + e_idx];
                    if (base_coll::check_error<Dtype>(value, rbuf_expected, comm)) {
                        std::cout << this->name() << " recv_bufs: buf_idx " << b_idx
                                  << ", rank_idx " << rank_idx << ", elem_idx " << e_idx
                                  << ", expected " << rbuf_expected << ", got " << value
                                  << std::endl;
                        ASSERT(0, "unexpected value");
                    }
                }
            }
        }
    }
};

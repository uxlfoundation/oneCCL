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

#include "coll_type.hpp"

struct scatter_strategy_impl {
    static constexpr const char* class_name() {
        return "scatter";
    }

    static constexpr coll_type class_id() {
        return coll_type::scatter;
    }

    // Multipliers scale the per-rank element count into buffer element counts.
    // Root's send buffer holds one count-sized block per destination rank
    // (nranks * count); every rank's recv buffer holds count elements.
    // NOTE: cpu_coll/sycl_coll allocate scatter buffers rank-aware (only root
    // gets the send buffer), so these multipliers size the buffers but do not
    // by themselves decide which ranks allocate them.
    size_t get_send_multiplier() {
        return transport_data::get_comm_size();
    }

    size_t get_recv_multiplier() {
        return 1;
    }

    static const ccl::scatter_attr& get_op_attr(const bench_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::scatter_attr>();
    }

    template <class Dtype, class... Args>
    void start_internal(ccl::communicator& comm,
                        size_t count,
                        const Dtype send_buf,
                        Dtype recv_buf,
                        const bench_exec_attr& bench_attr,
                        req_list_t& reqs,
                        Args&&... args) {
        // count here is the per-rank receive count (max_elem_count passed from base).
        // send_buf is nullptr on non-root ranks; onecclScatter ignores it.
        auto recv_buf_adjusted = recv_buf;
        if (recv_buf == nullptr) {
            // In-place on root: recv aliases into send at the root's own block.
            recv_buf_adjusted = send_buf + count * COLL_ROOT;
        }
        reqs.push_back(ccl::scatter(send_buf,
                                    recv_buf_adjusted,
                                    count,
                                    get_ccl_dtype<Dtype>(),
                                    COLL_ROOT,
                                    comm,
                                    std::forward<Args>(args)...));
    }
};

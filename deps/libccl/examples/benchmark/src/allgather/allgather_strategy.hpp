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

struct allgather_strategy_impl {
    static constexpr const char* class_name() {
        return "allgather";
    }

    size_t get_send_multiplier() {
        return 1;
    }

    size_t get_recv_multiplier() {
        return transport_data::get_comm_size();
    }

    static const ccl::allgather_attr& get_op_attr(const bench_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::allgather_attr>();
    }

    template <class Dtype, class... Args>
    void start_internal(ccl::communicator& comm,
                        size_t count,
                        const Dtype send_buf,
                        Dtype recv_buf,
                        const bench_exec_attr& bench_attr,
                        req_list_t& reqs,
                        Args&&... args) {
        auto send_buf_adjusted = send_buf;
        if (send_buf == nullptr) {
            // In-place operation
            size_t offset_count = count * comm.rank();
            send_buf_adjusted = recv_buf + offset_count;
        }
        reqs.push_back(ccl::allgather(send_buf_adjusted,
                                      recv_buf,
                                      count,
                                      get_ccl_dtype<Dtype>(),
                                      comm,
                                      std::forward<Args>(args)...));
    }
};

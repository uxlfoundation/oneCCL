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

struct alltoall_strategy_impl {
    static constexpr const char* class_name() {
        return "alltoall";
    }

    size_t get_send_multiplier() {
        return transport_data::get_comm_size();
    }

    size_t get_recv_multiplier() {
        return transport_data::get_comm_size();
    }

    static const ccl::alltoall_attr& get_op_attr(const bench_exec_attr& bench_attr) {
        return bench_attr.get_attr<ccl::alltoall_attr>();
    }

    template <class Dtype, class... Args>
    void start_internal(ccl::communicator& comm,
                        size_t count,
                        const Dtype send_buf,
                        Dtype recv_buf,
                        const bench_exec_attr& bench_attr,
                        req_list_t& reqs,
                        Args&&... args) {
        reqs.push_back(ccl::alltoall(
            send_buf, recv_buf, count, get_ccl_dtype<Dtype>(), comm, std::forward<Args>(args)...));
    }
};

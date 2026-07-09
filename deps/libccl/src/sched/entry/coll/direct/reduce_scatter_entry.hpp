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

#include "sched/entry/coll/direct/base_coll_entry.hpp"

class reduce_scatter_entry : public base_coll_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "REDUCE_SCATTER";
    }

    reduce_scatter_entry() = delete;
    reduce_scatter_entry(ccl_sched* sched,
                         const ccl_buffer send_buf,
                         ccl_buffer recv_buf,
                         size_t recv_cnt,
                         const ccl_datatype& dtype,
                         ccl::reduction reduction,
                         ccl_comm* comm)
            : base_coll_entry(sched),
              send_buf(send_buf),
              recv_buf(recv_buf),
              recv_cnt(recv_cnt),
              dtype(dtype),
              op(reduction),
              comm(comm) {}

    void start() override {
        LOG_DEBUG("REDUCE_SCATTER entry req ", req, ", recv_cnt ", recv_cnt);

        size_t send_cnt = recv_cnt * comm->size();

        size_t send_bytes = send_cnt * dtype.size();
        size_t recv_bytes = recv_cnt * dtype.size();

        atl_status_t atl_status =
            comm->get_atl_comm()->reduce_scatter(sched->bin->get_atl_ep(),
                                                 send_buf.get_ptr(send_bytes),
                                                 recv_buf.get_ptr(recv_bytes),
                                                 recv_cnt,
                                                 dtype.atl_datatype(),
                                                 static_cast<atl_reduction_t>(op),
                                                 req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("REDUCE_SCATTER entry failed. atl_status: ", atl_status_to_str(atl_status));
        }
        else
            status = ccl_sched_entry_status_started;
    }

    void update() override {
        atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("REDUCE_SCATTER entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req.is_completed)
            status = ccl_sched_entry_status_complete;
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", recv_cnt ",
                           recv_cnt,
                           ", send_buf ",
                           send_buf,
                           ", recv_buf ",
                           recv_buf,
                           ", op ",
                           ccl_reduction_to_str(op),
                           ", comm_id ",
                           comm->get_comm_id(),
                           ", req ",
                           req,
                           "\n");
    }

private:
    ccl_buffer send_buf;
    ccl_buffer recv_buf;
    size_t recv_cnt;
    ccl_datatype dtype;
    ccl::reduction op;
    ccl_comm* comm;
    atl_req_t req{};
};

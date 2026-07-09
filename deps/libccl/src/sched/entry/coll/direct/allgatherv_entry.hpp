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

class allgatherv_entry : public base_coll_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ALLGATHERV";
    }

    allgatherv_entry() = delete;
    allgatherv_entry(ccl_sched* sched,
                     const ccl_buffer send_buf,
                     size_t send_count,
                     ccl_buffer recv_buf,
                     const size_t* recv_counts,
                     const ccl_datatype& dtype,
                     ccl_comm* comm)
            : base_coll_entry(sched),
              send_buf(send_buf),
              send_count(send_count),
              recv_buf(recv_buf),
              recv_counts(recv_counts, recv_counts + comm->size()),
              dtype(dtype),
              comm(comm),
              recv_bytes(comm->size()),
              offsets(comm->size()),
              sum_recv_bytes(0) {}

    void start() override {
        size_t dt_size = dtype.size();
        size_t send_bytes = send_count * dt_size;
        int comm_size = comm->size();
        int i;

        recv_bytes[0] = recv_counts[0] * dt_size;
        offsets[0] = 0;
        sum_recv_bytes = recv_bytes[0];

        for (i = 1; i < comm_size; i++) {
            recv_bytes[i] = recv_counts[i] * dt_size;
            offsets[i] = offsets[i - 1] + recv_bytes[i - 1]; // treat buffers as char buffers
            sum_recv_bytes += recv_bytes[i];
        }

        LOG_DEBUG("ALLGATHERV entry req ", req, ", send_bytes ", send_bytes);
        atl_status_t atl_status = comm->get_atl_comm()->allgatherv(sched->bin->get_atl_ep(),
                                                                   send_buf.get_ptr(send_bytes),
                                                                   send_bytes,
                                                                   recv_buf.get_ptr(sum_recv_bytes),
                                                                   recv_bytes.data(),
                                                                   offsets.data(),
                                                                   req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("ALLGATHERV entry failed. atl_status: ", atl_status_to_str(atl_status));
        }
        else
            status = ccl_sched_entry_status_started;
    }

    void update() override {
        atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("ALLGATHERV entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req.is_completed) {
            status = ccl_sched_entry_status_complete;
        }
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", send_count ",
                           send_count,
                           ", send_buf ",
                           send_buf,
                           ", recv_counts[0] ",
                           recv_counts[0],
                           ", recv_buf ",
                           recv_buf,
                           ", recv_bytes[0] ",
                           recv_bytes[0],
                           ", offsets[0] ",
                           offsets[0],
                           ", comm_id ",
                           comm->get_comm_id(),
                           ", req ",
                           req,
                           "\n");
    }

private:
    const ccl_buffer send_buf;
    const size_t send_count;
    const ccl_buffer recv_buf;
    const std::vector<size_t> recv_counts;
    const ccl_datatype dtype;
    const ccl_comm* comm;
    atl_req_t req{};

    std::vector<size_t> recv_bytes;
    std::vector<size_t> offsets;
    size_t sum_recv_bytes;
};

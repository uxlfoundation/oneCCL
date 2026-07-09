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

#include "comm/comm.hpp"
#include "sched/entry/coll/direct/base_coll_entry.hpp"

class alltoallv_entry : public base_coll_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ALLTOALLV";
    }

    alltoallv_entry() = delete;
    alltoallv_entry(ccl_sched* sched,
                    const ccl_buffer send_buf,
                    const size_t* send_counts,
                    ccl_buffer recv_buf,
                    const size_t* recv_counts,
                    const ccl_datatype& dtype,
                    ccl_comm* comm)
            : base_coll_entry(sched),
              send_buf(send_buf),
              send_counts(send_counts, send_counts + comm->size()),
              recv_buf(recv_buf),
              recv_counts(recv_counts, recv_counts + comm->size()),
              dtype(dtype),
              comm(comm),
              send_bytes(comm->size()),
              recv_bytes(comm->size()),
              send_offsets(comm->size()),
              recv_offsets(comm->size()),
              sum_send_bytes(0),
              sum_recv_bytes(0) {}

    void start() override {
        size_t dt_size = dtype.size();
        int comm_size = comm->size();
        int i;
        sum_recv_bytes = 0;
        sum_send_bytes = 0;

        send_bytes[0] = send_counts[0] * dt_size;
        recv_bytes[0] = recv_counts[0] * dt_size;
        send_offsets[0] = 0;
        recv_offsets[0] = 0;
        sum_send_bytes = send_bytes[0];
        sum_recv_bytes = recv_bytes[0];

        for (i = 1; i < comm_size; i++) {
            send_bytes[i] = send_counts[i] * dt_size;
            recv_bytes[i] = recv_counts[i] * dt_size;
            send_offsets[i] =
                send_offsets[i - 1] + send_bytes[i - 1]; // treat buffers as char buffers
            recv_offsets[i] = recv_offsets[i - 1] + recv_bytes[i - 1];
            sum_send_bytes += send_bytes[i];
            sum_recv_bytes += recv_bytes[i];
        }

        LOG_DEBUG("alltoallv entry req ", req, ", sum_send_bytes ", sum_send_bytes);

        atl_status_t atl_status = comm->get_atl_comm()->alltoallv(sched->bin->get_atl_ep(),
                                                                  send_buf.get_ptr(sum_send_bytes),
                                                                  send_bytes.data(),
                                                                  send_offsets.data(),
                                                                  recv_buf.get_ptr(sum_recv_bytes),
                                                                  recv_bytes.data(),
                                                                  recv_offsets.data(),
                                                                  req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("alltoallv entry failed. atl_status: ", atl_status_to_str(atl_status));
        }
        else
            status = ccl_sched_entry_status_started;
    }

    void update() override {
        atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("alltoallv entry failed. atl_status: ", atl_status_to_str(atl_status));
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
                           ", send_counts[0] ",
                           send_counts[0],
                           ", send_buf ",
                           send_buf,
                           ", send_bytes[0] ",
                           send_bytes[0],
                           ", send_offsets[0] ",
                           send_offsets[0],
                           ", recv_counts[0] ",
                           recv_counts[0],
                           ", recv_buf ",
                           recv_buf,
                           ", recv_bytes[0] ",
                           recv_bytes[0],
                           ", recv_offsets[0] ",
                           recv_offsets[0],
                           ", comm_id ",
                           comm->get_comm_id(),
                           ", req ",
                           req,
                           "\n");
    }

private:
    const ccl_buffer send_buf;
    const std::vector<size_t> send_counts;
    const ccl_buffer recv_buf;
    const std::vector<size_t> recv_counts;
    const ccl_datatype dtype;
    const ccl_comm* comm;
    atl_req_t req{};

    std::vector<size_t> send_bytes;
    std::vector<size_t> recv_bytes;
    std::vector<size_t> send_offsets;
    std::vector<size_t> recv_offsets;
    size_t sum_send_bytes;
    size_t sum_recv_bytes;
};

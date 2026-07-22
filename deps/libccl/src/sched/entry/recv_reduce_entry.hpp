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

#include "common/global/global.hpp"
#include "comp/comp.hpp"
#include "sched/entry/entry.hpp"
#include "sched/queue/queue.hpp"

#include <utility>

enum ccl_recv_reduce_result_buf_type { ccl_recv_reduce_local_buf, ccl_recv_reduce_comm_buf };

class recv_reduce_entry final : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "RECV_REDUCE";
    }

    recv_reduce_entry(const recv_reduce_entry& other) = delete;
    recv_reduce_entry& operator=(const recv_reduce_entry& other) = delete;
    recv_reduce_entry(ccl_sched* sched,
                      ccl_buffer inout_buf,
                      size_t cnt,
                      const ccl_datatype& dtype,
                      ccl::reduction reduction_op,
                      int src,
                      ccl_comm* comm,
                      ccl_buffer comm_buf = ccl_buffer(),
                      ccl_recv_reduce_result_buf_type result_buf_type = ccl_recv_reduce_local_buf)
            : sched_entry(sched),
              inout_buf(inout_buf),
              in_cnt(cnt),
              dtype(dtype),
              op(reduction_op),
              src(src),
              comm(comm),
              comm_buf(comm_buf),
              result_buf_type(result_buf_type),
              fn(sched->coll_attr.reduction_fn) {
        CCL_THROW_IF_NOT(op != ccl::reduction::custom || fn,
                         "custom reduction requires user provided callback",
                         ", op ",
                         ccl_reduction_to_str(op),
                         ", fn ",
                         fn);

        CCL_THROW_IF_NOT(
            (result_buf_type == ccl_recv_reduce_local_buf && inout_buf.get_ptr() != nullptr) ||
                (result_buf_type == ccl_recv_reduce_comm_buf && comm_buf.get_ptr() != nullptr),
            "result buffer should be non null");

        if ((comm_buf.get_ptr() == nullptr || comm_buf == inout_buf) && in_cnt) {
            this->comm_buf = sched->alloc_buffer({ in_cnt * dtype.size(), inout_buf });
        }
    }

    ~recv_reduce_entry() override {
        if (status == ccl_sched_entry_status_started) {
            size_t bytes = in_cnt * dtype.size();
            LOG_DEBUG(
                "cancel RECV in RECV_REDUCE entry, src ", src, ", req ", req, ", bytes", bytes);
            comm->get_atl_comm()->cancel(sched->bin->get_atl_ep(), req);
        }
    }

    void start() override {
        atl_tag = comm->get_atl_comm()->tag_creator->create(
            src, comm->get_comm_id(), sched->sched_id, sched->get_op_id());
        size_t bytes = in_cnt * dtype.size();
        LOG_DEBUG("starting RECV in RECV_REDUCE entry, src ",
                  src,
                  ", tag ",
                  atl_tag,
                  ", req ",
                  req,
                  ", bytes ",
                  bytes);

        atl_status_t atl_status = comm->get_atl_comm()->recv(
            sched->bin->get_atl_ep(), comm_buf.get_ptr(bytes), bytes, src, atl_tag, req);

        update_status(atl_status);
    }

    void update() override {
        atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("RECV_REDUCE entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (!req.is_completed) {
            return;
        }

        LOG_DEBUG("completed RECV in RECV_REDUCE entry, req=", req, ", starting REDUCE");
        size_t bytes = in_cnt * dtype.size();
        size_t offset = inout_buf.get_offset();

        const ccl::fn_context context = { sched->coll_attr.match_id.c_str(), offset };

        ccl_buffer reduce_in_buf =
            (result_buf_type == ccl_recv_reduce_local_buf) ? comm_buf : inout_buf;

        ccl_buffer reduce_inout_buf =
            (result_buf_type == ccl_recv_reduce_local_buf) ? inout_buf : comm_buf;

        ccl::status comp_status = ccl_comp_reduce(sched,
                                                  reduce_in_buf.get_ptr(bytes),
                                                  in_cnt,
                                                  reduce_inout_buf.get_ptr(bytes),
                                                  nullptr, /* out_count */
                                                  dtype,
                                                  op,
                                                  fn,
                                                  &context);

        CCL_ASSERT(comp_status == ccl::status::success, "bad status ", comp_status);
        status = ccl_sched_entry_status_complete;
        LOG_DEBUG("completed REDUCE in RECV_REDUCE entry");
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", inout_buf ",
                           inout_buf,
                           ", in_cnt ",
                           in_cnt,
                           ", op ",
                           ccl_reduction_to_str(op),
                           ", red_fn  ",
                           fn,
                           ", src ",
                           src,
                           ", atl_tag ",
                           atl_tag,
                           ", comm_id ",
                           comm->get_comm_id(),
                           ", comm_buf ",
                           comm_buf,
                           ", result_buf_type ",
                           result_buf_type,
                           ", req ",
                           req,
                           "\n");
    }

private:
    ccl_buffer inout_buf;
    size_t in_cnt;
    ccl_datatype dtype;
    ccl::reduction op;
    int src;
    ccl_comm* comm;
    ccl_buffer comm_buf;
    ccl_recv_reduce_result_buf_type result_buf_type;
    uint64_t atl_tag = 0;
    ccl::reduction_fn fn;
    atl_req_t req{};
};

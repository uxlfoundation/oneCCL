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

#include "sched/entry/entry.hpp"
#include "sched/queue/queue.hpp"

class write_entry : public sched_entry,
                    public postponed_fields<write_entry,
                                            ccl_sched_entry_field_src_mr,
                                            ccl_sched_entry_field_dst_mr> {
public:
    static constexpr const char* class_name() noexcept {
        return "WRITE";
    }

    write_entry() = delete;
    write_entry(const write_entry& other) = delete;
    write_entry& operator=(const write_entry& other) = delete;
    write_entry(ccl_sched* sched,
                ccl_buffer src_buf,
                atl_mr_t* src_mr,
                size_t cnt,
                const ccl_datatype& dtype,
                int dst,
                atl_mr_t* dst_mr,
                size_t dst_buf_off,
                ccl_comm* comm)
            : sched_entry(sched),
              src_buf(src_buf),
              src_mr(src_mr),
              cnt(cnt),
              dtype(dtype),
              dst(dst),
              dst_mr(dst_mr),
              dst_buf_off(dst_buf_off),
              comm(comm) {}

    ~write_entry() {
        if (status == ccl_sched_entry_status_started) {
            LOG_DEBUG("cancel WRITE entry dst ", dst, ", req ", req);
            comm->get_atl_comm()->cancel(sched->bin->get_atl_ep(), req);
        }
    }

    void start() override {
        update_fields();

        LOG_DEBUG("WRITE entry dst ", dst, ", req ", req);

        CCL_THROW_IF_NOT(src_buf && src_mr && dst_mr, "incorrect values");

        if (!cnt) {
            status = ccl_sched_entry_status_complete;
            return;
        }

        size_t bytes = cnt * dtype.size();
        atl_status_t atl_status = comm->get_atl_comm()->write(sched->bin->get_atl_ep(),
                                                              src_buf.get_ptr(bytes),
                                                              bytes,
                                                              src_mr,
                                                              (uint64_t)dst_mr->buf + dst_buf_off,
                                                              dst_mr->remote_key,
                                                              dst,
                                                              req);
        update_status(atl_status);
    }

    void update() override {
        atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("WRITE entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req.is_completed) {
            LOG_DEBUG("WRITE entry done, dst ", dst);
            status = ccl_sched_entry_status_complete;
        }
    }

    const char* name() const override {
        return class_name();
    }

    atl_mr_t*& get_field_ref(field_id_t<ccl_sched_entry_field_src_mr> id) {
        return src_mr;
    }

    atl_mr_t*& get_field_ref(field_id_t<ccl_sched_entry_field_dst_mr> id) {
        return dst_mr;
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(dtype),
                           ", cnt ",
                           cnt,
                           ", src_buf ",
                           src_buf,
                           ", src_mr ",
                           src_mr,
                           ", dst ",
                           dst,
                           ", dst_mr ",
                           dst_mr,
                           ", dst_off ",
                           dst_buf_off,
                           ", comm_id ",
                           comm->get_comm_id(),
                           ", req ",
                           req,
                           "\n");
    }

private:
    ccl_buffer src_buf;
    atl_mr_t* src_mr;
    size_t cnt;
    ccl_datatype dtype;
    int dst;
    atl_mr_t* dst_mr;
    size_t dst_buf_off;
    ccl_comm* comm;
    atl_req_t req{};
};

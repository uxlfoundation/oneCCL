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
#include "sched/entry/subsched_entry.hpp"

class coll_entry : public subsched_entry,
                   public postponed_fields<coll_entry,
                                           ccl_sched_entry_field_send_buf,
                                           ccl_sched_entry_field_recv_buf,
                                           ccl_sched_entry_field_cnt,
                                           ccl_sched_entry_field_dtype,
                                           ccl_sched_entry_field_send_count> {
public:
    static constexpr const char* class_name() noexcept {
        return "COLL";
    }

    coll_entry() = delete;
    coll_entry(ccl_sched* sched, const ccl_coll_param& param, ccl_op_id_t op_id = 0)
            : subsched_entry(
                  sched,
                  op_id,
                  [this](ccl_sched* s) {
                      coll_entry::build_sched(s, this->param);
                  },
                  "coll_entry",
                  true /*is_coll*/),
              param(param) {}

    void start() override;

    bool is_strict_order_satisfied() override {
#ifdef CCL_ENABLE_SYCL
        /* use more strict condition for SYCL build to handle async execution */
        return (status == ccl_sched_entry_status_complete);
#else // CCL_ENABLE_SYCL
        return (subsched) ? subsched->is_strict_order_satisfied() : false;
#endif // CCL_ENABLE_SYCL
    }

    const char* name() const override {
        return class_name();
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_send_buf> id) {
        return param.send_buf;
    }

    ccl_buffer& get_field_ref(field_id_t<ccl_sched_entry_field_recv_buf> id) {
        return param.recv_buf;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_cnt> id) {
        return param.count;
    }

    ccl_datatype& get_field_ref(field_id_t<ccl_sched_entry_field_dtype> id) {
        return param.dtype;
    }

    size_t& get_field_ref(field_id_t<ccl_sched_entry_field_send_count> id) {
        return param.send_count;
    }

    static ccl::status build_sched(ccl_sched* sched, const ccl_coll_param& param);

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str,
                           "dt ",
                           ccl::global_data::get().dtypes->name(param.dtype),
                           ", coll ",
                           ccl_coll_type_to_str(param.ctype),
                           ", send_buf ",
                           param.send_buf,
                           ", recv_buf ",
                           param.recv_buf,
                           ", cnt ",
                           param.count,
                           ", op ",
                           ccl_reduction_to_str(param.reduction),
                           ", comm ",
                           param.comm,
                           "\n");
    }

private:
    ccl_coll_param param;
};

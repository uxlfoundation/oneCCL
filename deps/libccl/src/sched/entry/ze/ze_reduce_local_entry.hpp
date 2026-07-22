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
#include "sched/entry/ze/ze_base_entry.hpp"

class ze_reduce_local_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_REDUCE_LOCAL";
    }

    const char* name() const override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    explicit ze_reduce_local_entry(ccl_sched* sched,
                                   const ccl_buffer in_buf,
                                   size_t in_cnt,
                                   ccl_buffer inout_buf,
                                   size_t* out_cnt,
                                   const ccl_datatype& dtype,
                                   ccl::reduction op,
                                   const std::vector<ze_event_handle_t>& wait_events = {});

    void init_ze_hook() override;
    void finalize_ze_hook() override;

private:
    const ccl_buffer in_buf;
    const size_t in_cnt;
    const ccl_buffer inout_buf;
    const ccl_datatype dtype;
    const ccl::reduction op;

    ze_module_handle_t module{};
    std::string kernel_name{};
};

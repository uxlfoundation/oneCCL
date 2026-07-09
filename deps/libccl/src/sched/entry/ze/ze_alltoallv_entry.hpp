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

#include "common/utils/buffer.hpp"
#include "comp/comp.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"

#include <atomic>
#include <sstream>

class ze_alltoallv_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_ALLTOALLV";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    ze_alltoallv_entry() = delete;
    explicit ze_alltoallv_entry(ccl_sched* sched,
                                std::vector<ccl_buffer> send_bufs,
                                std::vector<ccl_buffer> recv_bufs,
                                std::vector<size_t> counts,
                                size_t buf_idx_start,
                                const ccl_datatype& dtype,
                                ccl_comm* comm,
                                const std::vector<ze_event_handle_t>& wait_events = {});

    void init_ze_hook() override;
    void finalize_ze_hook() override;

    void start() override;
    void update() override;

protected:
    void dump_detail(std::stringstream& str) const override;

private:
    std::vector<ccl_buffer> send_bufs;
    std::vector<ccl_buffer> recv_bufs;

    std::vector<size_t> counts;

    size_t buf_idx_start;

    const ccl_datatype dtype;

    std::string kernel_name;
};

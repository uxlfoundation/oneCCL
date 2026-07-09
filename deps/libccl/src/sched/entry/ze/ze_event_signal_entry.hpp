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

class ccl_sched;

class ze_event_signal_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_EVENT_SIGNAL";
    }

    const char* name() const override {
        return class_name();
    }

    ze_event_signal_entry() = delete;
    explicit ze_event_signal_entry(ccl_sched* sched, ccl_sched* master_sched);
    explicit ze_event_signal_entry(ccl_sched* sched, ze_event_handle_t event);
    ze_event_signal_entry(const ze_event_signal_entry&) = delete;

    bool is_strict_order_satisfied() override {
        /* use more strict condition for SYCL build to handle async execution */
        return (status == ccl_sched_entry_status_complete);
    }

    void start() override;
    void update() override;

private:
    ccl_sched* const master_sched{};
    const ze_event_handle_t event{};

    void handle_sycl_event_status();
};

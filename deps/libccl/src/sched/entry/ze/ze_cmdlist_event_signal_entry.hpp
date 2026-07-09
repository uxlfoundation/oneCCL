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

#include "sched/entry/ze/ze_base_entry.hpp"

class ze_cmdlist_event_signal_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_CMDLIST_EVENT_SIGNAL_ENTRY";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    ze_cmdlist_event_signal_entry() = delete;
    explicit ze_cmdlist_event_signal_entry(ccl_sched* sched,
                                           ccl_comm* comm,
                                           ze_event_handle_t signal_event,
                                           const std::vector<ze_event_handle_t>& wait_events);

    void init_ze_hook() override;

private:
    ze_event_handle_t signal_event{};
};

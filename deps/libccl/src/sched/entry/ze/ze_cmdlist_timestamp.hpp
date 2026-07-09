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

class ze_cmdlist_timestamp : public ze_base_entry {
private:
    std::string text{};

public:
    static constexpr const char* class_name() noexcept {
        return "ZE_CMDLIST_TIMESTAMP";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    ze_cmdlist_timestamp() = delete;
    explicit ze_cmdlist_timestamp(ccl_sched* sched,
                                  ccl_comm* comm,
                                  std::string text,
                                  const std::vector<ze_event_handle_t>& wait_events);

    void init_ze_hook() override;
};

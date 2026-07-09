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

#include "sched/entry/factory/entry_factory.hpp"

class ze_membarrier_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_MEMBARRIER";
    }

    const char* name() const override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    explicit ze_membarrier_entry(ccl_sched* sched,
                                 const std::vector<size_t>& range_sizes,
                                 const std::vector<const void*>& ranges,
                                 const std::vector<ze_event_handle_t>& wait_events = {});

    explicit ze_membarrier_entry(ccl_sched* sched,
                                 size_t size,
                                 const void* range,
                                 const std::vector<ze_event_handle_t>& wait_events = {});

    void init_ze_hook() override;

private:
    std::vector<size_t> range_sizes;
    std::vector<const void*> ranges;
};

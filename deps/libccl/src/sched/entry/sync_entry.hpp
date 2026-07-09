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
#include "common/utils/sync_object.hpp"
#include "common/utils/yield.hpp"
#include "sched/entry/entry.hpp"

#include <memory>

class sync_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "SYNC";
    }

    sync_entry() = delete;
    explicit sync_entry(ccl_sched* sched, std::shared_ptr<sync_object> sync)
            : sched_entry(sched, true),
              sync(sync) {}

    void start() override {
        status = ccl_sched_entry_status_started;
    }

    void update() override {
        if ((sched->get_start_idx() == start_idx) && should_visit) {
            /* ensure intra-schedule barrier before inter-schedule barrier */
            sync->visit();
            should_visit = false;
        }

        auto counter = sync->value();
        if (counter == 0) {
            status = ccl_sched_entry_status_complete;
        }
        else {
            // LOG_TRACE("waiting SYNC entry cnt ", counter);
            ccl_yield(ccl::global_data::env().yield_type);
        }
    }

    void reset(size_t idx) override {
        sched_entry::reset(idx);
        sync->reset();
        should_visit = true;
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "counter ", sync->value(), "\n");
    }

private:
    std::shared_ptr<sync_object> sync;
    bool should_visit = true;
};

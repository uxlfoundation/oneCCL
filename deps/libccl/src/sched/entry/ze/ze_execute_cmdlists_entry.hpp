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

#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"

// These entries, ze_execute_cmdlists_on_*_entry, are used to submit commands
//   to their respective command lists/queues, execute them, and mark the
//   schedule as 'submitted to GPU'.
//
// The difference between both, lies on the moment when these steps are taken:
//  - on_init:  similar to when init_ze_hook() is executed; i.e., upon start()
//              of the first ze_base_entry of the schedule.
//  - on_start: upon start() of this entry.
//
// Typically, algorithms will want to use on_init. Only in specific cases,
//  such as when ze_commands are cached, reordered, dynamically submitted,
//  etc., on_start should be used.

class ze_execute_cmdlists_on_init_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZEEXEC_CMDLIST_INIT";
    }
    const char* name() const override {
        return class_name();
    }

    ze_execute_cmdlists_on_init_entry(ccl_sched* sched) : ze_base_entry(sched, {}) {}

    void init() override {
        LOG_DEBUG("execute cmdlists entry");
        if (sched->use_single_list && !sched->get_ze_commands_bypass_flag()) {
            // submit commands to command lists
            int cmd_counter = sched->ze_commands_submit();

            if (cmd_counter) {
                // once command lists have commands, execute their associated cmdqueues
                sched->get_memory().list_manager->execute(this);
            }

            sched->set_submitted_to_gpu(true);
        }
        is_initialized = true;
    }

    void update() override {
        ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
        ze_base_entry::update();
    }
};

typedef uint32_t (*ze_commands_submit_function_t)(ccl_sched*);

class ze_execute_cmdlists_on_start_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZEEXEC_CMDLIST_START";
    }
    const char* name() const override {
        return class_name();
    }

    ze_execute_cmdlists_on_start_entry(ccl_sched* sched,
                                       std::shared_ptr<sync_object> sync_obj = nullptr,
                                       ze_commands_submit_function_t submit_fn = nullptr)
            : sched_entry(sched),
              sync_obj(std::move(sync_obj)),
              submit_fn(submit_fn) {}

    void start() override {
        status = ccl_sched_entry_status_started;
    }

    void update() override {
        if (sync_obj && sync_obj->value() > 0) {
            return;
        }

        if (sched->use_single_list && !sched->get_ze_commands_bypass_flag()) {
            if (!commands_submitted_flag) {
                // submit commands to command lists
                LOG_DEBUG("submit commands to device");
                if (submit_fn) {
                    cmd_counter = submit_fn(sched);
                }
                else {
                    cmd_counter = sched->ze_commands_submit();
                }
                commands_submitted_flag = true;
            }

            if (cmd_counter > 0) {
                // once command lists have commands, execute their associated cmdqueues
                LOG_DEBUG("execute command lists. cmd_counter: ", cmd_counter);
                sched->get_memory().list_manager->execute(this);
            }

            sched->set_submitted_to_gpu(true);
        }

        status = ccl_sched_entry_status_complete;
    }

    void reset(size_t idx) override {
        sched_entry::reset(idx);
        if (sync_obj) {
            sync_obj->reset();
        }
    }

private:
    std::shared_ptr<sync_object> sync_obj;
    ze_commands_submit_function_t submit_fn;
    uint32_t cmd_counter{};
    bool commands_submitted_flag{ false };
};

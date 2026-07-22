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

#include "sched/entry/copy/copy_helper.hpp"
#include "sched/entry/entry.hpp"
#include "sched/queue/queue.hpp"

class ccl_sched;
class ccl_comm;

using namespace ccl::ze;

// ze_base_entry
class ze_base_entry : public sched_entry {
public:
    ze_base_entry() = delete;
    ze_base_entry(const ze_base_entry &) = delete;
    ze_base_entry &operator=(const ze_base_entry &) = delete;
    virtual ~ze_base_entry();

    static ze_event_handle_t create_event(ze_event_pool_handle_t event_pool,
                                          ze_event_desc_t event_desc);
    static bool is_event_completed(ze_event_handle_t event);

    virtual void start() override;
    virtual void update() override;

    ze_command_list_handle_t get_comp_list(uint32_t index = 0) const;
    ze_command_list_handle_t get_copy_list(
        copy_direction direction = copy_direction::d2d,
        uint32_t index = 0,
        queue_group_type force_queue_type = queue_group_type::unknown) const;

    ze_event_handle_t entry_event{};

    bool is_finalized{}; // used to detect entries that was not finalized

protected:
    explicit ze_base_entry(ccl_sched *sched,
                           const std::vector<ze_event_handle_t> &wait_events,
                           ccl_comm *comm = nullptr,
                           uint32_t add_event_count = 0);

    void init() override;
    void finalize() override;

    /* ze hooks which can be implemented in derived entry */
    virtual void init_ze_hook(){};
    virtual void finalize_ze_hook(){};

    void init_entries();
    void finalize_entries();

    ze_event_handle_t create_event();
    void reset_events();
    void destroy_events();

    ccl_comm *comm{};
    int comm_rank{};
    int comm_size{};

    size_t worker_idx{};

    bool is_initialized{};

    ze_module_handle_t module{};

    ze_device_handle_t device{};
    ze_context_handle_t context{};

    const bool use_single_list;

    using ze_events_t = std::vector<ze_event_handle_t>;
    ze_events_t wait_events;

private:
    uint32_t event_counter{};
    ze_event_pool_desc_t event_pool_desc{};
    ze_event_pool_handle_t event_pool{};
    std::vector<ze_event_handle_t> events;
};

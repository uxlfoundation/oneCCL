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
#include "sched/entry/ze/ze_primitives.hpp"

#include <list>
#include <memory>
#include <unordered_map>

struct ccl_sched_base;
class sched_entry;

namespace ccl {
namespace ze {

class list_info {
public:
    list_info() = default;

    ze_command_list_handle_t get_native() const;
    ze_command_list_handle_t* get_native_ptr();
    const ze_command_list_desc_t& get_desc() const;
    bool is_valid() const;
    bool is_copy() const;
    uint32_t get_queue_index() const;

    bool is_closed{};
    bool is_executed{};

#ifdef ENABLE_DEBUG
    std::string list_name{};
#endif //ENABLE_DEBUG

private:
    friend class list_factory;
    ze_command_list_handle_t list{};
    ze_command_list_desc_t desc{};
    bool is_copy_list{};
    uint32_t queue_index{};
};

using list_info_t = typename std::shared_ptr<list_info>;

class queue_info {
public:
    queue_info() = default;

    ze_command_queue_handle_t get_native() const;
    const ze_command_queue_desc_t& get_desc() const;
    queue_group_type get_type() const;
    bool is_valid() const;
    bool is_copy() const;

private:
    friend class queue_factory;
    ze_command_queue_handle_t queue{};
    ze_command_queue_desc_t desc{};
    bool is_copy_queue{};
    queue_group_type type{};
};

using queue_info_t = typename std::shared_ptr<queue_info>;

class queue_factory {
public:
    queue_factory(ze_device_handle_t device, ze_context_handle_t context, queue_group_type type);
    queue_factory& operator=(const queue_factory&) = delete;
    queue_factory(const queue_factory&) = delete;
    queue_factory& operator=(queue_factory&&) = delete;
    ~queue_factory();
    queue_info_t get(uint32_t index);
    void clear();

    uint32_t get_ordinal() const;

    static bool queue_group_usable(ze_device_handle_t device, queue_group_type type);
    static bool can_use_queue_group(ze_device_handle_t device,
                                    queue_group_type type,
                                    copy_engine_mode mode);

private:
    const ze_device_handle_t device;
    const ze_context_handle_t context;
    const bool is_copy_queue;
    const queue_group_type type;

    static constexpr ssize_t worker_idx = 0;

    uint32_t queue_ordinal = 0;
    std::vector<queue_info_t> queues;

    void destroy(queue_info_t& queue);
    const char* get_type_str() const;
    uint32_t get_max_available_queue_count() const;
    uint32_t get_queue_index(uint32_t requested_index) const;
};

class list_factory {
public:
    list_factory(ze_device_handle_t device, ze_context_handle_t context, bool is_copy);
    list_factory& operator=(const list_factory&) = delete;
    list_factory& operator=(list_factory&&) = delete;
    ~list_factory() = default;
    list_info_t get(const queue_info_t& queue);
    void destroy(list_info_t& list);

private:
    const ze_device_handle_t device;
    const ze_context_handle_t context;
    const bool is_copy_list;

    static constexpr ssize_t worker_idx = 0;

    const char* get_type_str() const;
};

class list_manager {
public:
    list_manager() = delete;
    explicit list_manager(const ccl_sched_base* sched, const ccl_stream* stream);
    list_manager(const list_manager&) = delete;
    list_manager& operator=(const list_manager&) = delete;
    explicit list_manager(list_manager&&) = default;
    ~list_manager();

    void execute(const sched_entry* entry = nullptr);

    ze_command_list_handle_t get_comp_list(const sched_entry* entry = nullptr,
                                           const std::vector<ze_event_handle_t>& wait_events = {},
                                           uint32_t index = 0);
    ze_command_list_handle_t get_copy_list(
        const sched_entry* entry = nullptr,
        const std::vector<ze_event_handle_t>& wait_events = {},
        copy_direction direction = copy_direction::d2d,
        uint32_t index = 0,
        queue_group_type force_queue_type = queue_group_type::unknown);

    void clear();
    void reset_execution_state();

    bool is_executed() const;

private:
    const ccl_sched_base* sched;
    const ze_device_handle_t device;
    const ze_context_handle_t context;
    std::unique_ptr<queue_factory> comp_queue_factory;
    std::unique_ptr<queue_factory> link_queue_factory;
    std::unique_ptr<queue_factory> main_queue_factory;
    std::unique_ptr<list_factory> comp_list_factory;
    std::unique_ptr<list_factory> copy_list_factory;

    static constexpr ssize_t worker_idx = 0;

    // support of single_list mode
    // this maps contain info:
    // - queue index
    // - list for this queue index
    // - which entries use that list
    // list is only one always because this is single_list mode
    using queue_map_t = typename std::unordered_map<
        uint32_t,
        std::pair<list_info_t, std::unordered_map<const sched_entry*, bool>>>;
    queue_map_t comp_queue_map;
    queue_map_t link_queue_map;
    queue_map_t main_queue_map;

    // support of non single_list mode
    // in non single_list mode we execute lists
    // for specific entry on every execute(entry) method call
    // so remember entries and lists from that entries
    std::unordered_map<const sched_entry*, std::list<std::pair<queue_info_t, list_info_t>>>
        entry_map;

    std::list<std::pair<queue_info_t, list_info_t>> access_list;
    bool executed = false;
    bool main_queue_available = false;
    bool link_queue_available = false;
    bool main_queue_usable = false;

    std::pair<queue_factory*, queue_map_t*> get_factory_and_map(
        bool is_copy,
        copy_direction direction,
        queue_group_type force_type = queue_group_type::unknown) const;
    list_info_t get_list(const sched_entry* entry,
                         uint32_t index,
                         bool is_copy,
                         const std::vector<ze_event_handle_t>& wait_events,
                         copy_direction direction,
                         queue_group_type force_type = queue_group_type::unknown);

    void execute_list(queue_info_t& queue, list_info_t& list);

    void print_dump() const;
};

} // namespace ze
} // namespace ccl

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

#include <list>
#include <unordered_map>
#include "common/api_wrapper/ze_api_wrapper.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

class ccl_stream;

namespace ccl {
namespace ze {

class event_pool {
public:
    event_pool(ze_context_handle_t context);
    event_pool(ze_context_handle_t context, const ze_event_pool_desc_t& pool_desc);
    event_pool(ze_context_handle_t context,
               const ze_event_pool_desc_t& pool_desc,
               const ze_event_desc_t& event_desc);

    event_pool(const event_pool&) = delete;
    event_pool(event_pool&&) = default;
    event_pool& operator=(const event_pool&) = delete;
    event_pool& operator=(event_pool&&) = delete;

    virtual ~event_pool();

    operator ze_event_pool_handle_t() const;

    ze_event_handle_t create_event();
    ze_event_handle_t create_event(ze_event_desc_t& desc);

    size_t size() const;
    size_t capacity() const;

    void reset();
    void clear();

private:
    static constexpr size_t worker_idx{};

    const ze_context_handle_t context;
    const ze_event_pool_desc_t pool_desc;
    const ze_event_desc_t event_desc;

    ze_event_pool_handle_t pool{};
    std::list<ze_event_handle_t> events;

    void create_pool();
};

class event_manager {
public:
    event_manager(ze_context_handle_t context);
    event_manager(const ccl_stream* stream);

    event_manager(const event_manager&) = delete;
    event_manager(event_manager&&) = default;
    event_manager& operator=(const event_manager&) = delete;
    event_manager& operator=(event_manager&&) = default;

    virtual ~event_manager();

    ze_event_handle_t create(ze_event_desc_t desc = get_default_event_desc());
    std::vector<ze_event_handle_t> create(size_t count,
                                          ze_event_desc_t desc = get_default_event_desc());

    void reset();
    void clear();

    static ze_event_desc_t get_default_event_desc();
    static ze_event_pool_desc_t get_default_event_pool_desc();

protected:
    static constexpr size_t default_pool_size{ 50 };

    ze_context_handle_t context{};
    std::list<event_pool> pools;

    event_pool* add_pool(ze_event_pool_desc_t desc = get_default_event_pool_desc(),
                         ze_event_desc_t event_desc = get_default_event_desc());
};

// allows to dynamically allocate events by managing multiple event pools
// the basic idea is to have a list of event pools(depending on the number
// of requested events). For each pool we keep track of slots with allocated
// events and slots without them.
// note: this is relatively similar to event_manager class above, except it
// allows to release events one by one, but with additional overhead of status
// tracking. Potentially these 2 classes could be merged into one, but keep
// them separate for now for different use-cases.
class dynamic_event_pool {
public:
    dynamic_event_pool(const ccl_stream* stream);
    ~dynamic_event_pool();

    dynamic_event_pool(const dynamic_event_pool&) = delete;
    dynamic_event_pool(dynamic_event_pool&&) = delete;

    dynamic_event_pool& operator=(const dynamic_event_pool&) = delete;
    dynamic_event_pool& operator=(dynamic_event_pool&&) = delete;

    ze_event_handle_t get_event();
    void put_event(ze_event_handle_t event);

private:
    struct event_pool_info {
        ze_event_pool_handle_t pool;
        // number of allocated events from the pool
        size_t num_alloc_events;
        // vector of flags(true - slot is occupied, false - slot is free)
        std::vector<bool> event_alloc_status;
    };

    struct event_info {
        // position of the event's pool in the list
        std::list<event_pool_info>::iterator pool;
        // index inside the pool, necessary to track free/non-free status
        size_t pool_idx;
    };

    bool find_free_slot(event_info& slot);
    ze_event_handle_t create_event(const event_info& slot);

    // TODO: make some parameters configurable
    // TODO: check if another value would be better, as this one is chosen quite arbitrary
    static constexpr size_t event_pool_size{ 100 };
    int event_pool_request_idx{ 0 };

    static ze_event_pool_desc_t get_default_event_pool_desc();
    static const ze_event_pool_desc_t common_pool_desc;

    static constexpr size_t worker_idx{};

    ze_context_handle_t context;
    std::mutex lock;
    // map to keep allocation information for each event so we can properly track
    // free/non-free slots
    std::unordered_map<ze_event_handle_t, event_info> event_alloc_info;
    // list of all allocated event pools
    std::list<event_pool_info> event_pools;
    // store currently used event pool to avoid searching
    std::list<event_pool_info>::iterator current_pool{ event_pools.end() };
};

} // namespace ze
} // namespace ccl

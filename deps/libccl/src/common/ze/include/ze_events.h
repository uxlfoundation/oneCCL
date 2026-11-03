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
#include <vector>
#include <memory>
#include <sycl/sycl.hpp>
#include "ze_inc.h"

class ze_context_t;
class ze_event_pool_t;
class ze_cmd_list_t;

constexpr uint32_t ZE_EVENT_INVALID = (uint32_t)-1;

class ze_event_t : public ze_handle_t<ze_event_handle_t>
{
protected:
    ze_event_t() = default;
    ze_event_t(const ze_event_pool_t& pool,
               uint32_t               _index,
               ze_event_scope_flags_t signal_flags,
               ze_event_scope_flags_t wait_flags);

public:
    const uint32_t index = ZE_EVENT_INVALID;  // Default to invalid index
    ze_event_t(ze_event_handle_t handle) : ze_handle_t<ze_event_handle_t>(handle) {}
    virtual ze_result_t get_counter_addrs(uint64_t* counter_address) const { return ZE_RESULT_SUCCESS; } // Placeholder for getting counter addresses if needed;
    virtual ze_result_t signal(ze_cmd_list_t& list) const;
    std::shared_ptr<sycl::event> to_sycl_event(const sycl::context& ctx);
};

using ze_event_ptr = std::shared_ptr<ze_event_t>;
using ze_events_t  = std::vector<ze_event_ptr>;

// SYSTEM => // Host and Device
constexpr auto ZE_EVENT_FULL_SCOPE = ZE_EVENT_SCOPE_FLAG_HOST | ZE_EVENT_SCOPE_FLAG_DEVICE;

class ze_device_event_t : public ze_event_t
{
public:
    ze_device_event_t(const ze_event_pool_t& pool, uint32_t index)
    : ze_event_t(pool, index, ZE_EVENT_FULL_SCOPE, ZE_EVENT_FULL_SCOPE)
    {
    }
};

// ========================================================================================================

constexpr ze_event_pool_flags_t ZE_EVENT_POOL_FULL_SCOPE    = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_IPC;
constexpr uint32_t              ZE_EVENT_POOL_DEFAULT_COUNT = 1024;

class ze_event_pool_t : public ze_handle_t<ze_event_pool_handle_t>
{
public:
    ze_event_pool_t(const ze_context_t&   context,
                    uint32_t              count = ZE_EVENT_POOL_DEFAULT_COUNT,
                    ze_event_pool_flags_t flags = ZE_EVENT_POOL_FULL_SCOPE);

    ze_event_ptr create_event(uint32_t index);

protected:
    uint32_t m_count;
};

using ze_event_pool_ptr = std::shared_ptr<ze_event_pool_t>;
using ze_event_pools_t  = std::vector<ze_event_pool_t>;

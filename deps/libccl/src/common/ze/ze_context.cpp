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

#include "include/ze_context.h"
#include "include/ze_device.h"
#include "include/ze_cb_event.h"  // for CB Events

ze_context_t::ze_context_t(ze_driver_handle_t driver) : ZE_HANDLE(ze_context_handle_t, zeContextDestroy)
{
    ze_context_desc_t context_desc = {.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
                                      .pNext = nullptr,
                                      .flags = 0};

    ZE_NO_RET(zeContextCreate(driver, &context_desc, &m_handle));
}

ze_cmd_queue_ptr ze_context_t::create_queue(const ze_device_t& device, ze_engine_t engine, bool high_priority) const
{
    return std::make_shared<ze_cmd_queue_t>(device, engine, high_priority);
}

ze_cmd_queue_ptr
ze_context_t::create_queue(const ze_device_t& device, ze_engine_t engine, uint32_t queue_index, bool high_priority) const
{
    return std::make_shared<ze_cmd_queue_t>(device, engine, queue_index, high_priority);
}

ze_cmd_list_ptr ze_context_t::create_list(const ze_device_t& device,
                                      ze_engine_t          engine,
                                      bool               relaxed_order,
                                      bool               max_perf,
                                      bool               single_engine,
                                      bool               in_order,
                                      bool               cloneable) const
{
    // Create a normal command list
    return std::make_shared<ze_cmd_list_t>(device, engine, relaxed_order, max_perf, single_engine, in_order, cloneable);
}

ze_immediate_list_ptr ze_context_t::create_immediate_list(const ze_device_t& device,
                                                ze_engine_t          engine,
                                                bool               in_order,
                                                bool               high_priority) const
{
    return std::make_shared<ze_immediate_list_t>(device, engine, in_order, high_priority);
}

ze_immediate_list_ptr ze_context_t::create_immediate_list(const ze_device_t& device,
                                                ze_engine_t          engine,
                                                uint32_t           queue_index,
                                                bool               in_order,
                                                bool               high_priority) const
{
    return std::make_shared<ze_immediate_list_t>(device, engine, queue_index, in_order, high_priority);
}

ze_event_pool_ptr ze_context_t::create_event_pool(uint32_t count, ze_event_pool_flags_t flags) const
{
    return std::make_shared<ze_event_pool_t>(*this, count, flags);
}

ze_cb_event_ptr ze_context_t::create_cb_event(const ze_device_t& device) const
{
    return std::make_shared<ze_cb_event_t>(device);
}

ze_cb_event_ptr ze_context_t::create_cb_event(const ze_device_t& device, cb_event_arg_t& args) const
{
    return std::make_shared<ze_cb_event_t>(device, args);
}

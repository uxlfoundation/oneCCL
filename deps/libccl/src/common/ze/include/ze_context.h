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
#include "ze_events.h"
#include "ze_cb_event.h"
#include "ze_list.h"
#include "ze_queue.h"

class ze_device_t;

/**
 * @brief class representing a Level Zero context
 *
 * This class encapsulates the creation and management of Level Zero resources
 * such as command queues, command lists, and memory allocations.
 */
class ze_context_t : public ze_handle_t<ze_context_handle_t>
{
public:
    ze_context_t(ze_driver_handle_t driver);
    ze_context_t(ze_context_handle_t hdl){
        this->m_handle = hdl;
    }
    ze_cmd_queue_ptr create_queue(const ze_device_t& device, ze_engine_t engine, bool high_priority = false) const;

    ze_cmd_queue_ptr create_queue(const ze_device_t& device,
                              ze_engine_t engine,
                              uint32_t queue_index,
                              bool high_priority = false) const;

    ze_cmd_list_ptr create_list(const ze_device_t& device,
                            ze_engine_t          engine,
                            bool               relaxed_order = false,  // ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING
                            bool               max_perf      = false,  // ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT
                            bool               single_engine = false,  // ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY
                            bool               in_order      = true,  // ZE_COMMAND_LIST_FLAG_IN_ORDER
                            bool               cloneable     = true) const;              // ZE_COMMAND_LIST_FLAG_EXP_CLONEABLE

    ze_immediate_list_ptr create_immediate_list(const ze_device_t& device,
                                                ze_engine_t engine,
                                                bool in_order,
                                                bool high_priority = false) const;

    ze_immediate_list_ptr create_immediate_list(const ze_device_t& device,
                                                ze_engine_t          engine,
                                                uint32_t           queue_index,
                                                bool               in_order,
                                                bool               high_priority = false) const;

    ze_event_pool_ptr create_event_pool(uint32_t count, ze_event_pool_flags_t flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE | ZE_EVENT_POOL_FLAG_IPC) const;
    ze_cb_event_ptr   create_cb_event(const ze_device_t& device) const;
    ze_cb_event_ptr   create_cb_event(const ze_device_t& device, cb_event_arg_t& args) const;
};

using ze_context_ptr = std::shared_ptr<ze_context_t>;

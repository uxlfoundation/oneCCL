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
#include "ze_exp.h"
#include "ze_events.h"
#include "ze_list.h"

class ze_device_t;

typedef uint64_t cb_event_mem_t;

typedef struct ze_cb_event_arg
{
    uint64_t  completion;
    uint64_t* deviceStorage;
    uint64_t* hostStorage;
} cb_event_arg_t;

/**
 * @brief class representing a Level Zero counter-based event
 *
 * This class encapsulates the creation and management of counter-based events
 * in the Level Zero API.
 */
class ze_cb_event_t : public ze_event_t
{
public:

    ze_cb_event_t(const ze_device_t& device);
    ze_cb_event_t(const ze_device_t& device, const cb_event_arg_t& arg);
    // ze_cb_event_t(const ze_context_handle_t& ctxt, const ze_device_handle_t& device, const cb_event_arg_t& arg);

    // ze_result_t synchronize(uint64_t timeout = UINT64_MAX);
    ze_result_t get_completion_value(uint64_t* completion_value) const;
    ze_result_t get_counter(uint64_t* counter) const;
    ze_result_t get_counter_addrs(uint64_t* counter_address) const override;
    ze_result_t reset_counter(ze_cmd_list_t& list, const ze_sync_t& sync = {});
    ze_result_t signal(ze_cmd_list_t& list) const override;
protected:
    ze_cb_event_t() = default;
    // creates the underlying counter-based event; arg is optional external sync storage
    ze_result_t create(const ze_device_t& device, const cb_event_arg_t* arg);
    // queries both the completion value and the counter device address of this event
    ze_result_t get_device_address(uint64_t* completion_value, uint64_t* counter_address) const;
    // true when the event is backed by an external doorbell cell (see cb_event_arg_t),
    // false when the driver owns the counter
    bool has_ext_storage() const { return m_args.deviceStorage != nullptr; }
    const cb_event_arg_t m_args = {};
};

using ze_cb_event_ptr = std::shared_ptr<ze_cb_event_t>;


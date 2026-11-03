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
#include <memory>
#include <vector>
#include "ze_context.h"
#include "ze_group.h"
#include "ze_events.h"
#include "sched/entry/ze/ze_primitives.hpp"

constexpr uint32_t DEF_POOL_EVENTS = 1024;
/**
 * @brief Class representing a Level Zero device.
 *
 * Level Zero device represents a hardware device that can execute commands,
 * allocate memory, and create command lists and queues.
 */
class ze_device_t : public ze_handle_t<ze_device_handle_t>
{
public:
    ze_device_t(ze_device_handle_t device, ze_driver_handle_t driver, uint32_t device_index);
    ze_device_t(ze_device_handle_t device, ze_context_handle_t ctxt, ze_driver_handle_t driver, uint32_t device_index);

    void set_context(ze_context_handle_t ctxt);
    ze_result_t init();

    uint32_t index() const { return m_device_index; }

    const auto& properties() const { return m_properties; }

    std::string name() const { return m_properties.general.name; }

    const ze_groups_t& groups() const { return m_groups; }

    uint32_t ordinal(ze_engine_t type) const { return m_engine_ordinals[static_cast<size_t>(type)]; }

    const ze_context_ptr& context() const { return m_context; }

    ze_context_ptr create_context() const { return std::make_shared<ze_context_t>(m_driver); }

    ze_driver_handle_t get_native_driver() const { return m_driver; }

    uint32_t get_group_size(size_t bytes_per_item) const;

    ze_event_ptr create_event()
    {
        // recycle events from the pool in a round-robin fashion
        uint32_t index = event_id++ % DEF_POOL_EVENTS;
        return m_event_pool->create_event(index);
    }

    ze_cmd_queue_ptr create_queue(ze_engine_t engine) const {
        return context()->create_queue(*this, engine);
    }

    ze_cmd_list_ptr create_list(ze_engine_t engine, bool is_immediate) const {
        if(is_immediate) {
            return context()->create_immediate_list(*this, engine, true);
        }
        return context()->create_list(*this, engine);
    }
private:
    ze_driver_handle_t m_driver = nullptr;  // Driver handle for this device

    struct //cached properties
    {
        ze_device_properties_t general = ccl::ze::default_device_props;
        ze_device_compute_properties_t compute = ccl::ze::default_device_compute_properties;
    } m_properties;


    uint32_t m_device_index = 0;

    ze_context_ptr    m_context;      // default context for this device
    ze_event_pool_ptr m_event_pool ;  // default event pool for this device
    uint32_t          event_id = 0; // Counter for generating unique event IDs
    ze_groups_t       m_groups;

    using engine_ordinal_t = std::array<uint32_t, static_cast<size_t>(ze_engine_t::count)>;

    engine_ordinal_t m_engine_ordinals = {};
};

using ze_device_ptr = std::shared_ptr<ze_device_t>;
using ze_devices_t  = std::vector<ze_device_ptr>;

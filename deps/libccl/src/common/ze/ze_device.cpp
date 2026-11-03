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

#include "include/ze_device.h"
#include "include/ze_context.h"
#include "include/ze_exp.h"

// ze_device_t implementation
ze_device_t::ze_device_t(ze_device_handle_t device, ze_driver_handle_t driver, uint32_t device_index)
: ze_handle_t<ze_device_handle_t>(device, nullptr),
  m_driver(driver),
  m_device_index(device_index)
{
    m_engine_ordinals.fill(UINT32_MAX); // Initialize engine ordinals to an invalid value (-1)
    zexInit(m_driver);
}

ze_device_t::ze_device_t(ze_device_handle_t device, ze_context_handle_t ctxt, ze_driver_handle_t driver, uint32_t device_index)
: ze_handle_t<ze_device_handle_t>(device, nullptr),
  m_driver(driver),
  m_context{std::make_shared<ze_context_t>(ctxt)},
  m_device_index(device_index),
  m_event_pool{context()->create_event_pool(DEF_POOL_EVENTS, ZE_EVENT_POOL_FLAG_HOST_VISIBLE)}
{
    m_engine_ordinals.fill(UINT32_MAX); // Initialize engine ordinals to an invalid value (-1)
    zexInit(m_driver);
}

void ze_device_t::set_context(ze_context_handle_t ctxt)
{
    m_context = std::make_shared<ze_context_t>(ctxt);
    m_event_pool = context()->create_event_pool(DEF_POOL_EVENTS, ZE_EVENT_POOL_FLAG_HOST_VISIBLE);
}

ze_result_t ze_device_t::init()
{
    // Get device properties
    ZE_RET(zeDeviceGetProperties(m_handle, &m_properties.general));

    // Enumerate command queue groups
    uint32_t group_count = 0;
    ZE_RET(zeDeviceGetCommandQueueGroupProperties(m_handle, &group_count, nullptr));

    if (group_count == 0)
    {
        return ZE_RESULT_ERROR_UNKNOWN;
    }

    // Get group properties`
    std::vector<ze_command_queue_group_properties_t> group_properties(group_count);
    for (auto &prop : group_properties)
    {
        prop.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        prop.pNext = nullptr;
    }

    ZE_RET(zeDeviceGetCommandQueueGroupProperties(m_handle, &group_count, group_properties.data()));

    // Create groups
    for (uint32_t i = 0; i < group_count; i++)
    {
        auto group = std::make_shared<ze_group_t>(*this, i, group_properties[i]);
        m_groups.push_back(group);
        m_engine_ordinals[static_cast<size_t>(group->type)] = i;
    }

    ZE_RET(zeDeviceGetComputeProperties(m_handle, &m_properties.compute));

    return ZE_RESULT_SUCCESS;
}

uint32_t ze_device_t::get_group_size(size_t bytes_per_item) const
{
    const auto& compute_props = m_properties.compute;

    auto device_limit = std::min(compute_props.maxGroupSizeX, compute_props.maxTotalGroupSize);

    if (bytes_per_item > 0)
    {
        const auto max_local_items = (compute_props.maxSharedLocalMemory) / (static_cast<uint32_t>(bytes_per_item));
        device_limit = std::min(device_limit, max_local_items);
    }

    return device_limit;
}
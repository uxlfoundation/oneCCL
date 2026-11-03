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

#include "ze_inc.h"

class ze_device_t;  // Forward declaration

class ze_group_t
{
public:
    const ze_engine_t type = ze_engine_t::Unknown;  // Default to Unknown, will be set based on properties

    ze_group_t(const ze_device_t& device, uint32_t ordinal, const ze_command_queue_group_properties_t& properties);

    uint32_t ordinal() const { return m_ordinal; }
    uint32_t num_queues() const { return m_properties.numQueues; };
    uint32_t flags() const { return m_properties.flags; };

    const ze_device_t& device() const { return m_device; };

private:
    const ze_device_t&                        m_device;
    uint32_t                                  m_ordinal    = 0;   // Group index
    const ze_command_queue_group_properties_t m_properties = {};  // Properties of this group
};

using ze_group_ptr = std::shared_ptr<ze_group_t>;
using ze_groups_t  = std::vector<ze_group_ptr>;

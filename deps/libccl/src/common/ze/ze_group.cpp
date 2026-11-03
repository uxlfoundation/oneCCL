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

#include "include/ze_group.h"
#include "include/ze_device.h"

static inline ze_engine_t flags2type(uint32_t flags)
{
    if (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
        return ze_engine_t::Compute;

    if (flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)
        return ze_engine_t::Copy;

    return ze_engine_t::Unknown;
}

ze_group_t::ze_group_t(const ze_device_t&                         device,
                       uint32_t                                   ordinal,
                       const ze_command_queue_group_properties_t& properties)
: type(flags2type(properties.flags)), m_device(device), m_ordinal(ordinal), m_properties(properties)
{
}

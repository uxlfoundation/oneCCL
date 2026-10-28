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

#include "include/ze_queue.h"
#include "include/ze_context.h"
#include "include/ze_device.h"
#include "include/ze_fence.h"

ze_cmd_queue_t::ze_cmd_queue_t(const ze_device_t&      device,
                       ze_engine_t               engine,
                       bool                    high_priority,
                       bool                    explicit_que,
                       uint32_t                queue_index,
                       ze_command_queue_mode_t mode)
: ZE_HANDLE(ze_command_queue_handle_t, zeCommandQueueDestroy), type(engine), m_device(&device) 
{
    L0_TRACE("Construct queue", " engine=", static_cast<int>(engine),
             " high_priority=", high_priority,
             " explicit=", explicit_que,
             " index=", queue_index,
             " mode=", static_cast<int>(mode));

    ze_command_queue_desc_t desc = {
        .stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext    = nullptr,
        .ordinal  = device.ordinal(type),
        .index    = queue_index,
        .flags    = explicit_que ? static_cast<ze_command_queue_flags_t>(ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY)
                                 : static_cast<ze_command_queue_flags_t>(0),
        .mode     = mode,
        .priority = high_priority ? ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH : ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

    ZE_NO_RET(zeCommandQueueCreate(*device.context(), device, &desc, &m_handle));
}

ze_result_t ze_cmd_queue_t::execute(const ze_cmd_list_t& cmdlist, const ze_fence_t& fence) const
{
    L0_TRACE("Execute command list", " cmdlist=", static_cast<const void*>(cmdlist.handle()),
             " fence=", static_cast<const void*>(fence.handle()));

    ze_command_list_handle_t lists[] = {cmdlist};

    ZE_RET(zeCommandQueueExecuteCommandLists(m_handle, 1, lists, fence));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_queue_t::synchronize(uint64_t timeout) const
{
    L0_TRACE("Synchronize queue", " timeout=", timeout);
    ZE_RET(zeCommandQueueSynchronize(m_handle, timeout));
    return ZE_RESULT_SUCCESS;
}

ze_fence_ptr ze_cmd_queue_t::create_fence(bool signaled) const
{
    L0_TRACE("Create fence", " signaled=", signaled);
    return std::make_shared<ze_fence_t>(*this, signaled);
}

bool ze_cmd_queue_t::is_same_device(ze_device_t& device) const
{ 
    return m_device->handle() == device.handle();
}
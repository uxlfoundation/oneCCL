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
#include "ze_fence.h"

class ze_cmd_list_t;
class ze_fence_t;
class ze_device_t;

class ze_cmd_queue_t : public ze_handle_t<ze_command_queue_handle_t>
{
    const ze_device_t *m_device = nullptr;
protected:
    ze_cmd_queue_t(const ze_device_t&      device,
               ze_engine_t               engine,
               bool                    high_priority,
               bool                    explicit_que,
               uint32_t                queue_index,
               ze_command_queue_mode_t mode = ZE_COMMAND_QUEUE_MODE_DEFAULT);

public:
    ze_cmd_queue_t(const ze_device_t& device, ze_engine_t engine, bool high_priority)
    : ze_cmd_queue_t(device, engine, high_priority, false, 0)
    {
    }

    ze_cmd_queue_t(const ze_device_t& device, ze_engine_t engine, uint32_t queue_index, bool high_priority)
    : ze_cmd_queue_t(device, engine, high_priority, true, queue_index)
    {
    }

    ze_cmd_queue_t(ze_command_queue_handle_t hdl) {
        m_handle = hdl;
    }
    
    bool         is_same_device(ze_device_t& device) const;
    ze_result_t  execute(const ze_cmd_list_t& cmdlist, const ze_fence_t& fence = {}) const;
    ze_result_t  synchronize(uint64_t timeout = UINT64_MAX) const;
    ze_fence_ptr create_fence(bool signaled = false) const;

    const ze_engine_t type = ze_engine_t::Unknown;
};

using ze_cmd_queue_ptr = std::shared_ptr<ze_cmd_queue_t>;
using ze_cmd_queues_t  = std::vector<ze_cmd_queue_ptr>;
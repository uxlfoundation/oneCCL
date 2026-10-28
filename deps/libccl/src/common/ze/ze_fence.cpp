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

#include "include/ze_fence.h"
#include "include/ze_queue.h"

ze_fence_t::ze_fence_t(const ze_cmd_queue_t& que, bool signaled) : ze_fence_t()
{
    ze_fence_desc_t desc = {.stype = ZE_STRUCTURE_TYPE_FENCE_DESC,
                            .pNext = nullptr,
                            .flags = signaled ? static_cast<ze_fence_flags_t>(ZE_FENCE_FLAG_SIGNALED)
                                              : static_cast<ze_fence_flags_t>(0)};

    ZE_NO_RET(zeFenceCreate(que, &desc, &m_handle));
}

ze_result_t ze_fence_t::wait(uint64_t timeout) const
{
    ZE_RET(zeFenceHostSynchronize(m_handle, timeout));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_fence_t::query_status() const
{
    ZE_RET(zeFenceQueryStatus(m_handle));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_fence_t::reset() const
{
    ZE_RET(zeFenceReset(m_handle));
    return ZE_RESULT_SUCCESS;
}

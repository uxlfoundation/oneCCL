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
#include "include/ze_events.h"
#include "common/global/global.hpp"

ze_event_t::ze_event_t(const ze_event_pool_t& pool,
                       uint32_t               _index,
                       ze_event_scope_flags_t signal_flags,
                       ze_event_scope_flags_t wait_flags)
: ZE_HANDLE(ze_event_handle_t, zeEventDestroy), index(_index)
{
    ze_event_desc_t desc = {
        .stype  = ZE_STRUCTURE_TYPE_EVENT_DESC,
        .pNext  = nullptr,
        .index  = index,
        .signal = signal_flags,
        .wait   = wait_flags,
    };
    ZE_NO_RET(zeEventCreate(pool, &desc, &m_handle));
}

std::shared_ptr<sycl::event> ze_event_t::to_sycl_event(const sycl::context& ctx)
{
    // Convert the Level Zero event to a SYCL event and transfer ownership to SYCL. 
    // This allows the event to be used in SYCL command groups and ensures proper cleanup.
    sycl::event sycl_evt = sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
        {
            this->handle(), sycl::ext::oneapi::level_zero::ownership::transfer
        },
        ctx);
    LOG_TRACE("Convert L0 event {", this->handle(), "} to sycl event {", &sycl_evt, "}");

    // SYCL now owns the L0 event; prevent ze_event_t destructor from destroying it.
    this->m_handle = nullptr;
    
    return std::make_shared<sycl::event>(sycl_evt);
}

//-----------------------------------------------------------------------------------------------------------------------------------

ze_event_pool_t::ze_event_pool_t(const ze_context_t& context, uint32_t count, ze_event_pool_flags_t flags)
: ZE_HANDLE(ze_event_pool_handle_t, zeEventPoolDestroy), m_count(count)
{
    ze_event_pool_desc_t desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        .pNext = nullptr,
        .flags = flags,
        .count = count,
    };
    ZE_NO_RET(zeEventPoolCreate(context.handle(), &desc, 0, nullptr, &m_handle));
}

ze_event_ptr ze_event_pool_t::create_event(uint32_t index)
{
    CCL_ASSERT(index < m_count, "Event index out of bounds: ", index, " >= ", m_count);
    return std::make_shared<ze_device_event_t>(*this, index);
}

ze_result_t ze_event_t::signal(ze_cmd_list_t& list) const
{
    L0_TRACE("Signaling(", m_handle, ")");
    ZE_RET(zeCommandListAppendSignalEvent(list, m_handle));
    return ZE_RESULT_SUCCESS;
}
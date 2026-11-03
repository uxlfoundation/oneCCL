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

#include "include/ze_list.h"
#include "include/ze_context.h"
#include "include/ze_device.h"

ze_cmd_list_t::ze_cmd_list_t(const ze_device_t& device,
                     ze_engine_t          engine,
                     bool               weak_order,
                     bool               max_perf,
                     bool               single_engine,
                     bool               in_order,
                     bool               cloneable)
: ze_cmd_list_t(engine)
{
    L0_TRACE("device.index=",
             device.index(),
             " engine=",
             engine,
             " weak_order=",
             weak_order,
             " max_perf=",
             max_perf,
             " single_engine=",
             single_engine,
             " in_order=",
             in_order,
             " cloneable=",
             cloneable);

    uint32_t flags = 0;

    if (weak_order)
        flags |= ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING;

    if (max_perf)
        flags |= ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT;

    if (single_engine)
        flags |= ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY;

    if (in_order)
        flags |= ZE_COMMAND_LIST_FLAG_IN_ORDER;

    if (cloneable)
        flags |= ZE_COMMAND_LIST_FLAG_EXP_CLONEABLE;

    ze_command_list_desc_t cl_desc = {.stype                    = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
                                      .pNext                    = nullptr,
                                      .commandQueueGroupOrdinal = device.ordinal(type),
                                      .flags                    = static_cast<ze_command_list_flag_t>(flags)};

    ZE_NO_RET(zeCommandListCreate(*device.context(), device, &cl_desc, &m_handle));
}

ze_immediate_list_t::ze_immediate_list_t(const ze_device_t&      device,
                                         ze_engine_t               engine,
                                         bool                    in_order,
                                         bool                    high_priority,
                                         bool                    explicit_que,
                                         uint32_t                queue_index,
                                         ze_command_queue_mode_t mode)

: ze_cmd_list_t(engine)
{
    L0_TRACE("device.index=",
             device.index(),
             " engine=",
             engine,
             " high_priority=",
             high_priority,
             " in_order=",
             in_order,
             " explicit_que=",
             explicit_que,
             " queue_index=",
             queue_index,
             " mode=",
             mode);

    // Immediate lists are created with a command queue descriptor
    uint32_t flags = 0;

    if (in_order)
        flags |= ZE_COMMAND_QUEUE_FLAG_IN_ORDER;

    if (explicit_que)
        flags |= ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;

    ze_command_queue_desc_t desc = {.stype    = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
                                    .pNext    = nullptr,
                                    .ordinal  = device.ordinal(type),
                                    .index    = queue_index,
                                    .flags    = static_cast<ze_command_queue_flag_t>(flags),
                                    .mode     = mode,
                                    .priority = high_priority ? ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_HIGH
                                                              : ZE_COMMAND_QUEUE_PRIORITY_NORMAL};

    ZE_NO_RET(zeCommandListCreateImmediate(*device.context(), device, &desc, &m_handle));
    LOG_DEBUG("Created immediate command list with handle ", m_handle);
}
ze_result_t ze_cmd_list_t::mem_copy(void* dst, const void* src, size_t size, const ze_sync_t& sync)
{
    L0_TRACE("dst=",
             dst,
             " src=",
             src,
             " size=",
             size,
             " sync.signal=",
             sync.signal(),
             " sync.waits.size=",
             sync.waits().size(),
             " sync.waits=",
             sync.waits().data());

    ZE_RET(zeCommandListAppendMemoryCopy(m_handle,
                                          dst,
                                          src,
                                          size,
                                          sync.signal(),
                                          sync.waits().size(),
                                          (ze_event_handle_t*)sync.waits().data()));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::barrier(const ze_sync_t& sync)
{
    L0_TRACE("sync.signal=", sync.signal(), " sync.waits.size=", sync.waits().size());

    ZE_RET(zeCommandListAppendBarrier(m_handle,
                                       sync.signal(),
                                       sync.waits().size(),
                                       (ze_event_handle_t*)sync.waits().data()));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::mem_fill(void* dst, const void* pattern, size_t pattern_size, size_t size_bytes, const ze_sync_t& sync)
{
    L0_TRACE("dst=",
             (void*)dst,
             " pattern=",
             pattern,
             " pattern_size=",
             pattern_size,
             " size_bytes=",
             size_bytes,
             " sync.signal=",
             sync.signal(),
             " sync.waits.size=",
             sync.waits().size());

    ZE_RET(zeCommandListAppendMemoryFill(m_handle,
                                          dst,
                                          pattern,
                                          pattern_size,
                                          size_bytes,
                                          sync.signal(),
                                          sync.waits().size(),
                                          (ze_event_handle_t*)sync.waits().data()));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::mem_prefetch(const void* ptr, size_t size)
{
    L0_TRACE("ptr=", ptr, " size=", size);

    ZE_RET(zeCommandListAppendMemoryPrefetch(m_handle, ptr, size));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::mem_advise(const ze_device_t& device, const void* ptr, size_t size, ze_memory_advice_t advice)
{
    L0_TRACE("device.index=", device.index(), " ptr=", ptr, " size=", size, " advice=", advice);

    ZE_RET(zeCommandListAppendMemAdvise(m_handle, device, ptr, size, advice));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::signal_event(const ze_event_t& event)
{
    L0_TRACE("event=", (void*)event.handle());

    event.signal(*this);

    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::wait_event(const ze_event_t& event)
{
    L0_TRACE("event=", (void*)event);

    ze_event_handle_t events[] = {event};

    ze_result_t res = zeCommandListAppendWaitOnEvents(m_handle, 1, events);
    if (res != ZE_RESULT_SUCCESS)
    {
        L0_ERROR("Failed(", res, "), handle(", m_handle, "), event(", (void*)event, ")");
    }

    return res;
}

ze_result_t ze_cmd_list_t::event_reset(const ze_event_t& event)
{
    L0_TRACE("event=", (void*)event);

    ZE_RET(zeCommandListAppendEventReset(m_handle, event));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::close() const
{
    L0_TRACE("handle=", m_handle);

    ZE_RET(zeCommandListClose(m_handle));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::reset()
{
    L0_TRACE("handle=", m_handle);

    ZE_RET(zeCommandListReset(m_handle));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cmd_list_t::execute(const ze_cmd_queue_t& queue, bool _close, const ze_fence_t& fence) const
{
    L0_TRACE("queue=", &queue, " fence=", (void*)fence);

    if (_close)
        close();

    return queue.execute(*this, fence);
}

ze_result_t ze_immediate_list_t::execute_cmd_list_as_immediate(ze_cmd_list_ptr cmd_list, const ze_sync_t& sync) const
{
    L0_TRACE("handle=", m_handle);

    cmd_list->close();
    ze_command_list_handle_t cmd_lists[] = {cmd_list->handle()};
    ZE_RET(zeCommandListImmediateAppendCommandListsExp(m_handle, 
                                                        1, 
                                                        cmd_lists, 
                                                        sync.signal(), 
                                                        sync.waits().size(), 
                                                        (ze_event_handle_t*)sync.waits().data()));
    
    return ZE_RESULT_SUCCESS;
}
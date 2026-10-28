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

#include "include/ze_helpers.h"
#include "include/ze_cb_event.h"
#include "include/ze_device.h"

ze_cb_event_t::ze_cb_event_t(const ze_device_t& device)
{
    ZE_NO_RET(create(device, nullptr));
}

ze_cb_event_t::ze_cb_event_t(const ze_device_t& device, const cb_event_arg_t& arg) : m_args{arg}
{
    ZE_NO_RET(create(device, &arg));
}

// The zeEventCounterBased* entry points are optional in the Level Zero loader: on older
// loaders they are left null in ze_lib_ops, in which case we fall back to the equivalent
// zex extension functions resolved from the driver by zexInit().
ze_result_t ze_cb_event_t::create(const ze_device_t& device, const cb_event_arg_t* arg)
{
    // Whichever branch below creates the event, this object owns it and must destroy it.
    // Set before creating: on failure m_handle stays null and ~ze_handle_t does nothing.
    // ZE_HANDLE cannot be used here - ze_handle_t is an indirect base of this class.
    m_dtor = ZE_DTOR(ze_event_handle_t, zeEventDestroy);

    if (zeEventCounterBasedCreate != nullptr)
    {
        L0_DEBUG("zeEventCounterBasedCreate available, using official API");
        ze_event_counter_based_desc_t counter_desc = {
            .stype  = ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC,
            .pNext  = nullptr,
            .flags  = ZE_EVENT_COUNTER_BASED_FLAG_IMMEDIATE | ZE_EVENT_COUNTER_BASED_FLAG_NON_IMMEDIATE,
            .signal = ZE_EVENT_SCOPE_FLAG_DEVICE | ZE_EVENT_SCOPE_FLAG_HOST,
            .wait   = ZE_EVENT_SCOPE_FLAG_DEVICE | ZE_EVENT_SCOPE_FLAG_HOST,
        };

        ze_event_counter_based_external_sync_allocation_desc_t external_sync_desc = {};
        if (arg != nullptr)
        {
            external_sync_desc = {
                .stype           = ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_EXTERNAL_SYNC_ALLOCATION_DESC,
                .pNext           = nullptr,
                .deviceAddress   = arg->deviceStorage,
                .hostAddress     = arg->hostStorage,
                .completionValue = arg->completion,
            };
            counter_desc.pNext = &external_sync_desc;
        }

        ZE_RET(zeEventCounterBasedCreate(*device.context(), device, &counter_desc, &m_handle));
        return ZE_RESULT_SUCCESS;
    }

    L0_DEBUG("zeEventCounterBasedCreate is not available, falling back to zex extension");

    zex_counter_based_event_desc_t zex_counter_desc = {
        .stype       = ZEX_STRUCTURE_COUNTER_BASED_EVENT_DESC,
        .pNext       = nullptr,
        .flags       = ZEX_COUNTER_BASED_EVENT_FLAG_IMMEDIATE | ZEX_COUNTER_BASED_EVENT_FLAG_NON_IMMEDIATE,
        .signalScope = ZE_EVENT_SCOPE_FLAG_DEVICE | ZE_EVENT_SCOPE_FLAG_HOST,
        .waitScope   = ZE_EVENT_SCOPE_FLAG_DEVICE | ZE_EVENT_SCOPE_FLAG_HOST,
    };

    zex_counter_based_event_external_sync_alloc_properties_t zex_external_sync_desc = {};
    if (arg != nullptr)
    {
        zex_external_sync_desc = {
            .stype           = ZEX_STRUCTURE_COUNTER_BASED_EVENT_EXTERNAL_SYNC_ALLOC_PROPERTIES,
            .pNext           = nullptr,
            .deviceAddress   = arg->deviceStorage,
            .hostAddress     = arg->hostStorage,
            .completionValue = arg->completion,
        };
        zex_counter_desc.pNext = &zex_external_sync_desc;
    }

    ZE_RET(cclCounterBasedEventCreate2(*device.context(), device, &zex_counter_desc, &m_handle));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cb_event_t::get_device_address(uint64_t* completion_value, uint64_t* counter_address) const
{
    if (zeEventCounterBasedGetDeviceAddress != nullptr)
    {
        ZE_RET(zeEventCounterBasedGetDeviceAddress(m_handle, completion_value, counter_address));
        return ZE_RESULT_SUCCESS;
    }

    L0_DEBUG("zeEventCounterBasedGetDeviceAddress is not available, falling back to zex extension");
    ZE_RET(zexEventGetDeviceAddress(m_handle, completion_value, counter_address));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cb_event_t::get_completion_value(uint64_t* completion_value) const
{
    uint64_t counter_address = 0;
    ZE_RET(get_device_address(completion_value, &counter_address));
    L0_TRACE("CB event completion value: ", *completion_value);
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cb_event_t::get_counter(uint64_t* counter) const
{
    uint64_t completion_value = 0;
    uint64_t counter_address = 0;
    ZE_RET(get_device_address(&completion_value, &counter_address));
    L0_TRACE("CB event counter value: ", completion_value);
    *counter = completion_value;
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cb_event_t::get_counter_addrs(uint64_t* counter_address) const
{
    uint64_t completion_value = 0;
    ZE_RET(get_device_address(&completion_value, counter_address));
    L0_TRACE("CB event counter addrs: ", *counter_address);
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cb_event_t::signal(ze_cmd_list_t& list) const
{
    constexpr uint8_t VAL = 1UL;
    if(!has_ext_storage())
    {
        // No doorbell cell to write: the driver owns the counter, so the event itself is
        // signalled through a regular append-signal.
        return ze_event_t::signal(list);
    }
    // Fill exactly one byte, not the whole cb_event_mem_t: mem_fill replicates the
    // pattern across the range, so filling sizeof(cb_event_mem_t) with a 1-byte
    // pattern would store 0x0101010101010101 instead of VAL. The pattern is kept at
    // one byte because this runs on a copy engine, whose queue group may report
    // maxMemoryFillPatternSize == 1. The remaining bytes of the cell are already
    // zero - from the memset at allocation and from reset_counter() - so the value
    // the counter-based event waits on reads back as exactly VAL.
    if(m_args.completion != VAL)
    {
        L0_ERROR("Completion value ", m_args.completion, " does not match the signalled value ", VAL);
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
    ZE_RET(list.mem_fill(m_args.deviceStorage, &VAL, sizeof(VAL), sizeof(VAL)));
    return ZE_RESULT_SUCCESS;
}

ze_result_t ze_cb_event_t::reset_counter(ze_cmd_list_t& list, const ze_sync_t& sync)
{
    const size_t       event_size = sizeof(cb_event_mem_t);
    constexpr uint8_t RESET_VAL = 0UL;
    if(!has_ext_storage())
    {
        // The driver advances the completion value on every signal, so there is no stale
        // doorbell cell to clear between collectives.
        return ZE_RESULT_SUCCESS;
    }
    ZE_RET(list.mem_fill(m_args.deviceStorage, &RESET_VAL, sizeof(RESET_VAL), event_size, sync));
    return ZE_RESULT_SUCCESS;
}


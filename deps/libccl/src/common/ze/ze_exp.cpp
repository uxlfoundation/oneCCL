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

#include <mutex>
#include "include/ze_helpers.h"
#include "include/ze_exp.h"
#include "common/global/global.hpp"

static void* get_ext_func_pointer(ze_driver_handle_t driver, const char* func_name)
{
    void* func_ptr = nullptr;

    ze_result_t res = zeDriverGetExtensionFunctionAddress(driver, func_name, (void**)&func_ptr);

    if (res != ZE_RESULT_SUCCESS)
    {
        L0_ERROR("zeDriverGetExtensionFunctionAddress failed: ", res);
        return nullptr;
    }

    L0_INFO("Loaded ", func_name, " at ", (void*)func_ptr);

    return func_ptr;
};

class zex_loader_t
{
    using PFN_zexCounterBasedEventCreate2 = ze_result_t(ZE_APICALL*)(ze_context_handle_t,
                                                                     ze_device_handle_t,
                                                                     const zex_counter_based_event_desc_t*,
                                                                     ze_event_handle_t*);

    using PFN_zexEventGetDeviceAddress = ze_result_t (ZE_APICALL *)(
        ze_event_handle_t,
        uint64_t*,
        uint64_t*
    );

public:
    PFN_zexCounterBasedEventCreate2 CounterBasedEventCreate2 = nullptr;
    PFN_zexEventGetDeviceAddress EventGetDeviceAddress = nullptr;

    bool init(ze_driver_handle_t driver)
    {
        CounterBasedEventCreate2 =
            (PFN_zexCounterBasedEventCreate2)get_ext_func_pointer(driver, "zexCounterBasedEventCreate2");
        EventGetDeviceAddress =
            (PFN_zexEventGetDeviceAddress)get_ext_func_pointer(driver, "zexEventGetDeviceAddress");

        return CounterBasedEventCreate2 != nullptr && EventGetDeviceAddress != nullptr;
    }

} zex;

ze_result_t zexInit(ze_driver_handle_t driver)
{
    static ze_result_t res = ZE_RESULT_SUCCESS;

    static std::once_flag init_flag;
    std::call_once(init_flag, [&]() {
        if (!zex.init(driver))
        {
            L0_ERROR("Failed to initialize Level Zero experimental extension functions");
            res = ZE_RESULT_ERROR_UNINITIALIZED;
        }
        return res;
    });

    return res;
}

ze_result_t ZE_APICALL cclCounterBasedEventCreate2(ze_context_handle_t                   hContext,
                                                   ze_device_handle_t                    hDevice,
                                                   const zex_counter_based_event_desc_t* pDesc,
                                                   ze_event_handle_t*                    phEvent)
{
    if (zex.CounterBasedEventCreate2 == nullptr)
    {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    return zex.CounterBasedEventCreate2(hContext, hDevice, pDesc, phEvent);
}

ze_result_t ZE_APICALL zexEventGetDeviceAddress(ze_event_handle_t hEvent, uint64_t* completionValue, uint64_t* address)
{
    if (zex.EventGetDeviceAddress == nullptr)
    {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    return zex.EventGetDeviceAddress(hEvent, completionValue, address);
}

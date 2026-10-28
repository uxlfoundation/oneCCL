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

#include "include/ze_inc.h"
#include "include/ze_driver.h"

ze_result_t ze_driver_t::init()
{
    // Initialize the drivers
    ze_init_driver_type_desc_t initDesc = {
        .stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,
        .pNext = nullptr,
        .flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU  // Initialize GPU drivers only
    };

    uint32_t driver_count = 0;
    ZE_RET(zeInitDrivers(&driver_count, nullptr, &initDesc));

    // Get driver handles
    m_drivers.resize(driver_count);
    ZE_RET(zeInitDrivers(&driver_count, m_drivers.data(), &initDesc));

    // Enumerate devices for each driver
    for (uint32_t driver_idx = 0; driver_idx < driver_count; driver_idx++)
    {
        ze_api_version_t ze_version;
        ZE_RET(zeDriverGetApiVersion(m_drivers[driver_idx], &ze_version));
        L0_WARN("Driver ", driver_idx, " API version: ", ZE_MAJOR_VERSION(ze_version), ".", ZE_MINOR_VERSION(ze_version));

        ze_driver_properties_t driverProperties = {
            .stype = ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES,
            .pNext = nullptr
        };
        ZE_RET(zeDriverGetProperties(m_drivers[driver_idx], &driverProperties));
        L0_WARN("Driver ", driver_idx, " Version: ", (driverProperties.driverVersion & 0xffff));

        uint32_t device_count = 0;
        ZE_RET(zeDeviceGet(m_drivers[driver_idx], &device_count, nullptr));

        if (device_count == 0)
        {
            continue;
        }

        // Get device handles
        std::vector<ze_device_handle_t> device_handles(device_count);
        ZE_RET(zeDeviceGet(m_drivers[driver_idx], &device_count, device_handles.data()));

        // Create device objects
        for (uint32_t device_idx = 0; device_idx < device_count; device_idx++)
        {
            auto device =
                std::make_shared<ze_device_t>(device_handles[device_idx], m_drivers[driver_idx], m_devices.size());

            // Initialize device
            auto result = device->init();
            if (result != ZE_RESULT_SUCCESS)
            {
                return result;
            }

            m_devices.push_back(device);
        }
    }

    return ZE_RESULT_SUCCESS;
}

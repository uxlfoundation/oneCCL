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
#include "ze_device.h"

/**
 * @brief Class to manage Level Zero devices on the system.
 *
 * Level Zero driver manage all devices of same type
 * and creation of Level Zero context(s).
 */
class ze_driver_t
{
public:
    ze_driver_t()  = default;
    ~ze_driver_t() = default;

    // Initialize the manager (initialize drivers and enumerate devices)
    ze_result_t init();

    // Get the vector of devices
    const ze_devices_t& devices() const { return m_devices; }

private:
    using ze_drivers_t = std::vector<ze_driver_handle_t>;

    ze_drivers_t m_drivers;
    ze_devices_t m_devices;
};

using ze_device_manager_t = ze_driver_t;  // Alias for convenience, can be used interchangeably

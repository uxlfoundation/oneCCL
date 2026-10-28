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

#include <sycl/sycl.hpp>
#include "comm/comm.hpp"
#include "common/ze/ze_wrapper.hpp"

class ce_utils_t
{
public:
    // Constructor to initialize L0 wrapper objects 
    ce_utils_t(ccl_comm* comm, sycl::queue& q);
    // Prepare for a collective operation - synchronize all ranks
    ze_cmd_list_ptr prolog(ccl_comm* comm, sycl::queue& q);
    // Finalize after a collective operation - synchronize all ranks and return a sycl event
    std::shared_ptr<sycl::event> epilog(ccl_comm* comm, sycl::queue& q);    
    // Helper function to extract L0 device pointer from SYCL device
    static ze_device_ptr get_ze_device(sycl::device& dev, const ze_devices_t& ze_devices);
    // Helper function to extract L0 device pointer from SYCL queue
    static ze_device_ptr get_ze_device(sycl::queue& q, const ze_devices_t& ze_devices);
    // Helper function to get L0 device pointer from Level Zero device handle
    static ze_device_ptr get_ze_device(ze_device_handle_t ze_dev, const ze_devices_t& ze_devices);
    static void enumerate_ze_devices(ze_device_manager_t &device_manager);
private:
    ze_device_ptr ze_device;
    ze_cmd_queue_ptr  ze_queue;
    ze_cmd_list_ptr   cmd_list;
    ze_event_ptr  last_event_in_sycl_queue;
    // Helper function to get the L0 event associated with a SYCL event
    ze_event_handle_t get_ze_event(sycl::event& sycl_ev);
    // Find the copy engine queue associated with the L0 device
    ze_cmd_queue_ptr get_copy_engine_queue(ze_device_ptr ze_device, ccl_comm* comm); 
    void check_point(const char* msg);
};


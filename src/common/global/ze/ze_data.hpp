/*
 Copyright 2016-2020 Intel Corporation
 
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

#include <unordered_map>

#include "common/global/ze/ze_fd_manager.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/ze/ze_event_manager.hpp"

#include "sched/sched_timer.hpp"

namespace ccl {
namespace ze {

struct device_info {
    ze_device_handle_t device;
    uint32_t parent_idx;
    ze_device_uuid_t uuid;
    int physical_idx;
    uint32_t total_threads;
    int numSlices;
    int numSubslicesPerSlice;
    device_family family;
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
    ze_pci_address_ext_t pci;
#endif // ZE_PCI_PROPERTIES_EXT_NAME

    device_info(ze_device_handle_t dev, uint32_t parent_idx);
};

class global_data_desc {
public:
    std::vector<ze_driver_handle_t> drivers;
    std::vector<ze_context_handle_t> contexts;
    std::vector<device_info> devices;
    std::unique_ptr<ze::cache> cache;
    std::unordered_map<ze_context_handle_t, ccl::ze::dynamic_event_pool> dynamic_event_pools;

    std::atomic<size_t> kernel_counter{};

    global_data_desc();
    global_data_desc(const global_data_desc&) = delete;
    global_data_desc(global_data_desc&&) = delete;
    global_data_desc& operator=(const global_data_desc&) = delete;
    global_data_desc& operator=(global_data_desc&&) = delete;
    ~global_data_desc();

    void init_external_pointer_registration();
    bool external_pointer_registration_enabled{ false };
    void import_external_pointer(void* ptr, size_t size);
    void release_imported_pointer(void* ptr);

private:
    typedef ze_result_t (*pFnzexDriverImportExternalPointer)(ze_driver_handle_t, void*, size_t);
    typedef ze_result_t (*pFnzexDriverReleaseImportedPointer)(ze_driver_handle_t, void*);
    pFnzexDriverImportExternalPointer zexDriverImportExternalPointer = nullptr;
    pFnzexDriverReleaseImportedPointer zexDriverReleaseImportedPointer = nullptr;
};

} // namespace ze
} // namespace ccl

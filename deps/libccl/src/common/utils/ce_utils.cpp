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

#include "coll/algorithms/algorithm_utils.hpp"
#include "common/env/env.hpp"
#include "common/global/global.hpp"
// #if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
// #include "common/global/ze/ze_data.hpp"
// #ifdef CCL_ENABLE_UMF
// #include "umf/ipc.hpp"
// #endif // CCL_ENABLE_UMF
// #endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL
#include "common/ze/ze_wrapper.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "ce_utils.hpp"

#define DEBUG_CHECK_POINT_ENABLE 0

ce_utils_t::ce_utils_t(ccl_comm* comm, sycl::queue& q)
{
    // Get the L0 device associated with the SYCL queue
    ze_device = get_ze_device(q, comm->get_devices());
    CCL_THROW_IF_NOT(ze_device != nullptr, "unable to get L0 device from SYCL queue\n");
    #if 0
    // Extract native Level Zero command queue handle from SYCL queue
    // abandoning this approach for now as sycl queue contains several queues (compute, copy, etc.)
    auto ze_queue_handle = std::get<ze_command_queue_handle_t>(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q));
    ze_queue = std::make_shared<ze_cmd_queue_ptr>(ze_queue_handle);
    #else
    ze_queue = get_copy_engine_queue(ze_device, comm);
    #endif
    
    CCL_THROW_IF_NOT((ze_device->handle()==sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device())), "device handles mismatch\n");
    CCL_THROW_IF_NOT((ze_device->context()->handle()==sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context())), "context handles mismatch\n");
    cmd_list = ze_device->create_list(ze_engine_t::Copy, true);
    CCL_THROW_IF_NOT(cmd_list, "unable to create command list\n");
    // Get the last event from the SYCL queue
    sycl::event sycl_ev = get_last_event(q);
    // TODO: sycl event can be either device or host event. We need to handle both cases. For now we assume it's a device event.
    // Get the corresponding Level Zero event handle
    ze_event_handle_t ze_ev = get_ze_event(sycl_ev);
    last_event_in_sycl_queue = std::make_shared<ze_event_t>(ze_ev);
}

ze_cmd_list_ptr ce_utils_t::prolog(ccl_comm* comm, sycl::queue& q)
{
    LOG_DEBUG("start...");
    // append a wait for this event on the command list to ensure all prior SYCL operations are completed
    ZE_RET_NULL(cmd_list->wait_event(*last_event_in_sycl_queue));

    LOG_DEBUG("done");
    return cmd_list;
}

std::shared_ptr<sycl::event> ce_utils_t::epilog(ccl_comm* comm, sycl::queue& q)
{
    LOG_DEBUG("start...");

    // create event to sync completion from recv device to host
    ze_event_ptr completion_event = ze_device->create_event();
    CCL_THROW_IF_NOT(completion_event->is_valid(),"unable to create completion event\n");
    CCL_THROW_IF_NOT((cmd_list->signal_event(*completion_event) == ZE_RESULT_SUCCESS), "failed to signal completion event\n");

    std::shared_ptr<sycl::event> sycl_ev = completion_event->to_sycl_event(q.get_context());

    // Execute the command list on the copy engine queue
    cmd_list->execute(*ze_queue);

    if(DEBUG_CHECK_POINT_ENABLE) check_point("epilog done");

    LOG_DEBUG("done");
    return sycl_ev;
}

// Helper function to get L0 device pointer from Level Zero device handle
ze_device_ptr ce_utils_t::get_ze_device(ze_device_handle_t ze_dev, const ze_devices_t& ze_devices)
{
    for(const auto& device : ze_devices)
    {
        if(device->handle() == ze_dev)
        {
            return device;
        }
    }
    return nullptr;
}

// Helper function to extract L0 device pointer from SYCL device
ze_device_ptr ce_utils_t::get_ze_device(sycl::device& dev, const ze_devices_t& ze_devices)
{
    // Extract native Level Zero device handle from SYCL queue
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
    
    return get_ze_device(ze_dev, ze_devices);
}

// Helper function to extract L0 device pointer from SYCL queue
ze_device_ptr ce_utils_t::get_ze_device(sycl::queue& q, const ze_devices_t& ze_devices)
{
    // Extract native Level Zero device handle from SYCL queue
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    return get_ze_device(ze_dev, ze_devices);
}

static void print_detailed_device_info(const ze_device_t& device)
{
    const auto& properties = device.properties().general;
    const auto& groups = device.groups();
    uint32_t device_index = device.index();

    L0_PRINT("=============================================================");
    L0_PRINT("DETAILED DEVICE INFO: ", properties.name);
    L0_PRINT("=============================================================");

    // Print basic information
    L0_PRINT("Basic Information:");
    L0_PRINT("  Device Index: ", device_index);
    L0_PRINT("  Device Type: ", properties.type, " (",
          (properties.type == ZE_DEVICE_TYPE_GPU ? "GPU" : properties.type == ZE_DEVICE_TYPE_CPU ? "CPU"
                                                        : properties.type == ZE_DEVICE_TYPE_FPGA  ? "FPGA"
                                                        : properties.type == ZE_DEVICE_TYPE_MCA   ? "MCA"
                                                        : properties.type == ZE_DEVICE_TYPE_VPU   ? "VPU"
                                                        : "Unknown"),
          ")");
    L0_PRINT("  Vendor ID: 0x", std::hex, properties.vendorId, std::dec);
    L0_PRINT("  Device ID: 0x", std::hex, properties.deviceId, std::dec);

    // Print device flags
    L0_PRINT("");
    L0_PRINT("Device Flags (0x", std::hex, properties.flags, std::dec, "):");
    if ((properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) != 0)
    {
        L0_PRINT("  - Subdevice");
    }
    if ((properties.flags & ZE_DEVICE_PROPERTY_FLAG_ECC) != 0)
    {
        L0_PRINT("  - ECC");
    }
    if ((properties.flags & ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING) != 0)
    {
        L0_PRINT("  - On-demand paging");
    }
    if ((properties.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0)
    {
        L0_PRINT("  - Integrated");
    }
    else
    {
        L0_PRINT("  - Discrete (not integrated)");
    }

    // Print hardware architecture details
    L0_PRINT("");
    L0_PRINT("Hardware Architecture:");
    if ((properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) != 0)
    {
        L0_PRINT("  • Subdevice ID: ", properties.subdeviceId);
    }

    L0_PRINT("  • Core Clock Rate: ", properties.coreClockRate, " MHz");
    L0_PRINT("  • EU Topology:");
    L0_PRINT("      - ", properties.numEUsPerSubslice, " EUs per Subslice");
    L0_PRINT("      - ", properties.numSubslicesPerSlice, " Subslices per Slice");
    L0_PRINT("      - ", properties.numSlices, " Slices total");
    L0_PRINT("      - ", (properties.numEUsPerSubslice * properties.numSubslicesPerSlice * properties.numSlices),
          " EUs total");
    L0_PRINT("  • ", properties.numThreadsPerEU, " Hardware Threads per EU");
    L0_PRINT("  • ", properties.physicalEUSimdWidth, "-wide SIMD per EU");

    // Calculate total hardware threads
    const uint32_t total_threads = properties.numThreadsPerEU *
                            properties.numEUsPerSubslice *
                            properties.numSubslicesPerSlice *
                            properties.numSlices;
    L0_PRINT("  • Total Hardware Threads: ", total_threads);

    // Print memory information
    L0_PRINT("");
    L0_PRINT("Memory Information:");
    L0_PRINT("  • Max Memory Allocation Size: ", properties.maxMemAllocSize, " bytes (",
          (static_cast<double>(properties.maxMemAllocSize) / (1024.0 * 1024.0)), " MB)");

    // Print execution capabilities
    L0_PRINT("");
    L0_PRINT("Execution Capabilities:");
    L0_PRINT("  • Max Hardware Contexts: ", properties.maxHardwareContexts);
    L0_PRINT("  • Max Command Queue Priority: ", properties.maxCommandQueuePriority);

    // Print timing information
    L0_PRINT("");
    L0_PRINT("Timing Information:");
    L0_PRINT("  • Timer Resolution: ", properties.timerResolution,
          (properties.stype == ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES ? " ns" : " cycles/sec"));
    L0_PRINT("  • Timestamp Valid Bits: ", properties.timestampValidBits);
    L0_PRINT("  • Kernel Timestamp Valid Bits: ", properties.kernelTimestampValidBits);

    // Print UUID
    L0_PRINT("");
    std::ostringstream uuid_stream;
    uuid_stream << "Device UUID: ";
    for (int k = 0; k < ZE_MAX_DEVICE_UUID_SIZE; ++k)
    {
        uuid_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(properties.uuid.id[k]);
        if (k < ZE_MAX_DEVICE_UUID_SIZE - 1)
        {
            uuid_stream << ":";
        }
    }
    L0_PRINT(uuid_stream.str());

    // Print command queue groups
    L0_PRINT("");
    L0_PRINT("Command Queue Groups (", groups.size(), "):");
    for (size_t grp = 0; grp < groups.size(); grp++)
    {
        L0_PRINT("  Group ", grp, ":");
        const uint32_t flags = groups[grp]->flags();
        L0_PRINT("    • Flags: 0x", std::hex, flags, std::dec);

        std::ostringstream caps_stream;
        caps_stream << "    • Capabilities: ";
        if ((flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)!= 0)
        {
            caps_stream << "Compute ";
        }
        if ((flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)!= 0)
        {
            caps_stream << "Copy ";
        }
        if ((flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS)!= 0)
        {
            caps_stream << "CooperativeKernels ";
        }
        if ((flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS)!= 0)
        {
            caps_stream << "Metrics ";
        }
        L0_PRINT(caps_stream.str());

        L0_PRINT("    • Max Queues: ", groups[grp]->num_queues());
    }

    L0_PRINT("=============================================================");
}

void ce_utils_t::enumerate_ze_devices(ze_device_manager_t &device_manager)
{
    L0_PRINT("Enumerating L0 devices using class hierarchy...");
    // Create the device manager
    // Initialize the manager (this will enumerate all devices)
    const ze_result_t result = device_manager.init();
    if (result != ZE_RESULT_SUCCESS)
    {
        L0_PRINT("Failed to initialize L0 device manager: ", result);
        return;
    }

    for (const auto &device : device_manager.devices())
    {
        // Print detailed device properties
        print_detailed_device_info(*device);
    }

}

ze_event_handle_t ce_utils_t::get_ze_event(sycl::event& sycl_ev)
{
    // Convert SYCL event to native Level Zero event handle
    auto ze_event = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(sycl_ev);
    
    if (ze_event == nullptr)
    {
        CCL_THROW("failed to get native L0 event from SYCL event");
    }
    return ze_event;
}

// Find the copy engine queue associated with the L0 device
ze_cmd_queue_ptr ce_utils_t::get_copy_engine_queue(ze_device_ptr ze_device, ccl_comm* comm)
{
    return comm->get_ce_queue();
}

void ce_utils_t::check_point(const char* msg)
{
    static int i = 0;
    LOG_DEBUG("synchronize at ", msg);
    if(cmd_list->is_immediate())
    {
        // LOG_DEBUG("immediate command list, synchronizing...");
        cmd_list->synchronize(*ze_queue);
        // LOG_DEBUG("immediate command list synchronized");
    }
    else
    {
        // LOG_DEBUG("non-immediate command list, synchronizing...");
        cmd_list->execute(*ze_queue);
        cmd_list->synchronize(*ze_queue);
        // LOG_DEBUG("non-immediate command list synchronized");
        cmd_list->reset();
    }
}   

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
#include <algorithm>
#include <fstream>
#include <unordered_map>

#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "common/utils/utils.hpp"
#include "common/utils/version.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

namespace ccl {

namespace ze {

std::map<copy_engine_mode, std::string> copy_engine_names = {
    std::make_pair(copy_engine_mode::none, "none"),
    std::make_pair(copy_engine_mode::main, "main"),
    std::make_pair(copy_engine_mode::link, "link"),
    std::make_pair(copy_engine_mode::auto_mode, "auto")
};

std::map<h2d_copy_engine_mode, std::string> h2d_copy_engine_names = {
    std::make_pair(h2d_copy_engine_mode::none, "none"),
    std::make_pair(h2d_copy_engine_mode::main, "main"),
    std::make_pair(h2d_copy_engine_mode::auto_mode, "auto")
};

std::map<d2d_copy_engine_mode, std::string> d2d_copy_engine_names = {
    std::make_pair(d2d_copy_engine_mode::none, "none"),
    std::make_pair(d2d_copy_engine_mode::main, "main"),
};

std::string get_build_log_string(ze_module_build_log_handle_t build_log) {
    size_t log_size{};
    ZE_CALL(zeModuleBuildLogGetString, (build_log, &log_size, nullptr));

    if (!log_size) {
        LOG_DEBUG(log_size, "empty build log");
        return {};
    }

    std::string log(log_size, '\0');
    ZE_CALL(zeModuleBuildLogGetString, (build_log, &log_size, const_cast<char*>(log.data())));
    return log;
}

static void load_file(std::ifstream& file, std::vector<uint8_t>& bytes) {
    file.seekg(0, file.end);
    size_t file_size = file.tellg();
    file.seekg(0, file.beg);

    bytes.resize(file_size);
    file.read(reinterpret_cast<char*>(bytes.data()), file_size);
}

void load_module(const std::string& file_path,
                 ze_device_handle_t device,
                 ze_context_handle_t context,
                 ze_module_handle_t* module) {
    bool compiling_spirv_module = false;

    ze_module_build_log_handle_t build_log{};
    ze_module_format_t format{};
    std::vector<uint8_t> module_data{};

    // Prepare name for cached module in format:
    // /tmp/ccl-module-cache-{UID}-{CCL version hash}
    size_t version_hash = std::hash<std::string>{}(utils::get_library_version().full);

    std::stringstream ss;
    ss << "/tmp/ccl-module-cache-" << getuid() << "-" << std::hex << version_hash;
    const std::string cached_module_path = ss.str();

    std::ifstream cached_module_file(cached_module_path, std::ios_base::in | std::ios_base::binary);

    if (cached_module_file.good() && ccl::global_data::env().kernel_module_cache) {
        LOG_DEBUG("|MODULE CACHE| Using cached module at: ", cached_module_path);

        load_file(cached_module_file, module_data);
        cached_module_file.close();

        format = ZE_MODULE_FORMAT_NATIVE;
        compiling_spirv_module = false;
    }
    else {
        LOG_DEBUG("|MODULE CACHE| SPIR-V module loading started, file: ", file_path);
        CCL_THROW_IF_NOT(!file_path.empty(), "incorrect path to SPIR-V module.");

        std::ifstream spirv_module_file(file_path, std::ios_base::in | std::ios_base::binary);
        CCL_THROW_IF_NOT(spirv_module_file.good(),
                         "failed to load file containing oneCCL SPIR-V kernels, file: ",
                         file_path);

        load_file(spirv_module_file, module_data);
        spirv_module_file.close();

        format = ZE_MODULE_FORMAT_IL_SPIRV;
        compiling_spirv_module = true;
    }

    ze_module_desc_t desc{ default_module_desc };
    desc.format = format;
    desc.inputSize = module_data.size();
    desc.pInputModule = reinterpret_cast<const uint8_t*>(module_data.data());

    try {
        ZE_CALL(zeModuleCreate, (context, device, &desc, module, &build_log));
    }
    catch (std::string& error_message) {
        CCL_THROW("failed to create module: ",
                  file_path,
                  "error message: ",
                  error_message,
                  ", log: ",
                  get_build_log_string(build_log));
    }

    if (compiling_spirv_module && ccl::global_data::env().kernel_module_cache) {
        // We had to compile SPIR-V binary to gpu specific ISA, so now we can cache the module
        LOG_DEBUG("|MODULE CACHE| Caching compiled module to: ", cached_module_path);
        std::ofstream cached_module_file_new(cached_module_path,
                                             std::ios_base::out | std::ios_base::binary);

        size_t binary_size = 0;
        ZE_CALL(zeModuleGetNativeBinary, (*module, &binary_size, nullptr));

        std::vector<uint8_t> compiled_module_data(binary_size);
        ZE_CALL(zeModuleGetNativeBinary, (*module, &binary_size, compiled_module_data.data()));

        cached_module_file_new.write(reinterpret_cast<char*>(compiled_module_data.data()),
                                     binary_size);
    }

    ZE_CALL(zeModuleBuildLogDestroy, (build_log));
}

void create_kernel(ze_module_handle_t module, std::string kernel_name, ze_kernel_handle_t* kernel) {
    ze_kernel_desc_t desc = default_kernel_desc;
    // convert to lowercase
    std::transform(kernel_name.begin(), kernel_name.end(), kernel_name.begin(), ::tolower);
    desc.pKernelName = kernel_name.c_str();
    ze_result_t res = ZE_CALL(zeKernelCreate, (module, &desc, kernel));
    if (res != ZE_RESULT_SUCCESS) {
        CCL_THROW("error at zeKernelCreate: kernel name: ", kernel_name, " ret: ", to_string(res));
    }
}

void get_suggested_group_size(ze_kernel_handle_t kernel,
                              size_t elem_count,
                              ze_group_size_t* group_size) {
    group_size->groupSizeX = group_size->groupSizeY = group_size->groupSizeZ = 1;
    if (!elem_count) {
        return;
    }

    if (ccl::global_data::env().kernel_group_size == 0) {
        ZE_CALL(zeKernelSuggestGroupSize,
                (kernel,
                 elem_count,
                 1,
                 1,
                 &group_size->groupSizeX,
                 &group_size->groupSizeY,
                 &group_size->groupSizeZ));
    }
    else {
        group_size->groupSizeX = ccl::global_data::env().kernel_group_size;
    }

    CCL_THROW_IF_NOT(group_size->groupSizeX >= 1,
                     "wrong group size calculation: size: ",
                     to_string(*group_size),
                     ", elem_count: ",
                     elem_count);
}

void get_suggested_group_count(const ze_group_size_t& group_size,
                               size_t elem_count,
                               ze_group_count_t* group_count) {
    group_count->groupCountX = std::max((elem_count / group_size.groupSizeX), 1ul);
    group_count->groupCountY = 1;
    group_count->groupCountZ = 1;

    auto rem = elem_count % group_size.groupSizeX;

    //check whether any remaining elements are left and
    //add an additional group to account for that
    if (ccl::global_data::env().kernel_group_size != 0 && rem != 0) {
        group_count->groupCountX = group_count->groupCountX + 1;
        rem = 0;
    }
    CCL_THROW_IF_NOT(group_count->groupCountX >= 1 && rem == 0,
                     "wrong group calculation: size: ",
                     to_string(group_size),
                     ", count: ",
                     to_string(*group_count),
                     ", elem_count: ",
                     std::to_string(elem_count),
                     ", rem: ",
                     rem);
}

void set_kernel_args(ze_kernel_handle_t kernel, const std::vector<ze_kernel_arg_t>& kernel_args) {
    uint32_t idx = 0;
    for (const auto& arg : kernel_args) {
        if (arg.is_skip_arg()) {
            // skip argument - don't call zeKernelSetArgumentValue
            ++idx;
            continue;
        }
        for (const auto& elem : arg.elems) {
            auto ptr = elem.get();
            auto res = ZE_CALL(zeKernelSetArgumentValue, (kernel, idx, arg.size, ptr));
            if (res != ZE_RESULT_SUCCESS) {
                CCL_THROW("zeKernelSetArgumentValue failed with error ",
                          to_string(res),
                          " on idx ",
                          idx,
                          " with value ",
                          *((void**)ptr));
            }
            ++idx;
        }
    }
}

void get_queues_properties(ze_device_handle_t device, ze_queue_properties_t* props) {
    uint32_t queue_group_count = 0;
    ZE_CALL(zeDeviceGetCommandQueueGroupProperties, (device, &queue_group_count, nullptr));

    CCL_THROW_IF_NOT(queue_group_count != 0, "no queue groups found");

    props->resize(queue_group_count);
    ZE_CALL(zeDeviceGetCommandQueueGroupProperties, (device, &queue_group_count, props->data()));
}

std::string to_string(queue_group_type type) {
    switch (type) {
        case queue_group_type::compute: return "compute";
        case queue_group_type::main: return "main";
        case queue_group_type::link: return "link";
        default: return "unknown";
    }
}

queue_group_type get_queue_group_type(const ze_queue_properties_t& props, uint32_t ordinal) {
    CCL_THROW_IF_NOT(ordinal < props.size(),
                     "wrong queue group ordinal or properties size: { ordinal: ",
                     ordinal,
                     ", size: ",
                     props.size(),
                     " }");
    const auto& prop = props[ordinal];
    queue_group_type type = queue_group_type::unknown;

    if (prop.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
        type = queue_group_type::compute;
    }
    else if ((prop.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
             ((prop.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0)) {
        type = (prop.numQueues == 1) ? queue_group_type::main : queue_group_type::link;
    }

    return type;
}

uint32_t get_queue_group_ordinal(const ze_queue_properties_t& props, queue_group_type type) {
    for (uint32_t ordinal = 0; ordinal < props.size(); ordinal++) {
        if (get_queue_group_type(props, ordinal) == type) {
            return ordinal;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

bool get_buffer_context_and_device(const void* buf,
                                   ze_context_handle_t* context,
                                   ze_device_handle_t* device,
                                   ze_memory_allocation_properties_t* props) {
    CCL_THROW_IF_NOT(context, "no context");
    bool success{};
    const auto& contexts = global_data::get().ze_data->contexts;
    auto mem_alloc_props = default_alloc_props;
    for (auto ctx : contexts) {
        ze_device_handle_t dev{};
        ze_result_t res = ZE_CALL(zeMemGetAllocProperties, (ctx, buf, &mem_alloc_props, &dev));
        if (res == ZE_RESULT_SUCCESS) {
            *context = ctx;
            if (device) {
                *device = dev;
            }
            if (props) {
                *props = mem_alloc_props;
            }
            success = true;
            break;
        }
    }
    return success;
}

bool get_context_global_id(ze_context_handle_t context, ssize_t* id) {
    CCL_THROW_IF_NOT(context, "no context");
    CCL_THROW_IF_NOT(id, "no id");
    bool success{};
    const auto& contexts = global_data::get().ze_data->contexts;
    auto found = std::find(contexts.begin(), contexts.end(), context);
    if (found != contexts.end()) {
        *id = std::distance(contexts.begin(), found);
        success = true;
    }
    return success;
}

int get_fd_from_handle(const ze_ipc_mem_handle_t& handle) {
    return *(reinterpret_cast<const int*>(handle.data));
}

void close_handle_fd(const ze_ipc_mem_handle_t& handle) {
    int fd = get_fd_from_handle(handle);
    ccl::utils::close_fd(fd);
}

ze_ipc_mem_handle_t get_handle_from_fd(int fd) {
    ze_ipc_mem_handle_t handle{};
    *(reinterpret_cast<int*>(handle.data)) = fd;
    return handle;
}

bool get_device_global_id(ze_device_handle_t device, ssize_t* id) {
    CCL_THROW_IF_NOT(device, "no device");
    CCL_THROW_IF_NOT(id, "no id");
    bool success{};
    const auto& devices = global_data::get().ze_data->devices;
    auto found =
        std::find_if(devices.begin(), devices.end(), [&device](const device_info& info) -> bool {
            return info.device == device;
        });
    if (found != devices.end()) {
        *id = std::distance(devices.begin(), found);
        success = true;
    }
    return success;
}

uint32_t get_parent_device_id(ze_device_handle_t device) {
    ssize_t dev_id = ccl::utils::invalid_device_id;
    ccl::ze::get_device_global_id(device, &dev_id);
    CCL_THROW_IF_NOT(dev_id != ccl::utils::invalid_device_id, "unexpected dev_id");
    LOG_DEBUG("device_id: ", dev_id);
    return ccl::global_data::get().ze_data->devices[dev_id].parent_idx;
}

uint32_t get_physical_device_id(ze_device_handle_t device) {
    ssize_t dev_id = ccl::utils::invalid_device_id;
    ccl::ze::get_device_global_id(device, &dev_id);
    auto parent_idx = get_parent_device_id(device);
    auto idx = ccl::global_data::get().ze_data->devices[parent_idx].physical_idx;
    LOG_DEBUG("physical_idx ", idx, ", dev_id: ", dev_id, ", parent_idx: ", parent_idx);
    return idx;
}

uint32_t get_device_id(ze_device_handle_t device) {
    // here we can control which device idx we can take
    // parent_idx which is based on logical idx
    // or physical idx, which is based on BDF
    auto dev_id = get_physical_device_id(device);
    if (!ccl::global_data::env().ze_drm_bdf_support ||
        (int)dev_id == fd_manager::invalid_physical_idx) {
        dev_id = get_parent_device_id(device);
    }
    return dev_id;
}

device_family get_device_family(ze_device_handle_t device) {
    ze_device_properties_t dev_props = ccl::ze::default_device_props;
    ZE_CALL(zeDeviceGetProperties, (device, &dev_props));
    uint32_t id = dev_props.deviceId & 0xfff0;
    using enum_t = typename std::underlying_type<device_family>::type;

    switch (id) {
        case static_cast<enum_t>(device_id::id1): return device_family::family1;
        case static_cast<enum_t>(device_id::id2): return device_family::family2;
        case static_cast<enum_t>(device_id::id3): return device_family::family3;
        case static_cast<enum_t>(device_id::id4): return device_family::family4;
        case static_cast<enum_t>(device_id::id5): return device_family::family5;
        case static_cast<enum_t>(device_id::id6): return device_family::family6;
        case static_cast<enum_t>(device_id::id7): return device_family::family7;
        default: {
            LOG_DEBUG("get_device_family returns unknown device_family: ", id);
            return device_family::unknown;
        }
    }
}

#ifdef ZE_PCI_PROPERTIES_EXT_NAME
bool is_same_pci_addr(const ze_pci_address_ext_t& addr1, const ze_pci_address_ext_t& addr2) {
    bool result = true;
    if (!(addr1.domain == addr2.domain && addr1.bus == addr2.bus && addr1.device == addr2.device &&
          addr1.function == addr2.function)) {
        result = false;
        //LOG_DEBUG("pci addresses are not the same:"
        //          " addr1: ",
        //          ccl::ze::to_string(addr1),
        //          " addr2: ",
        //          ccl::ze::to_string(addr2));
    }
    return result;
}
#endif // ZE_PCI_PROPERTIES_EXT_NAME

bool is_same_dev_uuid(const ze_device_uuid_t& uuid1, const ze_device_uuid_t& uuid2) {
    bool result = true;
    std::string state = "device uuids";
    if (std::memcmp(&uuid1, &uuid2, sizeof(ze_device_uuid_t))) {
        result = false;
        state += " are not the same:";
    }
    else {
        state += " are the same:";
    }
    // LOG_DEBUG(state, " uuid1: ", ccl::ze::to_string(uuid1), ", uuid2: ", ccl::ze::to_string(uuid2));
    return result;
}

bool is_same_fabric_port(const zes_fabric_port_id_t& port1, const zes_fabric_port_id_t& port2) {
    bool result = true;
    if (!(port1.fabricId == port2.fabricId && port1.attachId == port2.attachId &&
          port1.portNumber == port2.portNumber)) {
        result = false;
        // LOG_DEBUG("fabric ports are not the same:"
        //           " port1: ",
        //           ccl::ze::to_string(port1),
        //           " port2: ",
        //           ccl::ze::to_string(port2));
    }
    return result;
}

#ifdef ZE_PCI_PROPERTIES_EXT_NAME
bool pci_address_comparator::operator()(const ze_pci_address_ext_t& a,
                                        const ze_pci_address_ext_t& b) const {
    if (a.domain == b.domain) {
        if (a.bus == b.bus) {
            if (a.device == b.device) {
                if (a.function == b.function) {
                    return false;
                }
                else {
                    return (a.function < b.function);
                }
            }
            else {
                return (a.device < b.device);
            }
        }
        else {
            return (a.bus < b.bus);
        }
    }
    else {
        return (a.domain < b.domain);
    }
}
#endif // ZE_PCI_PROPERTIES_EXT_NAME

bool fabric_port_comparator::operator()(const zes_fabric_port_id_t& a,
                                        const zes_fabric_port_id_t& b) const {
    if (a.fabricId == b.fabricId) {
        if (a.attachId == b.attachId) {
            if (a.portNumber == b.portNumber) {
                return false;
            }
            else {
                return (a.portNumber < b.portNumber);
            }
        }
        else {
            return (a.attachId < b.attachId);
        }
    }
    else {
        return (a.fabricId < b.fabricId);
    }
}

std::string to_string(ze_event_scope_flag_t scope_flag) {
    switch (scope_flag) {
        case ZE_EVENT_SCOPE_FLAG_SUBDEVICE: return "ZE_EVENT_SCOPE_FLAG_SUBDEVICE";
        case ZE_EVENT_SCOPE_FLAG_DEVICE: return "ZE_EVENT_SCOPE_FLAG_DEVICE";
        case ZE_EVENT_SCOPE_FLAG_HOST: return "ZE_EVENT_SCOPE_FLAG_HOST";
        default:
            return "unknown ze_event_scope_flag_t value: " +
                   std::to_string(static_cast<uint32_t>(scope_flag));
    }
}

std::string to_string(ze_event_scope_flags_t _scope_flags) {
    auto scope_flags = _scope_flags;
    std::string out;
    while (scope_flags) {
        if (out.size())
            out += "|";
        if (scope_flags & ZE_EVENT_SCOPE_FLAG_SUBDEVICE) {
            out += to_string(ZE_EVENT_SCOPE_FLAG_SUBDEVICE);
            scope_flags &= ~ZE_EVENT_SCOPE_FLAG_SUBDEVICE;
        }
        else if (scope_flags & ZE_EVENT_SCOPE_FLAG_DEVICE) {
            out += to_string(ZE_EVENT_SCOPE_FLAG_DEVICE);
            scope_flags &= ~ZE_EVENT_SCOPE_FLAG_DEVICE;
        }
        else if (scope_flags & ZE_EVENT_SCOPE_FLAG_HOST) {
            out += to_string(ZE_EVENT_SCOPE_FLAG_HOST);
            scope_flags &= ~ZE_EVENT_SCOPE_FLAG_HOST;
        }
        else {
            return "unknown ze_event_scope_flag_t value: " +
                   std::to_string(static_cast<uint32_t>(_scope_flags));
        }
    }
    return out;
}

std::string to_string(ze_result_t result) {
    switch (result) {
        case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
        case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
        case ZE_RESULT_ERROR_MODULE_LINK_FAILURE: return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
            return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
        case ZE_RESULT_ERROR_NOT_AVAILABLE: return "ZE_RESULT_ERROR_NOT_AVAILABLE";
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
            return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION: return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE: return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_INVALID_SIZE: return "ZE_RESULT_ERROR_INVALID_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
        case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
            return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
        case ZE_RESULT_ERROR_INVALID_ENUMERATION: return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
            return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
            return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
        case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY: return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME: return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
        case ZE_RESULT_ERROR_INVALID_KERNEL_NAME: return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
        case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME: return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
        case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
            return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
            return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
            return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
            return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
            return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
        case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
            return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
        case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
            return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
        case ZE_RESULT_ERROR_OVERLAPPING_REGIONS: return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
        case ZE_RESULT_ERROR_UNKNOWN: return "ZE_RESULT_ERROR_UNKNOWN";
        case ZE_RESULT_FORCE_UINT32: return "ZE_RESULT_FORCE_UINT32";
        default: return "unknown ze_result_t value: " + std::to_string(static_cast<int>(result));
    }
}

std::string to_string(const ze_group_size_t& group_size) {
    std::stringstream ss;
    ss << "{ x: " << group_size.groupSizeX << ", y: " << group_size.groupSizeY
       << ", z: " << group_size.groupSizeZ << " }";
    return ss.str();
}

std::string to_string(const ze_group_count_t& group_count) {
    std::stringstream ss;
    ss << "{ x: " << group_count.groupCountX << ", y: " << group_count.groupCountY
       << ", z: " << group_count.groupCountZ << " }";
    return ss.str();
}

std::string to_string(const ze_kernel_args_t& kernel_args) {
    std::stringstream ss;
    ss << "{\n";
    size_t idx = 0;
    for (const auto& arg : kernel_args) {
        // TODO: can we distinguish argument types in order to properly print them instead of printing
        // as a void* ptr?
        for (const auto& elem : arg.elems) {
            ss << "  idx: " << idx << ", ptr: " << elem.get() << "\n";
            ++idx;
        }
    }
    ss << "}";
    return ss.str();
}

std::string to_string(ze_device_property_flag_t flag) {
    switch (flag) {
        case ZE_DEVICE_PROPERTY_FLAG_INTEGRATED: return "ZE_DEVICE_PROPERTY_FLAG_INTEGRATED";
        case ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE: return "ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE";
        case ZE_DEVICE_PROPERTY_FLAG_ECC: return "ZE_DEVICE_PROPERTY_FLAG_ECC";
        case ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING:
            return "ZE_DEVICE_PROPERTY_FLAG_ONDEMANDPAGING";
        case ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32: return "ZE_DEVICE_PROPERTY_FLAG_FORCE_UINT32";
        default:
            return "unknown ze_device_property_flag_t value: " +
                   std::to_string(static_cast<int>(flag));
    }
}

std::string to_string(ze_command_queue_group_property_flag_t flag) {
    switch (flag) {
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE";
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY";
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS";
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS";
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32";
        default:
            return "unknown ze_command_queue_group_property_flag_t value: " +
                   std::to_string(static_cast<int>(flag));
    }
}

std::string to_string(const ze_command_queue_group_properties_t& props) {
    std::stringstream ss;
    ss << "flags: " << flags_to_string<ze_command_queue_group_property_flag_t>(props.flags)
       << ", max_memory_fill_pattern_size: " << props.maxMemoryFillPatternSize
       << ", num_queues: " << props.numQueues;
    return ss.str();
}

#ifdef ZE_PCI_PROPERTIES_EXT_NAME
std::string to_string(const ze_pci_address_ext_t& addr) {
    std::stringstream ss;
    ss << "{ " << addr.domain << ", " << addr.bus << ", " << addr.device << ", " << addr.function
       << " }";
    return ss.str();
}
#endif // ZE_PCI_PROPERTIES_EXT_NAME

std::string to_string(const ze_device_uuid_t& uuid) {
    std::string str{};
    std::string tmp{};
    for (auto& entry : uuid.id) {
        tmp += std::to_string(entry);
        tmp += ", ";
    }
    str += "{ " + tmp.substr(0, tmp.size() - 2) + " }";
    return str;
}

std::string to_string(const zes_fabric_port_id_t& port) {
    std::stringstream ss;
    ss << "{ " << port.fabricId << " " << port.attachId << " " << (int)port.portNumber << " }";
    return ss.str();
}

std::string to_string(zes_fabric_port_status_t status) {
    switch (status) {
        case ZES_FABRIC_PORT_STATUS_UNKNOWN: return "unknown";
        case ZES_FABRIC_PORT_STATUS_HEALTHY: return "healthy";
        case ZES_FABRIC_PORT_STATUS_DEGRADED: return "degraded";
        case ZES_FABRIC_PORT_STATUS_FAILED: return "failed";
        case ZES_FABRIC_PORT_STATUS_DISABLED: return "disabled";
        default: return "unexpected";
    }
}

std::string to_string(zes_fabric_port_qual_issue_flag_t flag) {
    switch (flag) {
        case ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_LINK_ERRORS: return "link errors";
        case ZES_FABRIC_PORT_QUAL_ISSUE_FLAG_SPEED: return "speed";
        default: return "unexpected";
    }
}

std::string to_string(zes_fabric_port_failure_flag_t flag) {
    switch (flag) {
        case ZES_FABRIC_PORT_FAILURE_FLAG_FAILED: return "failed";
        case ZES_FABRIC_PORT_FAILURE_FLAG_TRAINING_TIMEOUT: return "training timeout";
        case ZES_FABRIC_PORT_FAILURE_FLAG_FLAPPING: return "flapping";
        default: return "unexpected";
    }
}

std::string to_string(const zes_fabric_port_state_t& state) {
    std::stringstream ss;

    ss << "{ status: " << to_string(state.status);

    if (state.status == ZES_FABRIC_PORT_STATUS_DEGRADED) {
        ss << ", details: "
           << flags_to_string<zes_fabric_port_qual_issue_flag_t>(state.qualityIssues);
    }

    if (state.status == ZES_FABRIC_PORT_STATUS_FAILED) {
        ss << ", details: "
           << flags_to_string<zes_fabric_port_failure_flag_t>(state.failureReasons);
    }
    else {
        ss << ", remote_port_id: " << to_string(state.remotePortId);
    }

    ss << " }";

    return ss.str();
}

bool is_arc_card(device_family family) {
    return family == ccl::device_family::family4 || family == ccl::device_family::family5 ||
           family == ccl::device_family::family6 || family == ccl::device_family::family7 ||
           ccl::global_data::env().sycl_force_pcie;
}

bool should_disable_rdma(const ze_device_handle_t& device) {
    static bool checked = false;
    static bool disable_rdma = false;

    // If we've already made a determination, return the cached result
    if (checked) {
        return disable_rdma;
    }

    ze_device_properties_t dev_props = ccl::ze::default_device_props;
    ZE_CALL(zeDeviceGetProperties, (device, &dev_props));
    LOG_DEBUG("Device ID: 0x", std::hex, dev_props.deviceId, std::dec);

    switch (dev_props.deviceId) {
        // Disable RDMA for some B-series cards
        case 0xE20B: // DT6
        case 0xE20C: // DT7
        case 0xE20D: // DT8
        case 0xE210: // DT6+
        case 0xE212: // WKS-B93
        case 0xE220: // DT1+
        case 0xE221: // DT2
        case 0xE223: // DT1+ WKS
            LOG_DEBUG("Disabling RDMA for device ID: 0x", std::hex, dev_props.deviceId, std::dec);
            disable_rdma = true;
            break;

        // Don't disable RDMA for these devices
        case 0xE211: // DT6-
        case 0xE216: // NEX2
        case 0xE222: // DT3-
            LOG_DEBUG(
                "Not disabling RDMA for device ID: 0x", std::hex, dev_props.deviceId, std::dec);
            disable_rdma = false;
            break;

        default:
            // For unknown device IDs, do not disable RDMA
            LOG_DEBUG("Unknown device ID: 0x",
                      std::hex,
                      dev_props.deviceId,
                      std::dec,
                      " - do not disable RDMA");
            disable_rdma = false;
            break;
    }

    checked = true;
    return disable_rdma;
}

} // namespace ze
} // namespace ccl

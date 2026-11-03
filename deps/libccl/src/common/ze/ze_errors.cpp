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

#include <unordered_map>
#include "include/ze_errors.h"

// Map for converting ze_result_t to string
static const std::unordered_map<ze_result_t, const char*> kL0Error = {
    {ZE_RESULT_SUCCESS, "ZE_RESULT_SUCCESS: [Core] success"},
    {ZE_RESULT_NOT_READY, "ZE_RESULT_NOT_READY: [Core] synchronization primitive not signaled"},
    {ZE_RESULT_ERROR_DEVICE_LOST,
     "ZE_RESULT_ERROR_DEVICE_LOST: [Core] device hung, reset, was removed, or "
     "driver update occurred"},
    {ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY,
     "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: [Core] insufficient host memory to "
     "satisfy call"},
    {ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY,
     "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: [Core] insufficient device memory "
     "to satisfy call"},
    {ZE_RESULT_ERROR_MODULE_BUILD_FAILURE,
     "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: [Core] error occurred when "
     "building module, see build log for details"},
    {ZE_RESULT_ERROR_MODULE_LINK_FAILURE,
     "ZE_RESULT_ERROR_MODULE_LINK_FAILURE: [Core] error occurred when linking "
     "modules, see build log for details"},
    {ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET, "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET: [Core] device requires a reset"},
    {ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE,
     "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE: [Core] device currently in "
     "low power state"},
    {ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX,
     "ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX: [Core, Experimental] device is "
     "not represented by a fabric vertex"},
    {ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE,
     "ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE: [Core, Experimental] fabric "
     "vertex does not represent a device"},
    {ZE_RESULT_EXP_ERROR_REMOTE_DEVICE,
     "ZE_RESULT_EXP_ERROR_REMOTE_DEVICE: [Core, Experimental] fabric vertex "
     "represents a remote device or subdevice"},
    {ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE,
     "ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE: [Core, Experimental] operands "
     "of comparison are not compatible"},
    {ZE_RESULT_EXP_RTAS_BUILD_RETRY,
     "ZE_RESULT_EXP_RTAS_BUILD_RETRY: [Core, Experimental] ray tracing "
     "acceleration structure build operation failed due to insufficient "
     "resources, retry with a larger acceleration structure buffer allocation"},
    {ZE_RESULT_EXP_RTAS_BUILD_DEFERRED,
     "ZE_RESULT_EXP_RTAS_BUILD_DEFERRED: [Core, Experimental] ray tracing "
     "acceleration structure build operation deferred to parallel operation "
     "join"},
    {ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS,
     "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS: [Sysman] access denied due to "
     "permission level"},
    {ZE_RESULT_ERROR_NOT_AVAILABLE,
     "ZE_RESULT_ERROR_NOT_AVAILABLE: [Sysman] resource already in use and "
     "simultaneous access not allowed or resource was removed"},
    {ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE,
     "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE: [Common] external required "
     "dependency is unavailable or missing"},
    {ZE_RESULT_WARNING_DROPPED_DATA, "ZE_RESULT_WARNING_DROPPED_DATA: [Tools] data may have been dropped"},
    {ZE_RESULT_ERROR_UNINITIALIZED, "ZE_RESULT_ERROR_UNINITIALIZED: [Validation] driver is not initialized"},
    {ZE_RESULT_ERROR_UNSUPPORTED_VERSION,
     "ZE_RESULT_ERROR_UNSUPPORTED_VERSION: [Validation] generic error code for "
     "unsupported versions"},
    {ZE_RESULT_ERROR_UNSUPPORTED_FEATURE,
     "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: [Validation] generic error code for "
     "unsupported features"},
    {ZE_RESULT_ERROR_INVALID_ARGUMENT,
     "ZE_RESULT_ERROR_INVALID_ARGUMENT: [Validation] generic error code for "
     "invalid arguments"},
    {ZE_RESULT_ERROR_INVALID_NULL_HANDLE,
     "ZE_RESULT_ERROR_INVALID_NULL_HANDLE: [Validation] handle argument is not "
     "valid"},
    {ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE,
     "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE: [Validation] object pointed to by "
     "handle still in-use by device"},
    {ZE_RESULT_ERROR_INVALID_NULL_POINTER,
     "ZE_RESULT_ERROR_INVALID_NULL_POINTER: [Validation] pointer argument may "
     "not be nullptr"},
    {ZE_RESULT_ERROR_INVALID_SIZE,
     "ZE_RESULT_ERROR_INVALID_SIZE: [Validation] size argument is invalid "
     "(e.g., must not be zero)"},
    {ZE_RESULT_ERROR_UNSUPPORTED_SIZE,
     "ZE_RESULT_ERROR_UNSUPPORTED_SIZE: [Validation] size argument is not "
     "supported by the device (e.g., too large)"},
    {ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT,
     "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: [Validation] alignment argument "
     "is not supported by the device (e.g., too small)"},
    {ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT,
     "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT: [Validation] "
     "synchronization object in invalid state"},
    {ZE_RESULT_ERROR_INVALID_ENUMERATION,
     "ZE_RESULT_ERROR_INVALID_ENUMERATION: [Validation] enumerator argument is "
     "not valid"},
    {ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION,
     "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION: [Validation] enumerator "
     "argument is not supported by the device"},
    {ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT,
     "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT: [Validation] image format is "
     "not supported by the device"},
    {ZE_RESULT_ERROR_INVALID_NATIVE_BINARY,
     "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY: [Validation] native binary is not "
     "supported by the device"},
    {ZE_RESULT_ERROR_INVALID_GLOBAL_NAME,
     "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME: [Validation] global variable is not "
     "found in the module"},
    {ZE_RESULT_ERROR_INVALID_KERNEL_NAME,
     "ZE_RESULT_ERROR_INVALID_KERNEL_NAME: [Validation] kernel name is not "
     "found in the module"},
    {ZE_RESULT_ERROR_INVALID_FUNCTION_NAME,
     "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME: [Validation] function name is not "
     "found in the module"},
    {ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION,
     "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION: [Validation] group size "
     "dimension is not valid for the kernel or device"},
    {ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION,
     "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION: [Validation] global "
     "width dimension is not valid for the kernel or device"},
    {ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,
     "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX: [Validation] kernel "
     "argument index is not valid for kernel"},
    {ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE,
     "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE: [Validation] kernel "
     "argument size does not match kernel"},
    {ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE,
     "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE: [Validation] value of "
     "kernel attribute is not valid for the kernel or device"},
    {ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED,
     "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED: [Validation] module with "
     "imports needs to be linked before kernels can be created from it"},
    {ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE,
     "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE: [Validation] command list "
     "type does not match command queue type"},
    {ZE_RESULT_ERROR_OVERLAPPING_REGIONS,
     "ZE_RESULT_ERROR_OVERLAPPING_REGIONS: [Validation] copy operations do not "
     "support overlapping regions of memory"},
    {ZE_RESULT_WARNING_ACTION_REQUIRED,
     "ZE_RESULT_WARNING_ACTION_REQUIRED: [Sysman] an action is required to "
     "complete the desired operation"},
    {ZE_RESULT_ERROR_INVALID_KERNEL_HANDLE,
     "ZE_RESULT_ERROR_INVALID_KERNEL_HANDLE: [Core, Validation] kernel handle "
     "is invalid for the operation"},
    {ZE_RESULT_ERROR_UNKNOWN, "ZE_RESULT_ERROR_UNKNOWN: [Core] unknown or internal error"}};

// Helper function to get string representation of ze_result_t
const char* onecclZeResultToString(ze_result_t result)
{
    auto itr = kL0Error.find(result);
    return (itr != kL0Error.end()) ? itr->second : "UNKNOWN_ZE_RESULT";
}

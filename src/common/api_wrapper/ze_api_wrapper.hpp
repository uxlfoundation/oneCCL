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

#include "oneapi/ccl/config.h"

#include <string>
#include <vector>

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

#include <ze_api.h>
#include <zes_api.h>
#include "layers/zel_tracing_api.h"
#include "layers/zel_tracing_register_cb.h"

namespace ccl {

typedef struct ze_lib_ops {
    decltype(zeInit) *zeInit;
    decltype(zesInit) *zesInit;
    decltype(zeDriverGet) *zeDriverGet;
    decltype(zeDriverGetApiVersion) *zeDriverGetApiVersion;
    decltype(zeMemGetAllocProperties) *zeMemGetAllocProperties;
    decltype(zeMemGetAddressRange) *zeMemGetAddressRange;
    decltype(zeMemAllocHost) *zeMemAllocHost;
    decltype(zeMemAllocDevice) *zeMemAllocDevice;
    decltype(zeMemAllocShared) *zeMemAllocShared;
    decltype(zeMemFree) *zeMemFree;
    decltype(zeMemOpenIpcHandle) *zeMemOpenIpcHandle;
    decltype(zeMemCloseIpcHandle) *zeMemCloseIpcHandle;
    decltype(zeMemGetIpcHandle) *zeMemGetIpcHandle;
    decltype(zeMemPutIpcHandle) *zeMemPutIpcHandle;
    decltype(zeDeviceGet) *zeDeviceGet;
    decltype(zeDeviceGetProperties) *zeDeviceGetProperties;
    decltype(zeDeviceCanAccessPeer) *zeDeviceCanAccessPeer;
    decltype(zeDeviceGetCommandQueueGroupProperties) *zeDeviceGetCommandQueueGroupProperties;
    decltype(zeDeviceGetP2PProperties) *zeDeviceGetP2PProperties;
    decltype(zeDeviceGetGlobalTimestamps) *zeDeviceGetGlobalTimestamps;
    decltype(zeDriverGetProperties) *zeDriverGetProperties;
    decltype(zeDriverGetIpcProperties) *zeDriverGetIpcProperties;
    decltype(zeCommandQueueCreate) *zeCommandQueueCreate;
    decltype(zeCommandQueueExecuteCommandLists) *zeCommandQueueExecuteCommandLists;
    decltype(zeCommandQueueSynchronize) *zeCommandQueueSynchronize;
    decltype(zeCommandQueueDestroy) *zeCommandQueueDestroy;
    decltype(zeCommandListCreate) *zeCommandListCreate;
    decltype(zeCommandListCreateImmediate) *zeCommandListCreateImmediate;
    decltype(zeCommandListAppendMemoryCopy) *zeCommandListAppendMemoryCopy;
    decltype(zeCommandListAppendLaunchKernel) *zeCommandListAppendLaunchKernel;
    decltype(zeCommandListAppendWaitOnEvents) *zeCommandListAppendWaitOnEvents;
    decltype(zeCommandListAppendSignalEvent) *zeCommandListAppendSignalEvent;
    decltype(zeCommandListAppendBarrier) *zeCommandListAppendBarrier;
    decltype(zeCommandListAppendMemoryRangesBarrier) *zeCommandListAppendMemoryRangesBarrier;
    decltype(zeCommandListAppendWriteGlobalTimestamp) *zeCommandListAppendWriteGlobalTimestamp;
    decltype(zeCommandListClose) *zeCommandListClose;
    decltype(zeCommandListReset) *zeCommandListReset;
    decltype(zeCommandListDestroy) *zeCommandListDestroy;
    decltype(zeContextCreate) *zeContextCreate;
    decltype(zeContextSystemBarrier) *zeContextSystemBarrier;
    decltype(zeContextDestroy) *zeContextDestroy;
    decltype(zeEventPoolCreate) *zeEventPoolCreate;
    decltype(zeEventCreate) *zeEventCreate;
    decltype(zeEventQueryStatus) *zeEventQueryStatus;
    decltype(zeEventHostSynchronize) *zeEventHostSynchronize;
    decltype(zeEventHostReset) *zeEventHostReset;
    decltype(zeEventHostSignal) *zeEventHostSignal;
    decltype(zeEventDestroy) *zeEventDestroy;
    decltype(zeEventPoolOpenIpcHandle) *zeEventPoolOpenIpcHandle;
    decltype(zeEventPoolCloseIpcHandle) *zeEventPoolCloseIpcHandle;
    decltype(zeEventPoolGetIpcHandle) *zeEventPoolGetIpcHandle;
    decltype(zeEventQueryKernelTimestamp) *zeEventQueryKernelTimestamp;
    decltype(zeEventPoolDestroy) *zeEventPoolDestroy;
    decltype(zeFenceHostSynchronize) *zeFenceHostSynchronize;
    decltype(zeFenceCreate) *zeFenceCreate;
    decltype(zeKernelCreate) *zeKernelCreate;
    decltype(zeKernelSetArgumentValue) *zeKernelSetArgumentValue;
    decltype(zeKernelSuggestGroupSize) *zeKernelSuggestGroupSize;
    decltype(zeKernelSetGroupSize) *zeKernelSetGroupSize;
    decltype(zeKernelDestroy) *zeKernelDestroy;
    decltype(zeModuleCreate) *zeModuleCreate;
    decltype(zeModuleDestroy) *zeModuleDestroy;
    decltype(zeModuleBuildLogGetString) *zeModuleBuildLogGetString;
    decltype(zeModuleBuildLogDestroy) *zeModuleBuildLogDestroy;
    decltype(zeModuleGetNativeBinary) *zeModuleGetNativeBinary;
    decltype(zeDeviceGetComputeProperties) *zeDeviceGetComputeProperties;
    decltype(zeDeviceGetMemoryAccessProperties) *zeDeviceGetMemoryAccessProperties;
    decltype(zeDeviceGetMemoryProperties) *zeDeviceGetMemoryProperties;
    decltype(zeDeviceGetSubDevices) *zeDeviceGetSubDevices;
    decltype(zesDevicePciGetProperties) *zesDevicePciGetProperties;
    decltype(zesDeviceEnumFabricPorts) *zesDeviceEnumFabricPorts;
    decltype(zesFabricPortGetConfig) *zesFabricPortGetConfig;
    decltype(zesFabricPortGetProperties) *zesFabricPortGetProperties;
    decltype(zesFabricPortGetState) *zesFabricPortGetState;
    decltype(zesDriverGet) *zesDriverGet;
    decltype(zesDeviceGet) *zesDeviceGet;
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
    decltype(zeDevicePciGetPropertiesExt) *zeDevicePciGetPropertiesExt;
#endif // ZE_PCI_PROPERTIES_EXT_NAME
    decltype(zesDriverGetDeviceByUuidExp) *zesDriverGetDeviceByUuidExp;
    decltype(zeDriverGetExtensionFunctionAddress) *zeDriverGetExtensionFunctionAddress;
    decltype(zeFabricVertexGetExp) *zeFabricVertexGetExp;
    decltype(zeFabricVertexGetSubVerticesExp) *zeFabricVertexGetSubVerticesExp;
    decltype(zeFabricEdgeGetExp) *zeFabricEdgeGetExp;
    decltype(zeFabricEdgeGetPropertiesExp) *zeFabricEdgeGetPropertiesExp;
    decltype(zeDeviceGetFabricVertexExp) *zeDeviceGetFabricVertexExp;
    decltype(zelTracerCreate) *zelTracerCreate;
    decltype(zelTracerMemFreeRegisterCallback) *zelTracerMemFreeRegisterCallback;
    decltype(zelTracerSetEnabled) *zelTracerSetEnabled;
} ze_lib_ops_t;

static std::vector<std::string> ze_fn_names = {
    "zeInit",
    "zesInit",
    "zeDriverGet",
    "zeDriverGetApiVersion",
    "zeMemGetAllocProperties",
    "zeMemGetAddressRange",
    "zeMemAllocHost",
    "zeMemAllocDevice",
    "zeMemAllocShared",
    "zeMemFree",
    "zeMemOpenIpcHandle",
    "zeMemCloseIpcHandle",
    "zeMemGetIpcHandle",
    "zeMemPutIpcHandle",
    "zeDeviceGet",
    "zeDeviceGetProperties",
    "zeDeviceCanAccessPeer",
    "zeDeviceGetCommandQueueGroupProperties",
    "zeDeviceGetP2PProperties",
    "zeDeviceGetGlobalTimestamps",
    "zeDriverGetProperties",
    "zeDriverGetIpcProperties",
    "zeCommandQueueCreate",
    "zeCommandQueueExecuteCommandLists",
    "zeCommandQueueSynchronize",
    "zeCommandQueueDestroy",
    "zeCommandListCreate",
    "zeCommandListCreateImmediate",
    "zeCommandListAppendMemoryCopy",
    "zeCommandListAppendLaunchKernel",
    "zeCommandListAppendWaitOnEvents",
    "zeCommandListAppendSignalEvent",
    "zeCommandListAppendBarrier",
    "zeCommandListAppendMemoryRangesBarrier",
    "zeCommandListAppendWriteGlobalTimestamp",
    "zeCommandListClose",
    "zeCommandListReset",
    "zeCommandListDestroy",
    "zeContextCreate",
    "zeContextSystemBarrier",
    "zeContextDestroy",
    "zeEventPoolCreate",
    "zeEventCreate",
    "zeEventQueryStatus",
    "zeEventHostSynchronize",
    "zeEventHostReset",
    "zeEventHostSignal",
    "zeEventDestroy",
    "zeEventPoolOpenIpcHandle",
    "zeEventPoolCloseIpcHandle",
    "zeEventPoolGetIpcHandle",
    "zeEventQueryKernelTimestamp",
    "zeEventPoolDestroy",
    "zeFenceHostSynchronize",
    "zeFenceCreate",
    "zeKernelCreate",
    "zeKernelSetArgumentValue",
    "zeKernelSuggestGroupSize",
    "zeKernelSetGroupSize",
    "zeKernelDestroy",
    "zeModuleCreate",
    "zeModuleDestroy",
    "zeModuleBuildLogGetString",
    "zeModuleBuildLogDestroy",
    "zeModuleGetNativeBinary",
    "zeDeviceGetComputeProperties",
    "zeDeviceGetMemoryAccessProperties",
    "zeDeviceGetMemoryProperties",
    "zeDeviceGetSubDevices",
    "zesDevicePciGetProperties",
    "zesDeviceEnumFabricPorts",
    "zesFabricPortGetConfig",
    "zesFabricPortGetProperties",
    "zesFabricPortGetState",
    "zesDriverGet",
    "zesDeviceGet",
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
    "zeDevicePciGetPropertiesExt",
#endif // ZE_PCI_PROPERTIES_EXT_NAME
    "zesDriverGetDeviceByUuidExp",
    "zeDriverGetExtensionFunctionAddress",
    "zeFabricVertexGetExp",
    "zeFabricVertexGetSubVerticesExp",
    "zeFabricEdgeGetExp",
    "zeFabricEdgeGetPropertiesExp",
    "zeDeviceGetFabricVertexExp",
    "zelTracerCreate",
    "zelTracerMemFreeRegisterCallback",
    "zelTracerSetEnabled",
};

extern ccl::ze_lib_ops_t ze_lib_ops;

#define zeInit                  ccl::ze_lib_ops.zeInit
#define zesInit                 ccl::ze_lib_ops.zesInit
#define zeDriverGet             ccl::ze_lib_ops.zeDriverGet
#define zeDriverGetApiVersion   ccl::ze_lib_ops.zeDriverGetApiVersion
#define zeMemGetAllocProperties ccl::ze_lib_ops.zeMemGetAllocProperties
#define zeMemGetAddressRange    ccl::ze_lib_ops.zeMemGetAddressRange
#define zeMemAllocHost          ccl::ze_lib_ops.zeMemAllocHost
#define zeMemAllocDevice        ccl::ze_lib_ops.zeMemAllocDevice
#define zeMemAllocShared        ccl::ze_lib_ops.zeMemAllocShared
#define zeMemFree               ccl::ze_lib_ops.zeMemFree
#define zeMemOpenIpcHandle      ccl::ze_lib_ops.zeMemOpenIpcHandle
#define zeMemCloseIpcHandle     ccl::ze_lib_ops.zeMemCloseIpcHandle
#define zeMemGetIpcHandle       ccl::ze_lib_ops.zeMemGetIpcHandle
#define zeMemPutIpcHandle       ccl::ze_lib_ops.zeMemPutIpcHandle
#define zeDeviceGet             ccl::ze_lib_ops.zeDeviceGet
#define zeDeviceGetProperties   ccl::ze_lib_ops.zeDeviceGetProperties
#define zeDeviceCanAccessPeer   ccl::ze_lib_ops.zeDeviceCanAccessPeer
#define zeDeviceGetCommandQueueGroupProperties \
    ccl::ze_lib_ops.zeDeviceGetCommandQueueGroupProperties
#define zeDeviceGetP2PProperties          ccl::ze_lib_ops.zeDeviceGetP2PProperties
#define zeDeviceGetGlobalTimestamps       ccl::ze_lib_ops.zeDeviceGetGlobalTimestamps
#define zeDriverGetProperties             ccl::ze_lib_ops.zeDriverGetProperties
#define zeDriverGetIpcProperties          ccl::ze_lib_ops.zeDriverGetIpcProperties
#define zeCommandQueueCreate              ccl::ze_lib_ops.zeCommandQueueCreate
#define zeCommandQueueExecuteCommandLists ccl::ze_lib_ops.zeCommandQueueExecuteCommandLists
#define zeCommandQueueSynchronize         ccl::ze_lib_ops.zeCommandQueueSynchronize
#define zeCommandQueueDestroy             ccl::ze_lib_ops.zeCommandQueueDestroy
#define zeCommandListCreate               ccl::ze_lib_ops.zeCommandListCreate
#define zeCommandListCreateImmediate      ccl::ze_lib_ops.zeCommandListCreateImmediate
#define zeCommandListAppendMemoryCopy     ccl::ze_lib_ops.zeCommandListAppendMemoryCopy
#define zeCommandListAppendLaunchKernel   ccl::ze_lib_ops.zeCommandListAppendLaunchKernel
#define zeCommandListAppendWaitOnEvents   ccl::ze_lib_ops.zeCommandListAppendWaitOnEvents
#define zeCommandListAppendSignalEvent    ccl::ze_lib_ops.zeCommandListAppendSignalEvent
#define zeCommandListAppendBarrier        ccl::ze_lib_ops.zeCommandListAppendBarrier
#define zeCommandListAppendMemoryRangesBarrier \
    ccl::ze_lib_ops.zeCommandListAppendMemoryRangesBarrier
#define zeCommandListAppendWriteGlobalTimestamp \
    ccl::ze_lib_ops.zeCommandListAppendWriteGlobalTimestamp
#define zeCommandListClose                ccl::ze_lib_ops.zeCommandListClose
#define zeCommandListReset                ccl::ze_lib_ops.zeCommandListReset
#define zeCommandListDestroy              ccl::ze_lib_ops.zeCommandListDestroy
#define zeContextCreate                   ccl::ze_lib_ops.zeContextCreate
#define zeContextSystemBarrier            ccl::ze_lib_ops.zeContextSystemBarrier
#define zeContextDestroy                  ccl::ze_lib_ops.zeContextDestroy
#define zeEventPoolCreate                 ccl::ze_lib_ops.zeEventPoolCreate
#define zeEventCreate                     ccl::ze_lib_ops.zeEventCreate
#define zeEventQueryStatus                ccl::ze_lib_ops.zeEventQueryStatus
#define zeEventHostSynchronize            ccl::ze_lib_ops.zeEventHostSynchronize
#define zeEventHostReset                  ccl::ze_lib_ops.zeEventHostReset
#define zeEventHostSignal                 ccl::ze_lib_ops.zeEventHostSignal
#define zeEventDestroy                    ccl::ze_lib_ops.zeEventDestroy
#define zeEventPoolOpenIpcHandle          ccl::ze_lib_ops.zeEventPoolOpenIpcHandle
#define zeEventPoolCloseIpcHandle         ccl::ze_lib_ops.zeEventPoolCloseIpcHandle
#define zeEventPoolGetIpcHandle           ccl::ze_lib_ops.zeEventPoolGetIpcHandle
#define zeEventQueryKernelTimestamp       ccl::ze_lib_ops.zeEventQueryKernelTimestamp
#define zeEventPoolDestroy                ccl::ze_lib_ops.zeEventPoolDestroy
#define zeFenceHostSynchronize            ccl::ze_lib_ops.zeFenceHostSynchronize
#define zeFenceCreate                     ccl::ze_lib_ops.zeFenceCreate
#define zeKernelCreate                    ccl::ze_lib_ops.zeKernelCreate
#define zeKernelSetArgumentValue          ccl::ze_lib_ops.zeKernelSetArgumentValue
#define zeKernelSuggestGroupSize          ccl::ze_lib_ops.zeKernelSuggestGroupSize
#define zeKernelSetGroupSize              ccl::ze_lib_ops.zeKernelSetGroupSize
#define zeKernelDestroy                   ccl::ze_lib_ops.zeKernelDestroy
#define zeModuleCreate                    ccl::ze_lib_ops.zeModuleCreate
#define zeModuleDestroy                   ccl::ze_lib_ops.zeModuleDestroy
#define zeModuleBuildLogGetString         ccl::ze_lib_ops.zeModuleBuildLogGetString
#define zeModuleBuildLogDestroy           ccl::ze_lib_ops.zeModuleBuildLogDestroy
#define zeModuleGetNativeBinary           ccl::ze_lib_ops.zeModuleGetNativeBinary
#define zeDeviceGetComputeProperties      ccl::ze_lib_ops.zeDeviceGetComputeProperties
#define zeDeviceGetMemoryAccessProperties ccl::ze_lib_ops.zeDeviceGetMemoryAccessProperties
#define zeDeviceGetMemoryProperties       ccl::ze_lib_ops.zeDeviceGetMemoryProperties
#define zeDeviceGetSubDevices             ccl::ze_lib_ops.zeDeviceGetSubDevices
#define zesDevicePciGetProperties         ccl::ze_lib_ops.zesDevicePciGetProperties
#define zesDeviceEnumFabricPorts          ccl::ze_lib_ops.zesDeviceEnumFabricPorts
#define zesFabricPortGetConfig            ccl::ze_lib_ops.zesFabricPortGetConfig
#define zesFabricPortGetProperties        ccl::ze_lib_ops.zesFabricPortGetProperties
#define zesFabricPortGetState             ccl::ze_lib_ops.zesFabricPortGetState
#define zesDriverGet                      ccl::ze_lib_ops.zesDriverGet
#define zesDeviceGet                      ccl::ze_lib_ops.zesDeviceGet
#ifdef ZE_PCI_PROPERTIES_EXT_NAME
#define zeDevicePciGetPropertiesExt ccl::ze_lib_ops.zeDevicePciGetPropertiesExt
#endif // ZE_PCI_PROPERTIES_EXT_NAME
#define zesDriverGetDeviceByUuidExp         ccl::ze_lib_ops.zesDriverGetDeviceByUuidExp
#define zeDriverGetExtensionFunctionAddress ccl::ze_lib_ops.zeDriverGetExtensionFunctionAddress
#define zeFabricVertexGetExp                ccl::ze_lib_ops.zeFabricVertexGetExp
#define zeFabricVertexGetSubVerticesExp     ccl::ze_lib_ops.zeFabricVertexGetSubVerticesExp
#define zeFabricEdgeGetExp                  ccl::ze_lib_ops.zeFabricEdgeGetExp
#define zeFabricEdgeGetPropertiesExp        ccl::ze_lib_ops.zeFabricEdgeGetPropertiesExp
#define zeDeviceGetFabricVertexExp          ccl::ze_lib_ops.zeDeviceGetFabricVertexExp
#define zelTracerCreate                     ccl::ze_lib_ops.zelTracerCreate
#define zelTracerMemFreeRegisterCallback    ccl::ze_lib_ops.zelTracerMemFreeRegisterCallback
#define zelTracerSetEnabled                 ccl::ze_lib_ops.zelTracerSetEnabled

bool ze_api_init();
void ze_api_fini();

} //namespace ccl

#endif //CCL_ENABLE_SYCL && CCL_ENABLE_ZE

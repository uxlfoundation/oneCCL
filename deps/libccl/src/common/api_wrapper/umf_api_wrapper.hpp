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

#include "oneapi/ccl/config.h"

#include <string>
#include <vector>

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
#include <umf.h>
#include <umf/ipc.h>
#include <umf/providers/provider_level_zero.h>
#include <umf/pools/pool_proxy.h>
// #include <umf/pools/pool_disjoint.h>
#include <umf/memory_provider.h>
namespace ccl {

typedef struct umf_lib_ops {
    decltype(umfPoolGetIPCHandler) *umfPoolGetIPCHandler;
    decltype(umfGetIPCHandle) *umfGetIPCHandle;
    decltype(umfPutIPCHandle) *umfPutIPCHandle;
    decltype(umfOpenIPCHandle) *umfOpenIPCHandle;
    decltype(umfCloseIPCHandle) *umfCloseIPCHandle;
    decltype(umfPoolDestroy) *umfPoolDestroy;
    decltype(umfPoolCreate) *umfPoolCreate;
    decltype(umfPoolMalloc) *umfPoolMalloc;
    decltype(umfPoolAlignedMalloc) *umfPoolAlignedMalloc;
    decltype(umfPoolCalloc) *umfPoolCalloc;
    decltype(umfPoolFree) *umfPoolFree;
    decltype(umfMemoryProviderCreate) *umfMemoryProviderCreate;
    decltype(umfLevelZeroMemoryProviderOps) *umfLevelZeroMemoryProviderOps;
    decltype(umfProxyPoolOps) *umfProxyPoolOps;
    decltype(umfLevelZeroMemoryProviderParamsCreate) *umfLevelZeroMemoryProviderParamsCreate;
    decltype(umfLevelZeroMemoryProviderParamsSetMemoryType)
        *umfLevelZeroMemoryProviderParamsSetMemoryType;
    decltype(umfLevelZeroMemoryProviderParamsSetDevice) *umfLevelZeroMemoryProviderParamsSetDevice;
    decltype(
        umfLevelZeroMemoryProviderParamsSetContext) *umfLevelZeroMemoryProviderParamsSetContext;
} umf_lib_ops_t;

static std::vector<std::string> umf_fn_names = { "umfPoolGetIPCHandler",
                                                 "umfGetIPCHandle",
                                                 "umfPutIPCHandle",
                                                 "umfOpenIPCHandle",
                                                 "umfCloseIPCHandle",
                                                 "umfPoolDestroy",
                                                 "umfPoolCreate",
                                                 "umfPoolMalloc",
                                                 "umfPoolAlignedMalloc",
                                                 "umfPoolCalloc",
                                                 "umfPoolFree",
                                                 "umfMemoryProviderCreate",
                                                 "umfLevelZeroMemoryProviderOps",
                                                 "umfProxyPoolOps",
                                                 "umfLevelZeroMemoryProviderParamsCreate",
                                                 "umfLevelZeroMemoryProviderParamsSetMemoryType",
                                                 "umfLevelZeroMemoryProviderParamsSetDevice",
                                                 "umfLevelZeroMemoryProviderParamsSetContext" };

extern ccl::umf_lib_ops_t umf_lib_ops;

#define umfPoolGetIPCHandler          ccl::umf_lib_ops.umfPoolGetIPCHandler
#define umfGetIPCHandle               ccl::umf_lib_ops.umfGetIPCHandle
#define umfPutIPCHandle               ccl::umf_lib_ops.umfPutIPCHandle
#define umfOpenIPCHandle              ccl::umf_lib_ops.umfOpenIPCHandle
#define umfCloseIPCHandle             ccl::umf_lib_ops.umfCloseIPCHandle
#define umfPoolDestroy                ccl::umf_lib_ops.umfPoolDestroy
#define umfPoolCreate                 ccl::umf_lib_ops.umfPoolCreate
#define umfPoolMalloc                 ccl::umf_lib_ops.umfPoolMalloc
#define umfPoolAlignedMalloc          ccl::umf_lib_ops.umfPoolAlignedMalloc
#define umfPoolCalloc                 ccl::umf_lib_ops.umfPoolCalloc
#define umfPoolFree                   ccl::umf_lib_ops.umfPoolFree
#define umfMemoryProviderCreate       ccl::umf_lib_ops.umfMemoryProviderCreate
#define umfLevelZeroMemoryProviderOps ccl::umf_lib_ops.umfLevelZeroMemoryProviderOps
#define umfProxyPoolOps               ccl::umf_lib_ops.umfProxyPoolOps
#define umfLevelZeroMemoryProviderParamsCreate \
    ccl::umf_lib_ops.umfLevelZeroMemoryProviderParamsCreate
#define umfLevelZeroMemoryProviderParamsSetMemoryType \
    ccl::umf_lib_ops.umfLevelZeroMemoryProviderParamsSetMemoryType
#define umfLevelZeroMemoryProviderParamsSetDevice \
    ccl::umf_lib_ops.umfLevelZeroMemoryProviderParamsSetDevice
#define umfLevelZeroMemoryProviderParamsSetContext \
    ccl::umf_lib_ops.umfLevelZeroMemoryProviderParamsSetContext

bool umf_api_init();
void umf_api_fini();

} //namespace ccl
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF

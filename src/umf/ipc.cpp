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
#include "oneapi/ccl/config.h"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
#include "umf/ipc.hpp"
#include "common/stream/stream.hpp"
#include "common/global/global.hpp"

#include "comm/comm_interface.hpp"
#include "comm/comm.hpp"

#include <vector>
#include <map>
#include <iostream>
#include <cstring>
#include <cstdlib>

std::map<int, std::vector<umf_ipc_handle_t>> ipc_handle_map;
umf_memory_pool_handle_t* umf_memory_pools = nullptr;
std::vector<std::pair<ze_device_handle_t, int>> ze_devices;

int create_umf_ipc_handle(const void* ptr, gpu_umf_ipc_mem_handle_t* ipc_handle) {
    umf_result_t umf_result;

    std::memset(ipc_handle, 0, sizeof(gpu_umf_ipc_mem_handle_t));
    umf_result = umfGetIPCHandle(ptr, &ipc_handle->handle, &ipc_handle->size);

    if (umf_result != UMF_RESULT_SUCCESS) {
        CCL_THROW("Failed to get UMF IPC handle: ", umf_result);
    }
    return 0;
}

void umf_ipc_exchange(ccl_comm* comm, ccl_stream* stream, std::vector<void*> ptrs) {
    ccl_comm* node_comm = comm->get_node_comm().get();
    std::vector<gpu_umf_ipc_mem_handle_t> ipc_handles(ptrs.size());
    std::vector<size_t> umf_sizes(ptrs.size());

    // Create IPC handles for each pointer and record their sizes
    for (size_t i = 0; i < ptrs.size(); ++i) {
        create_umf_ipc_handle(ptrs[i], &ipc_handles[i]);
        umf_sizes[i] = ipc_handles[i].size;
    }

    // Calculate total memory required to store all IPC handles and their sizes
    size_t total_umf_size = 0;
    for (const auto& size : umf_sizes) {
        total_umf_size += size;
    }

    // Allocate memory for all handles and an array to store sizes
    void* mem = new char[total_umf_size];
    void* all_mem = new char[total_umf_size * node_comm->size()];

    // Copy local IPC handles to the memory buffer
    size_t offset = 0;
    for (size_t i = 0; i < ipc_handles.size(); ++i) {
        memcpy(static_cast<char*>(mem) + offset, ipc_handles[i].handle, umf_sizes[i]);
        offset += umf_sizes[i];
    }

    // Perform allgather
    ccl::utils::allgather(node_comm->get_atl_comm(), mem, all_mem, total_umf_size);

    // Process the received IPC handles and store them in the map
    for (int rank = 0; rank < node_comm->size(); rank++) {
        size_t rank_offset = rank * total_umf_size;
        offset = 0;
        std::vector<umf_ipc_handle_t> rank_ipc_handles;
        for (size_t handle_idx = 0; handle_idx < ipc_handles.size(); handle_idx++) {
            char* ipc_handle = static_cast<char*>(all_mem) + rank_offset + offset;
            rank_ipc_handles.push_back(
                (umf_ipc_handle_t)(ipc_handle)); // Store the handle in the vector
            offset += umf_sizes[handle_idx];
        }
        ipc_handle_map[rank] = std::move(rank_ipc_handles);
    }
}

int init_umf_mem_pools(std::vector<ccl::ze::device_info> ze_devs) {
    if (ccl::global_data::env().umf_enable == 0) {
        CCL_THROW("umf is not initialized");
    }

    // auto devices = q.get_context().get_devices();
    for (int dev_idx = 0; dev_idx < ze_devs.size(); ++dev_idx) {
        ze_devices.emplace_back(ze_devs[dev_idx].device, dev_idx);
    }

    umf_memory_pools = static_cast<umf_memory_pool_handle_t*>(
        std::calloc(ze_devs.size(), sizeof(umf_memory_pool_handle_t)));
    if (!umf_memory_pools) {
        CCL_THROW("Failed to allocate UMF memory pools");
        return 1;
    }

    return 0;
}

int get_device_index(sycl::queue q, sycl::device target_device) {
    auto devices = q.get_context().get_devices();
    for (int dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        if (devices[dev_idx] == target_device) {
            return dev_idx;
        }
    }
    CCL_THROW("Failed to find device index");
    return -1;
}

int destroy_umf_mem_pools(std::vector<ccl::ze::device_info> ze_devs) {
    if (umf_memory_pools) {
        auto num_devices = ze_devs.size();
        for (size_t i = 0; i < num_devices; ++i) {
            if (umf_memory_pools[i]) {
                umfPoolDestroy(umf_memory_pools[i]);
                umf_memory_pools[i] = nullptr;
            }
        }
        std::free(umf_memory_pools);
        umf_memory_pools = nullptr;
    }
    return 0;
}

int create_level_zero_pool(sycl::queue q,
                           ze_device_handle_t device,
                           umf_memory_pool_handle_t* pool) {
    umf_memory_provider_handle_t ze_provider = nullptr;
    umf_level_zero_memory_provider_params_handle_t level_zero_params = nullptr;

    const umf_memory_provider_ops_t* ze_ops = umfLevelZeroMemoryProviderOps();
    sycl::context ctx = q.get_context();
    ze_context_handle_t ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

    umf_result_t umf_result = umfLevelZeroMemoryProviderParamsCreate(&level_zero_params);
    if (umf_result != UMF_RESULT_SUCCESS) {
        CCL_THROW("Failed umfLevelZeroMemoryProviderParamsCreate: ", umf_result);
    }

    umf_result = umfLevelZeroMemoryProviderParamsSetContext(level_zero_params, ze_ctx);
    if (umf_result != UMF_RESULT_SUCCESS) {
        CCL_THROW("Failed: umfLevelZeroMemoryProviderParamsSetContext: ", umf_result);
    }

    umf_result = umfLevelZeroMemoryProviderParamsSetDevice(level_zero_params, device);
    if (umf_result != UMF_RESULT_SUCCESS) {
        CCL_THROW("Failed: umfLevelZeroMemoryProviderParamsSetDevice: ", umf_result);
    }

    umf_result =
        umfLevelZeroMemoryProviderParamsSetMemoryType(level_zero_params, UMF_MEMORY_TYPE_DEVICE);
    if (umf_result != UMF_RESULT_SUCCESS) {
        CCL_THROW("Failed: umfLevelZeroMemoryProviderParamsSetMemoryType: ", umf_result);
    }

    umf_result = umfMemoryProviderCreate(ze_ops, level_zero_params, &ze_provider);
    if (umf_result != UMF_RESULT_SUCCESS) {
        CCL_THROW("Failed to create Level Zero provider: ", umf_result);
    }

    umf_pool_create_flags_t flags = UMF_POOL_CREATE_FLAG_OWN_PROVIDER;
    const umf_memory_pool_ops_t* pool_ops = umfProxyPoolOps();

    umf_result = umfPoolCreate(pool_ops, ze_provider, nullptr, flags, pool);
    if (umf_result != UMF_RESULT_SUCCESS) {
        CCL_THROW("Failed to create Level Zero Pool: ", umf_result);
    }

    return 0;
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF

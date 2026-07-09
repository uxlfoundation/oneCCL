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

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_UMF)
#include "common/log/log.hpp"
#include "common/api_wrapper/umf_api_wrapper.hpp"
#include <sycl/sycl.hpp>
#include <map>
#include <vector>

// Forward declarations to avoid circular includes
class ccl_comm;
class ccl_stream;
namespace ccl {
namespace ze {
struct device_info;
}
} // namespace ccl
typedef struct gpu_umf_ipc_mem_handle_t {
    umf_ipc_handle_t handle;
    size_t size;
} gpu_umf_ipc_mem_handle_t;

extern std::map<int, std::vector<umf_ipc_handle_t>> ipc_handle_map;
extern umf_memory_pool_handle_t *umf_memory_pools;
extern std::vector<std::pair<ze_device_handle_t, int>> ze_devs;

void umf_ipc_exchange(ccl_comm *comm, ccl_stream *stream, std::vector<void *> ptrs);
int create_umf_ipc_handle(const void *ptr, gpu_umf_ipc_mem_handle_t *ipc_handle);

int init_umf_mem_pools(std::vector<ccl::ze::device_info> ze_devs);
int destroy_umf_mem_pools(std::vector<ccl::ze::device_info> ze_devs);

int get_device_index(sycl::queue q, sycl::device target_device);

int create_level_zero_pool(sycl::queue q,
                           ze_device_handle_t device,
                           umf_memory_pool_handle_t *pool);

// Helper functions for UMF memory allocation
umf_memory_pool_handle_t get_or_create_umf_pool(sycl::queue &q);
void *umf_alloc_device(sycl::queue &q, size_t size);
void *umf_alloc_device_aligned(sycl::queue &q, size_t alignment, size_t size);
void umf_free(void *ptr);

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE && CCL_ENABLE_UMF

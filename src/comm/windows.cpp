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

#include <unistd.h>
#include <cstdint>
#include <utility>
#include "comm/comm.hpp"
#include "comm/windows.hpp"

std::vector<std::vector<void *>> do_simple_ipc_exchange(ccl_comm *comm,
                                                        const std::vector<void *> &ptrs,
                                                        const std::vector<size_t> &sizes);

int ccl_window::register_buf(void *ptr, size_t size) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    int comm_size = comm->size();
    int comm_rank = comm->rank();

    ze_context_handle_t context{};
    ze_device_handle_t device{};
    ze_memory_allocation_properties_t mem_alloc_props{};
    if (!ccl::ze::get_buffer_context_and_device(ptr, &context, &device, &mem_alloc_props)) {
        LOG_WARN("unable to get context from ptr\n");
        return 0;
    }
    void *base_ptr;
    size_t total_size, offset;
    ZE_CALL(zeMemGetAddressRange, (context, ptr, &base_ptr, &total_size));
    if ((size_t)((char *)ptr + size) > (size_t)((char *)base_ptr + total_size)) {
        LOG_WARN("Invalid pointer registered. ");
        return 0;
    }
    offset = (char *)ptr - (char *)base_ptr;

    std::vector<void *> in_buffers(1);
    std::vector<size_t> in_buffers_sizes(1);
    in_buffers[0] = ptr;
    in_buffers_sizes[0] = size;
    LOG_DEBUG("calling do_simple_ipc_exchange on:", std::hex, base_ptr);
    std::vector<std::vector<void *>> ipc_ptrs =
        do_simple_ipc_exchange(comm, in_buffers, in_buffers_sizes);

    registered_ptrs.resize(comm_size);
    for (int r = 0; r < comm_size; r++) {
        registered_ptrs[r].ptr = r == comm_rank ? ptr : ipc_ptrs[r][0];
        registered_ptrs[r].size = size;
        LOG_DEBUG("register ipc ptr: rank: ", r, " ptr: ", std::hex, registered_ptrs[r].ptr);
    }
    return 1;
#else
    return 0;
#endif
}

int ccl_window::deregister_buf() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    int comm_rank = comm->rank();
    int comm_size = comm->size();

    ze_context_handle_t context{};
    ze_device_handle_t device{};
    ze_memory_allocation_properties_t mem_alloc_props{};
    if (!ccl::ze::get_buffer_context_and_device(
            registered_ptrs[comm_rank].ptr, &context, &device, &mem_alloc_props)) {
        LOG_WARN("unable to get context from ptr\n");
        return 0;
    }

    for (int r = 0; r < comm_size; r++) {
        if (r == comm->rank())
            continue;
        ZE_CALL(zeMemCloseIpcHandle, (context, registered_ptrs[r].ptr));
    }
    return 1;
#else
    return 0;
#endif
}

bool ccl_window::is_registered(const void *ptr, size_t size, size_t &offset) {
    int comm_rank = comm->rank();
    if (ptr >= registered_ptrs[comm_rank].ptr &&
        static_cast<const char *>(ptr) + size <=
            static_cast<char *>(registered_ptrs[comm_rank].ptr) + registered_ptrs[comm_rank].size) {
        offset =
            static_cast<const char *>(ptr) - static_cast<char *>(registered_ptrs[comm_rank].ptr);
        LOG_DEBUG("found registration ptr: ",
                  std::hex,
                  registered_ptrs[comm_rank].ptr,
                  " offset: ",
                  offset);
        return true;
    }
    return false;
}

std::vector<void *> ccl_window::get_ptrs(size_t offset) {
    std::vector<void *> ptrs(comm->size());
    for (int r = 0; r < comm->size(); r++) {
        ptrs[r] = static_cast<char *>(registered_ptrs[r].ptr) + offset;
        LOG_DEBUG("get ptr: rank: ",
                  r,
                  " ptr: ",
                  std::hex,
                  registered_ptrs[r].ptr,
                  " with offset: ",
                  offset);
    }
    return ptrs;
}

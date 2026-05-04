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
#include "atl/mpi/atl_mpi_ctx.hpp"
#include "coll/coll_util.hpp"
#include "comm/comm.hpp"

struct ipc_handle_desc {
    ze_ipc_mem_handle_t ipc_handle{};
    size_t mem_offset{};
    void *mem_ptr{};
    size_t size;
};

struct payload_t {
    ze_context_handle_t context{};
    ze_device_handle_t device{};
    ze_ipc_mem_handle_t ipc_handle{};
    size_t mem_offset{};
    size_t size;
};

// assuming base pointers
// no caching
std::vector<std::vector<void *>> do_simple_ipc_exchange(ccl_comm *comm,
                                                        const std::vector<void *> &ptrs,
                                                        const std::vector<size_t> &sizes) {
    int comm_rank = comm->rank();
    int comm_size = comm->size();
    int buf_count = ptrs.size();

    std::vector<payload_t> all_payloads(comm_size * buf_count);
    std::vector<payload_t> local_payloads(buf_count);
    std::vector<std::vector<void *>> handles(comm_size);

    for (int i = 0; i < buf_count; i++) {
        auto ptr = ptrs[i];
        void *base_ptr;
        size_t total_size;
        if (ptr == NULL)
            continue;
        payload_t payload;
        //int mem_handle = ccl::utils::invalid_mem_handle;

        ze_context_handle_t context{};
        ze_device_handle_t device{};
        ze_memory_allocation_properties_t mem_alloc_props{};
        if (!ccl::ze::get_buffer_context_and_device(ptr, &context, &device, &mem_alloc_props)) {
            CCL_THROW("unable to get context from ptr\n");
        }
        ZE_CALL(zeMemGetAddressRange, (context, ptr, &base_ptr, &total_size));

        ze_ipc_mem_handle_t ipc_handle{};
        ZE_CALL(zeMemGetIpcHandle, (context, base_ptr, &ipc_handle));
        payload.ipc_handle = ipc_handle;
        payload.size = sizes[i];
        payload.mem_offset = (char *)ptr - (char *)base_ptr;
        payload.context = context;
        payload.device = device;
        local_payloads[i] = payload;
    }

    // call allgather
    if (!(ccl::utils::allgather(comm->get_atl_comm(),
                                local_payloads.data(),
                                all_payloads.data(),
                                sizeof(payload_t) * buf_count))) {
        CCL_THROW("allgather exchange is failed");
    }

    for (int r = 0; r < comm_size; r++) {
        if (r == comm_rank) {
            handles[r] = ptrs;
        }
        else {
            handles[r].resize(buf_count);
            for (int buf_idx = 0; buf_idx < buf_count; buf_idx++) {
                int payload_idx = buf_count * r + buf_idx;
                if (all_payloads[payload_idx].size != local_payloads[buf_idx].size) {
                    CCL_THROW("not same size\n");
                }
                void *ptr;
                ZE_CALL(zeMemOpenIpcHandle,
                        (local_payloads[buf_idx].context,
                         local_payloads[buf_idx].device,
                         all_payloads[payload_idx].ipc_handle,
                         0 /* cache allocation */,
                         &ptr));
                handles[r][buf_idx] = (char *)ptr + all_payloads[payload_idx].mem_offset;
            }
        }
    }
    return handles;
}

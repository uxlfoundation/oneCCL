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
#include "comm/comm.hpp"
#include "common/global/global.hpp"
#include "sched/entry/ze/ze_call.hpp"
#include "sched/ze/ze_handle_manager.hpp"

namespace ccl {

namespace ze {

static void cast_mem_to_pool_handle(ze_ipc_event_pool_handle_t* pool,
                                    const ze_ipc_mem_handle_t* mem) {
    static_assert(sizeof(ze_ipc_event_pool_handle_t) == sizeof(ze_ipc_mem_handle_t));
    memcpy(pool, mem, sizeof(*mem));
}

std::string to_string(ipc_mem_type mem_type) {
    switch (mem_type) {
        case ipc_mem_type::memory: return "buffer";
        case ipc_mem_type::pool: return "pool";
        default: return "unknown";
    }
}

std::string to_string(const ze_ipc_mem_handle_t& handle) {
    return ccl::utils::to_hex(handle.data, ZE_MAX_IPC_HANDLE_SIZE);
}

// ipc_handle_desc
ipc_handle_desc::ipc_handle_desc() {
    memset(&ipc_handle, 0, sizeof(ipc_handle));
}

ze_ipc_mem_handle_t ipc_handle_desc::mem_to_ipc_handle() const {
    if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::sockets ||
        ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::none) {
        // for the sockets mode, we don't need to do mem_handle_to_fd
        // we return immediately the ipc handle which was already inited
        return ipc_handle;
    }

    int fd = ccl::utils::invalid_fd;
    if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::drmfd) {
        CCL_THROW_IF_NOT(device_fd != ccl::utils::invalid_fd, "device_fd is invalid value");
        fd = ccl::ze::fd_manager::mem_handle_to_fd(device_fd, mem_handle);
        LOG_DEBUG("device_fd: ", device_fd, " gotten fd from mem_handle_to_fd: ", fd);
    }
    else if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::pidfd) {
        CCL_THROW_IF_NOT(remote_pid != ccl::utils::invalid_pid, "remote_pid is invalid value");
        int opened_fd = ccl::ze::fd_manager::pidfd_open(remote_pid);
        fd = ccl::ze::fd_manager::mem_handle_to_fd(opened_fd, mem_handle);
        ccl::utils::close_fd(opened_fd);
        LOG_DEBUG("remote_pid: ", remote_pid, ", gotten fd from mem_handle_to_fd: ", fd);
    }
    else {
        CCL_THROW("unexpected ipc_exchange_mode");
    }

    CCL_THROW_IF_NOT(fd != ccl::utils::invalid_fd, "mem_handle_to_fd: invalid fd: ", fd);
    LOG_DEBUG("mem_handle: ", mem_handle, ", fd: ", fd);
    return ccl::ze::get_handle_from_fd(fd);
}

// ipc_handle_manager
ipc_handle_manager::~ipc_handle_manager() {
    clear();
}

void ipc_handle_manager::init(const ccl_comm* init_comm, const ccl_stream* init_stream) {
    LOG_DEBUG("init");
    CCL_THROW_IF_NOT(init_comm, "no comm");
    CCL_THROW_IF_NOT(init_stream, "no stream");

    comm = const_cast<ccl_comm*>(init_comm);

    for (int idx = 0; idx < comm->size(); idx++) {
        rank_map.insert({ comm->get_global_rank(idx), idx });
    }

    device = init_stream->get_ze_device();
    context = init_stream->get_ze_context();

    LOG_DEBUG("init completed");
}

void ipc_handle_manager::clear() {
    for (int rank = 0; rank < static_cast<int>(handles.size()); rank++) {
        for (size_t buf_idx = 0; buf_idx < handles[rank].size(); buf_idx++) {
            const auto& handle_info = handles[rank][buf_idx];
            ze_ipc_mem_handle_t ipc_handle = handle_info.ipc_handle;
            auto mem_ptr = handle_info.mem_ptr;
            auto mem_type = handle_info.mem_type;
            size_t mem_offset = handle_info.mem_offset;

            // when closing the ipc_handle we need to take care of pointers that points to the
            // same level zero allocation. They're simply offsetted from some base pointer
            // although represented by different FDs. If we close this base pointer,
            // all the derived pointers are closed(unmapped) as well. To handle this case
            // we ignore the result of close function which would fail if we close a pointer
            // which is already closed. The function has semantic of free() call, so the result
            // is not much useful anyway.
            if (mem_ptr) {
                ze_result_t res{};
                if (handle_info.is_cached) {
                    // skip close, assume that ipc_handle will be closed in the cache
                    res = ZE_RESULT_SUCCESS;
                }
                else if (mem_type == ipc_mem_type::memory) {
                    // There is a bug in L0 that results in hang in this function
                    // when we use kernel output event, as a workaround skip it
                    // if the knob is set
                    if (global_data::env().ze_close_ipc_wa) {
                        res = ZE_RESULT_SUCCESS;
                    }
                    else {
                        LOG_DEBUG("close ipc_handle: { base_ptr: ",
                                  mem_ptr,
                                  ", offset: ",
                                  mem_offset,
                                  ", fd: ",
                                  get_fd_from_handle(ipc_handle),
                                  ", rank: ",
                                  rank,
                                  ", buf_idx: ",
                                  buf_idx,
                                  " }");
                        res = zeMemCloseIpcHandle(context, mem_ptr);
                    }
                }
                else if (mem_type == ipc_mem_type::pool) {
                    res = zeEventPoolCloseIpcHandle((ze_event_pool_handle_t)mem_ptr);
                }
                else {
                    CCL_THROW("unknown memory type");
                }

                if (res != ZE_RESULT_SUCCESS) {
                    LOG_TRACE("unable to close memory ipc_handle: ",
                              "level-zero res: ",
                              to_string(res),
                              ", rank: ",
                              rank,
                              ", buf_idx: ",
                              buf_idx,
                              ", ptr: ",
                              mem_ptr);
                }
            }
        }
    }

    if (!handles.empty()) {
        LOG_DEBUG("handles are cleared successfully");
    }

    handles.clear();
}

void ipc_handle_manager::set(const mem_handle_map_t& handles_arg, bool pt2pt_op) {
    CCL_THROW_IF_NOT(!handles_arg.empty(), "handles_arg argument is empty");
    if (pt2pt_op) {
        CCL_THROW_IF_NOT(handles_arg.size() == pt2pt_handles_size,
                         "handles_arg (",
                         handles_arg.size(),
                         ") and handle_pt2pt_size (",
                         pt2pt_handles_size,
                         "), but it must be equal");
    }
    else {
        CCL_THROW_IF_NOT(handles_arg.size() == static_cast<size_t>(comm->size()),
                         "handles_arg and comm sizes should be equal");
    }

    CCL_THROW_IF_NOT(handles.empty(), "handles should be empty before set");

    handles = handles_arg;
    LOG_DEBUG("handles are set successfully, size of handles: ", handles.size());
}

void* ipc_handle_manager::get_ptr(int rank,
                                  size_t buf_idx,
                                  const ccl_comm* map_comm,
                                  bool pt2pt_op,
                                  bool to_cache) {
    check_rank(rank, (map_comm) ? map_comm : comm, pt2pt_op);
    if (map_comm && (map_comm->id() != comm->id())) {
        int old_rank = rank;
        rank = map_comm->get_global_rank(rank);
        auto rank_it = rank_map.find(rank);
        if (rank_it == rank_map.end()) {
            CCL_THROW("handle manager can not handle global rank ", rank);
        }
        rank = rank_it->second;
        LOG_DEBUG("convert rank: old_rank: ",
                  old_rank,
                  " old_comm: id: ",
                  map_comm->id(),
                  ", size: ",
                  map_comm->size(),
                  ", new_rank: ",
                  rank,
                  " new_comm: id: ",
                  comm->id(),
                  ", size: ",
                  comm->size());
        check_rank(rank, comm, pt2pt_op);
    }
    CCL_THROW_IF_NOT(buf_idx < handles[rank].size(), "buf_idx is not valid value: ", buf_idx);

    // must be a non-const ref so it can be updated when ipc_handle is opened
    ipc_handle_desc& handle_info = handles[rank][buf_idx];
    auto& ipc_handle = handle_info.ipc_handle;
    auto& mem_ptr = handle_info.mem_ptr;
    auto mem_type = handle_info.mem_type;

    LOG_DEBUG("context: ", context, ", device: ", device, ", rank: ", rank, ", buf_idx: ", buf_idx);

    if (mem_ptr == nullptr) {
        if (mem_type == ipc_mem_type::memory) {
            open_handle(handle_info, &mem_ptr, to_cache);
        }
        else if (mem_type == ipc_mem_type::pool) {
            ze_ipc_event_pool_handle_t pool_handle;
            cast_mem_to_pool_handle(&pool_handle, &ipc_handle);
            open_handle(pool_handle, (ze_event_pool_handle_t*)&mem_ptr);
        }
        else {
            CCL_THROW("unknown memory type");
        }
    }

    LOG_DEBUG("get handle: { mem_ptr: ",
              mem_ptr,
              ", fd: ",
              get_fd_from_handle(ipc_handle),
              ", rank: ",
              rank,
              ", buf_idx: ",
              buf_idx,
              " }");

    // add offset that we received along with the ipc_handle
    if (mem_type == ipc_mem_type::pool) {
        CCL_THROW_IF_NOT(handle_info.mem_offset == 0, "offsets should be 0 for event pool");
    }
    return static_cast<void*>(static_cast<char*>(mem_ptr) + handle_info.mem_offset);
}

void ipc_handle_manager::get(int rank,
                             size_t buf_idx,
                             ccl_buffer& buf,
                             const ccl_comm* map_comm,
                             bool pt2pt_op,
                             bool to_cache) {
    buf.set(get_ptr(rank, buf_idx, map_comm, pt2pt_op, to_cache));
}

void ipc_handle_manager::get(int rank,
                             size_t buf_idx,
                             ze_event_pool_handle_t& buf,
                             const ccl_comm* map_comm,
                             bool pt2pt_op) {
    buf = (ze_event_pool_handle_t)get_ptr(rank, buf_idx, map_comm, pt2pt_op);
}

void ipc_handle_manager::get_handle(void* ptr, ze_ipc_mem_handle_t* ipc_handle, size_t* handle_id) {
    CCL_THROW_IF_NOT(ptr, "no mem pointer");
    if (global_data::env().enable_ze_cache && global_data::env().enable_ze_cache_get_ipc_handles) {
        ze_memory_allocation_properties_t alloc_props = ccl::ze::default_alloc_props;
        ZE_CALL(zeMemGetAllocProperties, (context, ptr, &alloc_props, &device));

        ipc_get_handle_desc ipc_desc = { ptr, alloc_props.id };
        ipc_entry_t tmp_ipc{};
        global_data::get().ze_data->cache->get(context, device, ipc_desc, &tmp_ipc);
        *ipc_handle = tmp_ipc.handle;
        *handle_id = tmp_ipc.handle_id;
    }
    else {
        ZE_CALL(zeMemGetIpcHandle, (context, ptr, ipc_handle));
    }
}

void ipc_handle_manager::get_handle(ze_event_pool_handle_t pool,
                                    ze_ipc_event_pool_handle_t* ipc_handle) {
    CCL_THROW_IF_NOT(pool, "no pool");
    ZE_CALL(zeEventPoolGetIpcHandle, (pool, ipc_handle));
}

void ipc_handle_manager::open_handle(ipc_handle_desc& info, void** ptr, bool to_cache) {
    // 1. If enable_ze_cache == false, code not executed
    // 2. If enable_ze_cache_open_ipc_handles == false, code not executed
    // 3. If (all conditions) == false, code not executed
    // 4. If all conditions true, code executed
    if (global_data::env().enable_ze_cache && global_data::env().enable_ze_cache_open_ipc_handles &&
        to_cache) {
        mem_handle_cache::value_t value{};
        global_data::get().ze_data->cache->get(context, device, info, &value);
        CCL_THROW_IF_NOT(value != nullptr, "unable to open ipc_handle");
        *ptr = const_cast<void*>(value->get_ptr());
        cached_handles.push_back(value);
        info.is_cached = true;
    }
    else {
        auto handle = info.mem_to_ipc_handle();
        ZE_CALL(zeMemOpenIpcHandle, (context, device, handle, 0 /* cache allocation */, ptr));
        if (!to_cache) {
            // used by Sycl kernels, IPC handle needs to be kept open
            info.is_cached = true;
        }
        else {
            close_handle_fd(handle);
        }
    }
}

void ipc_handle_manager::open_handle(const ze_ipc_event_pool_handle_t& ipc_handle,
                                     ze_event_pool_handle_t* pool) {
    ZE_CALL(zeEventPoolOpenIpcHandle, (context, ipc_handle, pool));
}

void ipc_handle_manager::get_address_range(const void* ptr, void** base_ptr, size_t* size) {
    ZE_CALL(zeMemGetAddressRange, (context, ptr, base_ptr, size));
    LOG_DEBUG("zeMemGetAddressRange: ptr: ",
              ptr,
              ", base ptr: ",
              *base_ptr,
              ", offset: ",
              ccl::utils::get_ptr_diff(*base_ptr, ptr),
              ", size: ",
              *size);
}

void ipc_handle_manager::check_rank(int rank, const ccl_comm* check_comm, bool pt2pt_op) {
    if (pt2pt_op) {
        CCL_THROW_IF_NOT((rank == 0) && (rank < static_cast<int>(handles.size())),
                         "expect 0 handle idx (rank) to get ptr for pt2pt_op: rank: ",
                         rank);
    }
    else {
        CCL_THROW_IF_NOT(
            (rank >= 0) && (rank < static_cast<int>(handles.size())) && (rank < check_comm->size()),
            "invalid rank: ",
            rank,
            ", handles.size: ",
            handles.size(),
            ", comm.size: ",
            check_comm->size());
        CCL_THROW_IF_NOT(
            rank != check_comm->rank(), "do not expect to open ipc_handle for own rank: ", rank);
    }
}

} // namespace ze
} // namespace ccl

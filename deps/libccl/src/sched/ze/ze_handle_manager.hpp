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

#include "common/log/log.hpp"
#include "common/stream/stream.hpp"
#include "common/utils/buffer.hpp"
#include "common/utils/utils.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <unordered_map>
#include "common/api_wrapper/ze_api_wrapper.hpp"

class ccl_comm;

namespace ccl {

namespace ze {

enum class ipc_mem_type : int { unknown = 0, memory, pool };

std::string to_string(ipc_mem_type type);

std::string to_string(const ze_ipc_mem_handle_t& handle);

struct ipc_get_handle_desc {
    void* ptr{ nullptr };
    uint64_t mem_id{};
};

struct ipc_handle_desc {
    ze_ipc_mem_handle_t ipc_handle{};
    size_t mem_offset{};
    void* mem_ptr{};
    void* remote_ptr{};
    ipc_mem_type mem_type{};
    size_t handle_id{ ccl::utils::initial_handle_id_value };
    int mem_handle{ ccl::utils::invalid_mem_handle };
    pid_t remote_pid{ ccl::utils::invalid_pid };
    ssize_t remote_context_id{ ccl::utils::invalid_context_id };
    uint64_t remote_mem_alloc_id{};
    ssize_t remote_device_id{ ccl::utils::invalid_device_id };
    int device_fd{ ccl::utils::invalid_fd };

    bool is_cached = false;

    ipc_handle_desc();
    ipc_handle_desc(const ipc_handle_desc&) = default;
    ipc_handle_desc& operator=(const ipc_handle_desc&) = default;

    ze_ipc_mem_handle_t mem_to_ipc_handle() const;
};

class ipc_handle_manager {
public:
    // for pt2pt ops, it's assumed to have only 1 buffer for communication
    // it means, only one handle is expected for each rank which is participated
    // in pt2pt communication
    static constexpr int pt2pt_handles_size = 1;

    // matrix with ipc handles, row - rank, column - buf_idx
    using mem_handle_map_t = typename std::vector<std::vector<ipc_handle_desc>>;

    ipc_handle_manager() = default;
    ipc_handle_manager(const ipc_handle_manager&) = delete;
    ipc_handle_manager& operator=(const ipc_handle_manager&) = delete;
    ~ipc_handle_manager();

    void init(const ccl_comm* comm, const ccl_stream* stream);
    void clear();

    void set(const mem_handle_map_t& handles_arg, bool pt2pt_op = false);

    void* get_ptr(int rank,
                  size_t buf_idx,
                  const ccl_comm* map_comm,
                  bool pt2pt_op = false,
                  bool to_cache = true);
    void get(int rank,
             size_t buf_idx,
             ccl_buffer& buf,
             const ccl_comm* map_comm = nullptr,
             bool pt2pt_op = false,
             bool to_cache = true /*can control unique buf to cache*/);
    void get(int rank,
             size_t buf_idx,
             ze_event_pool_handle_t& buf,
             const ccl_comm* map_comm,
             bool pt2pt_op = false);

    void get_handle(void* ptr, ze_ipc_mem_handle_t* ipc_handle, size_t* handle_id);
    void get_handle(ze_event_pool_handle_t pool, ze_ipc_event_pool_handle_t* ipc_handle);
    void open_handle(ipc_handle_desc& info, void** ptr, bool to_cache);
    void open_handle(const ze_ipc_event_pool_handle_t& ipc_handle, ze_event_pool_handle_t* pool);

    void get_address_range(const void* ptr, void** base_ptr, size_t* size);

private:
    ze_context_handle_t context{};
    ze_device_handle_t device{};
    ccl_comm* comm{};
    std::unordered_map<int, int> rank_map{};
    mem_handle_map_t handles;

    /**
     * The value can be destroyed in the cache if the cache reaches its limit.
     * This can happen at a time when the ipc_handle is really needed.
     * We can run a lot of ranks and get fail here.
     * Instead, the value will be popped from the cache, but only destroyed when not needed.
     * We rely on the smart pointer to work.
     * So, in cached_handles we just save handles to increase ref counter
     */
    std::list<mem_handle_cache::value_t> cached_handles;

    void check_rank(int rank, const ccl_comm* check_comm, bool pt2pt_op);
};

} // namespace ze
} // namespace ccl

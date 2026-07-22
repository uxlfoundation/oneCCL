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
#include "common/utils/hash.hpp"

#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/cache/ze_device_cache.hpp"

#include <unordered_map>

namespace ccl {
namespace ze {

class kernel_cache {
public:
    kernel_cache() = default;
    kernel_cache(const kernel_cache&) = delete;
    kernel_cache& operator=(const kernel_cache&) = delete;
    ~kernel_cache();

    void clear();

    void get(ze_module_handle_t module, const std::string& kernel_name, ze_kernel_handle_t* kernel);
    void push(ze_module_handle_t module, const std::string& kernel_name, ze_kernel_handle_t kernel);

private:
    using key_t = typename std::tuple<ze_module_handle_t, std::string>;
    using value_t = ze_kernel_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

// TODO: need to improve with ability to save list with commands for specific algo
class list_cache {
public:
    list_cache() = default;
    list_cache(const list_cache&) = delete;
    list_cache& operator=(const list_cache&) = delete;
    ~list_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_command_list_desc_t& list_desc,
             ze_command_list_handle_t* list);
    void push(ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_command_list_desc_t& list_desc,
              ze_command_list_handle_t list);

private:
    using key_t = typename std::
        tuple<ze_context_handle_t, ze_device_handle_t, uint32_t, ze_command_list_flags_t>;
    using value_t = ze_command_list_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

class queue_cache {
public:
    queue_cache() = default;
    queue_cache(const queue_cache&) = delete;
    queue_cache& operator=(const queue_cache&) = delete;
    ~queue_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_command_queue_desc_t& queue_desc,
             ze_command_queue_handle_t* queue);
    void push(ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_command_queue_desc_t& queue_desc,
              ze_command_queue_handle_t queue);

private:
    using key_t = typename std::tuple<ze_context_handle_t,
                                      ze_device_handle_t,
                                      uint32_t,
                                      uint32_t,
                                      ze_command_queue_flags_t,
                                      ze_command_queue_mode_t,
                                      ze_command_queue_priority_t>;
    using value_t = ze_command_queue_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

class event_pool_cache {
public:
    event_pool_cache() = default;
    event_pool_cache(const event_pool_cache&) = delete;
    event_pool_cache& operator=(const event_pool_cache&) = delete;
    ~event_pool_cache();

    void clear();

    void get(ze_context_handle_t context,
             const ze_event_pool_desc_t& pool_desc,
             ze_event_pool_handle_t* event_pool);

    void push(ze_context_handle_t context,
              const ze_event_pool_desc_t& pool_desc,
              ze_event_pool_handle_t event_pool);

private:
    using key_t = typename std::tuple<ze_context_handle_t, ze_event_pool_flags_t, uint32_t>;
    using value_t = ze_event_pool_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

struct ipc_handle_desc;

class module_cache {
public:
    module_cache() = default;
    module_cache(const module_cache&) = delete;
    module_cache& operator=(const module_cache&) = delete;
    ~module_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const std::string& spv_name,
             ze_module_handle_t* module);

private:
    using key_t = typename std::tuple<ze_device_handle_t, std::string>;
    using value_t = ze_module_handle_t;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;

    void load(ze_context_handle_t context,
              ze_device_handle_t device,
              const std::string& spv_name,
              ze_module_handle_t* module);
};

class mem_handle_cache {
public:
    class handle_desc {
    public:
        handle_desc() = delete;
        handle_desc(ze_context_handle_t remote_context,
                    const ze_ipc_mem_handle_t& handle,
                    const void* ptr,
                    size_t handle_id,
                    uint64_t remote_mem_alloc_id);
        handle_desc(const handle_desc&) = delete;
        handle_desc& operator=(const handle_desc&) = delete;
        ~handle_desc();

        const void* get_ptr() const;

    private:
        friend class mem_handle_cache;
        const ze_context_handle_t remote_context;
        const ze_ipc_mem_handle_t handle;
        const void* ptr{};
        uint64_t remote_mem_alloc_id;
        size_t handle_id;

        void close_handle() const;
    };

    using value_t = typename std::shared_ptr<const handle_desc>;

    mem_handle_cache();
    mem_handle_cache(const mem_handle_cache& other) = delete;
    mem_handle_cache& operator=(const mem_handle_cache& other) = delete;
    ~mem_handle_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ipc_handle_desc& info,
             value_t* out_value);

private:
    using key_t = typename std::
        tuple<pid_t, void*, ssize_t, ssize_t, ze_context_handle_t, ze_device_handle_t>;

    enum class key_id : size_t {
        remote_pid,
        remote_ptr,
        remote_context_id,
        remote_device_id,
        context,
        device
    };

    // LRU cache
    std::list<std::pair<key_t, value_t>> cache_list;
    std::unordered_map<key_t, decltype(cache_list.begin()), utils::tuple_hash> cache;
    std::mutex mutex;
    size_t threshold{};

    void push(ze_device_handle_t device,
              key_t&& key,
              const ipc_handle_desc& info,
              value_t* out_value);
    void make_clean(size_t limit);
    bool fd_is_valid(int fd);
};

struct ipc_get_handle_desc;

struct ipc_entry_t {
    ze_ipc_mem_handle_t handle{};
    uint64_t mem_id{};
    size_t handle_id{ ccl::utils::initial_handle_id_value };
};

class ipc_handle_cache {
public:
    using value_t = ipc_entry_t;

    ipc_handle_cache() = default;
    ipc_handle_cache(const ipc_handle_cache&) = delete;
    ipc_handle_cache& operator=(const ipc_handle_cache&) = delete;
    ~ipc_handle_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ipc_get_handle_desc& ipc_info,
             value_t* out_value);

    void evict(void* ptr);

private:
    using key_t = void*;
    std::unordered_multimap<key_t, value_t> cache;
    std::list<key_t> lru_order;
    std::mutex mutex;

    void push(ze_context_handle_t context,
              ze_device_handle_t device,
              key_t key,
              const ipc_get_handle_desc& ipc_info,
              value_t* out_value);

    void update_lru_order(key_t key);
};

class cache {
public:
    cache(size_t instance_count);
    cache(const cache&) = delete;
    cache& operator=(const cache&) = delete;
    ~cache();

    /* get */
    void get(size_t instance_idx,
             ze_module_handle_t module,
             const std::string& kernel_name,
             ze_kernel_handle_t* kernel) {
        kernels.at(instance_idx).get(module, kernel_name, kernel);
    }

    void get(size_t instance_idx,
             ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_command_list_desc_t& list_desc,
             ze_command_list_handle_t* list) {
        lists.at(instance_idx).get(context, device, list_desc, list);
    }

    void get(size_t instance_idx,
             ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_command_queue_desc_t& queue_desc,
             ze_command_queue_handle_t* queue) {
        queues.at(instance_idx).get(context, device, queue_desc, queue);
    }

    void get(size_t instance_idx,
             ze_context_handle_t context,
             const ze_event_pool_desc_t& pool_desc,
             ze_event_pool_handle_t* event_pool) {
        event_pools.at(instance_idx).get(context, pool_desc, event_pool);
    }

    void get(size_t instance_idx,
             ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
             size_t bytes,
             size_t alignment,
             void** pptr);

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const std::string& spv_name,
             ze_module_handle_t* module) {
        modules.get(context, device, spv_name, module);
    }

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ipc_handle_desc& info,
             mem_handle_cache::value_t* out_value) {
        mem_handles.get(context, device, info, out_value);
    }

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ipc_get_handle_desc& ipc_info,
             ipc_handle_cache::value_t* out_value) {
        ipc_handles.get(context, device, ipc_info, out_value);
    }

    /* push */
    void push(size_t instance_idx,
              ze_module_handle_t module,
              const std::string& kernel_name,
              ze_kernel_handle_t kernel) {
        kernels.at(instance_idx).push(module, kernel_name, kernel);
    }

    void push(size_t instance_idx,
              ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_command_list_desc_t& list_desc,
              ze_command_list_handle_t list) {
        lists.at(instance_idx).push(context, device, list_desc, list);
    }

    void push(size_t instance_idx,
              ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_command_queue_desc_t& queue_desc,
              ze_command_queue_handle_t queue) {
        queues.at(instance_idx).push(context, device, queue_desc, queue);
    }

    void push(size_t instance_idx,
              ze_context_handle_t context,
              const ze_event_pool_desc_t& pool_desc,
              ze_event_pool_handle_t event_pool) {
        event_pools.at(instance_idx).push(context, pool_desc, event_pool);
    }

    void push(size_t instance_idx,
              ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
              size_t bytes,
              size_t alignment,
              void* ptr);

    void evict(void* ptr) {
        ipc_handles.evict(ptr);
    }

private:
    const size_t instance_count;
    std::vector<kernel_cache> kernels;
    std::vector<list_cache> lists;
    std::vector<queue_cache> queues;
    std::vector<event_pool_cache> event_pools;
    std::vector<std::unique_ptr<device_mem_cache>> device_mems;
    module_cache modules{};
    mem_handle_cache mem_handles{};
    ipc_handle_cache ipc_handles{};
};

} // namespace ze
} // namespace ccl

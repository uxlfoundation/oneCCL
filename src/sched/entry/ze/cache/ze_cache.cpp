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
#include "common/global/global.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"

#include <fcntl.h>
#include <iterator>

namespace ccl {
namespace ze {

template <class map_t, class... keys_t>
bool get_from_cache(map_t& cache, typename map_t::mapped_type& object, keys_t... keys) {
    bool success{};

    if (!global_data::env().enable_ze_cache)
        return success;

    typename map_t::key_type key(keys...);
    auto key_value = cache.find(key);
    if (key_value != cache.end()) {
        object = key_value->second;
        cache.erase(key_value);
        LOG_DEBUG("loaded from cache: object: ", object);
        success = true;
    }
    return success;
}

template <class map_t, class... keys_t>
bool push_to_cache(map_t& cache, const typename map_t::mapped_type& object, keys_t... keys) {
    bool success{};

    if (!global_data::env().enable_ze_cache)
        return success;

    typename map_t::key_type key(keys...);
    auto range = cache.equal_range(key);
    auto range_len = std::distance(range.first, range.second);
    if (range_len > 0) {
        LOG_DEBUG("cache already contain ", range_len, " objects with the same key");
        for (auto i = range.first; i != range.second; ++i) {
            CCL_THROW_IF_NOT(i->second != object, "trying to push object that already exists");
        }
    }
    cache.insert({ std::move(key), object });
    LOG_DEBUG("inserted to cache: object: ", object);
    success = true;
    return success;
}

// kernel_cache
kernel_cache::~kernel_cache() {
    if (!cache.empty()) {
        LOG_WARN("kernel cache is not empty, size: ", cache.size());
        clear();
    }
}

void kernel_cache::clear() {
    LOG_DEBUG("clear kernel cache: size: ", cache.size());
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& key_value : cache) {
        ZE_CALL(zeKernelDestroy, (key_value.second));
    }
    cache.clear();
}

void kernel_cache::get(ze_module_handle_t module,
                       const std::string& kernel_name,
                       ze_kernel_handle_t* kernel) {
    CCL_THROW_IF_NOT(module);
    CCL_THROW_IF_NOT(!kernel_name.empty());
    CCL_THROW_IF_NOT(kernel);
    std::lock_guard<std::mutex> lock(mutex);
    if (!get_from_cache(cache, *kernel, module, kernel_name)) {
        create_kernel(module, kernel_name, kernel);
    }
}

void kernel_cache::push(ze_module_handle_t module,
                        const std::string& kernel_name,
                        ze_kernel_handle_t kernel) {
    CCL_THROW_IF_NOT(module);
    CCL_THROW_IF_NOT(!kernel_name.empty());
    CCL_THROW_IF_NOT(kernel);
    std::lock_guard<std::mutex> lock(mutex);
    if (!push_to_cache(cache, kernel, module, kernel_name)) {
        ZE_CALL(zeKernelDestroy, (kernel));
    }
}

// list_cache
list_cache::~list_cache() {
    if (!cache.empty()) {
        LOG_WARN("list cache is not empty, size: ", cache.size());
        clear();
    }
}

void list_cache::clear() {
    LOG_DEBUG("clear list cache: size: ", cache.size());
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& key_value : cache) {
        ZE_CALL(zeCommandListDestroy, (key_value.second));
    }
    cache.clear();
}

void list_cache::get(ze_context_handle_t context,
                     ze_device_handle_t device,
                     const ze_command_list_desc_t& list_desc,
                     ze_command_list_handle_t* list) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(list);
    std::lock_guard<std::mutex> lock(mutex);
    if (global_data::env().enable_ze_cache_cmdlists &&
        get_from_cache(
            cache, *list, context, device, list_desc.commandQueueGroupOrdinal, list_desc.flags)) {
        ZE_CALL(zeCommandListReset, (*list));
    }
    else {
        ZE_CALL(zeCommandListCreate, (context, device, &list_desc, list));
    }
}

void list_cache::push(ze_context_handle_t context,
                      ze_device_handle_t device,
                      const ze_command_list_desc_t& list_desc,
                      ze_command_list_handle_t list) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(list);
    std::lock_guard<std::mutex> lock(mutex);
    if (!global_data::env().enable_ze_cache_cmdlists ||
        !push_to_cache(
            cache, list, context, device, list_desc.commandQueueGroupOrdinal, list_desc.flags)) {
        ZE_CALL(zeCommandListDestroy, (list));
    }
}

// queue_cache
queue_cache::~queue_cache() {
    if (!cache.empty()) {
        LOG_WARN("queue cache is not empty, size: ", cache.size());
        clear();
    }
}

void queue_cache::clear() {
    LOG_DEBUG("clear queue cache: size: ", cache.size());
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& key_value : cache) {
        ZE_CALL(zeCommandQueueDestroy, (key_value.second));
    }
    cache.clear();
}

void queue_cache::get(ze_context_handle_t context,
                      ze_device_handle_t device,
                      const ze_command_queue_desc_t& queue_desc,
                      ze_command_queue_handle_t* queue) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(queue);
    std::lock_guard<std::mutex> lock(mutex);
    if (!global_data::env().enable_ze_cache_cmdqueues || !get_from_cache(cache,
                                                                         *queue,
                                                                         context,
                                                                         device,
                                                                         queue_desc.index,
                                                                         queue_desc.ordinal,
                                                                         queue_desc.flags,
                                                                         queue_desc.mode,
                                                                         queue_desc.priority)) {
        ZE_CALL(zeCommandQueueCreate, (context, device, &queue_desc, queue));
    }
}

void queue_cache::push(ze_context_handle_t context,
                       ze_device_handle_t device,
                       const ze_command_queue_desc_t& queue_desc,
                       ze_command_queue_handle_t queue) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(queue);
    std::lock_guard<std::mutex> lock(mutex);
    if (!global_data::env().enable_ze_cache_cmdqueues || !push_to_cache(cache,
                                                                        queue,
                                                                        context,
                                                                        device,
                                                                        queue_desc.index,
                                                                        queue_desc.ordinal,
                                                                        queue_desc.flags,
                                                                        queue_desc.mode,
                                                                        queue_desc.priority)) {
        ZE_CALL(zeCommandQueueDestroy, (queue));
    }
}

// event_pool_cache
event_pool_cache::~event_pool_cache() {
    if (!cache.empty()) {
        LOG_WARN("event pool cache is not empty, size: ", cache.size());
        clear();
    }
}

void event_pool_cache::clear() {
    LOG_DEBUG("clear event pool cache: size: ", cache.size());
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& key_value : cache) {
        ZE_CALL(zeEventPoolDestroy, (key_value.second));
    }
    cache.clear();
}

void event_pool_cache::get(ze_context_handle_t context,
                           const ze_event_pool_desc_t& pool_desc,
                           ze_event_pool_handle_t* event_pool) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(event_pool);
    std::lock_guard<std::mutex> lock(mutex);
    // TODO: we can potentially use pool with count >= pool_desc.count
    if (!global_data::env().enable_ze_cache_event_pools ||
        !get_from_cache(cache, *event_pool, context, pool_desc.flags, pool_desc.count)) {
        ZE_CALL(zeEventPoolCreate, (context, &pool_desc, 0, nullptr, event_pool));
    }
}

void event_pool_cache::push(ze_context_handle_t context,
                            const ze_event_pool_desc_t& pool_desc,
                            ze_event_pool_handle_t event_pool) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(event_pool);
    std::lock_guard<std::mutex> lock(mutex);
    if (!global_data::env().enable_ze_cache_event_pools ||
        !push_to_cache(cache, event_pool, context, pool_desc.flags, pool_desc.count)) {
        ZE_CALL(zeEventPoolDestroy, (event_pool));
    }
}

// module_cache
module_cache::~module_cache() {
    if (!cache.empty()) {
        LOG_WARN("module cache is not empty, size: ", cache.size());
        clear();
    }
}

void module_cache::clear() {
    LOG_DEBUG("clear module cache: size: ", cache.size());
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& key_value : cache) {
        ZE_CALL(zeModuleDestroy, (key_value.second));
    }
    cache.clear();
}

void module_cache::get(ze_context_handle_t context,
                       ze_device_handle_t device,
                       const std::string& spv_name,
                       ze_module_handle_t* module) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(!spv_name.empty());
    CCL_THROW_IF_NOT(module);
    std::lock_guard<std::mutex> lock(mutex);
    key_t key(device, spv_name);
    auto key_value = cache.find(key);
    if (key_value != cache.end()) {
        *module = key_value->second;
        LOG_DEBUG("loaded from cache: module: ", *module);
    }
    else {
        load(context, device, spv_name, module);
        cache.insert({ std::move(key), *module });
        LOG_DEBUG("inserted to cache: module: ", *module);
    }
}

void module_cache::load(ze_context_handle_t context,
                        ze_device_handle_t device,
                        const std::string& spv_name,
                        ze_module_handle_t* module) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(!spv_name.empty());
    CCL_THROW_IF_NOT(module);
    const std::string& modules_dir = global_data::env().kernel_path;
    std::string file_path = modules_dir + spv_name;
    load_module(file_path, device, context, module);
}

// mem_handle cache
mem_handle_cache::mem_handle_cache() {
    if (!global_data::env().enable_ze_cache_open_ipc_handles) {
        return;
    }

    threshold = global_data::env().ze_cache_open_ipc_handles_threshold;
    cache.reserve(threshold + 1);
    LOG_DEBUG("cache threshold: ", threshold);
}

mem_handle_cache::~mem_handle_cache() {
    if (!cache.empty()) {
        LOG_WARN("mem handle cache is not empty, size: ", cache.size());
        clear();
    }
}

void mem_handle_cache::clear() {
    std::lock_guard<std::mutex> lock(mutex);
    LOG_DEBUG("clear cache size: ", cache.size());
    make_clean(0);
    cache.clear();
    cache_list.clear();
}

void mem_handle_cache::make_clean(size_t limit) {
    static bool limit_warning_printed = false;
    // the function cleans up the cache by removing elements
    // from it until its size reaches the specified limit
    while (cache.size() > limit) {
        auto it = cache_list.end();
        --it;
        // handle will be closed from handle_desc class destructor
        cache.erase(it->first);
        cache_list.pop_back();
        if (!limit_warning_printed && limit != 0) {
            LOG_WARN(
                "Receiver cache limit is reached: mem_handle_cache size: ",
                cache.size(),
                ", limit: ",
                limit,
                ", it will remove older elements from the cache."
                " Cache size can be increased with 'CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=value'");
            limit_warning_printed = true;
        }
    }
}

void mem_handle_cache::get(ze_context_handle_t context,
                           ze_device_handle_t device,
                           const ipc_handle_desc& info,
                           value_t* out_value) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(info.remote_context_id >= 0);
    CCL_THROW_IF_NOT(info.remote_device_id >= 0);
    std::lock_guard<std::mutex> lock(mutex);

    bool found{};
    // key_t should uniquely identify opened IPC handle for the memory buffer from the remote process.
    // Furthermore, there are two possible situations we should check in advance if keys are matched:
    // * the remote buffer is freed and a new buffer is allocated to the same memory address.
    // * the handle for the remote buffer is closed with zeMemPutIpcHandle(). A new handle_id is created
    //   everytime a call to zeMemGetIpcHandle() is done, to guarantee that the handle_id is unique
    // If it is the same we re-use entry from the cache. Otherwise, we remove the entry from the cache and create a new one.
    key_t key(info.remote_pid,
              info.remote_ptr,
              info.remote_context_id,
              info.remote_device_id,
              context,
              device);
    auto key_value = cache.find(key);
    if (key_value != cache.end()) {
        value_t& value = key_value->second->second;
        const auto& handle_from_cache = value->handle;
        int fd = get_fd_from_handle(handle_from_cache);

        // We need to check that our value in the cache matches the handle_id/mem_alloc_id that
        // is being requested. If it does not match, flush the entry from the cache.
        if ((value->handle_id == info.handle_id) &&
            (value->remote_mem_alloc_id == info.remote_mem_alloc_id)) {
            if (fd_is_valid(fd)) {
                cache_list.splice(cache_list.begin(), cache_list, key_value->second);
                *out_value = value;
                found = true;
            }
            else {
                LOG_DEBUG("handle is invalid in the cache");
                cache_list.erase(key_value->second);
                cache.erase(key_value);
                found = false;
            }
        }
        else {
            LOG_DEBUG("handle with different current handle_id: ",
                      info.handle_id,
                      ", found handle_id: ",
                      value->handle_id,
                      ", current remote_mem_alloc_id: ",
                      info.remote_mem_alloc_id,
                      ", found remote_mem_alloc_id: ",
                      value->remote_mem_alloc_id);
            cache_list.erase(key_value->second);
            cache.erase(key_value);
            found = false;
        }
    }

    if (!found) {
        push(device, std::move(key), info, out_value);
    }
}

void mem_handle_cache::push(ze_device_handle_t device,
                            key_t&& key,
                            const ipc_handle_desc& info,
                            value_t* out_value) {
    make_clean(threshold);

    // no mutex here because this method is private and called from get() only
    auto remote_context_id = std::get<static_cast<size_t>(key_id::remote_context_id)>(key);
    auto remote_context = global_data::get().ze_data->contexts.at(remote_context_id);

    void* ptr{};
    ze_ipc_mem_handle_t handle = info.mem_to_ipc_handle();
    ZE_CALL(zeMemOpenIpcHandle, (remote_context, device, handle, {}, &ptr));
    *out_value = std::make_shared<const handle_desc>(
        remote_context, handle, ptr, info.handle_id, info.remote_mem_alloc_id);
    cache_list.push_front(std::make_pair(key, *out_value));
    cache.insert(std::make_pair(key, cache_list.begin()));
}

bool mem_handle_cache::fd_is_valid(int fd) {
    return fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}

mem_handle_cache::handle_desc::handle_desc(ze_context_handle_t remote_context,
                                           const ze_ipc_mem_handle_t& handle,
                                           const void* ptr,
                                           size_t handle_id,
                                           uint64_t remote_mem_alloc_id)
        : remote_context(remote_context),
          handle(handle),
          ptr(ptr),
          handle_id(handle_id),
          remote_mem_alloc_id(remote_mem_alloc_id) {}

const void* mem_handle_cache::handle_desc::get_ptr() const {
    return ptr;
}

void mem_handle_cache::handle_desc::close_handle() const {
    CCL_THROW_IF_NOT(remote_context, " no remote context");

    if (global_data::env().ze_close_ipc_wa) {
        close_handle_fd(handle);
        return;
    }
    auto fd = get_fd_from_handle(handle);
    auto res = zeMemCloseIpcHandle(remote_context, ptr);
    if (res == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
        ccl::utils::close_fd(fd);
    }
    else if (res != ZE_RESULT_SUCCESS) {
        CCL_THROW("error at zeMemCloseIpcHandle, code: ", to_string(res));
    }

    if (ccl::global_data::env().ze_ipc_exchange != ccl::ze::ipc_exchange_mode::sockets &&
        ccl::global_data::env().ze_ipc_exchange != ccl::ze::ipc_exchange_mode::none) {
        close_handle_fd(handle);
    }
}

mem_handle_cache::handle_desc::~handle_desc() {
    close_handle();
}

void ipc_handle_cache::clear() {
    LOG_DEBUG("clear ipc_handle_cache: size: ", cache.size());
    std::lock_guard<std::mutex> lock(mutex);
    for (auto& key_value : cache) {
        close_handle_fd(key_value.second.handle);
    }
    cache.clear();
    lru_order.clear();
}

ipc_handle_cache::~ipc_handle_cache() {
    if (!cache.empty()) {
        LOG_WARN("ipc_handle_cache is not empty, size: ", cache.size());
        clear();
    }
}

void ipc_handle_cache::get(ze_context_handle_t context,
                           ze_device_handle_t device,
                           const ipc_get_handle_desc& ipc_desc,
                           value_t* out_value) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(ipc_desc.ptr);

    std::lock_guard<std::mutex> lock(mutex);

    bool found = false;
    auto key_value = cache.find(ipc_desc.ptr);
    if (key_value != cache.end()) {
        value_t& value = key_value->second;
        if (value.mem_id == ipc_desc.mem_id) {
            *out_value = value;
            found = true;
        }
        else {
            cache.erase(key_value);
            lru_order.remove(ipc_desc.ptr);
            found = false;
        }
    }

    if (!found) {
        LOG_DEBUG("ipc_handle is not found in the cache");
        push(context, device, ipc_desc.ptr, ipc_desc, out_value);
        update_lru_order(ipc_desc.ptr);
    }
}

void ipc_handle_cache::push(ze_context_handle_t context,
                            ze_device_handle_t device,
                            key_t key,
                            const ipc_get_handle_desc& ipc_desc,
                            value_t* out_value) {
    // global_handle_id ensures that each handle added to the cache gets
    // a unique identifier by incrementing a global counter each time
    // a new handle is created.
    static std::atomic<size_t> global_handle_id = 0;

    while (cache.size() >= (size_t)global_data::env().ze_cache_get_ipc_handles_threshold) {
        LOG_WARN(
            "Sender cache limit is reached: cache size: ",
            cache.size(),
            ", limit: ",
            global_data::env().ze_cache_get_ipc_handles_threshold,
            ", it will clear older elements from the cache."
            " The app can crash in a L0 call due to a L0 issue. There are three possible work-arounds:"
            "\n1) Run with \"export ZE_ENABLE_TRACING_LAYER=1\". This may have a performance impact."
            "\n2) Increase the cache size 'CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=value'. The application may run out of memory."
            "\n3) Run with \"export CCL_ZE_CLOSE_IPC_WA=1\". This may leak L0 internal graph objects");
        auto lru_key = lru_order.back();
        auto lru_iter = cache.find(lru_key);

        if (lru_iter != cache.end()) {
            ze_memory_allocation_properties_t alloc_props = ccl::ze::default_alloc_props;
            ZE_CALL(zeMemGetAllocProperties, (context, lru_key, &alloc_props, &device));
            if (alloc_props.type != ZE_MEMORY_TYPE_UNKNOWN &&
                lru_iter->second.mem_id == alloc_props.id) {
                ZE_CALL(zeMemPutIpcHandle, (context, lru_iter->second.handle));
            }
            else {
                LOG_WARN(
                    "ipc_handle_cache: handle type is unexpected. Not calling zeMemPutIpcHandle, mem_id: ",
                    lru_iter->second.mem_id);
            }
            lru_order.pop_back();
            cache.erase(lru_iter);
        }
    }

    ZE_CALL(zeMemGetIpcHandle, (context, ipc_desc.ptr, &out_value->handle));

    out_value->mem_id = ipc_desc.mem_id;
    // every call to zeMemGetIpcHandle results in increment of
    // global_handle_id to be unique
    out_value->handle_id = global_handle_id++;

    cache.insert({ key, *out_value });
    update_lru_order(key);
}

void ipc_handle_cache::update_lru_order(key_t key) {
    lru_order.remove(key);
    lru_order.push_front(key);
}

void ipc_handle_cache::evict(void* ptr) {
    CCL_THROW_IF_NOT(ptr);

    std::lock_guard<std::mutex> lock(mutex);

    auto key_value = cache.find(ptr);
    if (key_value != cache.end()) {
        value_t& value = key_value->second;
        ze_context_handle_t context = ccl::global_data::get().ze_data->contexts[0];
        ze_device_handle_t device;
        ze_memory_allocation_properties_t alloc_props = ccl::ze::default_alloc_props;
        ZE_CALL(zeMemGetAllocProperties, (context, ptr, &alloc_props, &device));

        if (value.mem_id == alloc_props.id) {
            LOG_DEBUG("ipc_handle is evicted in the cache", ptr);
            cache.erase(key_value);
            lru_order.remove(ptr);
        }
    }
}

// cache
cache::cache(size_t instance_count)
        : instance_count(instance_count),
          kernels(instance_count),
          lists(instance_count),
          queues(instance_count),
          event_pools(instance_count) {
    if (global_data::env().ze_device_cache_policy == device_cache_policy_mode::chunk) {
        for (size_t i = 0; i < instance_count; ++i) {
            device_mems.push_back(std::make_unique<chunk_device_mem_cache>());
        }
    }
    else if (global_data::env().ze_device_cache_policy == device_cache_policy_mode::plain) {
        for (size_t i = 0; i < instance_count; ++i) {
            device_mems.push_back(std::make_unique<plain_device_mem_cache>());
        }
    }
    LOG_DEBUG("device cache policy mode: ",
              device_cache_policy_names[global_data::env().ze_device_cache_policy],
              ", instance_count: ",
              instance_count);
}

cache::~cache() {
    for (size_t i = 0; i < instance_count; ++i) {
        kernels[i].clear();
        lists[i].clear();
        queues[i].clear();
        event_pools[i].clear();
        device_mems[i]->clear();
    }

    modules.clear();
    mem_handles.clear();
    ipc_handles.clear();
}

} // namespace ze
} // namespace ccl

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

#include "common/global/global.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/cache/ze_device_cache.hpp"

namespace ccl {
namespace ze {

static size_t current_allocated_memory = 0;
static std::unordered_map<void*, size_t> recorded_allocations;

void device_allocate(ze_context_handle_t context,
                     const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                     size_t bytes,
                     size_t alignment,
                     ze_device_handle_t device,
                     void** pptr) {
    current_allocated_memory += bytes;
    LOG_DEBUG("|MEMLOG| Allocating: ",
              bytes / 1024,
              "KB. Current memory footprint: ",
              current_allocated_memory / 1024,
              "KB");

    ZE_CALL(zeMemAllocDevice, (context, &device_mem_alloc_desc, bytes, alignment, device, pptr));
    auto [_, inserted] = recorded_allocations.try_emplace(*pptr, bytes);

    if (!inserted) {
        LOG_WARN(
            "Could not record device allocation. Memory footprint might not be representing real consumption!");
    }
}

void device_allocate_shared(ze_context_handle_t context,
                            const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                            const ze_host_mem_alloc_desc_t& host_mem_alloc_desc,
                            size_t bytes,
                            size_t alignment,
                            ze_device_handle_t device,
                            void** pptr) {
    current_allocated_memory += bytes;
    LOG_DEBUG("|MEMLOG| Allocating: ",
              bytes / 1024,
              "KB. Current memory footprint: ",
              current_allocated_memory / 1024,
              "KB");

    ZE_CALL(
        zeMemAllocShared,
        (context, &device_mem_alloc_desc, &host_mem_alloc_desc, bytes, alignment, device, pptr));
    auto [_, inserted] = recorded_allocations.try_emplace(*pptr, bytes);

    if (!inserted) {
        LOG_WARN(
            "Could not record device allocation. Memory footprint might not be representing real consumption!");
    }
}

void device_free(ze_context_handle_t context, void* ptr) {
    auto recorded_allocation = recorded_allocations.find(ptr);

    // bytes = ccl::utils::invalid_bytes_value indicate an error in our recorded_allocations map
    // this could be caused by improper usage of the device memory wrapper
    size_t bytes = ccl::utils::invalid_bytes_value;

    if (recorded_allocation != recorded_allocations.end()) {
        bytes = recorded_allocation->second;
        current_allocated_memory -= bytes;
        recorded_allocations.erase(recorded_allocation);
    }
    else {
        LOG_WARN(
            "Could not record device allocation. Memory footprint might not be representing real consumption!");
    }

    LOG_DEBUG("|MEMLOG| Freeing: ",
              bytes / 1024,
              "KB. Current memory footprint: ",
              current_allocated_memory / 1024,
              "KB");
    ZE_CALL(zeMemFree, (context, ptr));
}

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

// plain_device_mem_cache
plain_device_mem_cache::~plain_device_mem_cache() {
    if (!cache.empty()) {
        LOG_WARN("device memory cache is not empty, size: ", cache.size());
        clear();
    }
}

void plain_device_mem_cache::clear() {
    LOG_DEBUG("clear plain device memory cache: size: ", cache.size());
    std::lock_guard<std::mutex> lock(mutex);
    //for (auto& key_value : cache) {
    // TODO: there is a segfault on this call, when ~cache is invoked w/ or w/0 api cache.
    // But it passes, when CCL_ZE_CACHE=0 (calls of zeMemAllocDevice and ZeMemFree happen on every iteration).
    // We don't control destroying phase and may be key_value.second (mem_ptr) is already away to free?
    // ZE_CALL(zeMemFree, (std::get<0>(key_value.first), key_value.second))
    //}
    cache.clear();
}

void plain_device_mem_cache::get(ze_context_handle_t context,
                                 ze_device_handle_t device,
                                 const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                                 size_t bytes,
                                 size_t alignment,
                                 void** pptr) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(pptr);
    std::lock_guard<std::mutex> lock(mutex);
    if (!get_from_cache(cache,
                        *pptr,
                        context,
                        device,
                        bytes,
                        device_mem_alloc_desc.flags,
                        device_mem_alloc_desc.ordinal)) {
        device_allocate(context, device_mem_alloc_desc, bytes, alignment, device, pptr);
    }
}

void plain_device_mem_cache::push(ze_context_handle_t context,
                                  ze_device_handle_t device,
                                  const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                                  size_t bytes,
                                  size_t alignment,
                                  void* ptr) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(ptr);
    std::lock_guard<std::mutex> lock(mutex);
    if (!push_to_cache(cache,
                       ptr,
                       context,
                       device,
                       bytes,
                       device_mem_alloc_desc.flags,
                       device_mem_alloc_desc.ordinal)) {
        device_free(context, ptr);
    }
}

// chunk implementation
chunk_device_mem_cache::~chunk_device_mem_cache() {
    if (!memory_chunks.empty()) {
        LOG_WARN("device memory cache is not empty, size: ", memory_chunks.size());
        clear();
    }
}

void chunk_device_mem_cache::clear() {
    LOG_DEBUG("clear plain device memory cache: size: ", memory_chunks.size());
    std::lock_guard<std::mutex> lock(mutex);

    // TODO: there is a segfault on this call, when ~cache is invoked w/ or w/0 api cache.
    // free all memory chunks and reset the vector.
    // for (auto& chunk : memory_chunks) {
    //     ZE_CALL(zeMemFree, (std::get<0>(key_value.first), chunk.base_ptr));
    // }
    memory_chunks.clear();
}

// get a memory chunk from the cache
void chunk_device_mem_cache::get(ze_context_handle_t context,
                                 ze_device_handle_t device,
                                 const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                                 size_t bytes,
                                 size_t alignment,
                                 void** pptr) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(pptr);

    std::lock_guard<std::mutex> lock(mutex);

    if (global_data::env().enable_ze_cache) {
        // find a suitable memory chunk that has enough space.
        size_t block_size = bytes;
        for (auto& chunk : memory_chunks) {
            if (chunk.block_size < block_size) {
                LOG_DEBUG("skip chunks with different block size: chunk.block_size: ",
                          chunk.block_size,
                          ", block_size: ",
                          block_size);
                continue;
            }

            for (size_t block_idx = 0; block_idx < chunk.num_blocks; block_idx++) {
                if (!chunk.used_blocks[block_idx]) {
                    // found a free block in the chunk, use it.
                    chunk.used_blocks[block_idx] = true;
                    *pptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(chunk.base_ptr) +
                                                    block_idx * chunk.block_size);
                    LOG_DEBUG("loaded from cache: object: ", *pptr);
                    return;
                }
            }
        }

        // if no suitable block found, allocate a new chunk and use the first block.
        allocate_new_chunk(device_mem_alloc_desc, context, device, bytes, alignment);
        *pptr = memory_chunks.back().base_ptr;
        LOG_DEBUG("allocated new chunk: object: ", *pptr);

        // check memory usage and evict the smallest chunk if necessary
        if (get_total_cache_size() > global_data::env().ze_device_cache_upper_limit) {
            if (global_data::env().ze_device_cache_evict_smallest) {
                evict_smallest_chunk(context);
            }
            else {
                evict_largest_chunk(context);
            }
        }
    }
    else {
        device_allocate(context, device_mem_alloc_desc, bytes, alignment, device, pptr);
        LOG_DEBUG("allocated directly: object: ", *pptr);
    }
}

// push a memory chunk back to the cache
void chunk_device_mem_cache::push(ze_context_handle_t context,
                                  ze_device_handle_t device,
                                  const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                                  size_t bytes,
                                  size_t alignment,
                                  void* ptr) {
    CCL_THROW_IF_NOT(context);
    CCL_THROW_IF_NOT(device);
    CCL_THROW_IF_NOT(ptr);

    std::lock_guard<std::mutex> lock(mutex);

    if (global_data::env().enable_ze_cache) {
        // find the corresponding memory chunk and mark the block as free.
        for (auto& chunk : memory_chunks) {
            if (ptr >= chunk.base_ptr && ptr < (static_cast<char*>(chunk.base_ptr) + chunk.size)) {
                size_t offset =
                    reinterpret_cast<uintptr_t>(ptr) - reinterpret_cast<uintptr_t>(chunk.base_ptr);
                size_t block_index = offset / chunk.block_size;
                chunk.used_blocks[block_index] = false;
                LOG_DEBUG("pushed to cache: object: ", ptr);
                // check memory usage and evict the smallest chunk if necessary.
                if (get_total_cache_size() > global_data::env().ze_device_cache_upper_limit) {
                    if (global_data::env().ze_device_cache_evict_smallest) {
                        evict_smallest_chunk(context);
                    }
                    else {
                        evict_largest_chunk(context);
                    }
                }
                return;
            }
        }
    }

    // if the pointer does not belong to any existing chunk, free it directly.
    device_free(context, ptr);
    LOG_DEBUG("freed directly: object: ", ptr);
}

int chunk_device_mem_cache::get_total_cache_size() const {
    long total_size = 0;
    for (const auto& chunk : memory_chunks) {
        total_size += chunk.size;
    }
    return total_size;
}

template <typename ComparisonFunction>
void chunk_device_mem_cache::evict_chunk(ze_context_handle_t context, ComparisonFunction compFunc) {
    if (memory_chunks.empty()) {
        return;
    }

    auto chunk_it =
        std::max_element(memory_chunks.begin(),
                         memory_chunks.end(),
                         [this, &compFunc](const memory_chunk& a, const memory_chunk& b) {
                             return compFunc(a, b) && !is_chunk_used(a);
                         });

    if (chunk_it != memory_chunks.end() && !is_chunk_used(*chunk_it)) {
        device_free(context, chunk_it->base_ptr);
        memory_chunks.erase(chunk_it);
    }
}

void chunk_device_mem_cache::evict_smallest_chunk(ze_context_handle_t context) {
    evict_chunk(context, [](const memory_chunk& a, const memory_chunk& b) {
        return a.size < b.size;
    });
}

void chunk_device_mem_cache::evict_largest_chunk(ze_context_handle_t context) {
    evict_chunk(context, [](const memory_chunk& a, const memory_chunk& b) {
        return a.size > b.size;
    });
}

bool chunk_device_mem_cache::is_chunk_used(const memory_chunk& chunk) const {
    return std::any_of(chunk.used_blocks.begin(), chunk.used_blocks.end(), [](bool used) {
        return used;
    });
}

// allocate a new memory chunk
void chunk_device_mem_cache::allocate_new_chunk(
    const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
    ze_context_handle_t context,
    ze_device_handle_t device,
    size_t bytes,
    size_t alignment) {
    // define the chunk size as a multiple of the block size to avoid fragmentation.
    size_t block_size = bytes;
    size_t num_blocks_per_chunk =
        global_data::env()
            .ze_device_cache_num_blocks_in_chunk; // you can adjust this value based on your needs.
    size_t chunk_size = block_size * num_blocks_per_chunk;

    // allocate the memory chunk and create the memory_chunk structure.
    void* base_ptr;
    device_allocate(context, device_mem_alloc_desc, chunk_size, alignment, device, &base_ptr);
    memory_chunks.emplace_back(chunk_size, block_size);
    memory_chunks.back().base_ptr = base_ptr;
    memory_chunks.back().used_blocks[0] = true; // mark the first block as used
}

// cache
void cache::get(size_t instance_idx,
                ze_context_handle_t context,
                ze_device_handle_t device,
                const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                size_t bytes,
                size_t alignment,
                void** pptr) {
    device_mems.at(instance_idx % device_mems.size())
        ->get(context, device, device_mem_alloc_desc, bytes, alignment, pptr);
}

void cache::push(size_t instance_idx,
                 ze_context_handle_t context,
                 ze_device_handle_t device,
                 const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                 size_t bytes,
                 size_t alignment,
                 void* ptr) {
    device_mems.at(instance_idx % device_mems.size())
        ->push(context, device, device_mem_alloc_desc, bytes, alignment, ptr);
}

} // namespace ze
} // namespace ccl

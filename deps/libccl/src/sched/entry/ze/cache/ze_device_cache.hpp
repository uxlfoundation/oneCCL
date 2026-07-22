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

#include <unordered_map>

namespace ccl {
namespace ze {

void device_allocate(ze_context_handle_t context,
                     const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                     size_t bytes,
                     size_t alignment,
                     ze_device_handle_t device,
                     void** pptr);

void device_allocate_shared(ze_context_handle_t context,
                            const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                            const ze_host_mem_alloc_desc_t& host_mem_alloc_desc,
                            size_t bytes,
                            size_t alignment,
                            ze_device_handle_t device,
                            void** pptr);

void device_free(ze_context_handle_t context, void* ptr);

enum class device_cache_policy_mode : int { plain, chunk, none };
static std::map<device_cache_policy_mode, std::string> device_cache_policy_names = {
    std::make_pair(device_cache_policy_mode::plain, "plain"),
    std::make_pair(device_cache_policy_mode::chunk, "chunk"),
    std::make_pair(device_cache_policy_mode::none, "none")
};

class device_mem_cache {
public:
    device_mem_cache() = default;
    virtual ~device_mem_cache() = default;

    virtual void clear() = 0;

    virtual void get(ze_context_handle_t context,
                     ze_device_handle_t device,
                     const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                     size_t bytes,
                     size_t alignment,
                     void** pptr) = 0;

    virtual void push(ze_context_handle_t context,
                      ze_device_handle_t device,
                      const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                      size_t bytes,
                      size_t alignment,
                      void* ptr) = 0;
};

class plain_device_mem_cache : public device_mem_cache {
public:
    plain_device_mem_cache() = default;
    plain_device_mem_cache(const plain_device_mem_cache&) = delete;
    plain_device_mem_cache& operator=(const plain_device_mem_cache&) = delete;
    ~plain_device_mem_cache();

    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
             size_t bytes,
             size_t alignment,
             void** pptr);

    void push(ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
              size_t bytes,
              size_t alignment,
              void* ptr);

private:
    using key_t = typename std::tuple<ze_context_handle_t,
                                      ze_device_handle_t,
                                      size_t,
                                      ze_device_mem_alloc_flags_t,
                                      uint32_t>;
    using value_t = void*;
    std::unordered_multimap<key_t, value_t, utils::tuple_hash> cache;
    std::mutex mutex;
};

// chunk_device_mem_cache
class chunk_device_mem_cache : public device_mem_cache {
public:
    chunk_device_mem_cache() = default;
    chunk_device_mem_cache(const chunk_device_mem_cache&) = delete;
    chunk_device_mem_cache& operator=(const chunk_device_mem_cache&) = delete;
    ~chunk_device_mem_cache();
    void clear();

    void get(ze_context_handle_t context,
             ze_device_handle_t device,
             const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
             size_t bytes,
             size_t alignment,
             void** pptr);

    void push(ze_context_handle_t context,
              ze_device_handle_t device,
              const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
              size_t bytes,
              size_t alignment,
              void* ptr);

private:
    struct memory_chunk {
        size_t size;
        size_t block_size;
        size_t num_blocks;
        void* base_ptr;
        std::vector<bool> used_blocks;

        memory_chunk(size_t chunk_size, size_t block_size)
                : size(chunk_size),
                  block_size(block_size),
                  num_blocks(chunk_size / block_size),
                  base_ptr(nullptr),
                  used_blocks(chunk_size / block_size, false) {}
    };

    void allocate_new_chunk(const ze_device_mem_alloc_desc_t& device_mem_alloc_desc,
                            ze_context_handle_t context,
                            ze_device_handle_t device,
                            size_t bytes,
                            size_t alignment);

    template <typename ComparisonFunction>
    void evict_chunk(ze_context_handle_t context, ComparisonFunction compFunc);
    void evict_smallest_chunk(ze_context_handle_t context);
    void evict_largest_chunk(ze_context_handle_t context);

    int get_total_cache_size() const;
    bool is_chunk_used(const memory_chunk& chunk) const;
    std::vector<memory_chunk> memory_chunks;
    std::mutex mutex;
};

} // namespace ze
} // namespace ccl

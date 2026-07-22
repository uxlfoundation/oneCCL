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

#include <cstddef>

#include "common/api_wrapper/ze_api_wrapper.hpp"
#include "common/utils/sync_object.hpp"

class ccl_sched;
class ccl_comm;

class sched_group {
private:
    static size_t last_group_id;

    const size_t id;
    size_t chunks_count = 0;
    size_t submitted_chunks = 0;

    ccl_sched* sched = nullptr;
    ccl_comm* comm = nullptr;

    bool is_parallel = true;
    bool last_chunk_barrier_enabled = true;

    size_t alloc_count = 0;
    void* memory_context = nullptr;
    void* const memory_context_base;
    size_t memory_context_size = 0;

    std::shared_ptr<sync_object> sync{};
    std::mutex allocation_mutex{};

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    std::vector<ze_event_handle_t> events{};
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
public:
    sched_group() = delete;
    sched_group(const sched_group&);
    sched_group(ccl_sched* sched, ccl_comm* comm, void* memory_context, size_t memory_context_size);

    // Delete copy assignment operator
    sched_group operator=(const sched_group) = delete;
    sched_group& operator=(const sched_group&) = delete;

    // Delete move copy constructor and move assignment operator
    sched_group(const sched_group&&) = delete;
    sched_group& operator=(sched_group&&) = delete;

    bool operator==(const sched_group& other) const;

    ~sched_group() {}

    size_t get_id() const;
    void* allocate(size_t bytes, size_t alignment);
    bool parallelizable();
    bool is_pointer_within_memory_context(void* ptr) const;
    void set_sync_obj(std::shared_ptr<sync_object>);
    void disable_parallel_execution();
    void disable_last_chunk_barrier();
    void increase_chunk_count();
    void register_chunk_start(ccl_sched* subsched);
    void register_chunk_end(ccl_sched* subsched);

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    void register_chunk_end(ccl_sched* subsched, ze_event_handle_t event);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
};

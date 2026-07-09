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

#include <memory>
#include <unordered_map>

#include "comm/comm.hpp"
#include "common/log/log.hpp"
#include "sched.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/sched_group.hpp"

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "sched/entry/ze/ze_cmdlist_timestamp.hpp"
#include "sched/entry/ze/ze_cmdlist_event_signal_entry.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

sched_group::sched_group(ccl_sched* sched,
                         ccl_comm* comm,
                         void* memory_context,
                         size_t memory_context_size)
        : id(last_group_id++),
          chunks_count(0),
          submitted_chunks(0),
          sched(sched),
          comm(comm),
          is_parallel(true),
          alloc_count(0),
          memory_context(memory_context),
          memory_context_base(memory_context),
          memory_context_size(memory_context_size) {}

// Creates new instance of `sched_group` using the same
// `memory_context` as `other` with clean state of allocator.
// New instance will not account for previous allocations done
// in `other` schedule group.
sched_group::sched_group(const sched_group& other)
        : sched_group(other.sched,
                      other.comm,
                      other.memory_context_base,
                      other.memory_context_size) {}

bool sched_group::operator==(const sched_group& other) const {
    return (id == other.id);
}

size_t sched_group::get_id() const {
    return id;
}

void* sched_group::allocate(size_t bytes, size_t alignment) {
    std::lock_guard<std::mutex> lock(allocation_mutex);

    if (memory_context == nullptr) {
        LOG_DEBUG("|GROUPS| memory_context == nullptr, fallback to default schedule allocator");
        return nullptr;
    }

    void* ret = nullptr;
    void* old_ptr = memory_context;
    if (std::align(alignment, bytes, memory_context, memory_context_size)) {
        ret = memory_context;
        memory_context = static_cast<char*>(memory_context) + bytes;
        memory_context_size -= bytes;
        LOG_DEBUG("|GROUPS| Aligned allocation by: ", (size_t)ret - (size_t)old_ptr);
    }
    else {
        LOG_DEBUG(
            "|GROUPS| Could not allocate using supplied memory context! Falling back to default schedule alocator.");
        return nullptr;
    }

    alloc_count++;
    LOG_DEBUG("|GROUPS| Allocating[",
              id,
              "] => (offset: ",
              (size_t)ret - (size_t)memory_context_base,
              ", size: ",
              bytes,
              ", alloc_count: ",
              alloc_count,
              ")");
    return ret;
}

bool sched_group::parallelizable() {
    return is_parallel;
}

bool sched_group::is_pointer_within_memory_context(void* ptr) const {
    std::uintptr_t start = reinterpret_cast<std::uintptr_t>(memory_context_base);
    std::uintptr_t end = start + memory_context_size;
    std::uintptr_t checkPtr = reinterpret_cast<std::uintptr_t>(ptr);
    return checkPtr >= start && checkPtr < end;
}

void sched_group::set_sync_obj(std::shared_ptr<sync_object> new_sync) {
    sync = std::move(new_sync);
}

void sched_group::disable_parallel_execution() {
    is_parallel = false;
}

void sched_group::disable_last_chunk_barrier() {
    last_chunk_barrier_enabled = false;
}

void sched_group::increase_chunk_count() {
    chunks_count++;
}

void sched_group::register_chunk_start(ccl_sched* subsched) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ccl::global_data::get().env().debug_timestamps_level > 0) {
        std::ostringstream timestamp_text;
        std::vector<ze_event_handle_t> empty_events;
        LOG_DEBUG("Adding timestamp");
        timestamp_text << "Group[" << id << "] => "
                       << "Starting chunk " << submitted_chunks << "";
        entry_factory::create<ze_cmdlist_timestamp>(
            subsched, comm, timestamp_text.str(), empty_events);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    submitted_chunks++;
}

void sched_group::register_chunk_end(ccl_sched* subsched) {
    if (submitted_chunks == chunks_count) {
        LOG_DEBUG("|GROUPS| Group[", id, "] building complete");
    }
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

void sched_group::register_chunk_end(ccl_sched* subsched, ze_event_handle_t event) {
    if (sync) {
        entry_factory::create<sync_entry>(subsched, sync);
    }

    if (submitted_chunks == chunks_count && last_chunk_barrier_enabled) {
        LOG_DEBUG("|GROUPS| Group[", id, "] building complete, synchronizing commandlist");
        ze_event_handle_t chunk_completion_event = subsched->get_memory().event_manager->create();
        std::vector<ze_event_handle_t> dependencies = { event };
        entry_factory::create<ze_cmdlist_event_signal_entry>(
            subsched, comm, chunk_completion_event, dependencies);

        if (ccl::global_data::get().env().debug_timestamps_level > 0) {
            std::ostringstream timestamp_text;
            std::vector<ze_event_handle_t> wait_events{ chunk_completion_event };

            timestamp_text << "Group[" << id << "] => "
                           << "Finished";
            entry_factory::create<ze_cmdlist_timestamp>(
                subsched, comm, timestamp_text.str(), wait_events);
        }
    }
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

template <>
struct std::hash<sched_group> {
    std::size_t operator()(const sched_group& group) const {
        return std::hash<size_t>()(group.get_id());
    }
};

size_t sched_group::last_group_id = 0;

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

#include "ze_events.h"
#include "ze_helpers.h"
#include "ze_inc.h"
#include "ze_fence.h"
#include "ze_sync.h"

class ze_cmd_queue_t;
class ze_device_t;

// Use RELAXED_ORDERING when:
//  You have many independent operations (parallel memory copies)
//  Performance is critical and you can handle potential reordering
//  You're explicitly managing dependencies with events
//
// Use IN_ORDER when:
//  You need predictable, sequential execution
//  You want simplified programming without manual synchronization
//  You're using immediate command lists for simple sequential workflows
//  Debugging or when execution order matters for correctness

// Default Behavior (No Flags)
//
// Command Ordering:
//  Submission order preserved: Commands execute in the order they were appended to the command list
//  No automatic dependencies: The driver does NOT automatically insert synchronization between commands
//  Manual synchronization required: You must explicitly use events to enforce dependencies when needed
//
// Key Characteristics:
//  Predictable submission order: Commands A, B, C will be submitted to the GPU in that order
//  Potential overlap: Commands can still overlap in execution if they don't conflict and hardware allows it
//  No forced serialization: Unlike IN_ORDER, there's no automatic "wait for previous command to complete"
//  No reordering optimization: Unlike RELAXED_ORDERING, the driver won't actively reorder for performance

class ze_cmd_list_t : public ze_handle_t<ze_command_list_handle_t>
{
protected:
    ze_cmd_list_t(ze_engine_t engine) : ZE_HANDLE(ze_command_list_handle_t, zeCommandListDestroy), type(engine) {}

public:
    ze_cmd_list_t(const ze_device_t& device,
              ze_engine_t engine,
              bool relaxed_order = false, // ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING
              bool max_perf = false,      // ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT
              bool single_engine = false, // ZE_COMMAND_LIST_FLAG_EXPLICIT_ONLY
              bool in_order = false,      // ZE_COMMAND_LIST_FLAG_IN_ORDER
              bool cloneable = true);     // ZE_COMMAND_LIST_FLAG_EXP_CLONEABLE

// ----   commands appended to the list  ----

    virtual ze_result_t mem_prefetch(const void* ptr, size_t size);
    virtual ze_result_t mem_advise(const ze_device_t& device, const void* ptr, size_t size, ze_memory_advice_t advice);
    virtual ze_result_t mem_copy(void* dst, const void* src, size_t size, const ze_sync_t& sync = {});
    virtual ze_result_t barrier(const ze_sync_t& sync = {});
    virtual ze_result_t mem_fill(void* dst, const void* pattern, size_t pattern_size, size_t size_bytes, const ze_sync_t& sync = {});
    virtual ze_result_t signal_event(const ze_event_t& event);
    virtual ze_result_t wait_event(const ze_event_t& event);
    virtual ze_result_t event_reset(const ze_event_t& event);

// ------------------------------------------

    virtual ze_result_t close() const;
    virtual ze_result_t reset();
    virtual ze_result_t execute(const ze_cmd_queue_t& queue, bool _close = true, const ze_fence_t& fence = {}) const;
    virtual bool is_immediate() const { return false; }
    virtual void synchronize(const ze_cmd_queue_t& queue) const {
        LOG_ERROR("synchronize with queue is not supported for non-immediate lists - use ze_cmd_queue_t::synchronize instead");
    };
    const ze_engine_t type = ze_engine_t::Unknown; // Default to Unknown, will be set based on properties
};

using ze_cmd_list_ptr = std::shared_ptr<ze_cmd_list_t>;

class ze_immediate_list_t : public ze_cmd_list_t
{
protected:
    ze_immediate_list_t(const ze_device_t& device,
                        ze_engine_t engine,
                        bool in_order,
                        bool high_priority,
                        bool explicit_que,
                        uint32_t queue_index,
                        ze_command_queue_mode_t mode = ZE_COMMAND_QUEUE_MODE_DEFAULT);

public:
    ze_immediate_list_t(const ze_device_t& device, ze_engine_t engine, bool in_order, bool high_priority)
        : ze_immediate_list_t(device, engine, in_order, high_priority, false, 0) {}

    ze_immediate_list_t(const ze_device_t& device, ze_engine_t engine, uint32_t queue_index, bool in_order, bool high_priority)
        : ze_immediate_list_t(device, engine, in_order, high_priority, true, queue_index) {}

    virtual ze_result_t close() const override { return ZE_RESULT_SUCCESS; } // Immediate lists do not need to be closed;
    virtual ze_result_t reset() override { return ZE_RESULT_SUCCESS; } // Immediate lists do not need to be reset
    virtual ze_result_t execute(const ze_cmd_queue_t& queue, bool _close = true, const ze_fence_t& fence = {}) const override { return ZE_RESULT_SUCCESS; }
    ze_result_t execute_cmd_list_as_immediate(ze_cmd_list_ptr cmd_list, const ze_sync_t& sync = {}) const;
    virtual bool is_immediate() const override { return true; }
    virtual void synchronize(const ze_cmd_queue_t& queue) const override {
        ZE_NO_RET(zeCommandListHostSynchronize(m_handle, UINT64_MAX));
    };
};

using ze_immediate_list_ptr = std::shared_ptr<ze_immediate_list_t>;
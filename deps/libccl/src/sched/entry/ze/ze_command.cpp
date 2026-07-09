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

#include "sched/entry/factory/entry_factory.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/ze_command.hpp"
#include "common/global/global.hpp"

void ze_cmd_memory_copy::ze_call() {
    ZE_CALL(zeCommandListAppendMemoryCopy,
            (cmdlist, //ze_command_list_handle_t command_list_handle,
             dstptr, //void *dstptr,
             srcptr, //const void *srcptr,
             size, //size_t size,
             signal_event, //ze_event_handle_t signal_event,
             wait_events.size(), //uint32_t numwait_events,
             wait_events.data() //ze_event_handle_t *phwait_events);
             ));
}

void ze_cmd_launch_kernel::ze_call() {
    kernel.actually_call_ze(cmdlist, signal_event, wait_events);
}

void ze_cmd_barrier::ze_call() {
    ZE_CALL(zeCommandListAppendBarrier,
            (cmdlist, //ze_command_list_handle_t command_list_handle,
             signal_event, //ze_event_handle_t signal_event,
             wait_events.size(), //uint32_t numwait_events,
             wait_events.data() //ze_event_handle_t *phwait_events)
             ));
}

void ze_cmd_mem_range_barrier::ze_call() {
    ZE_CALL(
        zeCommandListAppendMemoryRangesBarrier,
        (cmdlist, //ze_command_list_handle_t command_list_handle,
         range_sizes.size(), //uint32_t numRanges,  [in] number of memory ranges
         &(range_sizes
               [0]), // const size_t* pRangerange_sizes, [in][range(0, numRanges)] array of range_sizes of memory range
         &(ranges[0]), // const void** pRanges, [in][range(0, numRanges)] array of memory ranges
         signal_event, //ze_event_handle_t signal_event,
         wait_events.size(), //uint32_t numwait_events,
         wait_events.data() //ze_event_handle_t *phwait_events)
         ));
}

void ze_cmd_wait_on_events::ze_call() {
    ZE_CALL(zeCommandListAppendWaitOnEvents,
            (cmdlist, //ze_command_list_handle_t command_list_handle,
             wait_events.size(), //uint32_t numwait_events,
             wait_events.data() //ze_event_handle_t *phwait_events)
             ));
}

void ze_cmd_signal_event::ze_call() {
    ZE_CALL(zeCommandListAppendSignalEvent,
            (cmdlist, //ze_command_list_handle_t command_list_handle,
             signal_event));
}

void ze_cmd_timestamp::ze_call() {
    ZE_CALL(zeCommandListAppendWriteGlobalTimestamp,
            (cmdlist, timestamp_ptr, signal_event, wait_events.size(), wait_events.data()));
}

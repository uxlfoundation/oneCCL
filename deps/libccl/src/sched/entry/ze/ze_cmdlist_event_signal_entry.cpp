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

#include "sched/entry/ze/ze_cmdlist_event_signal_entry.hpp"

ze_cmdlist_event_signal_entry::ze_cmdlist_event_signal_entry(
    ccl_sched* sched,
    ccl_comm* comm,
    ze_event_handle_t signal_event,
    const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, comm, 0 /* request additional events */),
          signal_event(signal_event) {
    CCL_THROW_IF_NOT(sched, "no sched");
}

void ze_cmdlist_event_signal_entry::init_ze_hook() {
    // Before signalling any event inside commandlist we append a barrier
    // to ensure that all previously appended commands are completed.
    // (empty `wait_events` means we wait for all prior tasks
    // https://spec.oneapi.io/level-zero/latest/core/api.html#zecommandlistappendbarrier)
    ZE_APPEND_CALL(ze_cmd_barrier, get_comp_list(), entry_event, (ze_events_t){});

    if (signal_event != nullptr) {
        ZE_APPEND_CALL(ze_cmd_signal_event, get_comp_list(), signal_event);
    }
}

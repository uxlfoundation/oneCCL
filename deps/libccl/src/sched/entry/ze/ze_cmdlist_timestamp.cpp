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

#include "sched/entry/ze/ze_cmdlist_timestamp.hpp"

ze_cmdlist_timestamp::ze_cmdlist_timestamp(ccl_sched* sched,
                                           ccl_comm* comm,
                                           std::string text,
                                           const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, comm, 0 /* request additional events */),
          text(std::move(text)) {
    CCL_THROW_IF_NOT(sched, "no sched");
    CCL_THROW_IF_NOT(comm, "no comm");
}

void ze_cmdlist_timestamp::init_ze_hook() {
    uint64_t* timestamp_ptr = nullptr;
    ze_device_mem_alloc_desc_t device_desc{};
    ze_host_mem_alloc_desc_t host_desc{};

    device_allocate_shared(context, device_desc, host_desc, 8, 8, device, (void**)&timestamp_ptr);
    ZE_APPEND_CALL(ze_cmd_timestamp, get_comp_list(), timestamp_ptr, entry_event, wait_events);
    ccl::global_data::get().timestamp_manager->add_timestamp(text, timestamp_ptr);
}

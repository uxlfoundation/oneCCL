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

#include "common/api_wrapper/ze_api_wrapper.hpp"
#include "sched/entry/ze/ze_membarrier_entry.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

ze_membarrier_entry::ze_membarrier_entry(ccl_sched* sched,
                                         size_t range_size,
                                         const void* range,
                                         const std::vector<ze_event_handle_t>& wait_events)
        : ze_membarrier_entry(sched,
                              std::vector<size_t>({ range_size }),
                              std::vector<const void*>({ range }),
                              wait_events) {}

ze_membarrier_entry::ze_membarrier_entry(ccl_sched* sched,
                                         const std::vector<size_t>& range_sizes,
                                         const std::vector<const void*>& ranges,
                                         const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, nullptr /*comm*/, 1 /*add_event_count*/),
          range_sizes(range_sizes),
          ranges(ranges) {
    CCL_THROW_IF_NOT(sched, "no sched");
}

void ze_membarrier_entry::init_ze_hook() {
    ze_command_list_handle_t list = ze_base_entry::get_copy_list();

    ZE_APPEND_CALL(ze_cmd_mem_range_barrier,
                   list,
                   range_sizes,
                   ranges,
                   ze_base_entry::entry_event,
                   wait_events);
}

std::string ze_membarrier_entry::name_ext() const {
    return name();
}

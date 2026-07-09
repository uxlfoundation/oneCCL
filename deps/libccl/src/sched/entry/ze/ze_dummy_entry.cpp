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
#include "sched/entry/ze/ze_dummy_entry.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"

ze_dummy_entry::ze_dummy_entry(ccl_sched* sched, const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, nullptr /*comm*/, 1 /*add_event_count*/) {
    CCL_THROW_IF_NOT(sched, "no sched");
}

void ze_dummy_entry::update() {
    ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
    ze_base_entry::update();
}

std::string ze_dummy_entry::name_ext() const {
    return name();
}

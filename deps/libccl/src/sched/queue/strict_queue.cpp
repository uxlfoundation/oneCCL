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

#include "sched/queue/strict_queue.hpp"

void ccl_strict_sched_queue::add(ccl_sched* sched) {
    CCL_ASSERT(sched);
    CCL_ASSERT(!sched->bin, "sched ", sched, ", bin ", sched->bin);
    CCL_ASSERT(sched->strict_order);

    std::lock_guard<sched_queue_lock_t> lock{ queue_guard };
    queue.push_back(sched);
    is_queue_empty = false;
}

void ccl_strict_sched_queue::clear() {
    std::lock_guard<sched_queue_lock_t> lock{ queue_guard };
    queue.clear();
    active_queue.clear();
    is_queue_empty = true;
}

sched_queue_t& ccl_strict_sched_queue::peek() {
    if (active_queue.empty()) {
        if (!is_queue_empty) {
            {
                std::lock_guard<sched_queue_lock_t> lock{ queue_guard };
                CCL_ASSERT(!queue.empty());
                active_queue.swap(queue);
                is_queue_empty = true;
            }

            for (const auto& sched : active_queue) {
                CCL_ASSERT(sched, "null sched");
                CCL_ASSERT(!sched->bin, "non null bin ", sched->bin);
                CCL_ASSERT(sched->get_in_bin_status() != ccl_sched_in_bin_added,
                           "unexpected sched in_bin_status ",
                           sched->get_in_bin_status());
                sched->set_in_bin_status(ccl_sched_in_bin_none);
            }
        }
    }

    return active_queue;
}

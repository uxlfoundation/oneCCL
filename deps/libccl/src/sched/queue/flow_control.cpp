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

#include "common/log/log.hpp"
#include "sched/queue/flow_control.hpp"

namespace ccl {

flow_control::flow_control()
        : max_credits(CCL_MAX_FLOW_CREDITS),
          min_credits(CCL_MAX_FLOW_CREDITS),
          credits(CCL_MAX_FLOW_CREDITS) {}

flow_control::~flow_control() {
    LOG_DEBUG("max used credits: ", (max_credits - min_credits));
}

void flow_control::set_max_credits(size_t value) {
    max_credits = min_credits = credits = value;
}

size_t flow_control::get_max_credits() const {
    return max_credits;
}

size_t flow_control::get_credits() const {
    return credits;
}

bool flow_control::take_credit() {
    if (credits) {
        credits--;
        CCL_THROW_IF_NOT(
            credits <= max_credits, "unexpected credits ", credits, ", max_credits ", max_credits);
        min_credits = std::min(min_credits, credits);
        return true;
    }
    else {
        LOG_TRACE("no available credits");
        return false;
    }
}

void flow_control::return_credit() {
    credits++;
    CCL_THROW_IF_NOT((credits > 0) && (credits <= max_credits) && (credits > min_credits),
                     "unexpected credits ",
                     credits,
                     ", max_credits ",
                     max_credits,
                     ", min_credits ",
                     min_credits);
}

} // namespace ccl

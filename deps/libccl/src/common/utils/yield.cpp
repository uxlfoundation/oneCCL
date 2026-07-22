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
#include "common/utils/yield.hpp"

std::map<ccl_yield_type, std::string> ccl_yield_type_names = {
    std::make_pair(ccl_yield_none, "none"),
    std::make_pair(ccl_yield_pause, "pause"),
    std::make_pair(ccl_yield_sleep, "sleep"),
    std::make_pair(ccl_yield_sched_yield, "sched_yield")
};

void ccl_yield(ccl_yield_type yield_type) {
    struct timespec sleep_time;

    switch (yield_type) {
        case ccl_yield_none: break;
        case ccl_yield_pause: _mm_pause(); break;
        case ccl_yield_sleep:
            sleep_time.tv_sec = 0;
            sleep_time.tv_nsec = 0;
            nanosleep(&sleep_time, nullptr);
            break;
        case ccl_yield_sched_yield: sched_yield(); break;
        default: break;
    }
}

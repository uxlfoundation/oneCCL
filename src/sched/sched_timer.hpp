/*
 Copyright 2016-2020 Intel Corporation
 
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

#include <chrono>
#include <string>

#ifdef CCL_ENABLE_ITT
#include "ittnotify.h"
#include "coll/algorithms/algorithm_utils.hpp"
#endif // CCL_ENABLE_ITT

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/api_wrapper/ze_api_wrapper.hpp"
#endif

namespace ccl {

class sched_timer {
public:
    sched_timer() = default;
    void start();
    void update();
    void reset();
    bool is_started() const;

    long double get_elapsed_usec() const;

private:
    bool started{};
    long double time_usec{};
    std::chrono::high_resolution_clock::time_point start_time{};
};

std::string to_string(const sched_timer& timer);

#ifdef CCL_ENABLE_ITT

namespace profile {
namespace itt {

void set_thread_name(const std::string& name);

void task_begin(const char* name);
void task_begin(const char* name, const char* meta_name, uint64_t value);
void task_begin(const char* name, const char* meta_name, uint64_t value, uint64_t comm_id);
void task_end();

static constexpr __itt_event invalid_event = -1;

__itt_event event_get(const char* name);
void event_start(__itt_event event);
void event_end(__itt_event event);

} // namespace itt
} // namespace profile

#endif // CCL_ENABLE_ITT

} //namespace ccl

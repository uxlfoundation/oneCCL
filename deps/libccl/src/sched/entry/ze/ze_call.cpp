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

#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "sched/entry/ze/ze_call.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

namespace ccl {
namespace ze {

std::mutex ze_call::mutex;

ze_call::ze_call() {
    if ((global_data::env().ze_serialize_mode & ze_call::serialize_mode::lock) != 0) {
        LOG_DEBUG("ze call is locked");
        mutex.lock();
    }
}

ze_call::~ze_call() {
    if ((global_data::env().ze_serialize_mode & ze_call::serialize_mode::lock) != 0) {
        LOG_DEBUG("ze call is unlocked");
        mutex.unlock();
    }
}

ze_result_t ze_call::do_call(ze_result_t ze_result, const char* ze_name) const {
    LOG_DEBUG("call ze function: ", ze_name);
    if (ze_result != ZE_RESULT_SUCCESS) {
        CCL_THROW("ze error at ", ze_name, ", code: ", to_string(ze_result));
    }
    return ze_result;
}

// provides different level zero synchronize methods
template <>
ze_result_t zeHostSynchronize(ze_event_handle_t handle) {
    return zeHostSynchronizeImpl(zeEventHostSynchronize, handle);
}

template <>
ze_result_t zeHostSynchronize(ze_command_queue_handle_t handle) {
    return zeHostSynchronizeImpl(zeCommandQueueSynchronize, handle);
}

template <>
ze_result_t zeHostSynchronize(ze_fence_handle_t handle) {
    return zeHostSynchronizeImpl(zeFenceHostSynchronize, handle);
}

} // namespace ze
} // namespace ccl

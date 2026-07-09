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

#pragma once

#include "common/api_wrapper/ze_api_wrapper.hpp"

namespace ccl {
namespace ze {

// class provides the serialization of level zero calls
class ze_call {
public:
    // rule level zero calls serialization
    enum serialize_mode : int {
        none, // no locking or blocking
        lock, // locking around each ZE_CALL
        block, // blocking ZE calls
    };

    ze_call();
    ze_call(const ze_call&) = delete;
    ze_call& operator=(const ze_call&) = delete;
    ~ze_call();
    ze_result_t do_call(ze_result_t ze_result, const char* ze_name) const;

private:
    // mutex that is used for total serialization
    static std::mutex mutex;
};

//host synchronize primitives
template <typename T>
ze_result_t zeHostSynchronize(T handle);
template <typename T, typename Func>
ze_result_t zeHostSynchronizeImpl(Func sync_func, T handle) {
    return sync_func(handle, std::numeric_limits<uint64_t>::max());
}

} // namespace ze
} // namespace ccl

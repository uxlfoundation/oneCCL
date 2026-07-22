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

#include "common/stream/stream.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include "common/api_wrapper/ze_api_wrapper.hpp"

namespace ccl {

namespace ze {

class ipc_event_pool_manager {
public:
    ipc_event_pool_manager() = default;
    ipc_event_pool_manager(const ipc_event_pool_manager&) = delete;
    ipc_event_pool_manager& operator=(const ipc_event_pool_manager&) = delete;
    ~ipc_event_pool_manager() {
        clear();
    }

    void init(const ccl_stream* init_stream);
    void clear();
    static void check_ipc_event_count(ccl_coll_type ctype,
                                      const size_t& ipc_event_count,
                                      const size_t& max_ipc_event_count);
    ze_event_pool_handle_t create(size_t event_count);

private:
    ze_context_handle_t context{};
    std::vector<std::pair<ze_event_pool_handle_t, ze_event_pool_desc_t>> event_pool_info{};
};

} // namespace ze
} // namespace ccl

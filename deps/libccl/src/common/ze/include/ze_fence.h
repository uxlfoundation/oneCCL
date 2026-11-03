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

#include <memory>
#include "ze_inc.h"

class ze_cmd_queue_t;

class ze_fence_t : public ze_handle_t<ze_fence_handle_t>
{
public:
    ze_fence_t() : ZE_HANDLE(ze_fence_handle_t, zeFenceDestroy) {}

    ze_fence_t(const ze_cmd_queue_t& que, bool signaled = false);

    ze_result_t wait(uint64_t timeout = UINT64_MAX) const;
    ze_result_t query_status() const;
    ze_result_t reset() const;
};

using ze_fence_ptr = std::shared_ptr<ze_fence_t>;
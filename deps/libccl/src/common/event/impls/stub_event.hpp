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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "common/event/impls/event_impl.hpp"

namespace ccl {

// event returned by stub backend
class stub_event_impl final : public event_impl {
public:
    stub_event_impl() = default;

    void wait() override {}

    bool test() override {
        return true;
    }

    bool cancel() override {
        return true;
    }

    event::native_t& get_native() override {
        throw ccl::exception(std::string(__FUNCTION__) + " - no native event for stub event");
    }

    ~stub_event_impl() override = default;
};

} // namespace ccl

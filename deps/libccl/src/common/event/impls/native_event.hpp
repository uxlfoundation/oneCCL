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
#include "common/event/ccl_event.hpp"

namespace ccl {

// event which wraps SYCL event to pass it to our API as a parameter
class native_event_impl final : public event_impl {
public:
    explicit native_event_impl(std::unique_ptr<ccl_event> ev);
    ~native_event_impl() override = default;

    void wait() override;
    bool test() override;
    bool cancel() override;
    event::native_t& get_native() override;

private:
    std::unique_ptr<ccl_event> ev = nullptr;
    bool completed = false;
};

} // namespace ccl

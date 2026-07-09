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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "common/event/impls/event_impl.hpp"
#include "common/event/impls/empty_event.hpp"
#include "common/event/impls/native_event.hpp"
#include "common/utils/version.hpp"

namespace ccl {

namespace v1 {

CCL_API event::event() noexcept : base_t(impl_value_t(new empty_event_impl())) {}
CCL_API event::event(event&& src) noexcept : base_t(std::move(src)) {}
CCL_API event::event(impl_value_t&& impl) noexcept : base_t(std::move(impl)) {}
CCL_API event::~event() noexcept {}

CCL_API event& event::operator=(event&& src) noexcept {
    this->acc_policy_t::create(this, std::move(src));
    return *this;
}

bool CCL_API event::operator==(const event& rhs) const noexcept {
    return this->get_impl() == rhs.get_impl();
}

bool CCL_API event::operator!=(const event& rhs) const noexcept {
    return this->get_impl() != rhs.get_impl();
}

CCL_API event::operator bool() {
    return this->test();
}

void CCL_API event::wait() {
    get_impl()->wait();
}

bool CCL_API event::test() {
    return get_impl()->test();
}

bool CCL_API event::cancel() {
    return get_impl()->cancel();
}

CCL_API event::native_t& event::get_native() {
    return const_cast<event::native_t&>(get_impl()->get_native());
}

CCL_API const event::native_t& event::get_native() const {
    return get_impl()->get_native();
}

event CCL_API event::create_from_native(native_t& native_event) {
    auto version = utils::get_library_version();

    auto ev = std::unique_ptr<ccl_event>(new ccl_event(native_event, version));

    return impl_value_t(new native_event_impl(std::move(ev)));
}

} // namespace v1

} // namespace ccl

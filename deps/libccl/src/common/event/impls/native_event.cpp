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

#include "common/event/impls/native_event.hpp"
#include "common/log/log.hpp"

namespace ccl {

native_event_impl::native_event_impl(std::unique_ptr<ccl_event> ev) : ev(std::move(ev)) {}

void native_event_impl::wait() {
    if (!completed) {
#ifdef CCL_ENABLE_SYCL
        auto native_event = get_native();
        native_event.wait();
#else // CCL_ENABLE_SYCL
        throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
#endif // CCL_ENABLE_SYCL
        completed = true;
    }
}

bool native_event_impl::test() {
    if (!completed) {
#ifdef CCL_ENABLE_SYCL
        auto native_event = get_native();
        completed = native_event.get_info<sycl::info::event::command_execution_status>() ==
                    sycl::info::event_command_status::complete;
#else // CCL_ENABLE_SYCL
        throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
#endif // CCL_ENABLE_SYCL
    }
    return completed;
}

bool native_event_impl::cancel() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

event::native_t& native_event_impl::get_native() {
    return ev->get_attribute_value(
        detail::ccl_api_type_attr_traits<ccl::event_attr_id, ccl::event_attr_id::native_handle>{});
}

} // namespace ccl

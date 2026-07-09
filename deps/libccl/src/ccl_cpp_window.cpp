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
#include "oneapi/ccl/environment.hpp"
#include "common/window/window_impl.hpp"
#include "common/log/log.hpp"

namespace ccl {

namespace v1 {

CCL_API window::window(impl_value_t&& impl) : base_t(std::move(impl)) {}

CCL_API window::window(window&& src) : base_t(std::move(src)) {}

CCL_API window& window::operator=(window&& src) {
    if (src.get_impl() != this->get_impl()) {
        src.get_impl().swap(this->get_impl());
        src.get_impl().reset();
    }
    return *this;
}

CCL_API window::~window() {}

} // namespace v1

} // namespace ccl

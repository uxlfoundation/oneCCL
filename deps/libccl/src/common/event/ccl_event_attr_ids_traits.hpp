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

namespace ccl {

namespace detail {

/**
 * Traits for stream attributes specializations
 */
template <>
struct ccl_api_type_attr_traits<event_attr_id, event_attr_id::version> {
    using type = ccl::library_version;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<event_attr_id, event_attr_id::native_handle> {
    using type = typename unified_event_type::ccl_native_t;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<event_attr_id, event_attr_id::context> {
    using type = typename unified_context_type::ccl_native_t;
    using handle_t = typename unified_context_type::ccl_native_t;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<event_attr_id, event_attr_id::command_type> {
    using type = uint32_t;
    using return_type = type;
};

template <>
struct ccl_api_type_attr_traits<event_attr_id, event_attr_id::command_execution_status> {
    using type = int64_t;
    using return_type = type;
};

} // namespace detail
} // namespace ccl

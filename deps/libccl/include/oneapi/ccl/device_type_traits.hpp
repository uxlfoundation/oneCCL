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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl_type_traits.hpp'"
#endif

#include "oneapi/ccl/native_device_api/export_api.hpp"

namespace ccl {

template <class native_stream>
constexpr bool is_stream_supported() {
    return api_type_info</*typename std::remove_pointer<typename std::remove_cv<*/
                         native_stream /*>::type>::type*/>::is_supported();
}

template <class native_event>
constexpr bool is_event_supported() {
    return api_type_info</*typename std::remove_pointer<typename std::remove_cv<*/
                         native_event /*>::type>::type*/>::is_supported();
}

template <class native_device>
constexpr bool is_device_supported() {
    return api_type_info<typename std::remove_pointer<typename std::remove_cv<
        typename std::remove_reference<native_device>::type>::type>::type>::is_supported();
}

template <class native_context>
constexpr bool is_context_supported() {
    return api_type_info<typename std::remove_pointer<typename std::remove_cv<
        typename std::remove_reference<native_context>::type>::type>::type>::is_supported();
}

/**
 * Export common native API supported types
 */
API_CLASS_TYPE_INFO(empty_t);
API_CLASS_TYPE_INFO(typename unified_device_type::ccl_native_t)
API_CLASS_TYPE_INFO(typename unified_context_type::ccl_native_t);
API_CLASS_TYPE_INFO(typename unified_stream_type::ccl_native_t);
API_CLASS_TYPE_INFO(typename unified_event_type::ccl_native_t);
} // namespace ccl

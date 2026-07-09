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

#include <tuple>
#include <type_traits>

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

#include "oneapi/ccl/lp_types.hpp"
#include "oneapi/ccl/types.hpp"

namespace ccl {
/**
 * Base type-trait helpers for "unknown" types
 */
template <ccl::datatype type>
struct type_info {
    static constexpr bool is_supported = false;
    static constexpr bool is_class = false;
};

template <class type>
struct native_type_info {
    static constexpr bool is_supported = false;
    static constexpr bool is_class = false;
};

#define CCL_TYPE_TRAITS(ccl_type, cpp_type, bytes, str) \
    template <> \
    struct type_info<ccl_type> \
            : public ccl_type_info_export<cpp_type, bytes, ccl_type, false, true> { \
        static constexpr const char* name() { \
            return #str; \
        } \
    }; \
    template <> \
    struct native_type_info<cpp_type> : public type_info<ccl_type> {};

#define CCL_CLASS_TYPE_TRAITS(ccl_type, cpp_type, bytes, str) \
    template <> \
    struct native_type_info<cpp_type> \
            : public ccl_type_info_export<cpp_type, bytes, ccl_type, true, true> { \
        static constexpr const char* name() { \
            return #str; \
        } \
    };

#define COMMA ,

/**
 * Enumeration of supported CCL API data types
 */

CCL_TYPE_TRAITS(ccl::datatype::int8, int8_t, sizeof(int8_t), int8)
CCL_TYPE_TRAITS(ccl::datatype::uint8, uint8_t, sizeof(uint8_t), uint8)
CCL_TYPE_TRAITS(ccl::datatype::int16, int16_t, sizeof(int16_t), int16)
CCL_TYPE_TRAITS(ccl::datatype::uint16, uint16_t, sizeof(uint16_t), uint16)
CCL_TYPE_TRAITS(ccl::datatype::int32, int32_t, sizeof(int32_t), int32)
CCL_TYPE_TRAITS(ccl::datatype::uint32, uint32_t, sizeof(uint32_t), uint32)
CCL_TYPE_TRAITS(ccl::datatype::int64, int64_t, sizeof(int64_t), int64)
CCL_TYPE_TRAITS(ccl::datatype::uint64, uint64_t, sizeof(uint64_t), uint64)
CCL_TYPE_TRAITS(ccl::datatype::float16, float16, sizeof(float16), float16)
CCL_TYPE_TRAITS(ccl::datatype::float32, float, sizeof(float), float32)
CCL_TYPE_TRAITS(ccl::datatype::float64, double, sizeof(double), float64)
CCL_TYPE_TRAITS(ccl::datatype::bfloat16, bfloat16, sizeof(bfloat16), bfloat16)

#ifdef CCL_ENABLE_SYCL
CCL_CLASS_TYPE_TRAITS(ccl::datatype::int8, sycl::buffer<int8_t COMMA 1>, sizeof(int8_t), int8)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::uint8, sycl::buffer<uint8_t COMMA 1>, sizeof(uint8_t), uint8)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::int16, sycl::buffer<int16_t COMMA 1>, sizeof(int16_t), int16)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::uint16,
                      sycl::buffer<uint16_t COMMA 1>,
                      sizeof(uint16_t),
                      uint16)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::int32, sycl::buffer<int32_t COMMA 1>, sizeof(int32_t), int32)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::uint32,
                      sycl::buffer<uint32_t COMMA 1>,
                      sizeof(uint32_t),
                      uint32)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::int64, sycl::buffer<int64_t COMMA 1>, sizeof(int64_t), int64)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::uint64,
                      sycl::buffer<uint64_t COMMA 1>,
                      sizeof(uint64_t),
                      uint64)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::float16,
                      sycl::buffer<float16 COMMA 1>,
                      sizeof(float16),
                      float16)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::float32, sycl::buffer<float COMMA 1>, sizeof(float), float32)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::float64, sycl::buffer<double COMMA 1>, sizeof(double), float64)
CCL_CLASS_TYPE_TRAITS(ccl::datatype::bfloat16,
                      sycl::buffer<bfloat16 COMMA 1>,
                      sizeof(bfloat16),
                      bfloat16)
#endif // CCL_ENABLE_SYCL

/**
 * Checks for supporting @c type in ccl API
 */
template <class type>
constexpr bool is_supported() {
    using clear_type = typename std::remove_pointer<type>::type;
    //    static_assert(native_type_info<clear_type>::is_supported, "type is not supported by ccl API");
    return native_type_info<clear_type>::is_supported;
}

/**
 * Checks is @c type a class
 */
template <class type>
constexpr bool is_class() {
    using clear_type = typename std::remove_pointer<type>::type;
    return native_type_info<clear_type>::is_class;
}

/**
 * SFINAE checks for supporting native type @c type in ccl API
 */
template <class type>
constexpr bool is_native_type_supported() {
    return (not is_class<type>() and is_supported<type>());
}

/**
  * SFINAE checks for supporting class @c type in ccl API
  */
template <class type>
constexpr bool is_class_supported() {
    return (is_class<type>() and is_supported<type>());
}

} // namespace ccl

#include "oneapi/ccl/device_type_traits.hpp"

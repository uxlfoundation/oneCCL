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
// Type traits, reduction-op functors, and bit-cast helper for SYCL kernels.

#include <sycl/sycl.hpp>
#include <type_traits>

#include "coll/reduction/reduction.hpp"

/* reduction operations supporting different data types */

// Trait for bfloat16
template <typename T>
struct is_sycl_bfloat16 : std::false_type {};
template <>
struct is_sycl_bfloat16<sycl::ext::oneapi::bfloat16> : std::true_type {};
template <typename T>
constexpr bool is_sycl_bfloat16_v = is_sycl_bfloat16<T>::value;

// Trait for floating-point types (float, double, half, bfloat16)
template <typename T>
struct is_sycl_floating_point
        : std::integral_constant<bool,
                                 std::is_floating_point_v<T> || std::is_same_v<T, sycl::half> ||
                                     std::is_same_v<T, sycl::ext::oneapi::bfloat16>> {};
template <typename T>
constexpr bool is_sycl_floating_point_v = is_sycl_floating_point<T>::value;

// Trait for integer types, excluding bfloat16
template <typename T>
constexpr bool is_sycl_integer_v = std::is_integral_v<T> && !is_sycl_bfloat16_v<T>;

/* internal operation types */

// generic max/min operations that works for all types
// it looks complex just because we use types like this sycl::marray<sycl::vec<T, vec_size>, 8>
struct sycl_max_op {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (is_sycl_bfloat16_v<T>) {
            // Use sycl::ext::oneapi::experimental::fmax for bfloat16
            return sycl::ext::oneapi::experimental::fmax(a, b);
        }
        else if constexpr (is_sycl_floating_point_v<T>) {
            // Use sycl::fmax for all other floating-point types
            return sycl::fmax(a, b);
        }
        else if constexpr (is_sycl_integer_v<T>) {
            // Use sycl::max for integer types
            return sycl::max(a, b);
        }
        else if constexpr (sycl::detail::is_vec_v<T> || sycl::detail::is_marray_v<T>) {
            // Recursively apply sycl_max_op elementwise for vectors and marrays
            T result;
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = (*this)(a[i], b[i]);
            return result;
        }
        else {
            // Fallback
            return a < b ? b : a;
        }
    }
};

struct sycl_min_op {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (is_sycl_bfloat16_v<T>) {
            // Use sycl::ext::oneapi::experimental::fmin for bfloat16
            return sycl::ext::oneapi::experimental::fmin(a, b);
        }
        else if constexpr (is_sycl_floating_point_v<T>) {
            // Use sycl::fmin for all other floating-point types
            return sycl::fmin(a, b);
        }
        else if constexpr (is_sycl_integer_v<T>) {
            // Use sycl::min for integer types
            return sycl::min(a, b);
        }
        else if constexpr (sycl::detail::is_vec_v<T> || sycl::detail::is_marray_v<T>) {
            // Recursively apply sycl_min_op elementwise for vectors and marrays
            T result;
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = (*this)(a[i], b[i]);
            return result;
        }
        else {
            // Fallback
            return a < b ? a : b;
        }
    }
};

struct sycl_prod_op {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a * b;
    }
};

// sycl_avg_op is the same as sycl_sum_op, but we will handle averaging separately
// in the reduce_average function, so we can keep it simple here
struct sycl_avg_op {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct sycl_pre_mul_sum_op {
    template <typename T, typename ScalarType>
    T pre_operation(const T &a, const ScalarType &scalar) const {
        if constexpr (sycl::detail::is_marray_v<T>) {
            T result;
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] * scalar;
            }
            return result;
        }
        else {
            return a * scalar;
        }
    }

    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct sycl_sum_op {
    static constexpr bool has_pre_operation = false;

    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

// generic operation
struct sycl_any_op {
    static constexpr bool has_pre_operation = true;

    template <typename T>
    T operator()(const ccl_reduction_data &reduction, const T &a, const T &b) const {
        if (reduction.op_type == ccl_reduction_type_internal::ccl_min) {
            return sycl_min_op()(a, b);
        }
        else if (reduction.op_type == ccl_reduction_type_internal::ccl_max) {
            return sycl_max_op()(a, b);
        }
        else if (reduction.op_type == ccl_reduction_type_internal::ccl_prod) {
            return sycl_prod_op()(a, b);
        }
        else if (reduction.op_type == ccl_reduction_type_internal::ccl_avg) {
            return sycl_avg_op()(a, b);
        }
        else if (reduction.op_type == ccl_reduction_type_internal::ccl_pre_mul_sum) {
            return sycl_pre_mul_sum_op()(a, b);
        }
        else {
            /* unknown/error */
            return a;
        }
    }
};

// helper trait to get the scalar type for vectors/marrays, otherwise T
template <typename T>
struct get_sycl_scalar_type {
    using type = T;
};
template <typename T, int N>
struct get_sycl_scalar_type<sycl::vec<T, N>> {
    using type = T;
};
template <typename T, int N>
struct get_sycl_scalar_type<sycl::marray<T, N>> {
    using type = T;
};
// specialization for sycl::marray<sycl::vec<T, N>, M>
template <typename T, int N, int M>
struct get_sycl_scalar_type<sycl::marray<sycl::vec<T, N>, M>> {
    using type = T;
};
// use to obtain T from sycl array types
template <typename T>
using get_sycl_scalar_type_t = typename get_sycl_scalar_type<T>::type;

// generic SYCL bit-cast helper for device code
template <typename T>
inline T sycl_bit_cast_device(uint64_t value) {
    using IntType = std::conditional_t<
        (sizeof(T) == 8),
        uint64_t,
        std::conditional_t<(sizeof(T) == 4),
                           uint32_t,
                           std::conditional_t<(sizeof(T) == 2), uint16_t, uint8_t>>>;
    IntType bits = static_cast<IntType>(value);
    return sycl::bit_cast<T>(bits);
}

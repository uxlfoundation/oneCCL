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

#include "common/api_wrapper/ze_api_wrapper.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#error "Unsupported compiler"
#endif

#if defined(__INTEL_LLVM_COMPILER)
#if (__INTEL_LLVM_COMPILER < 20230000)
#define CCL_USE_SYCL121_API 1
#else // (__INTEL_LLVM_COMPILER < 20230000)
#define CCL_USE_SYCL121_API 0
#endif // (__INTEL_LLVM_COMPILER < 20230000)
#elif defined(__LIBSYCL_MAJOR_VERSION)
#if (__LIBSYCL_MAJOR_VERSION < 6)
#define CCL_USE_SYCL121_API 1
#else // (__LIBSYCL_MAJOR_VERSION < 6)
#define CCL_USE_SYCL121_API 0
#endif // (__LIBSYCL_MAJOR_VERSION < 6)
#else // __INTEL_LLVM_COMPILER || __LIBSYCL_MAJOR_VERSION
// For open-source SYCL compilers, assume modern SYCL 2020 API
#if defined(SYCL_LANGUAGE_VERSION) && (SYCL_LANGUAGE_VERSION >= 202000L)
#define CCL_USE_SYCL121_API 0
#else
#define CCL_USE_SYCL121_API 1
#endif
#endif

#if CCL_USE_SYCL121_API
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/backend/level_zero.hpp>
#else // CCL_USE_SYCL121_API
#include <sycl/backend_types.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#endif // CCL_USE_SYCL121_API

#ifdef SYCL_LANGUAGE_VERSION
#define ICPX_VERSION __clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__
#else // SYCL_LANGUAGE_VERSION
#define ICPX_VERSION 0
#endif // SYCL_LANGUAGE_VERSION

class ccl_stream;

namespace ccl {
namespace utils {

bool is_sycl_event_completed(sycl::event event);
bool should_use_sycl_output_event(const ccl_stream* stream);

std::string usm_type_to_str(sycl::usm::alloc type);
std::string sycl_device_to_str(const sycl::device& dev);

constexpr sycl::backend get_level_zero_backend() {
#if ICPX_VERSION >= 140000
    return sycl::backend::ext_oneapi_level_zero;
#elif ICPX_VERSION < 140000
    return sycl::backend::level_zero;
#endif // ICPX_VERSION
}

sycl::event submit_barrier(sycl::queue queue);
sycl::event submit_barrier(sycl::queue queue, sycl::event event);

#ifdef CCL_ENABLE_SYCL_INTEROP_EVENT
sycl::event make_event(const sycl::context& context, const ze_event_handle_t& sync_event);
#endif // CCL_ENABLE_SYCL_INTEROP_EVENT

ze_event_handle_t get_native_event(sycl::event event);

} // namespace utils
} // namespace ccl

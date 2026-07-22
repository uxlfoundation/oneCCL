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

#include "oneapi/ccl/config.h"

#include <cstddef>
#include <vector>

class ccl_comm;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
class ccl_window;
#endif

namespace ccl {

/**
 * Internal window implementation class.
 * This contains the actual data structures that should be hidden from users.
 * The ccl_window_impl wraps the internal ccl_window and provides an opaque interface.
 */
class ccl_window_impl {
public:
    ccl_window_impl(ccl_comm* comm, void* buffer, size_t size, int win_flags);
    ~ccl_window_impl();

    // Prevent copying
    ccl_window_impl(const ccl_window_impl&) = delete;
    ccl_window_impl& operator=(const ccl_window_impl&) = delete;

    // Allow moving
    ccl_window_impl(ccl_window_impl&& other) noexcept;
    ccl_window_impl& operator=(ccl_window_impl&& other) noexcept;

    // Internal accessors (only for use within the library)
    void* get_buffer() const {
        return buffer_;
    }
    size_t get_size() const {
        return size_;
    }
    ccl_comm* get_comm() const {
        return comm_;
    }
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ccl_window* get_internal_window() const {
        return internal_window_;
    }
#endif

private:
    ccl_comm* comm_;
    void* buffer_;
    size_t size_;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ccl_window* internal_window_; // The actual internal window from comm
#else
    void* internal_window_; // Placeholder when SYCL/ZE not enabled
#endif
};

} // namespace ccl

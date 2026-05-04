/*
 Copyright 2016-2020 Intel Corporation
 
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
#include "oneapi/ccl/config.h"
#include "common/global/global.hpp"
#include "common/window/window_impl.hpp"
#include "common/log/log.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "comm/comm.hpp"
#include "comm/windows.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

namespace ccl {

ccl_window_impl::ccl_window_impl(ccl_comm* comm, void* buffer, size_t size, int win_flags)
        : comm_(comm),
          buffer_(buffer),
          size_(size),
          internal_window_(nullptr) {
    LOG_DEBUG("Creating ccl_window_impl, buffer: ", buffer, ", size: ", size);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    // Create and register the internal window
    if (win_flags == CCL_WIN_COLL_SYMMETRIC &&
        ze::is_arc_card(ccl::global_data::get().ze_data->devices[0].family)) {
        auto node_comm = comm->get_node_comm();
        internal_window_ = node_comm->window_register(buffer, size);
    }
    else {
        LOG_WARN("Window registration is not support, CCL_WIN_COLL_SYMMETRIC ignored");
    }
#else
    CCL_THROW("Window registration is only supported with SYCL and Level Zero enabled");
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

ccl_window_impl::~ccl_window_impl() {
    LOG_DEBUG("Destroying ccl_window_impl");
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (internal_window_) {
        auto node_comm = comm_->get_node_comm();
        node_comm->window_deregister(internal_window_);
        internal_window_ = nullptr;
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

ccl_window_impl::ccl_window_impl(ccl_window_impl&& other) noexcept
        : comm_(other.comm_),
          buffer_(other.buffer_),
          size_(other.size_),
          internal_window_(other.internal_window_) {
    other.comm_ = nullptr;
    other.buffer_ = nullptr;
    other.size_ = 0;
    other.internal_window_ = nullptr;
}

ccl_window_impl& ccl_window_impl::operator=(ccl_window_impl&& other) noexcept {
    if (this != &other) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        // Clean up current resources
        if (internal_window_) {
            internal_window_->deregister_buf();
            delete internal_window_;
        }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

        // Move resources
        comm_ = other.comm_;
        buffer_ = other.buffer_;
        size_ = other.size_;
        internal_window_ = other.internal_window_;

        // Reset source
        other.comm_ = nullptr;
        other.buffer_ = nullptr;
        other.size_ = 0;
        other.internal_window_ = nullptr;
    }
    return *this;
}

} // namespace ccl

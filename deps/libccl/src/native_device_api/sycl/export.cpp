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

#include "oneapi/ccl/config.h"
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

#include "oneapi/ccl/native_device_api/sycl/export.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "common/log/log.hpp"

namespace ccl {

/**
 * Context
 */
generic_context_type<cl_backend_type::dpcpp_sycl_l0>::generic_context_type() {}

generic_context_type<cl_backend_type::dpcpp_sycl_l0>::generic_context_type(ccl_native_t ctx)
        : context(ctx) {}

generic_context_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_context_type<cl_backend_type::dpcpp_sycl_l0>::get() noexcept {
    return const_cast<generic_context_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&>(
        static_cast<const generic_context_type<cl_backend_type::dpcpp_sycl_l0>*>(this)->get());
}

const generic_context_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_context_type<cl_backend_type::dpcpp_sycl_l0>::get() const noexcept {
    return context;
}

/**
 * Device
 */
generic_device_type<cl_backend_type::dpcpp_sycl_l0>::generic_device_type(
    device_index_type id,
    sycl::info::device_type type /* = info::device_type::gpu*/)
        : device() {
    if ((std::get<0>(id) == ccl::unused_index_value) &&
        (std::get<1>(id) == ccl::unused_index_value) &&
        (std::get<2>(id) == ccl::unused_index_value)) {
        return;
    }

    LOG_DEBUG("Try to find SYCL device by index: ",
              id,
              ", type: ",
              static_cast<typename std::underlying_type<sycl::info::device_type>::type>(type));

    auto platforms = sycl::platform::get_platforms();
    LOG_DEBUG("Found CL plalforms: ", platforms.size());
    auto platform_it =
        std::find_if(platforms.begin(), platforms.end(), [](const sycl::platform& pl) {
            return pl.get_info<sycl::info::platform::name>().find("Level-Zero") !=
                   std::string::npos;
            //or platform.get_backend() == sycl::backend::ext_oneapi_level_zero
        });
    if (platform_it == platforms.end()) {
        std::stringstream ss;
        ss << "cannot find Level-Zero platform. Supported platforms are:\n";
        for (const auto& pl : platforms) {
            ss << "Platform:\nprofile: " << pl.get_info<sycl::info::platform::profile>()
               << "\nversion: " << pl.get_info<sycl::info::platform::version>()
               << "\nname: " << pl.get_info<sycl::info::platform::name>()
               << "\nvendor: " << pl.get_info<sycl::info::platform::vendor>();
        }

        CCL_THROW("cannot find device by id: " + ccl::to_string(id) + ", reason:\n" + ss.str());
    }

    LOG_DEBUG("Platform:\nprofile: ",
              platform_it->get_info<sycl::info::platform::profile>(),
              "\nversion: ",
              platform_it->get_info<sycl::info::platform::version>(),
              "\nname: ",
              platform_it->get_info<sycl::info::platform::name>(),
              "\nvendor: ",
              platform_it->get_info<sycl::info::platform::vendor>());
}

generic_device_type<cl_backend_type::dpcpp_sycl_l0>::generic_device_type(
    const sycl::device& in_device)
        : device(in_device) {}

device_index_type generic_device_type<cl_backend_type::dpcpp_sycl_l0>::get_id() const {
    return {};
}

typename generic_device_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_device_type<cl_backend_type::dpcpp_sycl_l0>::get() noexcept {
    return device;
}

const typename generic_device_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_device_type<cl_backend_type::dpcpp_sycl_l0>::get() const noexcept {
    return device;
}

/**
 * Event
 */
generic_event_type<cl_backend_type::dpcpp_sycl_l0>::generic_event_type(ccl_native_t ev)
        : event(ev) {}

generic_event_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_event_type<cl_backend_type::dpcpp_sycl_l0>::get() noexcept {
    return const_cast<generic_event_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&>(
        static_cast<const generic_event_type<cl_backend_type::dpcpp_sycl_l0>*>(this)->get());
}

const generic_event_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_event_type<cl_backend_type::dpcpp_sycl_l0>::get() const noexcept {
    return event;
}

/**
 * Stream
 */
generic_stream_type<cl_backend_type::dpcpp_sycl_l0>::generic_stream_type(ccl_native_t q)
        : queue(q) {}

generic_stream_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_stream_type<cl_backend_type::dpcpp_sycl_l0>::get() noexcept {
    return const_cast<generic_stream_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&>(
        static_cast<const generic_stream_type<cl_backend_type::dpcpp_sycl_l0>*>(this)->get());
}

const generic_stream_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_stream_type<cl_backend_type::dpcpp_sycl_l0>::get() const noexcept {
    return queue;
}

/**
 * Platform
 */
generic_platform_type<cl_backend_type::dpcpp_sycl_l0>::generic_platform_type(ccl_native_t& pl)
        : platform(pl) {}

generic_platform_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_platform_type<cl_backend_type::dpcpp_sycl_l0>::get() noexcept {
    return const_cast<generic_platform_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&>(
        static_cast<const generic_platform_type<cl_backend_type::dpcpp_sycl_l0>*>(this)->get());
}

const generic_platform_type<cl_backend_type::dpcpp_sycl_l0>::ccl_native_t&
generic_platform_type<cl_backend_type::dpcpp_sycl_l0>::get() const noexcept {
    return platform;
}
} // namespace ccl
#endif //#if defined(CCL_ENABLE_SYCL) && defined (CCL_ENABLE_ZE)

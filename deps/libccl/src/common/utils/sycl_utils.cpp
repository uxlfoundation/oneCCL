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

#include "common/global/global.hpp"
#include "common/stream/stream.hpp"
#include "common/utils/sycl_utils.hpp"

namespace ccl {
namespace utils {

bool is_sycl_event_completed(sycl::event event) {
    return event.template get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete;
}

bool should_use_sycl_output_event(const ccl_stream* stream) {
    return (stream && stream->is_sycl_device_stream() && stream->is_gpu() &&
            ccl::global_data::env().enable_sycl_output_event &&
            (getenv("ONEAPI_DEVICE_SELECTOR") == nullptr ||
             !strcmp(getenv("ONEAPI_DEVICE_SELECTOR"), "level_zero:gpu")));
}

std::string usm_type_to_str(sycl::usm::alloc type) {
    switch (type) {
        case sycl::usm::alloc::host: return "host";
        case sycl::usm::alloc::device: return "device";
        case sycl::usm::alloc::shared: return "shared";
        case sycl::usm::alloc::unknown: return "unknown";
        default: CCL_THROW("unexpected USM type: ", static_cast<int>(type));
    }
}

std::string sycl_device_to_str(const sycl::device& dev) {
    if (dev.is_cpu()) {
        return "cpu";
    }
    else if (dev.is_gpu()) {
        return "gpu";
    }
    else if (dev.is_accelerator()) {
        return "accel";
    }
    else {
        CCL_THROW("unexpected device type");
    }
}

sycl::event submit_barrier(sycl::queue queue) {
#if ICPX_VERSION >= 140000
    return queue.ext_oneapi_submit_barrier();
#elif ICPX_VERSION < 140000
    return queue.submit_barrier();
#endif // ICPX_VERSION
}

sycl::event submit_barrier(sycl::queue queue, sycl::event event) {
#if ICPX_VERSION >= 140000
    return queue.ext_oneapi_submit_barrier({ std::move(event) });
#elif ICPX_VERSION < 140000
    return queue.submit_barrier({ std::move(event) });
#endif // ICPX_VERSION
}

#ifdef CCL_ENABLE_SYCL_INTEROP_EVENT
sycl::event make_event(const sycl::context& context, const ze_event_handle_t& sync_event) {
#if ICPX_VERSION >= 140000
    return sycl::make_event<sycl::backend::ext_oneapi_level_zero>(
        { sync_event, sycl::ext::oneapi::level_zero::ownership::keep }, context);
#elif ICPX_VERSION < 140000
    return sycl::level_zero::make<sycl::event>(
        context, sync_event, sycl::level_zero::ownership::keep);
#endif // ICPX_VERSION
}
#endif // CCL_ENABLE_SYCL_INTEROP_EVENT

ze_event_handle_t get_native_event(sycl::event event) {
    return sycl::get_native<ccl::utils::get_level_zero_backend()>(event);
}

} // namespace utils
} // namespace ccl

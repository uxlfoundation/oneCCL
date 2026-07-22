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

#include "common/stream/stream_selector.hpp"

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

// creation from sycl::queue
std::unique_ptr<ccl_stream> stream_selector::create(stream_native_t& native_stream,
                                                    const ccl::library_version& version) {
    stream_type type = stream_type::host;

#ifdef CCL_ENABLE_SYCL
    if (native_stream.get_device().is_cpu()) {
        type = stream_type::cpu;
    }
    else if (native_stream.get_device().is_gpu()) {
        type = stream_type::gpu;
    }
    else {
        throw ccl::invalid_argument(
            "core",
            "create_stream",
            std::string("unsupported SYCL queue's device type:\n") +
                native_stream.get_device().template get_info<sycl::info::device::name>() +
                std::string("supported types: host, cpu, gpu"));
    }
#endif // CCL_ENABLE_SYCL
    std::unique_ptr<ccl_stream> ret(new ccl_stream(type, native_stream, version));

    LOG_INFO("stream: ", ret->to_string());

    return ret;
}

stream_selector::stream_native_t stream_selector::get_native_stream() const {
    return native_stream;
}

#ifdef CCL_ENABLE_SYCL
stream_selector::stream_native_t* stream_selector::get_native_stream(size_t idx) {
    return &(native_streams.at(idx));
}
#endif // CCL_ENABLE_SYCL

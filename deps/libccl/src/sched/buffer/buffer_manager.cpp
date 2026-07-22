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
#include "sched/buffer/buffer_cache.hpp"
#include "sched/buffer/buffer_manager.hpp"

#ifdef CCL_ENABLE_ZE
#include "sched/entry/ze/cache/ze_cache.hpp"
#endif // CCL_ENABLE_ZE

namespace ccl {

std::string to_string(buffer_type type) {
    switch (type) {
        case buffer_type::regular: return "regular";
        case buffer_type::sycl: return "sycl";
        case buffer_type::ze: return "ze";
        default: return "unknown";
    }
}

std::string to_string(buffer_place place) {
    switch (place) {
        case buffer_place::host: return "host";
        case buffer_place::device: return "device";
        case buffer_place::shared: return "shared";
        default: return "unknown";
    }
}

std::string alloc_param::to_string() const {
    std::stringstream ss;

    ss << "{ bytes: " << bytes << ", type: " << ccl::to_string(buf_type)
       << ", place: " << ccl::to_string(buf_place) << ", is_managed: " << is_managed;

    if (stream) {
        ss << ", stream: " << stream->to_string();
    }

#ifdef CCL_ENABLE_SYCL
    if (hint_ptr) {
        ss << ", hint_ptr: " << hint_ptr;
    }
#endif // CCL_ENABLE_SYCL

    ss << "}";

    return ss.str();
}

std::string dealloc_param::to_string() const {
    std::stringstream ss;

    ss << "{ ptr: " << ptr << ", bytes: " << bytes << ", type: " << ccl::to_string(buf_type);

    if (stream) {
        ss << ", stream: " << stream->to_string();
    }

    ss << "}";

    return ss.str();
}

buffer_manager::~buffer_manager() {
    clear();
}

void buffer_manager::clear() {
    for (auto it = regular_buffers.begin(); it != regular_buffers.end(); it++) {
        global_data::get().buffer_cache->push(instance_idx, it->bytes, it->ptr);
    }
    regular_buffers.clear();

#ifdef CCL_ENABLE_SYCL
    for (auto it = sycl_buffers.begin(); it != sycl_buffers.end(); it++) {
        global_data::get().buffer_cache->push(instance_idx, it->bytes, it->ctx, it->ptr);
    }
    sycl_buffers.clear();
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_ZE
    for (auto it = ze_buffers.begin(); it != ze_buffers.end(); it++) {
        global_data::get().ze_data->cache->push(instance_idx,
                                                it->ctx,
                                                it->dev,
                                                ze::default_device_mem_alloc_desc,
                                                it->bytes,
                                                0,
                                                it->ptr);
    }
    ze_buffers.clear();
#endif // CCL_ENABLE_ZE
}

void* buffer_manager::alloc(const alloc_param& param) {
    LOG_DEBUG("{ idx: ", instance_idx, ", param: ", param.to_string(), " }");

    void* ptr{};
    size_t bytes = param.bytes;

    CCL_THROW_IF_NOT(bytes > 0, "unexpected request to allocate zero size buffer");
    CCL_THROW_IF_NOT(
        param.buf_type != buffer_type::unknown, "unexpected buf_type ", to_string(param.buf_type));
    CCL_THROW_IF_NOT(param.buf_place != buffer_place::unknown,
                     "unexpected buf_place ",
                     to_string(param.buf_place));

    if (param.buf_type == buffer_type::regular) {
        CCL_THROW_IF_NOT(param.buf_place == buffer_place::host,
                         "unexpected buf_place ",
                         to_string(param.buf_place));
        global_data::get().buffer_cache->get(instance_idx, bytes, &ptr);
        if (param.is_managed) {
            regular_buffers.emplace_back(ptr, bytes);
        }
    }
#ifdef CCL_ENABLE_SYCL
    else if (param.buf_type == buffer_type::sycl) {
        CCL_THROW_IF_NOT(param.buf_place == buffer_place::host,
                         "unexpected buf_place ",
                         to_string(param.buf_place));
        CCL_THROW_IF_NOT(param.stream, "null stream");
        sycl::context sycl_ctx = param.stream->get_native_stream().get_context();
        global_data::get().buffer_cache->get(instance_idx, bytes, sycl_ctx, &ptr);
        if (param.is_managed) {
            sycl_buffers.emplace_back(ptr, bytes, sycl_ctx);
        }
    }
#endif // CCL_ENABLE_SYCL
#ifdef CCL_ENABLE_ZE
    else if (param.buf_type == buffer_type::ze) {
        CCL_THROW_IF_NOT(param.buf_place == buffer_place::device,
                         "unexpected buf_place ",
                         to_string(param.buf_place));
        CCL_THROW_IF_NOT(param.stream, "null stream");

        auto context = param.stream->get_ze_context();
        auto device = param.stream->get_ze_device();
        global_data::get().ze_data->cache->get(
            instance_idx, context, device, ze::default_device_mem_alloc_desc, bytes, 0, &ptr);
        if (param.is_managed) {
            ze_buffers.emplace_back(ptr, bytes, context, device);
        }
    }
#endif // CCL_ENABLE_ZE

    CCL_THROW_IF_NOT(ptr, "null pointer");

    return ptr;
}

void buffer_manager::dealloc(const dealloc_param& param) {
    LOG_DEBUG("{ idx: ", instance_idx, ", param: ", param.to_string(), " }");

    void* ptr = param.ptr;
    size_t bytes = param.bytes;

    CCL_THROW_IF_NOT(ptr, "unexpected request to deallocate null ptr");
    CCL_THROW_IF_NOT(bytes > 0, "unexpected request to deallocate zero size buffer");
    CCL_THROW_IF_NOT(
        param.buf_type != buffer_type::unknown, "unexpected buf_type ", to_string(param.buf_type));

    if (param.buf_type == buffer_type::regular) {
        global_data::get().buffer_cache->push(instance_idx, bytes, ptr);
    }
#ifdef CCL_ENABLE_SYCL
    else if (param.buf_type == buffer_type::sycl) {
        CCL_THROW_IF_NOT(param.stream, "null stream");
        sycl::context sycl_ctx = param.stream->get_native_stream().get_context();
        ccl::global_data::get().buffer_cache->push(instance_idx, bytes, sycl_ctx, ptr);
    }
#endif // CCL_ENABLE_SYCL
#ifdef CCL_ENABLE_ZE
    else if (param.buf_type == buffer_type::ze) {
        CCL_THROW_IF_NOT(param.stream, "null stream");
        auto context = param.stream->get_ze_context();
        auto device = param.stream->get_ze_device();
        global_data::get().ze_data->cache->push(
            instance_idx, context, device, ze::default_device_mem_alloc_desc, bytes, 0, ptr);
    }
#endif // CCL_ENABLE_ZE
}

} // namespace ccl

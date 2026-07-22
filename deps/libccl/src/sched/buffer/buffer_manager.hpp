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

#ifdef CCL_ENABLE_ZE
#include "common/api_wrapper/ze_api_wrapper.hpp"
#endif // CCL_ENABLE_ZE
#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL
#include <list>

#include "common/stream/stream.hpp"
#include "common/utils/buffer.hpp"

namespace ccl {

enum class buffer_type : int { regular, sycl, ze, unknown };

enum class buffer_place : int { host, device, shared, unknown };

std::string to_string(buffer_type type);
std::string to_string(buffer_place place);

struct alloc_param {
    size_t bytes;
    buffer_type buf_type;
    buffer_place buf_place;
    bool is_managed;
    ccl_stream* stream;
    void* hint_ptr;

    alloc_param() = delete;

    alloc_param(size_t bytes,
#ifdef CCL_ENABLE_SYCL
                buffer_type buf_type = buffer_type::unknown,
                buffer_place buf_place = buffer_place::unknown,
#else // CCL_ENABLE_SYCL
                buffer_type buf_type = buffer_type::regular,
                buffer_place buf_place = buffer_place::host,
#endif // CCL_ENABLE_SYCL
                bool is_managed = true,
                ccl_stream* stream = nullptr,
                void* hint_ptr = nullptr)
            : bytes(bytes),
              buf_type(buf_type),
              buf_place(buf_place),
              is_managed(is_managed),
              stream(stream),
              hint_ptr(hint_ptr) {
    }

    alloc_param(size_t bytes, void* ptr) : alloc_param(bytes) {
        hint_ptr = ptr;
    }

    alloc_param(size_t bytes, ccl_buffer buf) : alloc_param(bytes, buf.get_ptr()) {}

    std::string to_string() const;
};

struct dealloc_param {
    void* ptr;
    size_t bytes;
    buffer_type buf_type;
    ccl_stream* stream;

    dealloc_param() = delete;

    dealloc_param(void* ptr, size_t bytes, buffer_type buf_type, ccl_stream* stream = nullptr)
            : ptr(ptr),
              bytes(bytes),
              buf_type(buf_type),
              stream(stream) {}

    std::string to_string() const;
};

struct buffer_desc {
    void* ptr;
    size_t bytes;

    buffer_desc() = delete;
    buffer_desc(void* ptr, size_t bytes) : ptr(ptr), bytes(bytes) {}
};

#ifdef CCL_ENABLE_SYCL
struct sycl_buffer_desc : public buffer_desc {
    const sycl::context ctx;

    sycl_buffer_desc() = delete;
    sycl_buffer_desc(void* ptr, size_t bytes, const sycl::context& ctx)
            : buffer_desc(ptr, bytes),
              ctx(ctx) {}
};
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_ZE
struct ze_buffer_desc : public buffer_desc {
    const ze_context_handle_t ctx;
    const ze_device_handle_t dev;

    ze_buffer_desc() = delete;
    ze_buffer_desc(void* ptr,
                   size_t bytes,
                   const ze_context_handle_t& ctx,
                   const ze_device_handle_t& dev)
            : buffer_desc(ptr, bytes),
              ctx(ctx),
              dev(dev) {}
};
#endif // CCL_ENABLE_ZE

class buffer_manager {
public:
    buffer_manager() = default;
    buffer_manager(const buffer_manager&) = delete;
    buffer_manager& operator=(const buffer_manager&) = delete;
    ~buffer_manager();

    void init(int idx) {
        instance_idx = idx;
    }

    void clear();

    void* alloc(const alloc_param& param);
    void dealloc(const dealloc_param& param);

private:
    size_t instance_idx{};

    std::list<buffer_desc> regular_buffers;

#ifdef CCL_ENABLE_SYCL
    std::list<sycl_buffer_desc> sycl_buffers;
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_ZE
    std::list<ze_buffer_desc> ze_buffers;
#endif // CCL_ENABLE_ZE
};

} // namespace ccl

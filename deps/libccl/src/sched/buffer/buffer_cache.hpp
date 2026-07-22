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

#include <unordered_map>
#include <vector>

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#include "common/utils/hash.hpp"
#endif // CCL_ENABLE_SYCL

#include "common/utils/spinlock.hpp"

namespace ccl {

class regular_buffer_cache;
#ifdef CCL_ENABLE_SYCL
class sycl_buffer_cache;
#endif // CCL_ENABLE_SYCL

class buffer_cache {
public:
    buffer_cache(size_t instance_count);
    buffer_cache(const buffer_cache&) = delete;
    buffer_cache& operator=(const buffer_cache&) = delete;
    ~buffer_cache();

    void get(size_t idx, size_t bytes, void** pptr);

    void push(size_t idx, size_t bytes, void* ptr);

#ifdef CCL_ENABLE_SYCL
    void get(size_t idx, size_t bytes, const sycl::context& ctx, void** pptr);

    void push(size_t idx, size_t bytes, const sycl::context& ctx, void* ptr);
#endif // CCL_ENABLE_SYCL

    using lock_t = ccl_spinlock;

private:
    std::vector<regular_buffer_cache> reg_buffers;
#ifdef CCL_ENABLE_SYCL
    std::vector<sycl_buffer_cache> sycl_buffers;
#endif // CCL_ENABLE_SYCL
};

class regular_buffer_cache {
public:
    regular_buffer_cache() = default;
    ~regular_buffer_cache();
    regular_buffer_cache(const regular_buffer_cache& other) = delete;
    regular_buffer_cache& operator=(const regular_buffer_cache& other) = delete;

    void clear();
    void get(size_t bytes, void** pptr);
    void push(size_t bytes, void* ptr);

private:
    buffer_cache::lock_t guard{};

    using key_t = size_t;
    using value_t = void*;
    std::unordered_multimap<key_t, value_t> cache;
};

#ifdef CCL_ENABLE_SYCL
class sycl_buffer_cache {
public:
    sycl_buffer_cache() = default;
    sycl_buffer_cache(const sycl_buffer_cache& other) = delete;
    sycl_buffer_cache& operator=(const sycl_buffer_cache& other) = delete;
    ~sycl_buffer_cache();

    void clear();
    void get(size_t bytes, const sycl::context& ctx, void** pptr);
    void push(size_t bytes, const sycl::context& ctx, void* ptr);

private:
    buffer_cache::lock_t guard{};

    using key_t = typename std::tuple<size_t, sycl::context>;
    using value_t = void*;
    std::unordered_multimap<key_t, value_t, ccl::utils::tuple_hash> cache;
};
#endif // CCL_ENABLE_SYCL

} // namespace ccl

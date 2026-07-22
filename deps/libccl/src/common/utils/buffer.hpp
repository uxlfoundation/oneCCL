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

#include <iostream>
#include <stddef.h>

#include "common/log/log.hpp"

enum class ccl_buffer_type { DIRECT, INDIRECT };

inline std::ostream& operator<<(std::ostream& os, const ccl_buffer_type& type) {
    os << static_cast<typename std::underlying_type<ccl_buffer_type>::type>(type);
    return os;
}

class ccl_buffer {
private:
    void* src;
    ssize_t size; /* max available size, for sanity checks */
    size_t offset;
    ccl_buffer_type type;

    bool check_offset(size_t access_size = 0) const {
        bool result = true;

        if ((std::numeric_limits<size_t>::max() - offset) < access_size) {
            result = false;
            LOG_ERROR("unexpected (offset + access_size): ",
                      ", offset ",
                      offset,
                      ", access_size ",
                      access_size);
        }

        if ((size != -1) && (offset + access_size > (size_t)size) && (access_size != 0)) {
            result = false;
            LOG_ERROR("unexpected (offset + access_size): ",
                      "size ",
                      size,
                      ", offset ",
                      offset,
                      ", access_size ",
                      access_size);
        }

        return result;
    }

public:
    ccl_buffer(void* src) = delete;
    // user is responsible for freeing the memory after all buffers
    // are destroyed
    ~ccl_buffer() = default;

    // buffer does not take *src* ownership, user is responsible for
    // freeing all memory after all buffers are destroyed
    ccl_buffer(void* src, ssize_t size, size_t offset, ccl_buffer_type type)
            : src(src),
              size(size),
              offset(offset),
              type(type) {
        LOG_TRACE("create: src ",
                  src,
                  ", size ",
                  size,
                  ", offset ",
                  offset,
                  ", type ",
                  type,
                  ", ptr ",
                  get_ptr());
        CCL_ASSERT(check_offset());
    }

    ccl_buffer() : ccl_buffer(nullptr, -1, 0, ccl_buffer_type::DIRECT) {}
    ccl_buffer(void* src, ssize_t size) : ccl_buffer(src, size, 0, ccl_buffer_type::DIRECT) {}
    ccl_buffer(void* src, ssize_t size, size_t offset)
            : ccl_buffer(src, size, offset, ccl_buffer_type::DIRECT) {}
    ccl_buffer(void* src, ssize_t size, ccl_buffer_type type) : ccl_buffer(src, size, 0, type) {}

    ccl_buffer(const ccl_buffer& buf)
            : src(buf.src),
              size(buf.size),
              offset(buf.offset),
              type(buf.type) {
        CCL_ASSERT(check_offset());
    }

    ccl_buffer& operator=(const ccl_buffer& other) {
        if (this != &other) {
            this->src = other.src;
            this->size = other.size;
            this->offset = other.offset;
            this->type = other.type;
            CCL_ASSERT(check_offset());
        }
        return *this;
    }

    void set(void* buf_src, ssize_t buf_size, size_t buf_offset, ccl_buffer_type buf_type) {
        LOG_TRACE("set: src ",
                  buf_src,
                  ", size ",
                  buf_size,
                  ", offset ",
                  buf_offset,
                  ", type ",
                  buf_type,
                  ", old src: ",
                  this->src);

        // check buf_src pointer only if buf_size > 0
        CCL_ASSERT(!(buf_size > 0) || buf_src, "new src is null");

        this->src = buf_src;
        this->size = buf_size;
        this->offset = buf_offset;
        this->type = buf_type;

        CCL_ASSERT(check_offset());
    }

    void set(void* buf_src) {
        set(buf_src, -1, 0, ccl_buffer_type::DIRECT);
    }
    void set(void* buf_src, ssize_t buf_size) {
        set(buf_src, buf_size, 0, ccl_buffer_type::DIRECT);
    }
    void set(void* buf_src, ssize_t buf_size, ccl_buffer_type buf_type) {
        set(buf_src, buf_size, 0, buf_type);
    }
    void set(void* buf_src, ssize_t buf_size, size_t buf_offset) {
        set(buf_src, buf_size, buf_offset, ccl_buffer_type::DIRECT);
    }

    void* get_src() const {
        return src;
    }
    ssize_t get_size() const {
        return size;
    }
    size_t get_offset() const {
        return offset;
    }
    ccl_buffer_type get_type() const {
        return type;
    }

    ccl_buffer operator+(size_t val) {
        return ccl_buffer(src, size, offset + val, type);
    }

    ccl_buffer operator+(size_t val) const {
        return ccl_buffer(src, size, offset + val, type);
    }

    ccl_buffer operator-(size_t val) {
        CCL_ASSERT(offset >= val, "unexpected decrement value ", val);
        return ccl_buffer(src, size, offset - val, type);
    }

    ccl_buffer operator+(int val) {
        return ccl_buffer(src, size, offset + val, type);
    }

    ccl_buffer operator-(int val) {
        CCL_ASSERT(offset >= (size_t)val, "unexpected decrement value ", val);
        return ccl_buffer(src, size, offset - val, type);
    }

    ccl_buffer& operator+=(size_t val) {
        offset += val;
        CCL_ASSERT(check_offset());
        return *this;
    }

    size_t get_difference(ccl_buffer buf) {
        CCL_ASSERT((get_ptr() >= buf.get_ptr()), "difference between pointers < 0");
        return (static_cast<char*>(get_ptr()) - static_cast<char*>(buf.get_ptr()));
    }

    void* get_ptr(size_t access_size = 0) const {
        CCL_ASSERT(check_offset(access_size));

        if (!src)
            return nullptr;

        if (type == ccl_buffer_type::DIRECT)
            return ((char*)src + offset);
        else {
            return (*((char**)src)) ? (*((char**)src) + offset) : nullptr;
        }
    }

    operator bool() const {
        if (type == ccl_buffer_type::DIRECT)
            return src;
        else
            return (src && (*(void**)src));
    }

    bool operator==(ccl_buffer const& other) const {
        return ((get_ptr() == other.get_ptr()) && (get_type() == other.get_type()));
    }

    bool operator!=(ccl_buffer const& other) const {
        return !(*this == other);
    }

    bool operator>(ccl_buffer const& other) const {
        CCL_ASSERT(get_type() == other.get_type(), "types should match");
        return (get_ptr() > other.get_ptr());
    }

    friend std::ostream& operator<<(std::ostream& out, const ccl_buffer& buf) {
        out << "(src: " << buf.get_src() << ", size " << buf.get_size() << ", off "
            << buf.get_offset() << ", type: " << buf.get_type() << ", ptr: " << buf.get_ptr()
            << ")";
        return out;
    }
};

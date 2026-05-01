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
#pragma once
#include <memory>
#include <vector>

class mem_block_t {
    using memory_t = std::shared_ptr<uint8_t[]>;

public:
    // Explicitly define copy and move operations
    mem_block_t() = default;
    mem_block_t(const mem_block_t&) = default;
    mem_block_t& operator=(const mem_block_t&) = default;
    mem_block_t(mem_block_t&&);
    mem_block_t& operator=(mem_block_t&&);

    // construct from pointer and size (static memory)
    mem_block_t(const void* ptr, size_t size) : ptr_((void*)ptr), size_(size) {}

    // construct with allocated size (dynamic memory)
    mem_block_t(size_t size) {
        alloc(size);
    }

    void* ptr() const {
        return ptr_;
    }
    size_t size() const {
        return size_;
    }

    operator void*() const {
        return ptr_;
    }
    operator const void*() const {
        return ptr_;
    }
    operator size_t() const {
        return size_;
    }

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;

    memory_t mem_;

    void alloc(size_t size) {
        mem_ = memory_t(new uint8_t[size]);
        ptr_ = mem_.get();
        size_ = size;
    }
};

// make sure default move operations are available
inline mem_block_t::mem_block_t(mem_block_t&&) = default;
inline mem_block_t& mem_block_t::operator=(mem_block_t&&) = default;

using payload_t = std::vector<mem_block_t>;

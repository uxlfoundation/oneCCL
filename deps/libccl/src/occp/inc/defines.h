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
#include <string.h>

#define RET_ON_FALSE(func) \
    if (!func) \
    return false

class no_copy_t {
protected:
    no_copy_t() = default;
    ~no_copy_t() = default;

public:
    // Explicitly allow move operations
    no_copy_t(no_copy_t&&) = default;
    no_copy_t& operator=(no_copy_t&&) = default;

    // Explicitly delete copy operations
    no_copy_t(const no_copy_t&) = delete;
    no_copy_t& operator=(const no_copy_t&) = delete;
};

class no_move_t {
protected:
    no_move_t() = default;
    ~no_move_t() = default;

public:
    // Explicitly allow copy operations
    no_move_t(const no_move_t&) = default;
    no_move_t& operator=(const no_move_t&) = default;

    // Explicitly delete move operations
    no_move_t(no_move_t&&) = delete;
    no_move_t& operator=(no_move_t&&) = delete;
};

class untransferable_t : private no_copy_t, private no_move_t {
protected:
    untransferable_t() = default;
    ~untransferable_t() = default;
};

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define FOR_I(N) for (decltype(N) i = 0; i < (N); i++)

#define HLLOG_DEFINE_OSTREAM_FORMATTER(T)

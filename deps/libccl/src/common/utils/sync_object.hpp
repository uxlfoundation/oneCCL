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

#include "common/log/log.hpp"

#include <atomic>

class sync_object {
public:
    explicit sync_object(size_t count) : initial_cnt(count), sync(count) {
        CCL_ASSERT(initial_cnt > 0, "count must be greater than 0");
    }

    void visit() {
        auto cnt = sync.fetch_sub(1, std::memory_order_release);
        CCL_ASSERT(cnt <= initial_cnt, "invalid count ", cnt);
    }

    void reset() {
        sync.store(initial_cnt, std::memory_order_release);
    }

    size_t value() const {
        return sync.load(std::memory_order_acquire);
    }

private:
    size_t initial_cnt{};
    std::atomic_size_t sync{};
};

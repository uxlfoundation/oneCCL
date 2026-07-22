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

#include "common/utils/spinlock.hpp"
#include "sched/sched.hpp"

#include <chrono>
#include <mutex>
#include <deque>

class ccl_fusion_manager {
public:
    ccl_fusion_manager();
    ~ccl_fusion_manager();

    ccl_fusion_manager(const ccl_fusion_manager& other) = delete;
    ccl_fusion_manager& operator=(const ccl_fusion_manager& other) = delete;

    void reset();
    bool can_reset();
    bool can_fuse(ccl_sched* sched);
    bool add(ccl_sched* sched);
    void execute();
    void release_buffer(void* buf);

private:
    ccl_sched* build_sched();
    void clear_exec_queue();
    void check_tracked_scheds(bool force_release = false);

    const size_t bytes_threshold;
    const size_t count_threshold;
    const size_t buffer_size;

    using lock_t = ccl_spinlock;
    lock_t guard{};

    using sched_queue_t = std::deque<ccl_sched*>;
    sched_queue_t postponed_queue{};
    sched_queue_t exec_queue{};

    size_t exec_queue_sum_bytes = 0;

    std::list<ccl_sched*> tracked_scheds{};

    std::chrono::steady_clock::duration cycle;
    std::chrono::steady_clock::time_point last_exec_time;

    size_t stat_fused_ops = 0;
    size_t stat_fused_bytes = 0;
    size_t stat_empty_exec_calls = 0;
    size_t stat_overlapped_exec_calls = 0;
};

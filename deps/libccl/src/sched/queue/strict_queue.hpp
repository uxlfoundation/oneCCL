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

using sched_queue_lock_t = ccl_spinlock;
using sched_queue_t = std::vector<ccl_sched*>;

/* used to ensure strict start ordering for transports w/o tagged direct collectives - i.e. for all transports */
class ccl_strict_sched_queue {
public:
    ccl_strict_sched_queue() {}
    ccl_strict_sched_queue(const ccl_strict_sched_queue& other) = delete;
    ccl_sched_queue& operator=(const ccl_strict_sched_queue& other) = delete;
    ~ccl_strict_sched_queue() {}

    void add(ccl_sched* sched);
    void clear();
    sched_queue_t& peek();

private:
    sched_queue_lock_t queue_guard{};

    std::atomic_bool is_queue_empty{ true };

    /* used to buffer schedules which require strict start ordering */
    sched_queue_t queue{};

    /* but real strict starting will happen from this queue */
    sched_queue_t active_queue{};
};

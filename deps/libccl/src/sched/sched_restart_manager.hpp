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

#include <list>
#include <utility>

#include "coll/coll_param.hpp"
#include "common/request/request.hpp"
#include "common/utils/spinlock.hpp"

class ccl_sched;

class sched_restart_manager {
public:
    sched_restart_manager(ccl_sched* sched) : sched(sched) {}

    void add_launch_params(const std::pair<ccl_coll_param, ccl_coll_attr>& params);

    void update_launch_params();

    // create a new empty request and put it to delayed queue
    ccl_request* make_delayed_request();

    ccl_request* get_delayed_request();
    bool has_delayed_requests() const;

    // check/updates sched launch status
    bool is_in_progress() const;
    void set_not_in_progress();
    void set_in_progress();

    // do some preprocessing work before starting the sched
    ccl_request* preprocess(bool restart);
    bool check_delayed_requests();

private:
    ccl_sched* sched;

    // when the execution is delayed, we need update the parameters
    // only once the corresponding execution is started(because there
    // is only a single sched state, updated on each execution).
    // for simplicity, delayed setting of parameters is used every
    // time the schedule is retrieved from the cache
    std::list<std::pair<ccl_coll_param, ccl_coll_attr>> launch_params;
    // list of requests which we need to start once the active is
    // completed(in the specified order)
    std::list<ccl_request*> delayed_requests;

    // indicates that a new sched/cached sched is currently executions,
    // so it must be delayed
    // protected by the mutex
    bool in_progress = false;

    // protects state during sched start from user/worker threads
    using lock_t = ccl_spinlock;
    lock_t lock{};
};

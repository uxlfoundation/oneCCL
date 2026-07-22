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

#include "atl/atl_base_comm.hpp"
#include "coll/coll.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "common/request/request.hpp"
#include "exec/thread/listener.hpp"
#include "sched/sched.hpp"
#include "internal_types.hpp"

#include <memory>
#include <vector>

class ccl_worker;
class ccl_service_worker;
class ccl_sched;

enum ccl_wait_result {
    ccl_wait_result_completed_released,
    ccl_wait_result_completed_not_released,
    ccl_wait_result_not_completed,
};

class alignas(CACHELINE_SIZE) ccl_executor {
    friend class ccl_listener;

public:
    ccl_executor(const ccl_executor& other) = delete;
    ccl_executor& operator=(const ccl_executor& other) = delete;
    ccl_executor(ccl_executor&& other) = delete;
    ccl_executor& operator=(ccl_executor&& other) = delete;

    ccl_executor(const char* main_addr = NULL);
    ~ccl_executor();

    struct worker_guard {
        ccl_executor* parent;
        worker_guard(ccl_executor* e) {
            parent = e;
            parent->lock_workers();
        }
        worker_guard(worker_guard&) = delete;
        worker_guard(worker_guard&& src) {
            parent = src.parent;
            src.parent = nullptr;
        }
        worker_guard& operator=(worker_guard&) = delete;
        worker_guard& operator=(worker_guard&& src) {
            parent = src.parent;
            src.parent = nullptr;
            return *this;
        }
        ~worker_guard() {
            if (parent)
                parent->unlock_workers();
        }
    };

    worker_guard get_worker_lock() {
        return worker_guard(this);
    }

    void start(ccl_sched* sched, bool extra_sched = false);
    void wait(const ccl_request* req);
    bool test(const ccl_request* req);
    void do_work();

    void start_workers(atl_proc_coord_t& coord);
    bool are_workers_started() {
        return workers_started;
    };
    size_t get_worker_count() const;
    void update_wait_condition(size_t idx,
                               ccl_base_thread::wait_data::update_type type,
                               size_t delta);

    // TODO: Rework to support listener
    //    ccl::status create_listener(ccl_resize_fn_t resize_func);
    void update_workers();
    void lock_workers();
    void unlock_workers();
    bool is_locked = false;

    static size_t calculate_atl_ep_count(size_t worker_count);
    static atl_attr_t generate_atl_attr(const ccl::env_data& env);

private:
    size_t get_worker_idx_round_robin(ccl_sched* sched);
    size_t get_worker_idx_by_sched_id(ccl_sched* sched);

    std::unique_ptr<ccl_sched_queue> create_sched_queue(size_t idx, size_t ep_per_worker);

    std::vector<std::unique_ptr<ccl_worker>> workers;
    // TODO: Rework to support listener
    //  std::unique_ptr<ccl_listener> listener;

    bool workers_started = false;
};

inline void ccl_release_sched(ccl_sched* sched) {
    if (sched->coll_attr.to_cache && sched->type != sched_type_t::extra) {
        ccl::global_data::get().sched_cache->release(sched);
    }
    else {
        delete sched;
    }
}

/// frees memory if and only if req is detached from its scheduler
inline void ccl_release_request(ccl_request* req) {
    auto* sched = req->get_sched();

    CCL_THROW_IF_NOT(sched, "sched is not set for request");

    // if the released request is not the current active one, then we need
    // to explicitly delete it, otherwise it's going to be deleted in sched's
    // destructor
    if (req != sched->get_request()) {
        LOG_DEBUG("deleting req ", req, " detached from sched ", sched);
        delete req;
    }

    ccl_release_sched(sched);
}

template <class sched_type = ccl_sched>
inline ccl_wait_result ccl_wait_impl(ccl_executor* exec, ccl_request* request) {
    ccl_wait_result ret = ccl_wait_result_not_completed;
    exec->wait(request);
    if (!exec->is_locked) {
        LOG_DEBUG(
            "req ",
            request,
            " completed, sched ",
            ccl_coll_type_to_str(static_cast<sched_type*>(request->get_sched())->coll_param.ctype));
        if (!request->get_sched()->coll_attr.synchronous) {
            ccl_release_request(request);
            ret = ccl_wait_result_completed_released;
        }
        else {
            ret = ccl_wait_result_completed_not_released;
        }
    }
    return ret;
}

template <class sched_type = ccl_sched>
inline bool ccl_test_impl(ccl_executor* exec, ccl_request* request) {
    bool completed = exec->test(request);

    if (completed) {
        sched_type* sched = static_cast<sched_type*>(request->get_sched());
        LOG_DEBUG(
            "req ", request, " completed, sched ", ccl_coll_type_to_str(sched->coll_param.ctype));

        if (!exec->is_locked) {
            ccl_release_request(request);
        }
    }

    return completed;
}

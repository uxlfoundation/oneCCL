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

#include "sched/sched.hpp"
#include "sched/sched_restart_manager.hpp"

void sched_restart_manager::add_launch_params(
    const std::pair<ccl_coll_param, ccl_coll_attr>& params) {
    std::lock_guard<ccl_spinlock> guard(lock);
    if (is_in_progress()) {
        // if the sched is in progress, we can't immediately lauch it, so we must save the params and
        // update them when the sched is restarted.
        launch_params.push_back(params);
    }
    else {
        // if the sched is not in progress, that means that it can start immediately, so the params can be
        // updated right now. This is important when using fusion, in that case the sched is never started,
        // but its parameters are used by another fusion sched.
        CCL_THROW_IF_NOT(delayed_requests.empty(),
                         "must not update params if there are any delayed requests");
        sched->update_coll_param_and_attr(params.first, params.second);
    }
}

void sched_restart_manager::update_launch_params() {
    // update parameters only if we have something in the list, otherwise just
    // use the ones that are already set.
    if (!launch_params.empty()) {
        auto param = launch_params.front();
        launch_params.pop_front();

        // update some parameters and attributes in existing schedule
        // as they could be changed since previous call
        sched->update_coll_param_and_attr(param.first, param.second);
    }
}

// must be protected by the mutex
ccl_request* sched_restart_manager::make_delayed_request() {
    auto req = new ccl_request(*sched);
    // increase the counter on the request after creation to mark it as
    // non-complete while it's delayed.
    req->set_counter(sched->calculate_request_count());
    delayed_requests.emplace_back(req);

    return req;
}

ccl_request* sched_restart_manager::get_delayed_request() {
    auto req = delayed_requests.front();
    delayed_requests.pop_front();
    return req;
}

bool sched_restart_manager::has_delayed_requests() const {
    return !delayed_requests.empty();
}

bool sched_restart_manager::is_in_progress() const {
    return in_progress;
}

void sched_restart_manager::set_not_in_progress() {
    in_progress = false;
}

void sched_restart_manager::set_in_progress() {
    in_progress = true;
}

ccl_request* sched_restart_manager::preprocess(bool restart) {
    // we might start the sched either from API call or automatically after the completion of
    // the pervious execution, in this case it's started on worker thread. To avoid race
    // condition, we need to put this section under mutex
    std::lock_guard<ccl_spinlock> lk(lock);
    if (restart) {
        sched->update_active_request(true);

        LOG_DEBUG("starting ", sched, " with request ", sched->get_request());
    }
    else if (is_in_progress()) {
        // cannot execute immediately, create a new request, put it into the delayed
        // queue and return to the user
        // note: even when we don't exactly use run multiple collectives without waiting
        // on them, we can have a case of delayed request: this arises from a race condition:
        // worker thread completes request and finalizes the sched, including resetting in_progress
        // flag, but right before it resets the flag, user thread waiting on the request can see
        // the completion and run a new collective and read in_progress == true. Working around
        // this issue would result in more complex code. Although this behavior is not an obvious
        // and expected one, keep it as it doesn't affect correctness of execution and also simplifies
        // the code.
        auto* new_req = make_delayed_request();

        LOG_DEBUG("cached schedule ",
                  sched,
                  " is already executing, will continue later, new request: ",
                  new_req);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        // we need to set output event and submit barrier immediately, otherwise q.wait()
        // called by a user can return earlier than we process all the delayed requests
        sched->create_sync_event(new_req);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

        return new_req;
    }

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (!sched->get_request()->has_output_event()) {
        sched->create_sync_event(sched->get_request());
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    set_in_progress();
    update_launch_params();

    return nullptr;
}

bool sched_restart_manager::check_delayed_requests() {
    std::lock_guard<ccl_spinlock> lg(lock);
    if (!has_delayed_requests()) {
        LOG_DEBUG("no more iterations to run for sched ", sched);
        // we need to move current request to completed list and replace it with
        // an empty one, otherwise we might end up with multiple ccl::events
        // referring to the same request
        sched->update_active_request(/* replace with new request */ false);
        // no more delayed executions, the new scheds can be executed right away
        set_not_in_progress();
        return false;
    }

    return true;
}

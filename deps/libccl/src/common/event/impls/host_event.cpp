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

#include "coll/coll_util.hpp"
#include "common/request/request.hpp"
#include "common/event/impls/host_event.hpp"
#include "exec/exec.hpp"
#include "sched/cache/recycle_storage.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

namespace ccl {

host_event_impl::host_event_impl(ccl_request* r, bool in_is_group_activated) : req(r) {
    is_group_activated = in_is_group_activated;
    if (!req) {
        completed = true;
        return;
    }
#ifdef CCL_ENABLE_SYCL
    native_event = req->share_native_event();
    sync_event = req->share_sync_event();
#ifdef CCL_ENABLE_ZE
    if (sync_event) {
        stream = req->get_sched()->coll_param.stream;
        ze_context = stream->get_ze_context();
    }
#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL
    if (req->get_sched()->coll_attr.synchronous) {
        if (!ccl::global_data::get().executor.get()->is_locked) {
            ccl_release_request(req);
        }
        // if the user calls collective with coll_attr->synchronous=1 then it will be progressed
        // in place and in this case we mark request as completed,
        // all calls to wait() or test() will do nothing
        completed = true;
    }
}

host_event_impl::~host_event_impl() {
    // TODO: need to find a way to synchronize these 2 statuses, right now there are
    // some issues, e.g. in case of pure host event get_native() is an empty sycl
    // event which always complete, this way LOG_ERROR is never called
#ifdef CCL_ENABLE_SYCL
    auto& recycle_storage = ccl::global_data::get().recycle_storage;
    bool native_event_completed = true;
    if (native_event) {
        native_event_completed = utils::is_sycl_event_completed(*native_event);
    }
#endif // CCL_ENABLE_SYCL
    if (!completed
#ifdef CCL_ENABLE_SYCL
        && !native_event_completed) {
        recycle_storage->store_request(req);
    }
#else // CCL_ENABLE_SYCL
    ) {
        LOG_ERROR("not completed event is destroyed");
        wait();
    }
#endif // NOT CCL_ENABLE_SYCL

    // call wait() to do a proper finalization and cleanup if the task is completed
    if (completed
#ifdef CCL_ENABLE_SYCL
        || (native_event && native_event_completed)
#endif // CCL_ENABLE_SYCL
    ) {
        if (!is_group_activated) {
            wait();
        }
    }

#ifdef CCL_ENABLE_SYCL
    if (sync_event) {
        auto& pools = ccl::global_data::get().ze_data->dynamic_event_pools;
        auto pool_it = pools.find(ze_context);
        if (pool_it == pools.end()) {
            LOG_ERROR("pool must be initialized for the context");
        }
        else {
            recycle_storage->store_events(&(pool_it->second), sync_event, native_event);
        }
    }
#endif // CCL_ENABLE_SYCL
}

void host_event_impl::wait() {
    if (is_group_activated) {
        LOG_WARN("ccl::event::wait() is not supported for collectives within group API");
    }

    if (!completed) {
        auto* exec = ccl::global_data::get().executor.get();
        auto wait_result = ccl_wait_impl(exec, req);
        if (wait_result == ccl_wait_result_completed_not_released) {
            ccl_release_request(req);
        }
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        if (native_event) {
            native_event->wait();
        }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
        completed = true;
    }
}

bool host_event_impl::test() {
    if (is_group_activated) {
        LOG_WARN("ccl::event::test is not supported for collectives within group API");
    }
    if (!completed) {
        completed = ccl_test_impl(ccl::global_data::get().executor.get(), req);
    }
    return completed;
}

bool host_event_impl::cancel() {
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
}

event::native_t& host_event_impl::get_native() {
#ifdef CCL_ENABLE_SYCL
    if (is_group_activated) {
        LOG_WARN("ccl::event::get_native is not supported for collectives within group API");
    }

    if (ccl::global_data::env().enable_sycl_output_event) {
        return *native_event;
    }
    else {
        CCL_THROW("get_native() is not available without CCL_SYCL_OUTPUT_EVENT=1 env variable");
    }
#else // CCL_ENABLE_SYCL
    throw ccl::exception(std::string(__FUNCTION__) + " - is not implemented");
#endif // CCL_ENABLE_SYCL
}

} // namespace ccl

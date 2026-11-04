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

#ifdef CCL_ENABLE_SYCL

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <sstream>

#include <sycl/sycl.hpp>

#include "atl/atl_def.h"
#include "common/log/log.hpp"

template <typename F>
inline sycl::event ccl_host_task_events(sycl::queue &q,
                                        const std::vector<sycl::event> &deps,
                                        F &&host_callable) {
    LOG_TRACE("host task: using q.submit, deps.size()=", deps.size());
    return q.submit([&](sycl::handler &h) {
        // there used to be a bug with empty deps list
        // leave a check just in case
        if (!deps.empty()) {
            h.depends_on(deps);
        }
        h.host_task(std::forward<F>(host_callable));
    });
}


template <typename F>
inline sycl::event ccl_host_task_eventless(sycl::queue &q,
                                           const std::vector<sycl::event> &deps,
                                           F &&host_callable) {
#if defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER > 20260100)
    LOG_TRACE("host task: using syclexp::host_task + submit_barrier, deps.size()=", deps.size());
    if (!deps.empty()) {
        q.ext_oneapi_submit_barrier(deps);
    }
    sycl::ext::oneapi::experimental::host_task(q, std::forward<F>(host_callable));
    return q.ext_oneapi_submit_barrier();
#else
    CCL_THROW("explicitly requested eventless API on unsupported compiler;"\
              " expects icpx compiler vesion 2026.01.01 or more recent");
    return sycl::event();
#endif
}


template <typename F>
inline sycl::event ccl_host_task(sycl::queue &q,
                                 const std::vector<sycl::event> &deps,
                                 F &&host_callable) {
    auto host_task_type = ccl::global_data::env().sycl_host_task_type;
    switch (host_task_type) {
        case host_task_type::auto_type:
        case host_task_type::event:
            return ccl_host_task_events(q, deps, std::forward<F>(host_callable));
        case host_task_type::eventless:
            return ccl_host_task_eventless(q, deps, std::forward<F>(host_callable));
        default:
            auto &names = ccl::env_data::host_task_type_names;
            auto fd = names.find(host_task_type);
            std::string name = fd != names.end() ? fd->second : "unknwon name";
            std::ostringstream oss;
            oss << "host_task_type [" << name << "]: handling unimplemented";
            CCL_THROW(oss.str());
    }
}

// With CCL_SYCL_SCALEOUT_SYNC=1 an eager scale-out step runs on the calling
// thread, after its dependencies and all earlier work on the queue, instead of
// in a SYCL host task.
template <typename F>
inline sycl::event ccl_scaleout_task(sycl::queue &q,
                                     const std::vector<sycl::event> &deps,
                                     bool recording,
                                     F &&host_callable) {
    if (ccl::global_data::env().sycl_scaleout_sync && !recording) {
        sycl::event::wait(deps);
        q.wait();
        host_callable();
        return sycl::event();
    }
    return ccl_host_task(q, deps, std::forward<F>(host_callable));
}

// MPI requires collectives on a communicator to be started in the same order
// on every rank, but host tasks may start out of order. An eager scale-out
// collective therefore posts its MPI call from a host task of its own that
// depends on the previous post, and waits for completion in a second host task.
// No host task blocks waiting for its turn, so this holds with a single-thread
// host-task pool, and a post that throws does not hold up the posts after it.
struct ccl_scaleout_order {
    std::mutex mutex;
    std::unordered_map<sycl::context, sycl::event> last_post;
};

inline ccl_scaleout_order &get_scaleout_order() {
    static ccl_scaleout_order *order = new ccl_scaleout_order;
    return *order;
}

template <typename Post, typename Wait>
inline sycl::event ccl_scaleout_ordered_task(sycl::queue &q,
                                             const std::vector<sycl::event> &deps,
                                             bool recording,
                                             Post post,
                                             Wait wait) {
    if (recording || ccl::global_data::env().sycl_scaleout_sync) {
        return ccl_scaleout_task(q, deps, recording, [=]() {
            atl_req_t req;
            post(req);
            wait(req);
        });
    }

    // the posted request may point into data owned by the post callable (counts,
    // offsets), so both host tasks share it and keep it alive until the wait ends
    struct post_state {
        Post post;
        atl_req_t req;
        bool posted = false;
    };
    auto state = std::make_shared<post_state>(post_state{ std::move(post), {}, false });
    auto &order = get_scaleout_order();
    std::lock_guard<std::mutex> lock(order.mutex);
    std::vector<sycl::event> post_deps = deps;
    auto last = order.last_post.find(q.get_context());
    if (last != order.last_post.end()) {
        post_deps.push_back(last->second);
    }
    sycl::event posted = ccl_host_task(q, post_deps, [=]() {
        state->post(state->req);
        state->posted = true;
    });
    order.last_post.insert_or_assign(q.get_context(), posted);
    return ccl_host_task(q, { posted }, [=]() {
        if (state->posted) {
            wait(state->req);
        }
    });
}
#endif // CCL_ENABLE_SYCL


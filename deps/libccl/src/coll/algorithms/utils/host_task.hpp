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

#include <vector>
#include <sstream>

#include <sycl/sycl.hpp>

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
#endif // CCL_ENABLE_SYCL


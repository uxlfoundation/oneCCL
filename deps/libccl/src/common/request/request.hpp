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

#include <atomic>
#include <functional>

#include "common/utils/utils.hpp"

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

class ccl_sched;

class alignas(CACHELINE_SIZE) ccl_request {
public:
    using dump_func = std::function<void(std::ostream&)>;

    ccl_request(ccl_sched& sched);
    ccl_request(const ccl_request& other) = delete;
    ccl_request& operator=(const ccl_request& other) = delete;

    virtual ~ccl_request();

    // decrements counter and return the current value
    int complete_counter();
    // decrements counter and return true if it's equal to 0
    bool complete();

    bool is_completed() const;

    void set_counter(int counter);

    void increase_counter(int increment);

    mutable bool urgent = false;

#ifdef CCL_ENABLE_SYCL
    void set_native_event(sycl::event new_event) {
        native_event = std::make_shared<sycl::event>(new_event);
    }

    sycl::event& get_native_event() {
        return *native_event;
    }

    std::shared_ptr<sycl::event>& share_native_event() {
        return native_event;
    }

    void set_sync_event(sycl::event new_event) {
        sync_event = std::make_shared<sycl::event>(new_event);
    }

    sycl::event& get_sync_event() {
        return *sync_event;
    }

    std::shared_ptr<sycl::event>& share_sync_event() {
        return sync_event;
    }

    bool has_output_event() const {
        // by default the event is empty
        if (!native_event)
            return false;
        // running on xpu it'd be true
        return true;
    }

    bool has_sync_event() const {
        if (!sync_event)
            return false;
        // running on xpu it'd be true
        return true;
    }
#endif // CCL_ENABLE_SYCL

    ccl_sched* get_sched() {
        return &sched;
    }

    const ccl_sched* get_sched() const {
        return &sched;
    }

    std::atomic_int completion_counter{ 0 };

private:
#ifdef ENABLE_DEBUG
    void set_dump_callback(dump_func&& callback);
#endif // ENABLE_DEBUG

#ifdef CCL_ENABLE_SYCL
    // The actual event from submit_barrier. It's returned to the user via ccl::event.get_native()
    std::shared_ptr<sycl::event> native_event;
    // This is basically a wrapped l0 event from sched_base, we need to keep as sycl object because its destructor
    // implies wait on the event, but in our case it's not yet completed(right after we created it from l0 event).
    // So just keep it here until we signal the corresponding l0 event.
    std::shared_ptr<sycl::event> sync_event;
#endif // CCL_ENABLE_SYCL

    // ref to sched as part of which the request is created, there must be 1-to-1 relation
    ccl_sched& sched;

#ifdef ENABLE_DEBUG
    dump_func dump_callback;
    mutable size_t complete_checks_count = 0;
    static constexpr const size_t CHECK_COUNT_BEFORE_DUMP = 40000000;
#endif // ENABLE_DEBUG
};

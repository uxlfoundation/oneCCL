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

#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "common/request/request.hpp"

#ifdef ENABLE_DEBUG
void ccl_request::set_dump_callback(dump_func&& callback) {
    dump_callback = std::move(callback);
}
#endif

ccl_request::ccl_request(ccl_sched& sched) : sched(sched) {
#ifdef ENABLE_DEBUG
    set_dump_callback([&sched](std::ostream& out) {
        sched.dump(out);
    });
#endif
}

ccl_request::~ccl_request() {
    auto counter = completion_counter.load(std::memory_order_acquire);
    LOG_DEBUG("delete req ", this, " with counter ", counter);
    if (counter != 0 && !ccl::global_data::get().is_ft_enabled) {
        LOG_WARN("unexpected completion_counter ", counter);
    }
}

bool ccl_request::complete() {
    return complete_counter() == 0;
}

int ccl_request::complete_counter() {
    int prev_counter = completion_counter.fetch_sub(1, std::memory_order_release);
    CCL_THROW_IF_NOT(prev_counter > 0, "unexpected prev_counter ", prev_counter, ", req ", this);
    LOG_DEBUG("req ", this, ", counter ", prev_counter - 1);
    return prev_counter - 1;
}

bool ccl_request::is_completed() const {
    auto counter = completion_counter.load(std::memory_order_acquire);

#ifdef ENABLE_DEBUG
    if (counter != 0) {
        ++complete_checks_count;
        if (complete_checks_count >= CHECK_COUNT_BEFORE_DUMP) {
            complete_checks_count = 0;
            dump_callback(std::cout);
        }
    }
#endif
    LOG_TRACE("req: ", this, ", counter ", counter);

    return counter == 0;
}

void ccl_request::set_counter(int counter) {
    // add +1 to the inital counter value, this allows us to order
    // finalization/cleanup work on the request and its schedule right
    // before its completion, like this:
    // if (req->complete_counter() == 1) {
    //    // do cleanup
    //    req->complete();
    // }
    // this is important because it protects request and sched from
    // being destroyed at the time of this finalization work by
    // a user thread waiting on the event: with this approach
    // user thread will detect event completion only after finalization
    // work is done
    int adjusted_counter = counter + 1;
    LOG_DEBUG("req: ", this, ", set count ", adjusted_counter);
    int current_counter = completion_counter.load(std::memory_order_acquire);
    CCL_THROW_IF_NOT(current_counter == 0, "unexpected counter ", current_counter);
    completion_counter.store(adjusted_counter, std::memory_order_release);
}

void ccl_request::increase_counter(int increment) {
    LOG_DEBUG("req: ", this, ", increment ", increment);
    int prev_counter = completion_counter.fetch_add(increment, std::memory_order_release);
    CCL_THROW_IF_NOT(prev_counter > 0, "unexpected prev_counter ", prev_counter, ", req ", this);
    LOG_DEBUG("req ", this, ", counter ", prev_counter + increment);
}

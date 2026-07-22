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
#include "sched/entry/entry.hpp"
#include "sched/sched.hpp"

sched_entry::sched_entry(ccl_sched* sched, bool is_barrier, bool is_coll, bool is_deps)
        : sched(sched),
          barrier(is_barrier),
          coll(is_coll),
          deps(is_deps) {
    use_total_timer = ccl::global_data::env().sched_profile;
    detect_update_time_expiration =
        ccl::global_data::env().entry_max_update_time_sec != CCL_ENV_SIZET_NOT_SPECIFIED;
    use_update_timer = ccl::global_data::env().sched_profile || detect_update_time_expiration;
}

void sched_entry::do_progress() {
    if (is_completed())
        return;

    if (status < ccl_sched_entry_status_started) {
        CCL_THROW_IF_NOT(
            status == ccl_sched_entry_status_not_started || status == ccl_sched_entry_status_again,
            "bad status ",
            status,
            "(",
            status_to_str(status),
            ")");

        bool took_credits = false;
        if (status == ccl_sched_entry_status_not_started) {
            took_credits = sched->flow_control.take_credit();
            if (took_credits && use_total_timer) {
                total_timer.start();
            }
        }
        else if (status == ccl_sched_entry_status_again) {
            took_credits = true;
        }

        if (!took_credits) {
            return;
        }

#ifdef CCL_ENABLE_ITT
        this->itt_event = ccl::profile::itt::event_get(this->name());
        ccl::profile::itt::event_start(this->itt_event);
#endif // CCL_ENABLE_ITT

        start();
        CCL_THROW_IF_NOT(status >= ccl_sched_entry_status_again,
                         "bad status ",
                         status,
                         "(",
                         status_to_str(status),
                         ")");
    }
    else if (status == ccl_sched_entry_status_started) {
        LOG_TRACE("update entry ", name());

        if (use_update_timer && !update_timer.is_started()) {
            update_timer.start();
        }
        else if (update_timer.is_started() && detect_update_time_expiration) {
            // do this before entry::update so entry can handle this state inside update
            long double seconds = update_timer.get_elapsed_usec() / 1000000;
            if (seconds >= ccl::global_data::env().entry_max_update_time_sec) {
                is_update_time_expired = true;
            }
        }

        update();

        if (use_update_timer) {
            update_timer.update();
        }

        // ignore timeout on coll entry
        // actual timeout will be reported from sub-entries
        if (!is_coll()) {
            CCL_THROW_IF_NOT(
                !is_update_time_expired, "entry ", name(), " ", this, " update time expired");
        }

        CCL_THROW_IF_NOT(status >= ccl_sched_entry_status_started,
                         "bad status ",
                         status,
                         "(",
                         status_to_str(status),
                         ")");
    }

    if (status == ccl_sched_entry_status_complete) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::event_end(this->itt_event);
#endif // CCL_ENABLE_ITT

        if (use_total_timer) {
            total_timer.update();
        }

        if (exec_mode == ccl_sched_entry_exec_once) {
            status = ccl_sched_entry_status_complete_once;
        }

        sched->flow_control.return_credit();
    }

    CCL_THROW_IF_NOT(
        status != ccl_sched_entry_status_failed && status != ccl_sched_entry_status_invalid,
        "bad status ",
        status,
        "(",
        status_to_str(status),
        ")");
}

bool sched_entry::is_completed() {
    return (status == ccl_sched_entry_status_complete ||
            status == ccl_sched_entry_status_complete_once);
}

void sched_entry::update() {
    /*
       update is required for async ops (atl, ze, sync)
       for other ops it is empty method
    */
}

void sched_entry::reset(size_t idx) {
    if (use_total_timer) {
        total_timer.reset();
    }

    if (use_update_timer) {
        update_timer.reset();
        is_update_time_expired = false;
    }

    if (status == ccl_sched_entry_status_complete_once) {
        return;
    }

    start_idx = idx;
    status = ccl_sched_entry_status_not_started;
}

bool sched_entry::is_strict_order_satisfied() {
    return (status >= ccl_sched_entry_status_started);
}

void sched_entry::dump(std::stringstream& str, size_t idx) const {
    // update with the longest name
    const int entry_name_w = 14;

    ccl_logger::format(str,
                       "[",
                       std::left,
                       std::setw(3),
                       idx,
                       "] ",
                       std::left,
                       std::setw(entry_name_w),
                       name(),
                       " entry, address ",
                       this,
                       ", status ",
                       status_to_str(status),
                       " is_barrier ",
                       std::left,
                       std::setw(5),
                       barrier ? "TRUE" : "FALSE",
                       " ");
    dump_detail(str);
}

void sched_entry::make_barrier() {
    barrier = true;
}

bool sched_entry::is_barrier() const {
    return barrier;
}

bool sched_entry::is_coll() const {
    return coll;
}

bool sched_entry::is_deps() const {
    return deps;
}

ccl_sched_entry_status sched_entry::get_status() const {
    return status;
}

void sched_entry::set_status(ccl_sched_entry_status s) {
    status = s;
}

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
const ze_commands_t& sched_entry::get_ze_commands() const {
    return ze_commands;
}

uint32_t sched_entry::ze_commands_submit() {
    uint32_t cmd_counter = 0;
    LOG_DEBUG("entry ", name(), " ze_commands.size() ", ze_commands.size());
    for (auto& command : ze_commands) {
        LOG_DEBUG("adding command ", command->name(), " to command list");
        command->ze_call();
        ++cmd_counter;
    }
    LOG_DEBUG("entry ", name(), " all commands submitted");
    // TODO: determine the effect of destroying the commands on the cache (e.g. kernel cache)
    ze_commands.clear();
    return cmd_counter;
}
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

ccl_sched* sched_entry::get_sched() const {
    return sched;
}

void sched_entry::set_exec_mode(ccl_sched_entry_exec_mode mode) {
    exec_mode = mode;
}

std::string sched_entry::name_ext() const {
    return std::string(name());
}

void sched_entry::dump_detail(std::stringstream& str) const {}

void sched_entry::update_status(atl_status_t atl_status) {
    if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
        if (atl_status == ATL_STATUS_AGAIN) {
            status = ccl_sched_entry_status_again;
            return;
        }

        std::stringstream ss;
        dump_detail(ss);
        CCL_THROW("entry: ",
                  name(),
                  " failed. atl_status: ",
                  atl_status_to_str(atl_status),
                  ". Entry data:\n",
                  ss.str());
    }
    else {
        status = ccl_sched_entry_status_started;
    }
}

const char* sched_entry::status_to_str(ccl_sched_entry_status status) {
    switch (status) {
        case ccl_sched_entry_status_not_started: return "NOT_STARTED";
        case ccl_sched_entry_status_again: return "AGAIN";
        case ccl_sched_entry_status_started: return "STARTED";
        case ccl_sched_entry_status_complete: return "COMPLETE";
        case ccl_sched_entry_status_complete_once: return "COMPLETE_ONCE";
        case ccl_sched_entry_status_failed: return "FAILED";
        case ccl_sched_entry_status_invalid: return "INVALID";
        default: return "UNKNOWN";
    }
}

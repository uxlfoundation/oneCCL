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

#include "common/stream/stream.hpp"
#include "common/utils/sycl_utils.hpp"
#include "comm/comm.hpp"
#include "common/global/global.hpp"
#include "common/api_wrapper/ze_api_wrapper.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/ze_call.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/sched.hpp"

using namespace ccl;
using namespace ccl::ze;

// ze_base_entry
ze_base_entry::ze_base_entry(ccl_sched *sched,
                             const std::vector<ze_event_handle_t> &wait_events,
                             ccl_comm *comm,
                             uint32_t add_event_count)
        : sched_entry(sched),
          comm(comm),
          use_single_list(sched->use_single_list),
          wait_events(wait_events) {
    if (!this->comm) {
        this->comm = sched->coll_param.comm;
    }
    CCL_THROW_IF_NOT(this->comm, "no comm");
    comm_rank = this->comm->rank();
    comm_size = this->comm->size();

    // we can be here in case of copy_entry which may not have ze backend here, so check it
    if (sched->coll_param.stream &&
        sched->coll_param.stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        entry_event = sched->get_memory().event_manager->create();
    }
    // remember all ze entries for schedule
    // it is needed to execution modes processing
    sched->append_to_ze_entries_list(this);
    events.resize(add_event_count, nullptr);
}

ze_base_entry::~ze_base_entry() {
    finalize();
}

void ze_base_entry::init() {
    if (is_initialized) {
        return;
    }

    LOG_DEBUG("init");

    worker_idx = sched->queue->get_idx();

    CCL_THROW_IF_NOT(sched->coll_param.stream, "no stream");
    device = sched->coll_param.stream->get_ze_device();
    context = sched->coll_param.stream->get_ze_context();

    if (!use_single_list) {
        /* create event pool */
        if (events.size() > 0) {
            event_pool_desc = default_event_pool_desc;
            event_pool_desc.count = events.size();
            global_data::get().ze_data->cache->get(
                worker_idx, context, event_pool_desc, &event_pool);
            LOG_DEBUG("get event pool: { max event count: ", event_pool_desc.count, " }");
        }
    }

    init_ze_hook();

    is_initialized = true;

    LOG_DEBUG("init completed");
}

void ze_base_entry::finalize() {
    if (!is_initialized) {
        return;
    }

    LOG_DEBUG("finalize");

    is_finalized = true;

    finalize_ze_hook();
    destroy_events();

    if (!use_single_list) {
        /* event pool */
        if (event_pool) {
            global_data::get().ze_data->cache->push(
                worker_idx, context, event_pool_desc, event_pool);
        }
    }

    is_initialized = false;

    LOG_DEBUG("finalize completed");
}

void ze_base_entry::init_entries() {
    auto &entries = sched->ze_entries;
    if (entries.front() == this) {
        LOG_DEBUG("init ", entries.size(), " entries");

        for (auto &entry : entries) {
            entry->init();
        }
        auto sync_obj = sched->get_init_ze_hook_sync_obj();
        if (sync_obj) {
            sync_obj->visit();
        }
    }
}

void ze_base_entry::finalize_entries() {
    auto &entries = sched->ze_entries;
    if (entries.back() == this) {
        LOG_DEBUG("finalize ", entries.size(), " entries");
        for (auto &entry : entries) {
            entry->finalize();
        }
    }
}

void ze_base_entry::start() {
    if (use_single_list) {
        init_entries();
    }
    else {
        init();
        reset_events();
    }

    // there are two execution modes: single_list and non-single list
    // in single_list mode globally we have only one list per queue, so execute it elsewhere
    // in non single_list mode each entry execute only own lists
    if ((use_single_list && sched->ze_entries.front() == this &&
         sched->get_ze_commands_bypass_flag()) ||
        !use_single_list) {
        sched_entry::ze_commands_submit();
        sched->get_memory().list_manager->execute(this);
        // in case we are not in single list mode, we can only set_submitted_to_gpu(true)
        // on submission of the last ze_entry, as it means that everything is submitted
        if (use_single_list || (!use_single_list && sched->ze_entries.back() == this)) {
            sched->set_submitted_to_gpu(true);
        }
    }

    status = ccl_sched_entry_status_started;
}

bool ze_base_entry::is_event_completed(ze_event_handle_t event) {
    ze_result_t res = zeEventQueryStatus(event);
    CCL_THROW_IF_NOT(res == ZE_RESULT_SUCCESS || res == ZE_RESULT_NOT_READY,
                     "unexpected result from zeEventQueryStatus: ",
                     to_string(res));
    return (res == ZE_RESULT_SUCCESS);
}

void ze_base_entry::update() {
    bool complete = is_event_completed(entry_event);
    if (is_update_time_expired) {
        size_t complete_event_count = 0;
        for (auto &event : wait_events) {
            if (is_event_completed(event)) {
                complete_event_count++;
            }
        }
        LOG_DEBUG("completed ",
                  complete_event_count,
                  " of ",
                  wait_events.size(),
                  " wait events. Entry event ",
                  entry_event,
                  " is ",
                  (complete) ? "completed" : "not completed");
    }

    if (complete) {
        LOG_DEBUG(name(), " ", this, " entry complete");
        status = ccl_sched_entry_status_complete;

        if (use_single_list) {
            reset_events();
        }

        // Finalize must go after all operation with the event because it's destroyed there.
        if (!sched->coll_attr.to_cache) {
            if (use_single_list) {
                finalize_entries();
            }
            else {
                finalize();
            }
        }
    }
    else {
        // just return in case if the kernel is not ready yet
        // will check again on the next iteration
        return;
    }
}

ze_command_list_handle_t ze_base_entry::get_comp_list(uint32_t index) const {
    return sched->get_memory().list_manager->get_comp_list(this, wait_events, index);
}

ze_command_list_handle_t ze_base_entry::get_copy_list(copy_direction direction,
                                                      uint32_t index,
                                                      queue_group_type force_queue_type) const {
    return sched->get_memory().list_manager->get_copy_list(
        this, wait_events, direction, index, force_queue_type);
}

ze_event_handle_t ze_base_entry::create_event(ze_event_pool_handle_t event_pool,
                                              ze_event_desc_t event_desc) {
    ze_event_handle_t event;
    ZE_CALL(zeEventCreate, (event_pool, &event_desc, &event));
    return event;
}

ze_event_handle_t ze_base_entry::create_event() {
    if (use_single_list) {
        return sched->get_memory().event_manager->create();
    }

    ze_event_desc_t event_desc{ default_event_desc };
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_DEVICE;
    event_desc.index = event_counter++;
    LOG_DEBUG("create event with index ", event_desc.index);
    // Note: if this exception is encountered, we may need to increase the limit
    //       in the derived class, e.g., increase event_group_count
    //       in ze_a2a_reduce_scatter_copy_entry
    CCL_THROW_IF_NOT(event_desc.index < event_pool_desc.count,
                     ", event creation limit exceeded: ",
                     event_desc.index,
                     ", event_pool_desc.count: ",
                     event_pool_desc.count);
    CCL_THROW_IF_NOT(event_desc.index < events.size());

    ze_event_handle_t event = create_event(event_pool, event_desc);
    events[event_desc.index] = event;

    return event;
}

void ze_base_entry::reset_events() {
    if (use_single_list) {
        // events will be reseted in the event manager
        return;
    }

    for (auto &event : events) {
        if (event != nullptr) {
            ZE_CALL(zeEventHostReset, (event));
        }
    }
}

void ze_base_entry::destroy_events() {
    if (use_single_list) {
        // events will be destroyed in the event manager
        events.clear();
        return;
    }

    for (auto &event : events) {
        if (event != nullptr) {
            ZE_CALL(zeEventDestroy, (event));
        }
    }
    events.clear();
}

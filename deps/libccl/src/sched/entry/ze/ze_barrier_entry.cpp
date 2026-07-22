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

#include "common/api_wrapper/ze_api_wrapper.hpp"
#include "sched/entry/ze/ze_barrier_entry.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

ze_barrier_entry::ze_barrier_entry(ccl_sched* sched,
                                   ccl_comm* comm,
                                   ze_event_pool_handle_t& local_pool,
                                   size_t wait_event_idx)
        : sched_entry(sched),
          comm(comm),
          rank(comm->rank()),
          comm_size(comm->size()),
          wait_event_idx(wait_event_idx),
          local_pool(local_pool) {
    LOG_DEBUG("initialization");
    CCL_THROW_IF_NOT(sched, "no sched");
    CCL_THROW_IF_NOT(comm, "no comm");
    CCL_THROW_IF_NOT(local_pool, "no local event pool");
}

ze_barrier_entry::~ze_barrier_entry() {
    finalize();
}

void ze_barrier_entry::finalize() {
    LOG_DEBUG("finalization");
    ZE_CALL(zeEventDestroy, (signal_event));

    for (const auto& wait_event : wait_events) {
        ZE_CALL(zeEventDestroy, (wait_event.second));
    }
    wait_events.clear();
    LOG_DEBUG("finalization completed");
}

void ze_barrier_entry::start() {
    LOG_DEBUG("start");

    if (signal_event) {
        ZE_CALL(zeEventHostReset, (signal_event));
    }

    last_completed_event_idx = 0;

    ze_event_desc_t event_desc = default_event_desc;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST; //TODO: DEVICE
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.index = wait_event_idx;

    signal_event = ze_base_entry::create_event(local_pool, event_desc);
    LOG_DEBUG("signal event is created for rank: ", rank);

    for (int peer_rank = 0; peer_rank < comm_size; peer_rank++) {
        if (peer_rank == rank) {
            continue;
        }
        ze_event_pool_handle_t event_pool{};
        sched->get_memory().handle_manager.get(peer_rank, 2, event_pool, comm);
        wait_events.push_back({ peer_rank, ze_base_entry::create_event(event_pool, event_desc) });
        LOG_DEBUG(
            "wait_events.size: ", wait_events.size(), ", rank: ", rank, ", peer_rank: ", peer_rank);
    }

    ZE_CALL(zeEventHostSignal, (signal_event));

    status = ccl_sched_entry_status_started;
    LOG_DEBUG("start completed");
}

void ze_barrier_entry::update() {
    for (size_t event_idx = last_completed_event_idx; event_idx < wait_events.size(); event_idx++) {
        ze_event_handle_t event = wait_events[event_idx].second;
        CCL_THROW_IF_NOT(event, "event is not available");
        if (ze_base_entry::is_event_completed(event)) {
            last_completed_event_idx++;
            if (last_completed_event_idx == wait_events.size()) {
                LOG_DEBUG("event is completed. last_completed_event_idx: ",
                          last_completed_event_idx);
                status = ccl_sched_entry_status_complete;
                return;
            }
        }
    }
}

void ze_barrier_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str, "comm ", comm->to_string(), ", wait_events ", wait_events.size(), "\n");
}

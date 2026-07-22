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
#include "sched/queue/queue.hpp"

size_t ccl_sched_queue::get_idx() const {
    return idx;
}

void ccl_sched_bin::add(ccl_sched* sched) {
    if (ccl::global_data::env().priority_mode != ccl_priority_none) {
        CCL_ASSERT(sched->coll_attr.priority == priority,
                   "unexpected sched priority ",
                   sched->coll_attr.priority,
                   " expected ",
                   priority);
    }
    CCL_ASSERT(sched);
    CCL_ASSERT(!sched->bin);
    sched->bin = this;
    sched->queue = queue;
    sched_list.add(sched);
}

size_t ccl_sched_bin::erase(size_t idx, size_t& next_idx) {
    ccl_sched* sched = nullptr;
    size_t size = 0;
    {
        std::lock_guard<sched_queue_lock_t> lock(sched_list.elem_guard);
        size = sched_list.elems.size();
        CCL_THROW_IF_NOT(size > 0, "unexpected sched_list size ", size);
        CCL_ASSERT(idx < size);
        sched = sched_list.elems[idx];
        sched->set_in_bin_status(ccl_sched_in_bin_erased);
        sched->bin = nullptr;
        size--;
        std::swap(sched_list.elems[size], sched_list.elems[idx]);
        sched_list.elems.resize(size);
        CCL_ASSERT(sched_list.elems.size() == (size));
        next_idx = idx;
    }
    return size;
}

ccl_sched_queue::ccl_sched_queue(size_t idx, std::vector<size_t> atl_eps)
        : idx(idx),
          atl_eps(atl_eps) {
    LOG_DEBUG("created sched_queue, idx ",
              idx,
              ", atl_eps count ",
              atl_eps.size(),
              ", atl_eps[0] ",
              atl_eps[0]);

    if (ccl::global_data::env().priority_mode != ccl_priority_none) {
        CCL_ASSERT(atl_eps.size() == CCL_PRIORITY_BUCKET_COUNT,
                   "unexpected atl_eps count ",
                   atl_eps.size(),
                   ", expected ",
                   CCL_PRIORITY_BUCKET_COUNT);
    }
    else
        CCL_ASSERT(!atl_eps.empty());
}

ccl_sched_queue::~ccl_sched_queue() {
    size_t expected_max_priority = 0;
    ccl_sched_bin* expected_cached_max_priority_bin = nullptr;

    if (bins.size() >= 1) {
        ccl_sched_bin* bin = &(bins.begin()->second);
        expected_max_priority = bin->priority;
        expected_cached_max_priority_bin = bin;
        if (bins.size() > 1)
            LOG_WARN("unexpected bins size ", bins.size(), ", expected <= 1");
    }

    if (max_priority != expected_max_priority)
        LOG_WARN("unexpected max_priority ", max_priority, ", expected ", expected_max_priority);

    if (cached_max_priority_bin != expected_cached_max_priority_bin)
        LOG_WARN("unexpected cached_max_priority_bin");

    clear();
}

void ccl_sched_queue::add(ccl_sched* sched) {
    CCL_ASSERT(sched);
    CCL_ASSERT(!sched->bin);

    size_t priority = sched->get_priority();
    if (ccl::global_data::env().priority_mode != ccl_priority_none) {
        if (sched->coll_param.ctype == ccl_coll_barrier) {
            priority = max_priority;
            sched->coll_attr.priority = priority;
        }
    }

    sched->set_in_bin_status(ccl_sched_in_bin_added);

    LOG_DEBUG("add to bin: sched ", sched, ", priority ", priority);

    ccl_sched_bin* bin = nullptr;

    std::lock_guard<sched_queue_lock_t> lock(bins_guard);

    sched_bin_list_t::iterator it = bins.find(priority);
    if (it != bins.end()) {
        bin = &(it->second);
        LOG_DEBUG("found bin ", bin);
        CCL_ASSERT(bin->priority == priority);
        bin->add(sched);
    }
    else {
        size_t atl_ep = 0;
        if (ccl::global_data::env().priority_mode == ccl_priority_none)
            atl_ep = atl_eps[0];
        else {
            size_t ep_idx = (priority / CCL_PRIORITY_BUCKET_SIZE) % CCL_PRIORITY_BUCKET_COUNT;
            atl_ep = atl_eps[ep_idx];
            LOG_DEBUG("priority ", priority, ", ep_idx ", ep_idx);
        }

        // in-place construct priority bin with added sched
        auto emplace_result = bins.emplace(std::piecewise_construct,
                                           std::forward_as_tuple(priority),
                                           std::forward_as_tuple(this, atl_ep, priority, sched));
        CCL_ASSERT(emplace_result.second);
        bin = &(emplace_result.first->second);

        if (priority >= max_priority) {
            max_priority = priority;
            cached_max_priority_bin = bin;
        }
        LOG_DEBUG("didn't find bin, emplaced new one ", bin, " , max_priority ", max_priority);
    }

    CCL_ASSERT(bin);
}

size_t ccl_sched_queue::erase(ccl_sched_bin* bin, size_t sched_idx) {
    CCL_ASSERT(bin);
    size_t bin_priority = bin->get_priority();

    LOG_DEBUG("queue ", this, ", bin ", bin);

    size_t next_idx = 0;

    // erase sched and check bin size after
    // no need to lock whole `bins` for single erase
    if (!bin->erase(sched_idx, next_idx)) {
        // 'bin 'looks like empty, we can erase it from 'bins'.
        // double check on bin.empty(), before remove it from whole table
        std::lock_guard<sched_queue_lock_t> lock{ bins_guard };
        {
            // no need to lock 'bin' here, because all adding are under bins_guard protection
            if (bin->sched_list.elems.empty() /* && (bins.size() > 1)*/) {
                bins.erase(bin_priority);

                // change priority
                if (bins.empty()) {
                    max_priority = 0;
                    cached_max_priority_bin = nullptr;
                }
                else if (bin_priority == max_priority) {
                    max_priority--;
                    sched_bin_list_t::iterator it;
                    while ((it = bins.find(max_priority)) == bins.end()) {
                        max_priority--;
                    }
                    cached_max_priority_bin = &(it->second);
                }
            } // or do nothing, because somebody added new element in bin while we getting a lock
        }
    }

    return next_idx;
}

ccl_sched_bin* ccl_sched_queue::peek() {
    return cached_max_priority_bin;
}

std::vector<ccl_sched_bin*> ccl_sched_queue::peek_all() {
    std::lock_guard<sched_queue_lock_t> lock{ bins_guard };
    std::vector<ccl_sched_bin*> result;
    result.reserve(bins.size());
    for (auto& bin : bins) {
        result.emplace_back(&(bin.second));
    }
    return result;
}

void ccl_sched_queue::clear() {
    cached_max_priority_bin = nullptr;
    bins.clear();
    max_priority = 0;
}

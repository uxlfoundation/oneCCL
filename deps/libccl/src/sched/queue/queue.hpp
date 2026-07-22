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

#include "common/utils/spinlock.hpp"
#include "exec/exec.hpp"
#include "sched/sched.hpp"

#include <deque>
#include <unordered_map>

#define CCL_SCHED_QUEUE_INITIAL_BIN_COUNT (1024)

using sched_container_t = std::vector<ccl_sched*>;
using sched_bin_list_t = std::unordered_map<size_t, ccl_sched_bin>; // key - priority
using sched_queue_lock_t = ccl_spinlock;

/* ATL EP is limited resource, each priority bucket consumes single ATL EP and uses it for all bins in bucket */
#define CCL_PRIORITY_BUCKET_COUNT (1)

/* the size of priority bucket, each bin in bucket use the same ATL EP although bins have different priorities */
#define CCL_PRIORITY_BUCKET_SIZE (8)

#define CCL_BUCKET_INITIAL_ELEMS_COUNT (1024)
class ccl_sched_list {
public:
    friend class ccl_sched_bin;
    friend class ccl_sched_queue;

    ccl_sched_list() {
        CCL_UNUSED(padding_queue);
        elems.reserve(CCL_BUCKET_INITIAL_ELEMS_COUNT);
    }

    ccl_sched_list(ccl_sched* sched) {
        CCL_ASSERT(sched);
        elems.reserve(CCL_BUCKET_INITIAL_ELEMS_COUNT);
        elems.push_back(sched);
    }

    ~ccl_sched_list() {
        if (!elems.empty()) {
            LOG_WARN("unexpected elem_count ", elems.size(), ", expected 0");
        }
        clear();
    }

    void clear() {
        elems.clear();
    }

    ccl_sched_list& operator=(const ccl_sched_list& other) = delete;

    ccl_sched_list(const ccl_sched_list&) = delete;

    ccl_sched_list(ccl_sched_list&& src) {
        {
            std::lock_guard<sched_queue_lock_t> lock(src.elem_guard);
            elems = std::move(src.elems);
        }
    }
    ccl_sched_list& operator=(ccl_sched_list&& other) {
        if (this != &other) {
            std::lock(this->elem_guard, other.elem_guard);
            elems = std::move(other.elems);
            // make sure both already-locked mutexes are unlocked at the end of scope
            std::lock_guard<ccl_spinlock> own_lock(this->elem_guard, std::adopt_lock);
            std::lock_guard<ccl_spinlock> other_lock(other.elem_guard, std::adopt_lock);
        }
        return *this;
    }

    void add(ccl_sched* sched) {
        {
            std::lock_guard<sched_queue_lock_t> lock(elem_guard);
            elems.emplace_back(sched);
        }
    }

    size_t size() const {
        {
            std::lock_guard<sched_queue_lock_t> lock(elem_guard);
            return elems.size();
        }
    }

    bool empty() {
        {
            std::lock_guard<sched_queue_lock_t> lock(elem_guard);
            return elems.empty();
        }
    }

    ccl_sched* get(size_t idx) {
        {
            std::lock_guard<sched_queue_lock_t> lock(elem_guard);
            CCL_ASSERT(idx < elems.size());
            return elems[idx];
        }
    }

    ccl_sched* remove(size_t idx, size_t& next_idx) {
        ccl_sched* ret = nullptr;
        {
            std::lock_guard<sched_queue_lock_t> lock(elem_guard);
            size_t size = elems.size();
            CCL_ASSERT(idx < size);
            ret = elems[idx];
            std::swap(elems[size - 1], elems[idx]);
            elems.resize(size - 1);
            CCL_ASSERT(elems.size() == (size - 1));
            next_idx = idx;
        }
        return ret;
    }

    void dump(std::ostream& out) const {
        {
            auto sched_dump = ccl::global_data::env().sched_dump;
            std::lock_guard<sched_queue_lock_t> lock(elem_guard);
            if (sched_dump) {
                for (auto& e : elems) {
                    e->dump(out);
                }
            }
            else {
                for (size_t idx = 0; idx < elems.size(); idx++) {
                    out << "    [" << idx
                        << "]: " << ccl_coll_type_to_str(elems[idx]->coll_param.ctype) << "\n";
                }
            }
        }
    }

private:
    mutable sched_queue_lock_t elem_guard{};
    sched_container_t elems;
    char padding_queue[CACHELINE_SIZE];
};

class ccl_sched_bin {
public:
    friend class ccl_sched_queue;
    ccl_sched_bin(ccl_sched_queue* queue, size_t atl_ep, size_t priority, ccl_sched* sched)
            : queue(queue),
              atl_ep(atl_ep),
              sched_list(sched),
              priority(priority) {
        CCL_ASSERT(queue);
        CCL_ASSERT(sched);
        sched->bin = this;
        sched->queue = queue;
    }

    ~ccl_sched_bin() {
        sched_list.clear();
    }

    ccl_sched_bin() = delete;
    ccl_sched_bin(const ccl_sched_bin& other) = delete;
    ccl_sched_bin& operator=(const ccl_sched_bin& other) = delete;

    ccl_sched_bin(ccl_sched_bin&& src) = default;
    ccl_sched_bin& operator=(ccl_sched_bin&& other) = default;

    size_t size() const {
        return sched_list.size();
    }
    size_t get_priority() {
        return priority;
    }
    size_t get_atl_ep() {
        return atl_ep;
    }
    ccl_sched_queue* get_queue() {
        return queue;
    }

    void add(ccl_sched* sched);
    size_t erase(size_t idx, size_t& next);
    ccl_sched* get(size_t idx) {
        return sched_list.get(idx);
    }

    void dump(std::ostream& out) const {
        sched_list.dump(out);
    }

private:
    ccl_sched_queue* queue = nullptr; //!< pointer to the queue which owns the bin
    size_t atl_ep; //!< ATL communication endpoint
    ccl_sched_list sched_list; //!< list of schedules
    size_t priority{}; //!< the single priority for all elems
};

class ccl_sched_queue {
public:
    ccl_sched_queue(size_t idx, std::vector<size_t> atl_eps);

    ccl_sched_queue() = delete;
    ccl_sched_queue(const ccl_sched_queue& other) = delete;
    ccl_sched_queue& operator=(const ccl_sched_queue& other) = delete;
    ~ccl_sched_queue();

    size_t get_idx() const;

    void add(ccl_sched* sched);
    size_t erase(ccl_sched_bin* bin, size_t sched_idx);
    void clear();

    /**
     * Retrieve a pointer to the bin with the highest priority and number of its elements
     * @param bin_size[out] the current number of elements in bin. May have a zero if the queue has no bins with elements
     * @return a pointer to the bin with the highest priority or nullptr if there is no bins with content
     */
    ccl_sched_bin* peek();

    std::vector<ccl_sched_bin*> peek_all();

    void dump(std::ostream& out) const {
        {
            std::lock_guard<sched_queue_lock_t> lock(bins_guard);
            out << "{\n";
            out << "  sched_queue: idx: " << idx << " size: " << bins.size() << "\n";
            size_t bin_idx = 0;
            for (auto& bin : bins) {
                out << "   bin: idx: " << bin_idx << " priority: " << bin.first
                    << " size: " << bin.second.size() << "\n";
                bin_idx++;
                bin.second.dump(out);
            }
            out << "}\n";
        }
    }

private:
    mutable sched_queue_lock_t bins_guard{};

    size_t idx;
    std::vector<size_t> atl_eps;
    sched_bin_list_t bins{ CCL_SCHED_QUEUE_INITIAL_BIN_COUNT };
    size_t max_priority = 0;
    std::atomic<ccl_sched_bin*> cached_max_priority_bin{};
};

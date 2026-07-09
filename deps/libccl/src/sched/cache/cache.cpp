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
#include "sched/cache/cache.hpp"

ccl_sched* ccl_sched_cache::find_unsafe(const ccl_sched_key& key) const {
    ccl_sched* sched = nullptr;
    {
        auto it = table.find(key);
        if (it != table.end()) {
            sched = it->second;
        }
    }

#ifdef ENABLE_DEBUG
    if (sched && ccl::global_data::env().cache_key_type != ccl_cache_key_full) {
        LOG_DEBUG("do sanity check for found sched ", sched);
        CCL_THROW_IF_NOT(key.check(sched->coll_param, sched->coll_attr));
        LOG_DEBUG("sanity check is passed for sched ", sched);
    }
#endif

    return sched;
}

void ccl_sched_cache::recache(const ccl_sched_key& old_key, ccl_sched_key&& new_key) {
    {
        std::lock_guard<sched_cache_lock_t> lock{ guard };
        auto it = table.find(old_key);
        if (it == table.end()) {
            std::string error_message = "old_key wasn't found";
            CCL_ASSERT(false, error_message, old_key.match_id);
            throw ccl::exception(error_message + old_key.match_id);
        }
        ccl_sched* sched = it->second;
        table.erase(it);
        auto emplace_result = table.emplace(std::move(new_key), sched);
        CCL_THROW_IF_NOT(emplace_result.second);
    }
}

void ccl_sched_cache::release(ccl_sched* sched) {
    reference_counter--;
    LOG_DEBUG("releasing sched to cache: ", sched);
    LOG_TRACE("reference_counter=", reference_counter);
}

bool ccl_sched_cache::try_flush() {
    if (!ccl::global_data::env().enable_cache_flush)
        return true;

    std::lock_guard<sched_cache_lock_t> lock{ guard };

    if (reference_counter == 0) {
        for (auto it = table.begin(); it != table.end(); ++it) {
            ccl_sched* sched = it->second;
            CCL_ASSERT(sched);
            LOG_DEBUG("remove sched ", sched, " from cache");
            delete sched;
        }
        table.clear();
        return true;
    }
    else {
        return false;
    }
}

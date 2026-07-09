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
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/ze/ze_ipc_event_pool_manager.hpp"

using namespace ccl;
using namespace ccl::ze;

void ipc_event_pool_manager::init(const ccl_stream* init_stream) {
    LOG_DEBUG("init");
    CCL_THROW_IF_NOT(init_stream, "no stream");

    context = init_stream->get_ze_context();
    CCL_THROW_IF_NOT(context, "context is not valid");
    LOG_DEBUG("init completed");
}

void ipc_event_pool_manager::clear() {
    for (const auto& pool_info : event_pool_info) {
        ccl::global_data::get().ze_data->cache->push(0, context, pool_info.second, pool_info.first);
    }
    event_pool_info.clear();
    LOG_DEBUG("finalize completed");
}

ze_event_pool_handle_t ipc_event_pool_manager::create(size_t event_count) {
    CCL_THROW_IF_NOT(context, "context is unavailable");

    ze_event_pool_desc_t event_pool_desc = default_event_pool_desc;
    event_pool_desc.count = event_count;
    event_pool_desc.flags = ZE_EVENT_POOL_FLAG_IPC | ZE_EVENT_POOL_FLAG_HOST_VISIBLE;

    ze_event_pool_handle_t event_pool{};
    ccl::global_data::get().ze_data->cache->get(0, context, event_pool_desc, &event_pool);
    CCL_THROW_IF_NOT(event_pool, "ipc event pool is unavailable");

    event_pool_info.push_back({ event_pool, event_pool_desc });

    LOG_DEBUG("created manager completed. event_pool_info.size: ", event_pool_info.size());
    return event_pool;
}

void ipc_event_pool_manager::check_ipc_event_count(ccl_coll_type ctype,
                                                   const size_t& ipc_event_count,
                                                   const size_t& max_ipc_event_count) {
    if (ccl::global_data::env().enable_ze_barrier) {
        CCL_THROW_IF_NOT(ipc_event_count <= max_ipc_event_count,
                         "coll:  ",
                         ccl_coll_type_to_str(ctype),
                         ", unexpected ipc_event_count: ",
                         ipc_event_count,
                         ", expected max: ",
                         max_ipc_event_count);
    }
}

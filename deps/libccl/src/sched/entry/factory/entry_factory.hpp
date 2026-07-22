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

#include "sched/entry/coll/coll_entry.hpp"
#include "sched/entry/coll/direct/allgatherv_entry.hpp"
#include "sched/entry/coll/direct/allreduce_entry.hpp"
#include "sched/entry/coll/direct/alltoall_entry.hpp"
#include "sched/entry/coll/direct/alltoallv_entry.hpp"
#include "sched/entry/coll/direct/barrier_entry.hpp"
#include "sched/entry/coll/direct/bcast_entry.hpp"
#include "sched/entry/coll/direct/reduce_entry.hpp"
#include "sched/entry/coll/direct/reduce_scatter_entry.hpp"

#include "sched/entry/factory/entry_factory.h"

#include "sched/entry/copy/copy_entry.hpp"
#include "sched/entry/deps_entry.hpp"
#include "sched/entry/deregister_entry.hpp"
#include "sched/entry/function_entry.hpp"
#include "sched/entry/probe_entry.hpp"
#include "sched/entry/recv_entry.hpp"
#include "sched/entry/recv_copy_entry.hpp"
#include "sched/entry/recv_reduce_entry.hpp"
#include "sched/entry/reduce_local_entry.hpp"
#include "sched/entry/register_entry.hpp"
#include "sched/entry/send_entry.hpp"
#include "sched/entry/subsched_entry.hpp"
#include "sched/entry/sync_entry.hpp"
#include "sched/entry/wait_value_entry.hpp"
#include "sched/entry/write_entry.hpp"

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "sched/entry/ze/allreduce/ze_a2a_allreduce_entry.hpp"
#include "sched/entry/ze/allreduce/ze_onesided_allreduce_entry.hpp"
#include "sched/entry/ze/ze_a2a_allgatherv_entry.hpp"
#include "sched/entry/ze/ze_a2a_reduce_scatter_entry.hpp"
#include "sched/entry/ze/ze_alltoallv_entry.hpp"
#include "sched/entry/ze/ze_barrier_entry.hpp"
#include "sched/entry/ze/ze_copy_entry.hpp"
#include "sched/entry/ze/ze_handle_exchange_entry.hpp"
#include "sched/entry/ze/ze_event_signal_entry.hpp"
#include "sched/entry/ze/ze_event_wait_entry.hpp"
#include "sched/entry/ze/ze_execute_cmdlists_entry.hpp"
#include "sched/entry/ze/ze_cmdlist_event_signal_entry.hpp"
#include "sched/entry/ze/ze_onesided_reduce_entry.hpp"
#include "sched/entry/ze/ze_reduce_local_entry.hpp"
#include "sched/entry/ze/ze_a2a_pipeline_reduce_scatter_entry.hpp"
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

#include "sched/sched.hpp"

namespace entry_factory {

template <class EntryType, class... Arguments>
EntryType* create(ccl_sched* sched, Arguments&&... args) {
    LOG_DEBUG("creating: ", EntryType::class_name(), " entry");
    EntryType* new_entry = detail::entry_creator<EntryType>::template make_entry<
        ccl_sched_add_mode::ccl_sched_add_mode_last_value>(sched, std::forward<Arguments>(args)...);
    LOG_DEBUG("created: ", EntryType::class_name(), ", entry: ", new_entry, ", sched: ", sched);
    return new_entry;
}

} // namespace entry_factory

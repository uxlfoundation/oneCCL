/*
 Copyright 2016-2020 Intel Corporation
 
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

#include "common/global/global.hpp"
#include "sched/entry/copy/copy_helper.hpp"
#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "sched/entry/ze/ze_handle_exchange_entry.hpp"
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

namespace ccl {

void add_coll_entry(ccl_sched* sched, const ccl_coll_param& param);

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
static constexpr int invalid_host_buf_size = 0;

void add_wait_events(ccl_sched* sched, const std::vector<ze_event_handle_t>& wait_events);
void add_signal_event(ccl_sched* sched, ze_event_handle_t signal_event);
ze_event_handle_t add_signal_event(ccl_sched* sched);

void add_comm_barrier(ccl_sched* sched,
                      ccl_comm* comm,
                      const std::vector<ze_event_handle_t>& wait_events,
                      ze_event_handle_t& out_event,
                      ze_event_pool_handle_t ipc_pool = {},
                      size_t ipc_event_idx = 0);

void add_handle_exchange(ccl_sched* sched,
                         ccl_comm* comm,
                         const std::vector<ze_event_handle_t>& wait_events,
                         ze_event_handle_t& out_event,
                         const std::vector<ze_handle_exchange_entry::mem_desc_t>& in_buffers,
                         int skip_rank = ccl_comm::invalid_rank,
                         ze_event_pool_handle_t pool = nullptr,
                         size_t event_idx = 0,
                         const ccl::utils::pt2pt_handle_exchange_info& info = {});

ze_event_handle_t add_coll(ccl_sched* sched,
                           const ccl_coll_param& param,
                           const std::vector<ze_event_handle_t>& wait_events);

void add_scaleout(ccl_sched* sched,
                  const ccl_coll_param& in_coll_param,
                  const bool is_single_node,
                  const std::vector<ze_event_handle_t>& wait_events,
                  ze_event_handle_t& out_event,
                  const copy_attr& h2d_copy_attr = copy_attr(copy_direction::h2d),
                  ccl_comm* global_comm = nullptr,
                  ccl_buffer global_recv = {},
                  int global_root = 0);

bool is_queue_in_order(const ccl_stream* s);
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

CCL_API bool is_allgatherv_inplace(const void* send_buf,
                                   const size_t send_count,
                                   const void* recv_buf,
                                   const size_t* recv_counts,
                                   const size_t dtype_size,
                                   const size_t rank,
                                   const size_t comm_size);
bool is_reduce_scatter_inplace(const void* send_buf,
                               const void* recv_buf,
                               const size_t recv_count,
                               const size_t dtype_size,
                               const size_t rank,
                               const size_t comm_size);

void enable_direct_fallback_for_pt2pt();
void restore_pt2pt_fallback_table();

} // namespace ccl

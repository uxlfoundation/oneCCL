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
#include "comp/comp.hpp"
#include "sched/entry/ze/allreduce/ze_a2a_allreduce_entry.hpp"
#include "sched/entry/ze/ze_a2a_allgatherv_entry.hpp"
#include "sched/entry/ze/ze_a2a_reduce_scatter_entry.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/queue/queue.hpp"

#include <algorithm>
#include <string>
#include <sstream>

using namespace ccl;
using namespace ccl::ze;

ze_a2a_allreduce_entry::ze_a2a_allreduce_entry(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t cnt,
                                               const ccl_datatype& dtype,
                                               reduction op,
                                               ccl_comm* comm,
                                               const std::vector<ze_event_handle_t>& wait_events,
                                               size_t send_buf_idx,
                                               size_t recv_buf_idx,
                                               size_t peer_buf_offset)
        : ze_base_entry(sched, wait_events, comm, comm->size() * event_group_count),
          send_buf(send_buf),
          recv_buf(recv_buf),
          cnt(cnt),
          dtype(dtype),
          op(op),
          send_buf_idx(send_buf_idx),
          recv_buf_idx(recv_buf_idx),
          peer_buf_offset(peer_buf_offset),
          peer_count(comm->size() - 1) {
    size_t segment_count = cnt / comm->size();
    bool count_check =
        (segment_count > 0) || (segment_count == 0 && static_cast<size_t>(comm->rank()) < cnt);
    skip_entry = !count_check || ((comm->size() == 1) && (send_buf == recv_buf));
}

void ze_a2a_allreduce_entry::init_ze_hook() {
    if (skip_entry) {
        if (wait_events.empty()) {
            ZE_APPEND_CALL(ze_cmd_signal_event, get_copy_list(), ze_base_entry::entry_event);
        }
        else {
            ZE_APPEND_CALL(
                ze_cmd_barrier, get_copy_list(), ze_base_entry::entry_event, wait_events);
        }
        return;
    }

    /* get peer buffers */
    std::vector<ccl_buffer> peer_send_bufs(peer_count);
    // allgatherv entry requires the peer_recv_bufs at the same index as rank
    std::vector<ccl_buffer> peer_recv_bufs(comm->size());

    for (int i = 0; i < peer_count; ++i) {
        int peer_rank = (comm_rank + i + 1) % comm->size();
        sched->get_memory().handle_manager.get(peer_rank, send_buf_idx, peer_send_bufs[i], comm);
        CCL_THROW_IF_NOT(peer_send_bufs[i].get_ptr(), "null IPC buffer is received");
        sched->get_memory().handle_manager.get(
            peer_rank, recv_buf_idx, peer_recv_bufs[peer_rank], comm);
        CCL_THROW_IF_NOT(peer_recv_bufs[peer_rank].get_ptr(), "null IPC buffer is received");
    }

    size_t main_block_count = cnt / comm_size;
    if (main_block_count == 0 && static_cast<size_t>(comm_rank) < cnt) {
        main_block_count = 1;
    }

    size_t block_count = main_block_count;
    if (comm_rank == comm_size - 1) {
        block_count += cnt - main_block_count * comm_size;
    }

    CCL_THROW_IF_NOT(main_block_count > 0, "wrong segment count");

    /* alloc temp buffer */
    size_t tmp_buf_bytes = peer_count * block_count * dtype.size();
    ccl::alloc_param alloc_param(tmp_buf_bytes, buffer_type::ze, buffer_place::device);
    ccl_buffer tmp_buf = sched->alloc_buffer(alloc_param);

    LOG_DEBUG("rank ",
              comm_rank,
              ", main_block_count: ",
              main_block_count,
              ", block_count: ",
              block_count,
              ", tmp buf size: ",
              tmp_buf_bytes,
              ", cnt: ",
              cnt);

    /* copy peer segments to temp buffer */

    // do no need separate memcpys when using monolithic kernel
    if (!ccl::global_data::env().reduce_scatter_monolithic_kernel) {
        pre_copy_events.resize(peer_count);
        for (auto& event : pre_copy_events) {
            event = ze_base_entry::create_event();
        }
    }

    if (ccl::global_data::env().reduce_scatter_monolithic_kernel) {
        // two kernels. one leftover kernel and an aligned kernel
        kernel_events.resize((int)ccl::utils::align_kernels::count);
    }
    else {
        // when kernel merge is used only one kernel is required
        kernel_events.resize(1);
    }

    for (auto& event : kernel_events) {
        event = ze_base_entry::create_event();
    }

    barrier_event = ze_base_entry::create_event();
    bool is_monolithic = ccl::global_data::env().reduce_scatter_monolithic_kernel;

    ze_a2a_reduce_scatter_entry::fill_list(this,
                                           send_buf.get_ptr(),
                                           tmp_buf.get_ptr(),
                                           tmp_buf.get_ptr(),
                                           peer_send_bufs,
                                           peer_count,
                                           comm_rank,
                                           block_count,
                                           comm_rank * main_block_count, // rank_buf_offset
                                           wait_events,
                                           pre_copy_events, // copy_events
                                           kernels,
                                           kernel_events,
                                           barrier_event,
                                           dtype,
                                           module,
                                           device,
                                           context,
                                           op,
                                           worker_idx,
                                           peer_buf_offset,
                                           is_monolithic);

    CCL_THROW_IF_NOT(!ccl::global_data::env().allgatherv_topo_read,
                     "ze_a2a_allreduce_entry with allgatherv_read not implemented for scaleup");
    // TODO: for doing allgatherv_read, we need to copy the reduced part from
    // tmp_buf to recv_bufs[comm_rank] and use in_place allgatherv because
    // we do not have the remote address of tmp_buf. Else use ipc exchange for tmp_buf.
    // also we need to do a comm_barrier before allgatherv entry to make sure
    // all remote ranks have finished reduce_scatter

    // for write, we can directly use tmp_buf and do not need in_place as true.

    bool is_monolithic_allgat = ccl::global_data::env().allgatherv_monolithic_kernel;
    // TODO: MLSL-1651 make int8 work with allgatherv write monolithic kernel
    if (dtype == ccl::datatype::int8) {
        is_monolithic_allgat = false;
    }
    if (is_monolithic_allgat) {
        // two for peer copy (unaligned and aligned kernel) and one for non-inplace tmp_buf copy
        post_copy_events.resize((int)ccl::utils::align_kernels::count + 1);
    }
    else {
        post_copy_events.resize(comm_size);
    }
    for (auto& event : post_copy_events) {
        event = ze_base_entry::create_event();
    }

    size_t main_block_bytes = main_block_count * dtype.size();
    std::vector<size_t> block_bytes(comm_size, main_block_bytes);
    // last rank chunk may have a different size due to leftover data
    block_bytes.back() += (cnt - main_block_count * comm_size) * dtype.size();

    std::vector<size_t> rank_buf_offsets(comm_size);
    rank_buf_offsets.at(comm_rank) = comm_rank * main_block_count;
    std::vector<ccl_buffer> recv_bufs;
    for (int i = 0; i < comm_size; i++) {
        recv_bufs.push_back(recv_buf + i * main_block_bytes);
    }

    std::vector<ccl_buffer> empty_bufs;
    std::vector<size_t> empty_counts;
    ze_a2a_allgatherv_op init_params(sched,
                                     this,
                                     comm,
                                     nullptr,
                                     dtype,
                                     tmp_buf,
                                     recv_bufs,
                                     peer_recv_bufs,
                                     empty_bufs,
                                     block_bytes,
                                     empty_counts,
                                     peer_count,
                                     rank_buf_offsets,
                                     peer_buf_offset,
                                     post_copy_events,
                                     kernel_events,
                                     ze_base_entry::entry_event,
                                     is_monolithic_allgat,
                                     false, // is_inplace
                                     false); // is_separate_block_handles
    ze_a2a_allgatherv_op::select(init_params, kernels);
}

void ze_a2a_allreduce_entry::start() {
    ze_base_entry::start();
}

void ze_a2a_allreduce_entry::update() {
    ze_base_entry::update();
}

std::string ze_a2a_allreduce_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << cnt * dtype.size();
    return out.str();
}

void ze_a2a_allreduce_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       "dt ",
                       ccl::global_data::get().dtypes->name(dtype),
                       ", cnt ",
                       cnt,
                       ", send_buf ",
                       send_buf,
                       ", recv_buf ",
                       recv_buf,
                       ", op ",
                       ccl_reduction_to_str(op),
                       ", comm ",
                       comm->to_string(),
                       ", context ",
                       context,
                       "\n");
}

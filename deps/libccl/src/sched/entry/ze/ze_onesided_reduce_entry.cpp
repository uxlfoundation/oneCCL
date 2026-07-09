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
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/ze_onesided_reduce_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/queue/queue.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#include <string>

using namespace ccl;
using namespace ccl::ze;

ze_onesided_reduce_entry::ze_onesided_reduce_entry(
    ccl_sched* sched,
    ccl_buffer send_buf,
    ccl_buffer recv_buf,
    size_t cnt,
    const ccl_datatype& dtype,
    reduction op,
    int root,
    ccl_comm* comm,
    const std::vector<ze_event_handle_t>& wait_events,
    size_t peer_buf_offset)
        : ze_base_entry(sched, wait_events, comm, 4 /* request additional events */),
          // one event for empty_kernel, one for kernel_1s_copy_ops and two for aligning monolithic kernel
          send_buf(send_buf),
          recv_buf(recv_buf),
          cnt(cnt),
          dtype(dtype),
          op(op),
          root(root),
          buf_size_bytes(dtype.size() * cnt),
          peer_buf_offset_bytes(dtype.size() * peer_buf_offset),
          empty_kernel_event(nullptr) {
    skip_entry = !cnt || ((comm->size() == 1) && (send_buf == recv_buf));
    if (skip_entry) {
        // skip entry init and finalize
        sched->ze_entries.pop_back();
    }
}

void ze_onesided_reduce_entry::init_ze_hook() {
    CCL_THROW_IF_NOT(comm_rank == root, "unexpected comm_rank ", comm_rank, ", expected ", root);

    send_buf_ptr = send_buf.get_ptr();
    recv_buf_ptr = recv_buf.get_ptr();

    if (comm->size() == 1) {
        ZE_APPEND_CALL(ze_cmd_memory_copy,
                       ze_base_entry::get_copy_list(),
                       recv_buf_ptr,
                       send_buf_ptr,
                       buf_size_bytes,
                       ze_base_entry::entry_event,
                       wait_events);
        return;
    }

    /* create kernels */
    ccl_buffer right_send_buf;
    int peer_rank = (comm_rank + 1) % comm_size;
    sched->get_memory().handle_manager.get(peer_rank, 0, right_send_buf, comm);
    LOG_DEBUG(
        "get IPC pointers from ", peer_rank, " by ", root, ", right_send_buf: ", right_send_buf);

    send_buf_ptr = send_buf.get_ptr();
    recv_buf_ptr = recv_buf.get_ptr();

    // TODO: in place case check! diff idx for handle_mngr

    right_send_buf_ptr = static_cast<char*>(right_send_buf.get_ptr()) + peer_buf_offset_bytes;

    void* kernel_input_buf2 = right_send_buf_ptr;
    if (global_data::env().enable_kernel_1s_copy_ops) {
        ccl::alloc_param alloc_param(buf_size_bytes, buffer_type::ze, buffer_place::device);
        void* tmp_buf_ptr = sched->alloc_buffer(alloc_param).get_ptr();
        kernel_input_buf2 = tmp_buf_ptr;
    }

    ccl::global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    main_kernel_name =
        "reduce_local_outofplace_kernel_" + to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
    LOG_DEBUG("get kernel: name: ", main_kernel_name);

    bool use_empty_kernel{};
    ze_event_handle_t kernel_wait_event = nullptr;
    if (ccl::global_data::env().enable_kernel_1s_ipc_wa) {
        use_empty_kernel = true;
        empty_kernel_event = ze_base_entry::create_event();

        LOG_DEBUG("append empty kernel");
        ze_group_count_t empty_group_count = { 1, 1, 1 };
        ze_kernel_args_t reduce_local_kernel_args{
            &comm_rank, &comm_size, &cnt, &send_buf_ptr, &kernel_input_buf2, &recv_buf_ptr
        };

        ze_kernel kernel(module,
                         "empty_kernel",
                         reduce_local_kernel_args,
                         cnt, // maybe 1 instead of cnt?
                         empty_group_count,
                         worker_idx);

        ZE_APPEND_CALL(ze_cmd_launch_kernel,
                       get_comp_list(),
                       std::move(kernel),
                       empty_kernel_event,
                       wait_events);

        kernel_wait_event = empty_kernel_event;
    }

    if (global_data::env().enable_kernel_1s_copy_ops) {
        LOG_DEBUG("one-sided multi-phase algorithm");

        copy_from_peer_event = ze_base_entry::create_event();
        // copy to tmp buf
        ZE_APPEND_CALL(ze_cmd_memory_copy,
                       ze_base_entry::get_copy_list(),
                       kernel_input_buf2,
                       right_send_buf_ptr,
                       buf_size_bytes,
                       copy_from_peer_event,
                       (use_empty_kernel) ? ze_events_t({ empty_kernel_event }) : wait_events);
        kernel_wait_event = copy_from_peer_event;
    }
    else {
        LOG_DEBUG("one-sided monolithic algorithm");
    }

    LOG_DEBUG("ze_onesided_reduce_entry with aligned monolithic kernels");
    // use send_buf_ptr instead of right_send_buf_ptr since we cannot make sure
    // if right_send_buf_ptr got using ipc has the same alignment as remote send_buf_ptr.
    // we assume local send_buf_ptr and remote send_buf_ptr has the same alignment
    unsigned long pre_align_offset_byte = ccl::utils::get_aligned_offset_byte(
        send_buf_ptr, buf_size_bytes, ccl::global_data::env().kernel_mem_align);

    // first kernel starts from location 0 to pre_align_offset_byte
    // and the second kernel starts from location pre_align_offset_byte to the rest
    constexpr int kernel_count = (int)ccl::utils::align_kernels::count;
    const unsigned long offsets[kernel_count] = { 0, pre_align_offset_byte };
    const unsigned long counts[kernel_count] = { pre_align_offset_byte / dtype.size(),
                                                 cnt - pre_align_offset_byte / dtype.size() };
    ze_event_handle_t events[kernel_count];
    size_t start_kernel_idx = 0;
    size_t end_kernel_idx = kernel_count;
    bool use_single_kernel = false;
    // when counts[0] is 0, only aligned kernel is needed
    if (counts[(int)ccl::utils::align_kernels::unaligned] == 0) {
        use_single_kernel = true;
        start_kernel_idx++;
    }
    // when counts[1] is 0, aligned kernel is not needed
    if (counts[(int)ccl::utils::align_kernels::aligned] == 0) {
        use_single_kernel = true;
        end_kernel_idx--;
    }

    // if the initial data is aligned, we need only one kernel
    // otherwise run two kernels, one for unaligned and one for aligned data
    for (size_t idx = start_kernel_idx; idx < end_kernel_idx; idx++) {
        void* send_buf_ptr_with_offset = static_cast<char*>(send_buf_ptr) + offsets[idx];
        void* recv_buf_ptr_with_offset = static_cast<char*>(recv_buf_ptr) + offsets[idx];
        void* kernel_input_buf2_with_offset = static_cast<char*>(kernel_input_buf2) + offsets[idx];
        ze_kernel_args_t main_kernel_args{ &comm_rank,
                                           &comm_size,
                                           &counts[idx],
                                           &send_buf_ptr_with_offset,
                                           &kernel_input_buf2_with_offset,
                                           &recv_buf_ptr_with_offset };

        ze_kernel kernel(module, main_kernel_name, main_kernel_args, counts[idx], worker_idx);

        events[idx] =
            (use_single_kernel) ? ze_base_entry::entry_event : ze_base_entry::create_event();

        ZE_APPEND_CALL(ze_cmd_launch_kernel,
                       ze_base_entry::get_comp_list(),
                       std::move(kernel),
                       events[idx],
                       (kernel_wait_event) ? ze_events_t({ kernel_wait_event }) : wait_events);
    }

    // use a barrier to combine the events of the unalinged and aligned kernel
    if (!use_single_kernel) {
        ZE_APPEND_CALL(ze_cmd_barrier,
                       ze_base_entry::get_comp_list(),
                       ze_base_entry::entry_event,
                       ze_events_t({ events[0], events[1] }));
    }
}

void ze_onesided_reduce_entry::finalize_ze_hook() {}

void ze_onesided_reduce_entry::start() {
    if (skip_entry) {
        ZE_CALL(zeEventHostSignal, (ze_base_entry::entry_event));
        status = ccl_sched_entry_status_complete;
        return;
    }

    size_t kernel_counter = 0;
    if (ccl::global_data::env().enable_kernel_sync) {
        kernel_counter = ccl::global_data::get().ze_data->kernel_counter++;
    }

    if (kernel_counter == 0) {
        ze_base_entry::start();
    }
    else {
        ccl::global_data::get().ze_data->kernel_counter--;
        status = ccl_sched_entry_status_again;
    }
}

void ze_onesided_reduce_entry::update() {
    ze_base_entry::update();

    if (ccl::global_data::env().enable_kernel_sync &&
        ccl::global_data::get().ze_data->kernel_counter > 0) {
        ccl::global_data::get().ze_data->kernel_counter--;
    }
}

std::string ze_onesided_reduce_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << cnt * dtype.size();
    return out.str();
}

void ze_onesided_reduce_entry::dump_detail(std::stringstream& str) const {
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

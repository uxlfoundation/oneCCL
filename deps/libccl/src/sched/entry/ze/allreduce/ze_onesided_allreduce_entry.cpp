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
#include "sched/entry/ze/allreduce/ze_onesided_allreduce_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/queue/queue.hpp"

#include <string>
#include <sstream>

using namespace ccl;
using namespace ccl::ze;

ze_onesided_allreduce_entry::ze_onesided_allreduce_entry(
    ccl_sched* sched,
    ccl_buffer send_buf,
    ccl_buffer recv_buf,
    size_t cnt,
    const ccl_datatype& dtype,
    reduction op,
    ccl_comm* comm,
    const std::vector<ze_event_handle_t>& wait_events,
    size_t peer_buf_offset)
        : ze_base_entry(sched, wait_events, comm, 5 /* request additional events */),
          send_buf(send_buf),
          recv_buf(recv_buf),
          cnt(cnt),
          dtype(dtype),
          op(op),
          buf_size_bytes(dtype.size() * cnt),
          buf_offset_bytes(dtype.size() * peer_buf_offset) {}

void ze_onesided_allreduce_entry::init_ze_hook() {
    /* create kernels */
    ccl_buffer right_send_buf;
    ccl_buffer right_recv_buf;
    int peer_rank = (comm_rank + 1) % comm_size;

    send_buf_ptr = send_buf.get_ptr();
    recv_buf_ptr = recv_buf.get_ptr();
    sched->get_memory().handle_manager.get(peer_rank, 0, right_send_buf, comm);
    sched->get_memory().handle_manager.get(peer_rank, 1, right_recv_buf, comm);

    right_send_buf_ptr = static_cast<char*>(right_send_buf.get_ptr()) + buf_offset_bytes;
    right_recv_buf_ptr = static_cast<char*>(right_recv_buf.get_ptr()) + buf_offset_bytes;

    void* tmp_buf_ptr{};

    if (global_data::env().enable_kernel_1s_copy_ops) {
        main_kernel_name = "reduce_local_outofplace_kernel_";
        ccl::alloc_param alloc_param(buf_size_bytes, buffer_type::ze, buffer_place::device);
        tmp_buf_ptr = sched->alloc_buffer(alloc_param).get_ptr();
    }
    else {
        main_kernel_name = "allreduce_kernel_";
    }

    main_kernel_name += to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
    LOG_DEBUG("get kernel: name: ", main_kernel_name);

    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    ze_kernel_args_t allreduce_kernel_args{ &comm_rank,         &comm_size,    &cnt,
                                            &send_buf_ptr,      &recv_buf_ptr, &right_send_buf_ptr,
                                            &right_recv_buf_ptr };

    bool use_empty_kernel{ false };

    if (global_data::env().enable_kernel_1s_ipc_wa) {
        use_empty_kernel = true;
    }

    if (use_empty_kernel) {
        empty_kernel_event = ze_base_entry::create_event();
    }

    if (global_data::env().enable_kernel_1s_copy_ops) {
        copy_from_peer_event = ze_base_entry::create_event();
        reduce_local_kernel_event = ze_base_entry::create_event();
    }

    /* do appends */
    if (use_empty_kernel) {
        LOG_DEBUG("append empty kernel");
        ze_group_count_t empty_group_count = { 1, 1, 1 };

        ze_kernel kernel(module,
                         "empty_kernel",
                         allreduce_kernel_args,
                         cnt, // TODO: shouldn't this be 1?
                         empty_group_count,
                         worker_idx);

        ZE_APPEND_CALL(ze_cmd_launch_kernel,
                       ze_base_entry::get_comp_list(),
                       std::move(kernel),
                       empty_kernel_event,
                       wait_events);
    }

    if (global_data::env().enable_kernel_1s_copy_ops) {
        LOG_DEBUG("one-sided multi-phase algorithm");

        ze_kernel_args_t main_kernel_args{ &comm_rank,    &comm_size,   &cnt,
                                           &send_buf_ptr, &tmp_buf_ptr, &recv_buf_ptr };

        ZE_APPEND_CALL(ze_cmd_memory_copy,
                       ze_base_entry::get_copy_list(),
                       tmp_buf_ptr,
                       right_send_buf_ptr,
                       buf_size_bytes,
                       copy_from_peer_event,
                       (empty_kernel_event) ? ze_events_t({ empty_kernel_event }) : wait_events);

        ze_kernel kernel(module, main_kernel_name, main_kernel_args, cnt, worker_idx);

        ZE_APPEND_CALL(ze_cmd_launch_kernel,
                       ze_base_entry::get_comp_list(),
                       std::move(kernel),
                       reduce_local_kernel_event,
                       ze_events_t({ copy_from_peer_event }));

        ZE_APPEND_CALL(ze_cmd_memory_copy,
                       ze_base_entry::get_copy_list(),
                       right_recv_buf_ptr,
                       recv_buf_ptr,
                       buf_size_bytes,
                       ze_base_entry::entry_event,
                       ze_events_t({ reduce_local_kernel_event }));
    }
    else {
        LOG_DEBUG("one-sided monolithic algorithm");

        // use recv_buf_ptr instead of right_recv_buf_ptr since we cannot make sure
        // right_recv_buf_ptr got using ipc has the same alignment as remote recv_buf_ptr.
        // we assume local recv_buf_ptr and remote recv_buf_ptr has the same alignment
        unsigned long pre_align_offset_byte = ccl::utils::get_aligned_offset_byte(
            recv_buf_ptr, buf_size_bytes, ccl::global_data::env().kernel_mem_align);

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
        CCL_ASSERT(kernel_count == 2);
        for (size_t i = start_kernel_idx; i < end_kernel_idx; i++) {
            // the block stores can not be used for unaligned data that's why
            // it's skipped taking into account that the first kernel, if they
            // are two ones, is always unaligned
            int can_use_block = 1;
            if (i == 0 && end_kernel_idx == 2) {
                can_use_block = 0;
            }
            void* send_buf_ptr_tmp = static_cast<char*>(send_buf_ptr) + offsets[i];
            void* recv_buf_ptr_tmp = static_cast<char*>(recv_buf_ptr) + offsets[i];
            void* right_send_buf_ptr_tmp = static_cast<char*>(right_send_buf_ptr) + offsets[i];
            void* right_recv_buf_ptr_tmp = static_cast<char*>(right_recv_buf_ptr) + offsets[i];
            ze_kernel_args_t main_kernel_args{ &comm_rank,
                                               &comm_size,
                                               &counts[i],
                                               &can_use_block,
                                               &send_buf_ptr_tmp,
                                               &recv_buf_ptr_tmp,
                                               &right_send_buf_ptr_tmp,
                                               &right_recv_buf_ptr_tmp };
            ze_kernel kernel(module, main_kernel_name, main_kernel_args, counts[i], worker_idx);

            events[i] =
                (use_single_kernel) ? ze_base_entry::entry_event : ze_base_entry::create_event();

            ZE_APPEND_CALL(ze_cmd_launch_kernel,
                           ze_base_entry::get_comp_list(),
                           std::move(kernel),
                           events[i],
                           (use_empty_kernel) ? ze_events_t({ empty_kernel_event }) : wait_events);
        }

        // use a barrier to combine the events of the unalinged and aligned kernel
        if (!use_single_kernel) {
            ZE_APPEND_CALL(ze_cmd_barrier,
                           ze_base_entry::get_comp_list(),
                           ze_base_entry::entry_event,
                           ze_events_t({ events[0], events[1] }));
        }
    }
}

void ze_onesided_allreduce_entry::finalize_ze_hook() {}

void ze_onesided_allreduce_entry::start() {
    size_t kernel_counter = 0;
    if (global_data::env().enable_kernel_sync) {
        kernel_counter = global_data::get().ze_data->kernel_counter++;
    }

    if (kernel_counter == 0) {
        ze_base_entry::start();
    }
    else {
        global_data::get().ze_data->kernel_counter--;
        status = ccl_sched_entry_status_again;
    }
}

void ze_onesided_allreduce_entry::update() {
    ze_base_entry::update();

    if (global_data::env().enable_kernel_sync && global_data::get().ze_data->kernel_counter > 0) {
        global_data::get().ze_data->kernel_counter--;
    }
}

std::string ze_onesided_allreduce_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << cnt * dtype.size();
    return out.str();
}

void ze_onesided_allreduce_entry::dump_detail(std::stringstream& str) const {
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

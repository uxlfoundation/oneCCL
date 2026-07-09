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

#include "comp/comp.hpp"
#include "sched/entry/ze/ze_a2a_reduce_scatter_entry.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#include <numeric>

using namespace ccl;
using namespace ccl::ze;

ze_a2a_reduce_scatter_entry::ze_a2a_reduce_scatter_entry(
    ccl_sched* sched,
    ccl_buffer send_buf,
    ccl_buffer recv_buf,
    const size_t* recv_counts,
    const ccl_datatype& dtype,
    reduction op,
    ccl_comm* comm,
    const std::vector<ze_event_handle_t>& wait_events,
    size_t peer_buf_idx,
    size_t peer_buf_offset)
        : ze_base_entry(sched, wait_events, comm, comm->size() * event_group_count),
          send_buf(send_buf),
          recv_buf(recv_buf),
          dtype(dtype),
          op(op),
          recv_counts(recv_counts, recv_counts + comm->size()),
          peer_buf_idx(peer_buf_idx),
          peer_buf_offset(peer_buf_offset),
          peer_count(comm->size() - 1) {}

void ze_a2a_reduce_scatter_entry::kernel_init(size_t rank_buf_offset,
                                              size_t block_count,
                                              void* send_buf,
                                              void* output_buf,
                                              void* base_ptr,
                                              const std::vector<ccl_buffer>& peer_send_bufs,
                                              int peer_count,
                                              const ccl_datatype& dtype,
                                              int comm_rank,
                                              std::vector<ze_kernel>& kernels,
                                              ze_module_handle_t module,
                                              ze_device_handle_t device,
                                              ze_context_handle_t context,
                                              ccl::reduction op,
                                              size_t worker_idx,
                                              size_t peer_buf_offset,
                                              bool is_monolithic) {
    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    unsigned long count = block_count;
    if (is_monolithic && peer_count <= (int)ccl::ze::max_peer_count) {
        std::string monolithic_kernel_name =
            "reduce_monolithic_kernel_" + std::to_string(peer_count) + "_" +
            to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
        LOG_DEBUG("Reduce scatter monolithic kernel name: ", monolithic_kernel_name);

        // use send_buf instead of peer_send_buf_ptr since we cannot make sure
        // peer_send_buf_ptr got using ipc has the same alignment as remote send_buf=.
        // we assume local send_buf and remote send_buf has the same alignment
        void* send_buf_ptr = static_cast<char*>(send_buf) + rank_buf_offset * dtype.size();
        size_t buf_size_bytes = count * dtype.size();
        unsigned long pre_align_offset_byte = ccl::utils::get_aligned_offset_byte(
            send_buf_ptr, buf_size_bytes, ccl::global_data::env().kernel_mem_align);

        // First kernel starts from location 0 to pre_align_offset_byte
        // and the second kernel starts from location pre_align_offset_byte to the rest
        constexpr int kernel_count = (int)ccl::utils::align_kernels::count;
        kernels.reserve(kernel_count);

        const unsigned long offsets[kernel_count] = { 0, pre_align_offset_byte };
        const unsigned long counts[kernel_count] = { pre_align_offset_byte / dtype.size(),
                                                     count - pre_align_offset_byte / dtype.size() };

        // Start two kernels, first kernel for the part of the array before the aligned start offset
        // and second kernel for the rest of the array from the aligned start offset to the end
        for (int i = 0; i < kernel_count; i++) {
            unsigned long count_local = counts[i];
            // data count is small and there is no need to excute the second aligned kernel
            if (i == (int)ccl::utils::align_kernels::aligned && count_local == 0) {
                break;
            }
            void* input_buf = static_cast<char*>(send_buf_ptr) + offsets[i];
            std::vector<void*> peer_bufs;
            peer_bufs.reserve(peer_count);
            for (auto& peer_send_buf : peer_send_bufs) {
                void* peer_buf = static_cast<char*>(peer_send_buf.get_ptr()) +
                                 (rank_buf_offset + peer_buf_offset) * dtype.size() + offsets[i];
                peer_bufs.push_back(peer_buf);
            }
            void* output_ptr = static_cast<char*>(output_buf) + offsets[i];

            ze_kernel_args_t kernel_args{ &count_local,
                                          &input_buf,
                                          peer_bufs, //peer_bufs_ze_arg,
                                          &output_ptr };

            kernels.emplace_back(
                module, monolithic_kernel_name, kernel_args, count_local, worker_idx);
        }
    }
    else {
        // fallback path for monolithic kernel if peer_count > max_peer_count
        if (is_monolithic) {
            LOG_WARN("monolithic kernel not supported for peer_count ",
                     peer_count,
                     " > ",
                     ccl::ze::max_peer_count);
        }
        std::string kernel_name = "reduce_single_local_inplace_kernel_" + to_string(dtype.idx()) +
                                  "_" + ccl_reduction_to_str(op);

        LOG_DEBUG("get kernel_name: ", kernel_name);
        // reduce peer values in tmp_buf and own values in send_buf into tmp_buf
        kernels.reserve(1);
        void* input_buf1 = static_cast<char*>(send_buf) + rank_buf_offset * dtype.size();
        void* input_buf2 = base_ptr;
        ze_kernel_args_t kernel_args{ &count, &peer_count, &input_buf1, &input_buf2, &output_buf };
        kernels.emplace_back(module, kernel_name, kernel_args, count, worker_idx);
    }
}

void ze_a2a_reduce_scatter_entry::fill_list(const ze_base_entry* entry,
                                            void* send_buf,
                                            void* output_buf,
                                            void* tmp_buf,
                                            const std::vector<ccl_buffer>& peer_send_bufs,
                                            int peer_count,
                                            int comm_rank,
                                            size_t block_count,
                                            size_t rank_buf_offset,
                                            std::vector<ze_event_handle_t>& wait_events,
                                            std::vector<ze_event_handle_t>& copy_events,
                                            std::vector<ze_kernel>& kernels,
                                            std::vector<ze_event_handle_t>& kernel_events,
                                            ze_event_handle_t& barrier_event,
                                            const ccl_datatype& dtype,
                                            ze_module_handle_t module,
                                            ze_device_handle_t device,
                                            ze_context_handle_t context,
                                            ccl::reduction op,
                                            size_t worker_idx,
                                            size_t peer_buf_offset,
                                            bool is_monolithic) {
    CCL_THROW_IF_NOT(peer_count > 0, "peer_count must be more than 0");

    kernel_init(rank_buf_offset,
                block_count,
                send_buf,
                output_buf,
                tmp_buf,
                peer_send_bufs,
                peer_count,
                dtype,
                comm_rank,
                kernels,
                module,
                device,
                context,
                op,
                worker_idx,
                peer_buf_offset,
                is_monolithic);

    CCL_ASSERT(kernels.size(),
               "expecting to launch monolithic kernel(s), but no kernels were passed");
    if (is_monolithic && peer_count <= (int)ccl::ze::max_peer_count) {
        LOG_DEBUG("reduce stage with kernels");
        for (size_t i = 0; i < kernels.size(); ++i) {
            ZE_APPEND_CALL_TO_ENTRY(entry,
                                    ze_cmd_launch_kernel,
                                    entry->get_comp_list(),
                                    std::move(kernels[i]),
                                    //kernels[i].get_group_count(),
                                    kernel_events.at(i),
                                    wait_events);
        }
        // if only unaligned kernel is executed, then fill the event for
        // aligned kernel also since calling function expect two events
        if (kernels.size() < (int)ccl::utils::align_kernels::count) {
            CCL_THROW_IF_NOT(kernel_events.size() == (int)ccl::utils::align_kernels::count,
                             "monolithic kernel event count ",
                             kernel_events.size(),
                             " != ",
                             (int)ccl::utils::align_kernels::count);
            // assign kernel_events[1] = kernel_events[0]
            kernel_events.back() = kernel_events.front();
        }
    }
    else {
        LOG_DEBUG("copy peer segments to temp buffer");
        size_t copy_bytes = block_count * dtype.size();
        /* copy peer segments to temp buffer */
        for (int i = 0; i < peer_count; i++) {
            void* src = static_cast<char*>(peer_send_bufs[i].get_ptr()) +
                        (rank_buf_offset + peer_buf_offset) * dtype.size();
            void* dst = static_cast<char*>(tmp_buf) + i * copy_bytes;
            // TODO: if we are on the same device, then use t2t direction
            auto list = entry->get_copy_list(copy_direction::c2c, i);
            ZE_APPEND_CALL_TO_ENTRY(entry,
                                    ze_cmd_memory_copy,
                                    list,
                                    dst,
                                    src,
                                    copy_bytes,
                                    copy_events.at(i),
                                    wait_events);
        }

        ZE_APPEND_CALL_TO_ENTRY(
            entry, ze_cmd_barrier, entry->get_comp_list(), barrier_event, copy_events);

        // when output_buf == tmp_buf, then fill_list is invoked from allreduce_entry
        // and in that case we cannot signal the entry since allreduce_entry has an
        // allgatherv to finish after this reduce_scatter
        const bool is_signal_entry = (output_buf != tmp_buf);

        /* reduce stage */
        for (size_t i = 0; i < kernels.size(); ++i) {
            ZE_APPEND_CALL_TO_ENTRY(
                entry,
                ze_cmd_launch_kernel,
                entry->get_comp_list(),
                std::move(kernels[i]),
                is_signal_entry ? entry->entry_event : kernel_events.at(i),
                ze_events_t({ (i == 0) ? barrier_event : kernel_events.at(i - 1) }));
            // TODO: Can we parallelize by only waiting on barrier_event?
        }
    }
}

void ze_a2a_reduce_scatter_entry::init_ze_hook() {
    /* get peer buffers */
    bool is_monolithic = ccl::global_data::env().reduce_scatter_monolithic_kernel;
    std::vector<ccl_buffer> peer_send_bufs(peer_count);

    for (int i = 0; i < peer_count; ++i) {
        int peer_rank = (comm_rank + i + 1) % comm->size();
        sched->get_memory().handle_manager.get(peer_rank, peer_buf_idx, peer_send_bufs[i], comm);
        CCL_THROW_IF_NOT(peer_send_bufs[i].get_ptr(), "null IPC buffer is received");
    }

    /* alloc temp buffer */
    size_t buf_bytes = dtype.size() * recv_counts[comm_rank];
    size_t tmp_buf_bytes = peer_count * buf_bytes;
    if (tmp_buf_bytes == 0) {
        if (!wait_events.empty()) {
            ZE_APPEND_CALL(ze_cmd_barrier,
                           ze_base_entry::get_copy_list(),
                           ze_base_entry::entry_event,
                           wait_events);
        }
        return;
    }
    void* output_buf = recv_buf.get_ptr();
    void* tmp_buf;
    ccl::alloc_param alloc_param(tmp_buf_bytes, buffer_type::ze, buffer_place::device);
    if (is_monolithic && peer_count <= (int)ccl::ze::max_peer_count) {
        tmp_buf = nullptr;
    }
    else {
        tmp_buf = sched->alloc_buffer(alloc_param).get_ptr();
    }

    LOG_DEBUG("rank ",
              comm_size,
              ", tmp buf size: ",
              tmp_buf_bytes,
              ", buf_count: ",
              recv_counts[comm_rank]);

    /* copy peer segments to temp buffer */

    // do no need separate memcpys when using monolithic kernel
    if (!is_monolithic) {
        pre_copy_events.resize(peer_count);
        for (auto& event : pre_copy_events) {
            event = ze_base_entry::create_event();
        }
    }

    if (is_monolithic) {
        // leftover kernel and aligned kernel
        kernel_events.resize((int)ccl::utils::align_kernels::count);
    }
    else {
        // when kernel merge is used only one kernel is required
        kernel_events.resize(1);
    }

    for (auto& event : kernel_events) {
        event = ze_base_entry::create_event();
    }

    size_t rank_buf_offset = std::accumulate(
        recv_counts.begin(), recv_counts.begin() + comm_rank, ccl::utils::initial_count_value);

    barrier_event = ze_base_entry::create_event();

    fill_list(this,
              send_buf.get_ptr(),
              output_buf,
              tmp_buf,
              peer_send_bufs,
              peer_count,
              comm_rank,
              recv_counts[comm_rank],
              rank_buf_offset,
              wait_events,
              pre_copy_events,
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

    if (is_monolithic && peer_count <= (int)ccl::ze::max_peer_count) {
        // in case of monolithic kernel, use a barrier to combine the
        // events from unaligned and aligned kernels
        CCL_THROW_IF_NOT(kernel_events.size() == (int)ccl::utils::align_kernels::count,
                         "unexpected kernel events size: ",
                         kernel_events.size());
        ZE_APPEND_CALL(ze_cmd_barrier,
                       ze_base_entry::get_copy_list(),
                       ze_base_entry::entry_event,
                       kernel_events);
    }
    else {
        CCL_THROW_IF_NOT(kernel_events.size() == 1,
                         "single kernel event expected, size: ",
                         kernel_events.size());
    }
}

void ze_a2a_reduce_scatter_entry::update() {
    ze_base_entry::update();
}

std::string ze_a2a_reduce_scatter_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":"
        << std::accumulate(
               recv_counts.begin(), recv_counts.end(), ccl::utils::initial_count_value) *
               dtype.size();
    return out.str();
}

ze_a2a_reduce_scatter_write_copy_entry::ze_a2a_reduce_scatter_write_copy_entry(
    ccl_sched* sched,
    reduce_scatter_args rs_args,
    reduce_scatter_bufs rs_bufs,
    const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, rs_args.comm, rs_args.comm->size() * event_group_count),
          rs_args(rs_args),
          rs_bufs(std::move(rs_bufs)),
          peer_count(comm->size() - 1) {}

void ze_a2a_reduce_scatter_write_copy_entry::fill_list_copy(
    const ze_base_entry* entry,
    const reduce_scatter_args rs_args,
    const reduce_scatter_bufs rs_bufs,
    const std::vector<ccl_buffer>& peer_recv_bufs,
    int peer_count,
    int comm_rank,
    int comm_size,
    std::vector<ze_event_handle_t>& copy_events,
    ze_event_handle_t& barrier_event,
    ze_module_handle_t module,
    ze_device_handle_t device,
    ze_context_handle_t context,
    size_t worker_idx,
    const std::vector<ze_event_handle_t>& wait_events) {
    LOG_DEBUG("use fill_list_copy");

    for (int i = 0; i < peer_count; i++) {
        const int peer_rank = (comm_rank + i + 1) % comm_size;
        size_t copy_bytes_peer = rs_args.recv_counts[peer_rank] * rs_args.dtype.size();
        size_t peer_rank_buf_offset = std::accumulate(rs_args.recv_counts.begin(),
                                                      rs_args.recv_counts.begin() + peer_rank,
                                                      ccl::utils::initial_count_value);
        void* src_write = static_cast<char*>(rs_bufs.send_buf.get_ptr()) +
                          (peer_rank_buf_offset + rs_bufs.send_buf_offset) * rs_args.dtype.size();
        // write to tmp_buffer without creating any gap in the buffer:
        // When remote ranks write to tmp buffer of a given rank R, each rank fills up the block
        // corresponding to its rank, but none of them write to the block corresponding to R.
        // So, a gap is created. We're removing such gap to make the buffer contiguous.
        // It helps for further processing, e.g., during local reduce.
        size_t peer_block_offset = (comm_rank > peer_rank) ? comm_rank - 1 : comm_rank;
        void* dst_write =
            static_cast<char*>(peer_recv_bufs[i].get_ptr()) + peer_block_offset * copy_bytes_peer;
        // request copy engine at even index, it can be helpful in certain situations

        auto copy_engine_idx = (i + 1) * 2;
        if (ccl::global_data::env().type2_mode == type2_tune_mode::detected ||
            ccl::global_data::env().type2_mode == type2_tune_mode::on) {
            copy_engine_idx = i * 2;
        }
        auto list = entry->get_copy_list(copy_direction::c2c, copy_engine_idx);

        ZE_APPEND_CALL_TO_ENTRY(entry,
                                ze_cmd_memory_copy,
                                list,
                                dst_write,
                                src_write,
                                copy_bytes_peer,
                                copy_events.at(i),
                                wait_events);
    }
    ZE_APPEND_CALL_TO_ENTRY(
        entry, ze_cmd_barrier, entry->get_comp_list(), barrier_event, copy_events);
}

void ze_a2a_reduce_scatter_write_copy_entry::init_ze_hook() {
    // get peer buffer pointers
    std::vector<ccl_buffer> peer_recv_bufs(peer_count);
    if ((int)rs_bufs.peer_write_buf_idx > 0) {
        for (int i = 0; i < peer_count; ++i) {
            int peer_rank = (comm_rank + i + 1) % comm->size();
            sched->get_memory().handle_manager.get(
                peer_rank, rs_bufs.peer_write_buf_idx, peer_recv_bufs[i], comm);
            CCL_THROW_IF_NOT(peer_recv_bufs[i].get_ptr(), "null IPC buffer is received");
        }
    }

    pre_copy_events.resize(peer_count);
    for (auto& event : pre_copy_events) {
        event = ze_base_entry::create_event();
    }

    barrier_event = ze_base_entry::create_event();

    // copy segments to peer buffer
    fill_list_copy(this,
                   rs_args,
                   rs_bufs,
                   peer_recv_bufs,
                   peer_count,
                   comm_rank,
                   comm_size,
                   pre_copy_events,
                   barrier_event,
                   module,
                   device,
                   context,
                   worker_idx,
                   wait_events);

    // wait for post_copy_events and signal entry_event
    ZE_APPEND_CALL(ze_cmd_barrier,
                   get_copy_list(),
                   ze_base_entry::entry_event,
                   ze_events_t({ barrier_event }));
}

void ze_a2a_reduce_scatter_write_copy_entry::update() {
    ze_base_entry::update();
}

std::string ze_a2a_reduce_scatter_write_copy_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":"
        << std::accumulate(rs_args.recv_counts.begin(),
                           rs_args.recv_counts.end(),
                           ccl::utils::initial_count_value) *
               rs_args.dtype.size();
    return out.str();
}

ze_a2a_reduce_scatter_write_kernel_entry::ze_a2a_reduce_scatter_write_kernel_entry(
    ccl_sched* sched,
    reduce_scatter_args rs_args,
    reduce_scatter_bufs rs_bufs,
    const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, rs_args.comm, rs_args.comm->size() * event_group_count),
          rs_args(rs_args),
          rs_bufs(std::move(rs_bufs)),
          peer_count(rs_args.comm->size() - 1) {}

void ze_a2a_reduce_scatter_write_kernel_entry::kernel_init(size_t rank_buf_offset,
                                                           reduce_scatter_args rs_args,
                                                           reduce_scatter_bufs rs_bufs,
                                                           int peer_count,
                                                           int comm_rank,
                                                           std::vector<ze_kernel>& kernels,
                                                           ze_module_handle_t module,
                                                           ze_device_handle_t device,
                                                           ze_context_handle_t context,
                                                           size_t worker_idx) {
    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    size_t block_count = rs_args.recv_counts[comm_rank];
    unsigned long count = block_count;

    std::string kernel_name = "reduce_single_local_inplace_kernel_" +
                              to_string(rs_args.dtype.idx()) + "_" +
                              ccl_reduction_to_str(rs_args.op);

    LOG_DEBUG("get kernel name: ", kernel_name);
    // reduce peer values in tmp_buf and own values in send_buf into tmp_buf
    kernels.reserve(1);
    void* input_buf1 = static_cast<char*>(rs_bufs.send_buf.get_ptr()) +
                       (rank_buf_offset + rs_bufs.send_buf_offset) * rs_args.dtype.size();
    void* input_buf2 = rs_bufs.tmp_write_buf.get_ptr();
    void* output_buf = rs_bufs.recv_buf.get_ptr();

    ze_kernel_args_t kernel_args{ &count, &peer_count, &input_buf1, &input_buf2, &output_buf };
    kernels.emplace_back(module, kernel_name, kernel_args, count, worker_idx);
}

void ze_a2a_reduce_scatter_write_kernel_entry::fill_list_kernel(
    const ze_base_entry* entry,
    const reduce_scatter_args rs_args,
    const reduce_scatter_bufs rs_bufs,
    int peer_count,
    int comm_rank,
    int comm_size,
    size_t rank_buf_offset,
    std::vector<ze_kernel>& kernels,
    std::vector<ze_event_handle_t>& kernel_events,
    ze_module_handle_t module,
    ze_device_handle_t device,
    ze_context_handle_t context,
    size_t worker_idx,
    const std::vector<ze_event_handle_t>& wait_events) {
    kernel_init(rank_buf_offset,
                rs_args,
                rs_bufs,
                peer_count,
                comm_rank,
                kernels,
                module,
                device,
                context,
                worker_idx);
    CCL_ASSERT(kernels.size(),
               "expecting to launch monolithic kernel(s), but no kernels were passed");

    for (size_t i = 0; i < kernels.size(); ++i) {
        ZE_APPEND_CALL_TO_ENTRY(entry,
                                ze_cmd_launch_kernel,
                                entry->get_comp_list(),
                                std::move(kernels[i]),
                                entry->entry_event,
                                (i == 0) ? wait_events : ze_events_t({ kernel_events.at(i - 1) }));
    }
}

void ze_a2a_reduce_scatter_write_kernel_entry::init_ze_hook() {
    size_t buf_bytes = rs_args.dtype.size() * rs_args.recv_counts[comm_rank];

    if (!buf_bytes) {
        ZE_APPEND_CALL(ze_cmd_barrier,
                       ze_base_entry::get_copy_list(),
                       ze_base_entry::entry_event,
                       wait_events);
        return;
    }

    kernel_events.resize(1);

    for (auto& event : kernel_events) {
        event = ze_base_entry::create_event();
    }

    size_t rank_buf_offset = std::accumulate(rs_args.recv_counts.begin(),
                                             rs_args.recv_counts.begin() + comm_rank,
                                             ccl::utils::initial_count_value);

    barrier_event = ze_base_entry::create_event();

    fill_list_kernel(this,
                     rs_args,
                     rs_bufs,
                     peer_count,
                     comm_rank,
                     comm_size,
                     rank_buf_offset,
                     kernels,
                     kernel_events,
                     module,
                     device,
                     context,
                     worker_idx,
                     wait_events);
}

void ze_a2a_reduce_scatter_write_kernel_entry::update() {
    ze_base_entry::update();
}

std::string ze_a2a_reduce_scatter_write_kernel_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":"
        << std::accumulate(rs_args.recv_counts.begin(),
                           rs_args.recv_counts.end(),
                           ccl::utils::initial_count_value) *
               rs_args.dtype.size();
    return out.str();
}

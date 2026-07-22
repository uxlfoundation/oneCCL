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

#include "sched/entry/ze/ze_a2a_allgatherv_entry.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <numeric>

using namespace ccl;
using namespace ccl::ze;

// ze_a2a_allgatherv_entry
ze_a2a_allgatherv_entry::ze_a2a_allgatherv_entry(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 size_t send_count,
                                                 std::vector<ccl_buffer> recv_bufs,
                                                 std::vector<size_t> recv_counts,
                                                 const ccl_datatype& dtype,
                                                 ccl_comm* comm,
                                                 const std::vector<ze_event_handle_t>& wait_events,
                                                 size_t peer_buf_idx,
                                                 size_t peer_buf_offset,
                                                 bool is_monolithic_pipeline,
                                                 ccl_comm* pipeline_comm,
                                                 bool is_separate_block_handles,
                                                 bool is_scaleout,
                                                 size_t scaleout_offset)
        : ze_base_entry(sched, wait_events, comm, comm->size() * event_group_count),
          send_buf(send_buf),
          send_count(send_count),
          recv_bufs(std::move(recv_bufs)),
          recv_counts(std::move(recv_counts)),
          dtype(dtype),
          peer_buf_idx(peer_buf_idx),
          peer_buf_offset(peer_buf_offset),
          peer_count(comm->size() - 1),
          is_monolithic_pipeline(is_monolithic_pipeline),
          pipeline_comm(pipeline_comm),
          is_separate_block_handles(is_separate_block_handles),
          is_scaleout(is_scaleout),
          scaleout_offset(scaleout_offset) {}

void ze_a2a_allgatherv_entry::init_ze_hook() {
    /* get peer recv buffers */
    std::vector<ccl_buffer> peer_recv_bufs(comm->size());
    std::vector<ccl_buffer> pair_peer_recv_bufs(comm->size());

    if (is_monolithic_pipeline &&
        (dtype == ccl::datatype::bfloat16 || dtype == ccl::datatype::float16)) {
        ccl::global_data::env().kernel_mem_align = 64;
    }

    for (int i = 0; i < peer_count; ++i) {
        const int peer_rank = (comm_rank + i + 1) % comm->size();
        int peer_global_rank = comm->get_global_rank(peer_rank);
        if (is_scaleout) {
            // recv_bufs.size() gives size of global communicator
            peer_global_rank = (peer_global_rank + scaleout_offset) % recv_bufs.size();
        }

        ccl_buffer buf{};
        if (!(is_monolithic_pipeline && recv_counts.at(peer_rank) == 0)) {
            if (is_scaleout) {
                sched->get_memory().handle_manager.get(peer_rank, 1 + peer_global_rank, buf, comm);
            }
            else {
                sched->get_memory().handle_manager.get(peer_rank, peer_buf_idx, buf, comm);
            }
            CCL_THROW_IF_NOT(buf.get_ptr(), "null IPC buffer is received");
        }
        peer_recv_bufs[peer_rank] = buf;

        if (is_monolithic_pipeline && pipeline_comm != nullptr) {
            // get peer buffer handles with pair_comm peer when pair_comm size > 1
            if (pipeline_comm->size() > 1) {
                ccl_buffer buf_pair{};
                // currently pipeline_comm is pair_comm and it can have a maximum of 2 ranks
                CCL_THROW_IF_NOT(pipeline_comm->size() == 2,
                                 "algorithm only supports pipeline_comm of size 2");
                const int pair_peer_rank = (pipeline_comm->rank() + 1) % pipeline_comm->size();
                // when separate handles are used, there is a different handle for each peer rank.
                // position 0 is for send buffer, therefore add 1 for recv_buffer index.
                // when separate handles are not there, use the idx parameter for all peers.
                const size_t pair_peer_buf_idx =
                    (is_separate_block_handles) ? 1 + peer_global_rank : peer_buf_idx;
                // only get the buffer if it exists; otherwise, segfault occurs
                if (recv_counts[pair_peer_rank] > 0) {
                    sched->get_memory().handle_manager.get(
                        pair_peer_rank, pair_peer_buf_idx, buf_pair, pipeline_comm);
                }
                pair_peer_recv_bufs[peer_rank] = buf_pair;
            }
            else {
                pair_peer_recv_bufs[peer_rank] = buf;
            }
        }
    }

    bool is_inplace{};
    // is_separate_block_handles is used from allgatherv
    // topo which performs copy incase data is not inplace
    // and therefore we do not need a copy here
    if (is_separate_block_handles || send_buf == recv_bufs.at(comm_rank)) {
        is_inplace = true;
    }
    std::vector<size_t> rank_buf_offsets(comm_size);
    for (int i = 1; i < comm_size; i++) {
        rank_buf_offsets[i] = rank_buf_offsets[i - 1] + recv_counts[i - 1];
    }

    CCL_THROW_IF_NOT(is_separate_block_handles || send_count == recv_counts[comm_rank],
                     "allgatherv send_count :",
                     send_count,
                     " and recv_count of rank ",
                     comm_rank,
                     ":",
                     recv_counts[comm_rank],
                     " does not match");

    std::vector<size_t> block_bytes(comm_size);
    for (int i = 0; i < comm_size; i++) {
        block_bytes[i] = recv_counts[i] * dtype.size();
    }

    LOG_DEBUG("rank: ", comm_rank, ", block_bytes: ", block_bytes.at(comm_rank));

    bool is_monolithic = ccl::global_data::env().allgatherv_monolithic_kernel;
    bool is_read = ccl::global_data::env().allgatherv_topo_read;
    // TODO: MLSL-1651 make int8 work with allgatherv write monolithic kernel
    if (dtype == ccl::datatype::int8) {
        is_monolithic = false;
    }

    // when monolithic pipelined kernel is used, two events
    // are needed for the unaligned and aligned kernels.
    // otherwise we use an event for copying from each peer.
    size_t copy_events_size = (is_monolithic_pipeline) ? 2 : peer_count;
    // write requires two kernels, unaligned and aligned kernel
    if (is_monolithic && !is_read) {
        copy_events_size = (int)ccl::utils::align_kernels::count;
    }
    // need additional memcpy for non inplace data
    if (!is_inplace && std::any_of(recv_counts.begin(), recv_counts.end(), [](auto& i) {
            return i > 0;
        })) {
        copy_events_size++;
    }
    copy_events.resize(copy_events_size);
    for (auto& event : copy_events) {
        event = ze_base_entry::create_event();
    }

    ze_a2a_allgatherv_op init_params(sched,
                                     this,
                                     comm,
                                     pipeline_comm,
                                     dtype,
                                     send_buf,
                                     recv_bufs,
                                     peer_recv_bufs,
                                     pair_peer_recv_bufs,
                                     block_bytes,
                                     recv_counts,
                                     peer_count,
                                     rank_buf_offsets,
                                     peer_buf_offset,
                                     copy_events,
                                     wait_events,
                                     ze_base_entry::entry_event,
                                     is_monolithic,
                                     is_monolithic_pipeline,
                                     is_inplace,
                                     is_separate_block_handles,
                                     is_scaleout,
                                     scaleout_offset);
    ze_a2a_allgatherv_op::select(init_params, kernels);
}

void ze_a2a_allgatherv_entry::update() {
    ze_base_entry::update();
}

std::string ze_a2a_allgatherv_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << send_count * dtype.size();
    return out.str();
}

void ze_a2a_allgatherv_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str, "comm ", comm->to_string(), "\n");
}

// ze_a2a_allgatherv_op
ze_a2a_allgatherv_op::ze_a2a_allgatherv_op(ccl_sched* sched,
                                           const ze_base_entry* entry,
                                           ccl_comm* comm,
                                           ccl_comm* pipeline_comm,
                                           const ccl_datatype& dtype,
                                           ccl_buffer send_buf,
                                           const std::vector<ccl_buffer>& recv_bufs,
                                           const std::vector<ccl_buffer>& peer_bufs,
                                           const std::vector<ccl_buffer>& pair_peer_bufs,
                                           const std::vector<size_t>& copy_bytes,
                                           const std::vector<size_t>& recv_counts,
                                           int peer_count,
                                           const std::vector<size_t>& rank_buf_offsets,
                                           size_t peer_buf_offset,
                                           std::vector<ze_event_handle_t>& copy_events,
                                           std::vector<ze_event_handle_t>& wait_events,
                                           ze_event_handle_t out_event,
                                           bool is_monolithic,
                                           bool is_monolithic_pipeline,
                                           bool is_inplace,
                                           bool is_separate_block_handles,
                                           bool is_scaleout,
                                           size_t scaleout_offset)
        : sched(sched),
          entry(entry),
          comm(comm),
          pipeline_comm(pipeline_comm),
          dtype(dtype),
          send_buf(send_buf),
          recv_bufs(recv_bufs),
          peer_bufs(peer_bufs),
          pair_peer_bufs(pair_peer_bufs),
          copy_bytes(copy_bytes),
          recv_counts(recv_counts),
          peer_count(peer_count),
          rank_buf_offsets(rank_buf_offsets),
          peer_buf_offset(peer_buf_offset),
          copy_events(copy_events),
          wait_events(wait_events),
          out_event(out_event),
          is_monolithic(is_monolithic),
          is_monolithic_pipeline(is_monolithic_pipeline),
          is_inplace(is_inplace),
          is_separate_block_handles(is_separate_block_handles),
          is_scaleout(is_scaleout),
          scaleout_offset(scaleout_offset) {}

// main function to choose read/write operation for a2a_allgatherv
void ze_a2a_allgatherv_op::select(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels) {
    size_t num_op_events = 0;
    if (args.is_monolithic_pipeline) {
        // read data using xelink and then write that data through mdfi
        // input events: wait_events
        // output event: copy_events[0..1]
        ze_a2a_allgatherv_op::read_write(args, kernels);
        num_op_events = 2;
    }
    else if (ccl::global_data::env().allgatherv_topo_read) {
        // input events: wait_events
        // output event(s): copy_events[0..peer_count-1]
        ze_a2a_allgatherv_op::read(args);
        num_op_events = args.peer_count;
    }
    else {
        // input events: wait_events
        // output event(s): copy_events[0..peer_count-1]
        ze_a2a_allgatherv_op::write(args, kernels);
        num_op_events = args.peer_count;
    }

    CCL_ASSERT(args.copy_events.size() >= num_op_events);
    std::vector<ze_event_handle_t> op_events(args.copy_events.begin(),
                                             args.copy_events.begin() + num_op_events);
    CCL_ASSERT(op_events.size() == num_op_events);

    auto list = args.entry->get_copy_list(copy_direction::t2t);

    if (!args.is_inplace && args.copy_bytes.at(args.comm->rank()) > 0) {
        // copy send_buf to my buffer
        void* dst = args.recv_bufs.at(args.comm->rank()).get_ptr();
        if (args.is_monolithic_pipeline) {
            // TODO: how is this going to work in all cases? what if comm is !world_comm?
            // Then my_global_rank will point to an incorrect rank.
            // Which, at the very least, can get us referencing "recv_bufs[much_larger_than_size]".
            const int my_global_rank = args.comm->get_global_rank(args.comm->rank());
            dst = args.recv_bufs.at(my_global_rank).get_ptr();
        }
        ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                ze_cmd_memory_copy,
                                list,
                                dst,
                                args.send_buf.get_ptr(), // src
                                args.copy_bytes.at(args.comm->rank()),
                                args.out_event,
                                op_events);
    }
    else {
        // case:
        //      copy of zero buffer, no ze_cmd_memory_copy occured
        //      signalling copy_events[i] by hand is required for sync purposes
        //      ze_cmd_memory_copy cannot be called:
        //          src_buf might be null, which causes segfault
        //      copy_event cannot be skipped - it is referenced later

        ZE_APPEND_CALL_TO_ENTRY(args.entry, ze_cmd_barrier, list, args.out_event, op_events);
    }
}

// prepare params for pipeline to peers via monolithic kernels
// monolithic kernel reads data using xelink and then write that data through mdfi
void ze_a2a_allgatherv_op::read_write(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels) {
    auto& a = args;
    auto device = a.sched->coll_param.stream->get_ze_device();
    auto context = a.sched->coll_param.stream->get_ze_context();
    auto worker_idx = a.sched->queue->get_idx();

    ze_module_handle_t module = nullptr;
    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
    std::string monolithic_kernel_name =
        "read_write_monolithic_kernel_" + std::to_string(a.peer_count) + "_" +
        to_string(a.dtype.idx()) + "_" + ccl_reduction_to_str(ccl::reduction::custom);
    LOG_DEBUG("Allgatherv monolithic pipeline kernel name: ", monolithic_kernel_name);

    constexpr int unaligned_kernel_idx = (int)ccl::utils::align_kernels::unaligned; // 0
    constexpr int aligned_kernel_idx = (int)ccl::utils::align_kernels::aligned; // 1
    constexpr int kernel_count = (int)ccl::utils::align_kernels::count; // 2
    std::vector<size_t> counts[kernel_count];

    // kernel reads from peer_even_bufs using xelnik and write to local_bufs
    // and then copies from local_bufs to peer_pair_bufs using MDFI
    std::vector<const void*> peer_even_bufs[kernel_count], local_bufs[kernel_count],
        peer_pair_bufs[kernel_count];

    for (int i = 0; i < a.peer_count; i++) {
        const int peer_rank = (a.comm->rank() + i + 1) % a.comm->size();

        size_t base_index_local = 0, offset = 0;
        if (a.is_separate_block_handles) {
            // when using separate handles use the global rank
            // since user passes the count array with global index
            base_index_local = a.comm->get_global_rank(peer_rank);
            if (a.is_scaleout) {
                // recv_bufs.size() gives size of global communicator
                base_index_local = (base_index_local + a.scaleout_offset) % a.recv_bufs.size();
            }
        }
        else {
            // when using non-separate handles we fill the data
            // using the peer_rank as index
            base_index_local = peer_rank;
            // when only single handle to the base of buffer is there,
            // we add the offset for each peer to the base address
            offset = (a.rank_buf_offsets.at(peer_rank) + a.peer_buf_offset) * a.dtype.size();
        }

        const size_t count = a.recv_counts[base_index_local];

        const void* peer_even_buf =
            a.peer_bufs[peer_rank] ? (a.peer_bufs[peer_rank] + offset).get_ptr() : nullptr;
        const void* local_buf = a.recv_bufs[base_index_local].get_ptr();
        const void* peer_pair_buf = a.pair_peer_bufs[peer_rank]
                                        ? (a.pair_peer_bufs[peer_rank] + offset).get_ptr()
                                        : nullptr;

        // based on the starting address of local recv buffer,
        // create the unaligned and aligned kernels, based on the assumption
        // local and remote buffers have same alignment.
        size_t offset_byte = ccl::utils::get_aligned_offset_byte(
            local_buf, count * a.dtype.size(), ccl::global_data::env().kernel_mem_align);
        const size_t offset_count = offset_byte / a.dtype.size();
        counts[unaligned_kernel_idx].emplace_back(offset_count);
        counts[aligned_kernel_idx].emplace_back(count - offset_count);

        peer_even_bufs[unaligned_kernel_idx].push_back(peer_even_buf);
        local_bufs[unaligned_kernel_idx].push_back(local_buf);
        peer_pair_bufs[unaligned_kernel_idx].push_back(peer_pair_buf);

        // when aligned kernel is of zero size, adding offset_byte
        // will go outside allocated memory and therefore pass the
        // original pointer as dummy value
        if (counts[aligned_kernel_idx].back() == 0) {
            offset_byte = 0;
        }

        peer_even_bufs[aligned_kernel_idx].push_back((char*)peer_even_buf + offset_byte);
        local_bufs[aligned_kernel_idx].push_back((char*)local_buf + offset_byte);
        peer_pair_bufs[aligned_kernel_idx].push_back((char*)peer_pair_buf + offset_byte);
    }
    for (int i = 0; i < kernel_count; i++) {
        int pipeline_comm_size = a.pipeline_comm->size();

        ze_kernel_args_t kernel_args{ &pipeline_comm_size };
        auto try_push_arg = [&kernel_args](const std::vector<const void*>& buffers) {
            for (auto& buffer : buffers) {
                if (buffer) {
                    kernel_args.push_back(&buffer);
                }
                else {
                    kernel_args.push_back({});
                }
            }
        };

        try_push_arg(peer_even_bufs[i]);
        try_push_arg(local_bufs[i]);
        try_push_arg(peer_pair_bufs[i]);
        kernel_args.push_back(counts[i]);

        auto this_count = *std::max_element(counts[i].begin(), counts[i].end());
        ze_kernel kernel(module, monolithic_kernel_name, kernel_args, this_count, worker_idx);

        ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                ze_cmd_launch_kernel,
                                a.entry->get_comp_list(),
                                std::move(kernel),
                                a.copy_events.at(i),
                                a.wait_events);
    }
}

// prepare params for read from peers via ze copy
void ze_a2a_allgatherv_op::read(ze_a2a_allgatherv_op& args) {
    auto& a = args;

    if (a.is_monolithic) {
        LOG_DEBUG("allgatherv read is not supported by monolithic kernels");
    }

    for (int i = 0; i < a.peer_count; ++i) {
        const int peer_rank = (a.comm->rank() + i + 1) % a.comm->size();
        void* src = a.peer_bufs[peer_rank].get_ptr();
        if (a.is_inplace) {
            // TODO: use peer_bufs directly without adding offset
            src = (a.peer_bufs[peer_rank] +
                   (a.rank_buf_offsets.at(peer_rank) + a.peer_buf_offset) * a.dtype.size())
                      .get_ptr();
        }

        auto list = a.entry->get_copy_list(copy_direction::c2c, i);

        if (a.copy_bytes.at(peer_rank)) {
            ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                    ze_cmd_memory_copy,
                                    list,
                                    a.recv_bufs[peer_rank].get_ptr(),
                                    src,
                                    a.copy_bytes.at(peer_rank),
                                    a.copy_events.at(i),
                                    a.wait_events);
        }
        else {
            // case:
            //      copy of zero buffer, no ze_cmd_memory_copy occured
            //      signalling copy_events[i] by hand is required for sync purposes
            //      ze_cmd_memory_copy cannot be called:
            //          src_buf might be null, which causes segfault
            //      copy_event cannot be skipped - it is referenced later
            ZE_APPEND_CALL_TO_ENTRY(
                args.entry, ze_cmd_barrier, list, a.copy_events.at(i), a.wait_events);
        }
    }
}

// prepare params for write to peers via monolithic kernels
void ze_a2a_allgatherv_op::write(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels) {
    auto& a = args;
    auto device = a.sched->coll_param.stream->get_ze_device();
    auto context = a.sched->coll_param.stream->get_ze_context();
    auto worker_idx = a.sched->queue->get_idx();

    // copy send_buf to peer buffers
    std::vector<ccl_buffer> peer_dst_bufs;
    ccl_buffer src_buf = a.send_buf;
    if (a.is_inplace) {
        src_buf = a.recv_bufs.at(a.comm->rank());
    }
    for (int i = 0; i < a.peer_count; ++i) {
        const int peer_rank = (a.comm->rank() + i + 1) % a.comm->size();
        // TODO: use peer_bufs directly without adding offset
        ccl_buffer dst_buf =
            a.peer_bufs[peer_rank] +
            (a.rank_buf_offsets.at(a.comm->rank()) + a.peer_buf_offset) * a.dtype.size();

        if (a.is_monolithic) {
            peer_dst_bufs.push_back(dst_buf);
        }
        else {
            // request copy engine at even index, can be helpful in certain situation
            // TODO: if we on the same device, then use t2t direction
            auto copy_engine_idx = (i + 1) * 2;
            if (ccl::global_data::env().type2_mode == type2_tune_mode::detected ||
                ccl::global_data::env().type2_mode == type2_tune_mode::on) {
                copy_engine_idx = i * 2;
            }

            auto list = a.entry->get_copy_list(copy_direction::c2c, copy_engine_idx);
            if (a.copy_bytes.at(a.comm->rank()) > 0) {
                ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                        ze_cmd_memory_copy,
                                        list,
                                        dst_buf.get_ptr(),
                                        src_buf.get_ptr(),
                                        a.copy_bytes.at(a.comm->rank()),
                                        a.copy_events.at(i),
                                        a.wait_events);
            }
            else {
                // case:
                // copy of zero buffer, no ze_cmd_memory_copy occured
                // signalling copy_events[i] by hand is required for sync purposes
                // ze_cmd_memory_copy cannot be called:
                //    src_buf might be null, which causes segfault
                // copy_event cannot be skipped - it is referenced later
                ZE_APPEND_CALL_TO_ENTRY(
                    args.entry, ze_cmd_barrier, list, a.copy_events.at(i), a.wait_events);
            }
        }
    }
    if (a.is_monolithic) {
        ze_module_handle_t module = nullptr;
        //TODO: add fallback path for peer_count > max_peer_count
        CCL_THROW_IF_NOT(size_t(a.peer_count) <= ccl::ze::max_peer_count,
                         "monolithic kernel not supported for peer_count ",
                         a.peer_count,
                         " > ",
                         ccl::ze::max_peer_count);
        global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
        std::string monolithic_kernel_name =
            "write_monolithic_kernel_" + std::to_string(a.peer_count) + "_" +
            to_string(a.dtype.idx()) + "_" + ccl_reduction_to_str(ccl::reduction::custom);
        LOG_DEBUG("allgatherv monolithic kernel name: ", monolithic_kernel_name);

        // use src instead of peer dst since we cannot make sure
        // peer dst got using ipc has the same alignment as remote buffer
        // we assume local buffer and remote buffer has the same alignment
        size_t buf_size_bytes = a.copy_bytes.at(a.comm->rank());
        unsigned long pre_align_offset_byte =
            ccl::utils::get_aligned_offset_byte(a.recv_bufs.at(a.comm->rank()).get_ptr(),
                                                buf_size_bytes,
                                                ccl::global_data::env().kernel_mem_align);

        // First kernel starts from location 0 to pre_align_offset_byte
        // and the second kernel starts from location pre_align_offset_byte to the rest
        const size_t copy_count = a.copy_bytes.at(a.comm->rank()) / a.dtype.size();
        constexpr int kernel_count = (int)ccl::utils::align_kernels::count;
        const unsigned long offsets[kernel_count] = { 0, pre_align_offset_byte };
        const unsigned long counts[kernel_count] = { pre_align_offset_byte / a.dtype.size(),
                                                     copy_count -
                                                         pre_align_offset_byte / a.dtype.size() };

        // start two kernels, first kernel for the part of the array before the aligned start offset
        // and second kernel for the rest of the array from the aligned start offset to the end
        for (int i = 0; i < kernel_count; i++) {
            unsigned long count_local = counts[i];
            // data count is small and there is no need to execute the second aligned kernel
            if (i == (int)ccl::utils::align_kernels::aligned && count_local == 0) {
                a.copy_events.at(i) = a.copy_events.at(i - 1);
                break;
            }
            void* src = (src_buf + offsets[i]).get_ptr();
            std::vector<void*> dsts;
            for (auto& peer_dst_buf : peer_dst_bufs) {
                dsts.push_back((peer_dst_buf + offsets[i]).get_ptr());
            }

            ze_kernel_args_t kernel_args{
                &count_local,
                &src,
                dsts // peer_bufs_ze_arg
            };
            kernels.emplace_back(
                module, monolithic_kernel_name, kernel_args, count_local, worker_idx);

            ZE_APPEND_CALL_TO_ENTRY(args.entry,
                                    ze_cmd_launch_kernel,
                                    a.entry->get_comp_list(),
                                    std::move(kernels.back()),
                                    a.copy_events.at(i),
                                    a.wait_events);
        }
    }
}

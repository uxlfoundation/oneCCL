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

/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include <numeric>

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll_util.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "sched/entry/ze/ze_dummy_entry.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

using namespace ccl::utils;

ccl::status ccl_coll_build_direct_alltoallv(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            const size_t* send_counts,
                                            ccl_buffer recv_buf,
                                            const size_t* recv_counts,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm) {
    LOG_DEBUG("build direct alltoallv");

    entry_factory::create<alltoallv_entry>(
        sched, send_buf, send_counts, recv_buf, recv_counts, dtype, comm);
    return ccl::status::success;
}

ccl::status ccl_coll_calculate_alltoallv_counts(const ccl_coll_param& coll_param,
                                                std::vector<size_t>& send_counts,
                                                std::vector<size_t>& recv_counts,
                                                std::vector<size_t>& send_offsets,
                                                std::vector<size_t>& recv_offsets,
                                                size_t& total_send_count,
                                                size_t& total_recv_count,
                                                size_t& total_send_bytes,
                                                size_t& total_recv_bytes) {
    ccl_coll_type coll_type = coll_param.ctype;
    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_size = comm->size();
    size_t dtype_size = dtype.size();

    if (coll_type == ccl_coll_alltoall) {
        send_counts.resize(comm_size, coll_param.get_send_count());
        recv_counts.resize(comm_size, coll_param.get_recv_count());
    }
    else if (coll_type == ccl_coll_alltoallv) {
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.send_counts.size()) == comm_size,
                         "unexpected send_counts size ",
                         coll_param.send_counts.size(),
                         ", expected ",
                         comm_size);
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_counts.size()) == comm_size,
                         "unexpected recv_counts size ",
                         coll_param.recv_counts.size(),
                         ", expected ",
                         comm_size);
        send_counts = coll_param.send_counts;
        recv_counts = coll_param.recv_counts;
    }

    send_offsets.resize(comm_size, 0);
    recv_offsets.resize(comm_size, 0);

    for (int idx = 1; idx < comm_size; idx++) {
        send_offsets[idx] = send_offsets[idx - 1] + send_counts[idx - 1] * dtype_size;
        recv_offsets[idx] = recv_offsets[idx - 1] + recv_counts[idx - 1] * dtype_size;
    }

    total_send_count =
        std::accumulate(send_counts.begin(), send_counts.end(), ccl::utils::initial_count_value);
    total_recv_count =
        std::accumulate(recv_counts.begin(), recv_counts.end(), ccl::utils::initial_count_value);

    total_send_bytes = total_send_count * dtype_size;
    total_recv_bytes = total_recv_count * dtype_size;

    LOG_DEBUG("total_send_count ",
              total_send_count,
              ", total_recv_count ",
              total_recv_count,
              ", total_send_bytes ",
              total_send_bytes,
              ", total_recv_bytes ",
              total_recv_bytes);

    return ccl::status::success;
}

ccl::status ccl_coll_build_naive_alltoallv(ccl_sched* main_sched,
                                           std::vector<ccl_sched*>& scheds,
                                           const ccl_coll_param& coll_param) {
    LOG_DEBUG("build naive alltoallv");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_rank = comm->rank();
    int comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    bool inplace = coll_param.is_inplace();

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    if (total_send_count + total_recv_count == 0) {
        return ccl::status::success;
    }

    if (!inplace && send_counts[comm_rank] && recv_counts[comm_rank]) {
        size_t sched_idx = (2 * comm_rank) % sched_count;
        entry_factory::create<copy_entry>(scheds[sched_idx],
                                          ccl_buffer(coll_param.get_send_buf_ptr(),
                                                     total_send_bytes,
                                                     send_offsets[comm_rank],
                                                     ccl_buffer_type::INDIRECT),
                                          ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                     total_recv_bytes,
                                                     recv_offsets[comm_rank],
                                                     ccl_buffer_type::INDIRECT),
                                          send_counts[comm_rank],
                                          dtype);
    }

    for (int idx = 0; idx < comm_size; idx++) {
        if (idx == comm_rank)
            continue;
        if (recv_counts[idx] == 0 || send_counts[idx] == 0)
            continue;

        size_t sched_idx = (comm_rank + idx) % sched_count;

        ccl_buffer recv_buf;
        if (recv_counts[idx] > 0) {
            if (inplace)
                recv_buf = scheds[sched_idx]->alloc_buffer(
                    { recv_counts[idx] * dtype_size, coll_param.get_recv_buf() });
            else
                recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                      total_recv_bytes,
                                      recv_offsets[idx],
                                      ccl_buffer_type::INDIRECT);

            entry_factory::make_chunked_recv_entry(
                scheds, sched_idx, recv_buf, recv_counts[idx], dtype, idx, comm);
        }

        if (send_counts[idx] > 0) {
            entry_factory::make_chunked_send_entry(scheds,
                                                   sched_idx,
                                                   ccl_buffer(coll_param.get_send_buf_ptr(),
                                                              total_send_bytes,
                                                              send_offsets[idx],
                                                              ccl_buffer_type::INDIRECT),
                                                   send_counts[idx],
                                                   dtype,
                                                   idx,
                                                   comm);
        }

        if (inplace && (total_recv_bytes > 0)) {
            scheds[sched_idx]->add_barrier();
            entry_factory::create<copy_entry>(scheds[sched_idx],
                                              recv_buf,
                                              ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                         total_recv_bytes,
                                                         recv_offsets[idx],
                                                         ccl_buffer_type::INDIRECT),
                                              recv_counts[idx],
                                              dtype);
            scheds[sched_idx]->add_barrier();
        }
    }

    return ccl::status::success;
}

ccl::status ccl_coll_build_scatter_alltoallv(ccl_sched* main_sched,
                                             std::vector<ccl_sched*>& scheds,
                                             const ccl_coll_param& coll_param) {
    LOG_DEBUG("build scatter alltoallv");

    ccl_comm* comm = coll_param.comm;
    const ccl_datatype& dtype = coll_param.dtype;

    int comm_rank = comm->rank();
    int comm_size = comm->size();
    size_t sched_count = scheds.size();
    size_t dtype_size = dtype.size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    ssize_t max_ops = ccl::global_data::env().alltoall_scatter_max_ops;
    if (max_ops != CCL_ENV_SIZET_NOT_SPECIFIED) {
        for (size_t idx = 0; idx < sched_count; idx++) {
            scheds[idx]->flow_control.set_max_credits(max_ops);
        }
    }

    bool inplace = coll_param.is_inplace();

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    if (total_send_count + total_recv_count == 0) {
        return ccl::status::success;
    }

    std::vector<ccl_buffer> recv_bufs;
    if (inplace)
        recv_bufs.resize(comm_size);

    std::vector<ccl_sched*> recv_scheds(sched_count);
    std::vector<ccl_sched*> send_scheds(sched_count);

    for (size_t idx = 0; idx < sched_count; idx++) {
        auto recv_sched = entry_factory::create<subsched_entry>(
                              scheds[idx], 0, [](ccl_sched* s) {}, "A2AV_RECV")
                              ->get_subsched();

        recv_scheds[idx] = recv_sched;

        auto send_sched = entry_factory::create<subsched_entry>(
                              scheds[idx], 0, [](ccl_sched* s) {}, "A2AV_SEND")
                              ->get_subsched();

        send_scheds[idx] = send_sched;
    }

    if (!inplace && send_counts[comm_rank] && recv_counts[comm_rank]) {
        size_t sched_idx = (2 * comm_rank) % sched_count;
        entry_factory::create<copy_entry>(recv_scheds[sched_idx],
                                          ccl_buffer(coll_param.get_send_buf_ptr(),
                                                     total_send_bytes,
                                                     send_offsets[comm_rank],
                                                     ccl_buffer_type::INDIRECT),
                                          ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                     total_recv_bytes,
                                                     recv_offsets[comm_rank],
                                                     ccl_buffer_type::INDIRECT),
                                          send_counts[comm_rank],
                                          dtype);
    }

    for (int idx = 0; idx < comm_size; idx++) {
        int src = (comm_rank + idx) % comm_size;
        if (src == comm_rank)
            continue;

        if (recv_counts[src] == 0) {
            continue;
        }

        size_t sched_idx = (comm_rank + src) % sched_count;

        ccl_buffer recv_buf;

        if (inplace) {
            recv_buf = scheds[sched_idx]->alloc_buffer(
                { recv_counts[src] * dtype_size, coll_param.get_recv_buf() });
            recv_bufs[src] = recv_buf;
        }
        else
            recv_buf = ccl_buffer(coll_param.get_recv_buf_ptr(),
                                  total_recv_bytes,
                                  recv_offsets[src],
                                  ccl_buffer_type::INDIRECT);

        entry_factory::make_chunked_recv_entry(
            recv_scheds, sched_idx, recv_buf, recv_counts[src], dtype, src, comm);
    }

    for (int idx = 0; idx < comm_size; idx++) {
        int dst = (comm_rank - idx + comm_size) % comm_size;
        if (dst == comm_rank)
            continue;
        if (send_counts[dst] == 0) {
            continue;
        }

        size_t sched_idx = (comm_rank + dst) % sched_count;
        entry_factory::make_chunked_send_entry(send_scheds,
                                               sched_idx,
                                               ccl_buffer(coll_param.get_send_buf_ptr(),
                                                          total_send_bytes,
                                                          send_offsets[dst],
                                                          ccl_buffer_type::INDIRECT),
                                               send_counts[dst],
                                               dtype,
                                               dst,
                                               comm);
    }

    if (!inplace)
        return ccl::status::success;

    if (main_sched) {
        main_sched->sync_subscheds();
    }

    for (int idx = 0; idx < comm_size; idx++) {
        int src = (comm_rank + idx) % comm_size;
        if (src == comm_rank)
            continue;
        if (recv_counts[src] == 0) {
            continue;
        }
        size_t sched_idx = (comm_rank + src) % sched_count;

        entry_factory::create<copy_entry>(scheds[sched_idx],
                                          recv_bufs[src],
                                          ccl_buffer(coll_param.get_recv_buf_ptr(),
                                                     total_recv_bytes,
                                                     recv_offsets[src],
                                                     ccl_buffer_type::INDIRECT),
                                          recv_counts[src],
                                          dtype);
    }

    return ccl::status::success;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_alltoallv(ccl_sched* main_sched,
                                          std::vector<ccl_sched*>& scheds,
                                          const ccl_coll_param& coll_param) {
    ccl_comm* comm = coll_param.comm;
    ccl_sched* sched = scheds.front();

    const ccl_datatype& dtype = coll_param.dtype;
    const bool is_inplace = coll_param.is_inplace();
    const bool is_read = ccl::global_data::env().alltoallv_topo_read;

    int comm_rank = comm->rank();
    int comm_size = comm->size();

    std::vector<size_t> send_counts, recv_counts, send_offsets, recv_offsets;
    size_t total_send_count = 0, total_recv_count = 0;
    size_t total_send_bytes = 0, total_recv_bytes = 0;

    ccl_coll_calculate_alltoallv_counts(coll_param,
                                        send_counts,
                                        recv_counts,
                                        send_offsets,
                                        recv_offsets,
                                        total_send_count,
                                        total_recv_count,
                                        total_send_bytes,
                                        total_recv_bytes);

    std::vector<ccl_buffer> send_bufs(comm_size);
    std::vector<ccl_buffer> recv_bufs(comm_size);
    std::vector<ccl_buffer> tmp_bufs;

    if (is_inplace) {
        CCL_THROW_IF_NOT(send_counts == recv_counts, "unexpected send_counts");
        for (int idx = 0; idx < comm_size; idx++) {
            if (recv_counts[idx] == 0) {
                recv_bufs[idx].set(nullptr, 0, 0);
            }
            else {
                recv_bufs[idx].set(coll_param.get_send_buf(), total_recv_bytes, recv_offsets[idx]);
            }
            send_bufs[idx] = recv_bufs[idx];
        }

        tmp_bufs.resize(comm_size);
        for (int idx = 0; idx < comm_size; idx++) {
            if (send_counts[idx] == 0) {
                tmp_bufs[idx].set(nullptr, 0, 0);
            }
            else {
                ccl::alloc_param alloc_param(send_counts[idx] * dtype.size(),
                                             ccl::buffer_type::ze,
                                             ccl::buffer_place::device);
                tmp_bufs[idx] = sched->alloc_buffer(alloc_param);
            }
        }
    }
    else {
        CCL_THROW_IF_NOT(send_counts[comm_rank] == recv_counts[comm_rank],
                         "unexpected send_counts");
        for (int idx = 0; idx < comm_size; idx++) {
            if (send_counts[idx] == 0) {
                send_bufs[idx].set(nullptr, 0, 0);
            }
            else {
                send_bufs[idx].set(coll_param.get_send_buf(), total_send_bytes, send_offsets[idx]);
            }

            if (recv_counts[idx] == 0) {
                recv_bufs[idx].set(nullptr, 0, 0);
            }
            else {
                recv_bufs[idx].set(coll_param.get_recv_buf(), total_recv_bytes, recv_offsets[idx]);
            }
        }
    }

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();
    ccl_comm* r2r_comm = comm->get_r2r_comm().get();

    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    bool is_single_node = topo_manager.is_single_node;

    // IPC exchange
    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers;

    const size_t send_buf_idx_start = in_buffers.size();
    in_buffers.reserve(send_buf_idx_start + send_bufs.size());
    for (const auto& buf : send_bufs) {
        in_buffers.push_back({ buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
    }

    const size_t recv_buf_idx_start = in_buffers.size();
    in_buffers.reserve(recv_buf_idx_start + recv_bufs.size());
    for (const auto& buf : recv_bufs) {
        in_buffers.push_back({ buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
    }

    size_t tmp_buf_idx_start = -1;
    if (is_inplace) {
        tmp_buf_idx_start = in_buffers.size();
        in_buffers.reserve(tmp_buf_idx_start + tmp_bufs.size());
        for (const auto& buf : tmp_bufs) {
            in_buffers.push_back({ buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
        }
    }

    sched->try_enable_ze_single_list();

    std::vector<ze_event_handle_t> wait_events;
    std::list<ze_event_handle_t> parallel_copy_events;
    ze_event_handle_t out_event;

    ccl::add_handle_exchange(sched, node_comm, wait_events, out_event, in_buffers);
    clear_and_push_back(wait_events, out_event);

    if (!sched->is_deps_barrier() && sched->has_deps_entry()) {
        // Submit dummy ze_entry for earlier L0 submission of the workload
        // This has to be done after handle exchange to ensure that the IPC handles are ready
        entry_factory::create<ze_dummy_entry>(sched);

        // Dependencies output signal event has to be among wait events for comm_barrier
        wait_events.push_back(sched->get_related_deps_out_event());

        // Submit comm_barrier to ensure synchronization for early submitted entries
        ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);
        clear_and_push_back(wait_events, out_event);
    }

    auto add_sched_barrier_for_parallel_copies = [&]() {
        wait_events.insert(
            wait_events.end(), parallel_copy_events.begin(), parallel_copy_events.end());
        parallel_copy_events.clear();
        sched->add_barrier();
    };

    auto copy_to_self = [&](ccl_buffer& send, ccl_buffer& recv, const size_t count) {
        if (count == 0) {
            return;
        }
        copy_attr attr{};
        attr.direction = copy_direction::d2d;
        auto entry = entry_factory::create<ze_copy_entry>(
            sched, send, recv, count, dtype, attr, wait_events);
        parallel_copy_events.push_back(entry->entry_event);
        LOG_DEBUG("copy_to_self phase done")
    };

    auto add_scaleout_op = [&](std::vector<ccl_buffer>& send_bufs,
                               std::vector<ccl_buffer>& recv_bufs) {
        LOG_DEBUG("topo/scale_out: scale-out op");
        // set it by default to use
        // scatter algo for scale out phase
        auto ze_multi_workers_saved = ccl::global_data::env().ze_multi_workers;
        ccl::global_data::env().ze_multi_workers = 1;

        auto tmp_send_counts = send_counts;
        auto tmp_recv_counts = recv_counts;

        // skip the parts for scale up phase via nullifying
        // those send/recv_counts which are placed on one node
        // checking if ranks from global comm is a local one
        auto rank_info = topo_manager.get_filtered_rank_info_vec(topo_manager.get_host_idx());
        for (int rank_idx = 0; rank_idx < comm_size; rank_idx++) {
            for (auto& local_info : rank_info) {
                if (rank_idx == local_info.rank) {
                    tmp_send_counts[rank_idx] = 0;
                    tmp_recv_counts[rank_idx] = 0;
                }
            }
        }

        auto in_bufs = send_bufs;
        auto out_bufs = recv_bufs;
        if (is_inplace) {
            // copy from all recvs to tmps
            for (int idx = 0; idx < comm->size(); idx++) {
                copy_to_self(recv_bufs[idx], tmp_bufs[idx], send_counts[idx]);
            }
            add_sched_barrier_for_parallel_copies();
            ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);
            clear_and_push_back(wait_events, out_event);

            in_bufs = tmp_bufs;
            out_bufs = recv_bufs;
        }

        // preparation for host alltoall coll
        ccl_coll_param host_coll_param{ false };
        host_coll_param.ctype = ccl_coll_alltoallv;
        host_coll_param.send_scale_out_bufs = std::move(in_bufs);
        host_coll_param.recv_scale_out_bufs = std::move(out_bufs);
        host_coll_param.send_counts = std::move(tmp_send_counts);
        host_coll_param.recv_counts = std::move(tmp_recv_counts);
        host_coll_param.dtype = dtype;
        host_coll_param.comm = comm;
        host_coll_param.hint_algo.alltoallv = ccl_coll_alltoallv_direct;

        // do alltoall on the host (scale out) using global comm
        ccl::add_scaleout(sched, host_coll_param, is_single_node, wait_events, out_event);
        CCL_THROW_IF_NOT(out_event, "scaleout should have been enabled but it hasn't");
        clear_and_push_back(wait_events, out_event);

        // returned back saved value
        ccl::global_data::env().ze_multi_workers = ze_multi_workers_saved;
    };
    // TODO: enable alltoallv vectorized support for monolithic kernel MLSL-1371
    bool can_use_monolithic = true;
    for (int idx = 0; idx < comm_size; idx++) {
        // alltoallv with equal counts functionaly the same as a simple alltoall
        if (send_counts[idx] != recv_counts[idx]) {
            can_use_monolithic = false;
            break;
        }
    }

    if (ccl::global_data::env().alltoallv_monolithic_kernel && can_use_monolithic) {
        auto add_scaleup_mono_op = [&](std::vector<ccl_buffer>& send_bufs,
                                       std::vector<ccl_buffer>& recv_bufs,
                                       size_t buf_idx_start,
                                       ccl_comm* comm) {
            auto counts = recv_counts;
            auto in_bufs = send_bufs;
            auto out_bufs = recv_bufs;

            if (is_inplace) {
                // copy from all recvs to tmps
                for (int idx = 0; idx < comm->size(); idx++) {
                    copy_to_self(recv_bufs[idx], tmp_bufs[idx], send_counts[idx]);
                }
                add_sched_barrier_for_parallel_copies();
                ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);
                clear_and_push_back(wait_events, out_event);

                buf_idx_start = tmp_buf_idx_start;
                counts = send_counts;
                in_bufs = tmp_bufs;
                out_bufs = recv_bufs;
            }
            // check if there is any data to exchnge before creating alltoallv entry
            bool counts_not_empty =
                std::any_of(counts.begin(), counts.end(), [](const size_t size) {
                    return size > 0;
                });
            if (counts_not_empty) {
                auto entry = entry_factory::create<ze_alltoallv_entry>(
                    sched, in_bufs, out_bufs, counts, buf_idx_start, dtype, comm, wait_events);
                clear_and_push_back(wait_events, entry->entry_event);
            } // else skip adding alltoallv_entry
            // ze_alltoallv_entry doesn't need all ranks to participate
            sched->add_barrier();
        };

        size_t buf_idx_start = send_buf_idx_start;
        if (!ccl::global_data::env().alltoallv_monolithic_read_kernel) {
            buf_idx_start = recv_buf_idx_start;
        }

        auto mono_comm = comm;
        if (!is_single_node) {
            add_scaleout_op(send_bufs, recv_bufs);
            mono_comm = node_comm;
        }
        add_scaleup_mono_op(send_bufs, recv_bufs, buf_idx_start, mono_comm);
    }
    else {
        auto copy_to_peers = [&](std::vector<ccl_buffer>& bufs,
                                 std::vector<size_t>& counts,
                                 ccl_comm* comm,
                                 int start_buf_idx,
                                 int offset) {
            auto card_count = even_comm->size();
            auto tile_count = pair_comm->size();
            for (int card_idx = 0; card_idx < card_count; card_idx++) {
                for (int tile_idx = 0; tile_idx < tile_count; tile_idx++) {
                    auto peer_rank = (card_idx * tile_count + tile_idx);
                    if (peer_rank == comm->rank())
                        continue;
                    if (counts[peer_rank] == 0)
                        continue;

                    copy_attr attr{};
                    attr.peer_rank = peer_rank;
                    attr.peer_buf_idx = start_buf_idx + offset;
                    attr.map_comm = comm;
                    auto copy_engine_idx = card_idx * 2;
                    if (ccl::global_data::env().type2_mode == ccl::type2_tune_mode::detected ||
                        ccl::global_data::env().type2_mode == ccl::type2_tune_mode::on) {
                        copy_engine_idx = parallel_copy_events.size() * 2;
                    }
                    attr.hint_queue_index = copy_engine_idx;
                    attr.direction = copy_direction::c2c;

                    if (!is_single_node) {
                        // in order to get the correct offset for peer rank for example(2)
                        peer_rank = peer_rank + (r2r_comm->rank() * node_comm->size());
                    }

                    auto src = bufs[peer_rank];
                    auto dst = ccl_buffer();
                    if (is_read) {
                        src = ccl_buffer();
                        dst = bufs[peer_rank];
                        LOG_DEBUG("ze_copy: read copy is enabled")
                    }

                    auto entry = entry_factory::create<ze_copy_entry>(
                        sched, src, dst, counts[peer_rank], dtype, attr, wait_events);
                    parallel_copy_events.push_back(entry->entry_event);
                }
            }
            LOG_DEBUG("copy_to_peers phase done")
        };

        auto inplace_mode = [&](ccl_comm* comm, int offset) {
            for (int idx = 0; idx < comm_size; idx++) {
                CCL_THROW_IF_NOT(send_bufs[idx].get_ptr() == recv_bufs[idx].get_ptr(),
                                 "unexpected send_buf ptr for inplace case");
            }

            // copy from all recvs to tmps
            for (int idx = 0; idx < comm_size; idx++) {
                copy_to_self(recv_bufs[idx], tmp_bufs[idx], send_counts[idx]);
            }
            add_sched_barrier_for_parallel_copies();
            ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);
            clear_and_push_back(wait_events, out_event);

            // copy from peer rank send to peer rank recv
            if (is_read) {
                copy_to_peers(recv_bufs, recv_counts, comm, tmp_buf_idx_start, offset);
            }
            else {
                copy_to_peers(tmp_bufs, send_counts, comm, recv_buf_idx_start, offset);
            }
        };

        // the case, when it only goes on a single node,
        // scale up is performing
        if (is_single_node) {
            LOG_DEBUG("topo/scale_up/intra: in single node case");
            if (is_inplace) {
                LOG_DEBUG("inplace/topo/scale_up/intra: in single node case");
                // do scale up for inplace on a global comm
                inplace_mode(comm, comm->rank());
            }
            else {
                LOG_DEBUG("outofplace/topo/scale_up/intra: in single node case");
                // do scale up for outofplace on a global comm
                // copy from own send to own recv
                copy_to_self(
                    send_bufs[comm->rank()], recv_bufs[comm->rank()], send_counts[comm->rank()]);
                // copy from peer rank send to peer rank recv
                if (is_read) {
                    copy_to_peers(recv_bufs, recv_counts, comm, send_buf_idx_start, comm->rank());
                }
                else {
                    copy_to_peers(send_bufs, send_counts, comm, recv_buf_idx_start, comm->rank());
                }
            }
        }

        // the last part of scale out case is intra scale up
        // it runs when it is multi
        if (!is_single_node) {
            // Here is example of the whole flow:
            // input data:      scale-out:(1)  scale-up:(2)   result:
            // 0r 0  1  2  3    |x  x  8  12   |0  4  x  x   |0  4  8  12
            // 1r 4  5  6  7    |x  x  9  13   |1  5  x  x   |1  5  9  13
            // 2r 8  9  10 11   |2  6  x  x    |x  x  10 14  |2  6  10 14
            // 3r 12 13 14 15   |3  7  x  x    |x  x  11 15  |3  7  11 15

            // add scaleout part
            add_scaleout_op(send_bufs, recv_bufs);

            LOG_DEBUG("topo/scale_up/intra: in scale out case");
            if (is_inplace) {
                LOG_DEBUG("inplace/topo/scale_up/intra: in multi node case");
                // do scale up for inplace on a node comm
                inplace_mode(node_comm, comm_rank);
            }
            else {
                LOG_DEBUG("outofplace/topo/scale_up/intra: in multi node case");
                // do scale up for outofplace on a node comm
                // copy from own send to own recv
                copy_to_self(
                    send_bufs[comm->rank()], recv_bufs[comm->rank()], send_counts[comm->rank()]);
                // copy from peer rank send to peer rank recv
                if (is_read) {
                    copy_to_peers(recv_bufs, recv_counts, node_comm, send_buf_idx_start, comm_rank);
                }
                else {
                    copy_to_peers(send_bufs, send_counts, node_comm, recv_buf_idx_start, comm_rank);
                }
            }
        }
        add_sched_barrier_for_parallel_copies();
    }
    ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);

    entry_factory::create<ze_execute_cmdlists_on_init_entry>(sched);

    return ccl::status::success;
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

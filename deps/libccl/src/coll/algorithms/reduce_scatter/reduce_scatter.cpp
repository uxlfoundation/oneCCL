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


/*
 *
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll_util.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "sched/entry/ze/ze_dummy_entry.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

using namespace ccl::utils;

ccl::status ccl_coll_build_direct_reduce_scatter(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 ccl_buffer recv_buf,
                                                 size_t recv_count,
                                                 const ccl_datatype& dtype,
                                                 ccl::reduction reduction,
                                                 ccl_comm* comm) {
    LOG_DEBUG("build direct reduce_scatter");

    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (recv_count == 0) {
        return ccl::status::success;
    }

    const bool inplace = ccl::is_reduce_scatter_inplace(send_buf.get_ptr(),
                                                        recv_buf.get_ptr(),
                                                        recv_count,
                                                        dtype.size(),
                                                        comm->rank(),
                                                        comm->size());

    // MPI "in-place" semantics described in 4.1 standard is different
    // from oneCCL/NCCL semantics. It expects "recv_buf == send_buf".
    // Therefore, for correctness, allocate additional buffer to mimic
    // "non-inplace" case and copy the data to the required destination afterwards.
    ccl_buffer recv_local_buf = recv_buf;
    if (inplace) {
        recv_local_buf = sched->alloc_buffer({ recv_count * dtype.size(), recv_buf });
    }

    entry_factory::create<reduce_scatter_entry>(
        sched, send_buf, recv_local_buf, recv_count, dtype, reduction, comm);

    if (inplace) {
        // MPI reduce-scatter direct has to be completed
        // before we start to do the copy. That is why the barrier is needed.
        sched->add_barrier();
        entry_factory::create<copy_entry>(sched, recv_local_buf, recv_buf, recv_count, dtype);
    }

    return ccl::status::success;
}

ccl::status ccl_coll_build_naive_reduce_scatter(ccl_sched* sched,
                                                ccl_buffer send_buf,
                                                ccl_buffer recv_buf,
                                                size_t recv_count,
                                                const ccl_datatype& dtype,
                                                ccl::reduction op,
                                                ccl_comm* comm) {
    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (recv_count == 0) {
        return ccl::status::success;
    }

    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     " recv ",
                     recv_buf);

    ccl::status status = ccl::status::success;

    const int comm_size = comm->size();
    const int rank = comm->rank();
    const size_t dtype_size = dtype.size();

    const bool inplace = ccl::is_reduce_scatter_inplace(
        send_buf.get_ptr(), recv_buf.get_ptr(), recv_count, dtype.size(), rank, comm_size);
    LOG_DEBUG("build naive reduce_scatter: ", inplace ? "in-place" : "out-of-place");

    // NOTE: Initial copy can be avoided if there is an entry which accepts 3 buffers:
    // tmp, send_buf as a reduction parameters, and recv_buff as an output buffer.
    if (!inplace) {
        /* copy local data into recv_buf */
        entry_factory::create<copy_entry>(
            sched, send_buf + rank * recv_count * dtype_size, recv_buf, recv_count, dtype);
    }

    /* allocate temporary buffer to store incoming data */
    ccl_buffer tmp_buf = sched->alloc_buffer({ recv_count * dtype_size, recv_buf });
    int idx, src, dst;

    for (idx = 1; idx < comm_size; idx++) {
        src = (comm_size + rank - idx) % comm_size;
        dst = (rank + idx) % comm_size;

        /* send the data that dst needs. recv data that this process
         * needs from src into tmp_buf */
        entry_factory::create<send_entry>(
            sched, send_buf + dst * recv_count * dtype_size, recv_count, dtype, dst, comm);

        entry_factory::create<recv_entry>(sched, tmp_buf, recv_count, dtype, src, comm);

        sched->add_barrier();

        entry_factory::create<reduce_local_entry>(
            sched, tmp_buf, recv_count, recv_buf, nullptr, dtype, op);
    }

    return status;
}

ccl::status ccl_coll_build_ring_reduce_scatter(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t recv_count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction op,
                                               ccl_comm* comm) {
    // count is the same for all ranks
    // if one rank skips mpi collectives, all ranks skip
    // this means we can safely skip all operations with zero count
    if (recv_count == 0) {
        return ccl::status::success;
    }

    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     " recv ",
                     recv_buf);

    // Note: current algorithm does not require the checking for in-place
    // case, because the temporary recv buffer has to be allocatd either way,
    // if we use a ccl_coll_build_reduce_scatter_block implementation.
    LOG_DEBUG("build ring reduce_scatter");

    const size_t dtype_size = dtype.size();
    const int comm_size = comm->size();
    const int rank = comm->rank();

    // Due to the implementation specifics of reduce_scatter_block,
    // receive buffer has to be the same size as send buffer.
    // TODO: in-place case modifies the send buffer, which is also not applicable if the reduce_scatter
    // is called from the user code. It is better to implement a separate ring algo for reduce_scatter.
    ccl_buffer recv_local_buf =
        sched->alloc_buffer({ recv_count * comm_size * dtype_size, recv_buf });

    ccl_coll_build_reduce_scatter_block(
        sched, send_buf, recv_local_buf, recv_count * comm_size, dtype, op, comm);

    // Copy the data to the user provided receive buffer
    entry_factory::create<copy_entry>(
        sched, recv_local_buf + rank * recv_count * dtype_size, recv_buf, recv_count, dtype);

    return ccl::status::success;
}

/* A building block for other collectives (used as a part of allreduce, reduce, reduce_scatter).
 * Implemented with "ring" algorithm, chunking/pipelining is supported.
 * Last block may contain more elements. */
ccl::status ccl_coll_build_reduce_scatter_block(ccl_sched* sched,
                                                ccl_buffer send_buf,
                                                ccl_buffer recv_buf,
                                                size_t recv_count,
                                                const ccl_datatype& dtype,
                                                ccl::reduction op,
                                                ccl_comm* comm) {
    LOG_DEBUG("build reduce_scatter block (ring)");

    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     " recv ",
                     recv_buf);

    ccl::status status = ccl::status::success;
    int comm_size, rank;
    size_t dtype_size = dtype.size();

    comm_size = comm->size();
    rank = comm->rank();

    int src = (comm_size + rank - 1) % comm_size;
    int dst = (comm_size + rank + 1) % comm_size;

    size_t count = recv_count;
    size_t bytes = count * dtype_size;

    if (comm_size == 0) {
        return status;
    }

    size_t chunk_count =
        (bytes >= ccl::global_data::env().rs_min_chunk_size &&
         count >= ccl::global_data::env().rs_chunk_count && (int)count >= comm_size)
            ? ccl::global_data::env().rs_chunk_count
            : 1;

    while ((chunk_count > 1) &&
           (bytes / (comm_size * chunk_count) < ccl::global_data::env().rs_min_chunk_size)) {
        chunk_count--;
    }

    if (chunk_count == 0) {
        LOG_ERROR("unexpected chunk_count");
        chunk_count = 1;
    }

    // Note: this is not a correct way to detect in-place case
    // (see ccl::is_reduce_scatter_inplace). However, this algo
    // used only as a part of other collectives, thus never called as a top
    // level function, where this requirement is mandatory. Also, other collectives
    // like allreduce have different in-place semantics, therefore
    // general send == recv case is supported here.
    int inplace = (send_buf == recv_buf) ? 1 : 0;
    LOG_DEBUG("build reduce_scatter block (ring): ",
              inplace ? "in-place" : "out-of-place",
              ", chunk_count ",
              chunk_count);

    if (comm_size == 1) {
        if (!inplace) {
            entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
            sched->add_barrier();
        }
        return ccl::status::success;
    }

    ccl_buffer tmp_buf;

    if (inplace) {
        tmp_buf = sched->alloc_buffer({ count * dtype_size, recv_buf });
    }

    ccl_buffer sbuf, rbuf;
    ccl_buffer reduce_in_buf, reduce_inout_buf;
    ccl_buffer recv_reduce_local_buf, recv_reduce_comm_buf;

    /* start send and recv from such positions to have */
    /* the final reduction result on last iteration in corresponsing block */

    /* block = group of ~ equal-sized chunks */
    int block_idx = (rank + comm_size - 1) % comm_size;
    size_t main_block_size = count / comm_size;
    size_t last_block_size = main_block_size + count % comm_size;
    int send_block_idx, recv_block_idx;
    size_t send_block_size, recv_block_size;
    size_t send_block_offset, recv_block_offset;

    size_t send_main_chunk_size, send_last_chunk_size;
    size_t recv_main_chunk_size, recv_last_chunk_size;

    size_t send_chunk_size, recv_chunk_size = 0, reduce_chunk_size;
    size_t send_chunk_offset, recv_chunk_offset = 0, reduce_chunk_offset;

    /* if chunk_count > 1 then make reduction with 1 chunk delay to get comp/comp overlapping */
    bool use_prev = (chunk_count > 1) ? true : false;
    size_t prev_recv_chunk_size, prev_recv_chunk_offset;

    ccl_recv_reduce_result_buf_type recv_reduce_result_type;

    for (int idx = 0; idx < (comm_size - 1); idx++) {
        send_block_idx = block_idx;
        recv_block_idx = (comm_size + block_idx - 1) % comm_size;

        send_block_size = (send_block_idx == (comm_size - 1)) ? last_block_size : main_block_size;
        recv_block_size = (recv_block_idx == (comm_size - 1)) ? last_block_size : main_block_size;

        send_block_offset = main_block_size * send_block_idx * dtype_size;
        recv_block_offset = main_block_size * recv_block_idx * dtype_size;

        send_main_chunk_size = send_block_size / chunk_count;
        send_last_chunk_size = send_main_chunk_size + send_block_size % chunk_count;

        recv_main_chunk_size = recv_block_size / chunk_count;
        recv_last_chunk_size = recv_main_chunk_size + recv_block_size % chunk_count;

        for (size_t chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) {
            prev_recv_chunk_size = recv_chunk_size;
            send_chunk_size =
                (chunk_idx == (chunk_count - 1)) ? send_last_chunk_size : send_main_chunk_size;
            recv_chunk_size =
                (chunk_idx == (chunk_count - 1)) ? recv_last_chunk_size : recv_main_chunk_size;
            reduce_chunk_size = (use_prev) ? prev_recv_chunk_size : recv_chunk_size;

            prev_recv_chunk_offset = recv_chunk_offset;
            send_chunk_offset = send_block_offset + send_main_chunk_size * chunk_idx * dtype_size;
            recv_chunk_offset = recv_block_offset + recv_main_chunk_size * chunk_idx * dtype_size;
            reduce_chunk_offset = (use_prev) ? prev_recv_chunk_offset : recv_chunk_offset;

            if (inplace) {
                sbuf = recv_buf;
                rbuf = tmp_buf;

                reduce_in_buf = tmp_buf;
                reduce_inout_buf = recv_buf;

                recv_reduce_local_buf = recv_buf;
                recv_reduce_comm_buf = rbuf;
                recv_reduce_result_type = ccl_recv_reduce_local_buf;
            }
            else {
                sbuf = (idx == 0) ? send_buf : recv_buf;
                rbuf = recv_buf;

                reduce_in_buf = send_buf;
                reduce_inout_buf = recv_buf;

                recv_reduce_local_buf = send_buf;
                recv_reduce_comm_buf = rbuf;
                recv_reduce_result_type = ccl_recv_reduce_comm_buf;
            }

            sbuf += send_chunk_offset;
            rbuf += recv_chunk_offset;

            reduce_in_buf += reduce_chunk_offset;
            reduce_inout_buf += reduce_chunk_offset;

            recv_reduce_local_buf += reduce_chunk_offset;
            recv_reduce_comm_buf += reduce_chunk_offset;

            entry_factory::create<send_entry>(sched, sbuf, send_chunk_size, dtype, dst, comm);

            if (!use_prev) {
                CCL_ASSERT(recv_chunk_size == reduce_chunk_size);
                entry_factory::create<recv_reduce_entry>(sched,
                                                         recv_reduce_local_buf,
                                                         recv_chunk_size,
                                                         dtype,
                                                         op,
                                                         src,
                                                         comm,
                                                         recv_reduce_comm_buf,
                                                         recv_reduce_result_type);
            }
            else {
                entry_factory::create<recv_entry>(sched, rbuf, recv_chunk_size, dtype, src, comm);

                if (idx + chunk_idx > 0) {
                    entry_factory::create<reduce_local_entry>(sched,
                                                              reduce_in_buf,
                                                              reduce_chunk_size,
                                                              reduce_inout_buf,
                                                              nullptr,
                                                              dtype,
                                                              op);
                    sched->add_barrier();
                }

                if ((idx == comm_size - 2) && (chunk_idx == chunk_count - 1)) {
                    /* tail reduction for last recv operation */
                    sched->add_barrier();

                    if (inplace) {
                        reduce_in_buf = tmp_buf;
                        reduce_inout_buf = recv_buf;
                    }
                    else {
                        reduce_in_buf = send_buf;
                        reduce_inout_buf = recv_buf;
                    }

                    reduce_in_buf += recv_chunk_offset;
                    reduce_inout_buf += recv_chunk_offset;

                    entry_factory::create<reduce_local_entry>(sched,
                                                              reduce_in_buf,
                                                              recv_chunk_size,
                                                              reduce_inout_buf,
                                                              nullptr,
                                                              dtype,
                                                              op);
                }
            }

            sched->add_barrier();
        }

        /* move blocks left */
        block_idx = (comm_size + block_idx - 1) % comm_size;
    }

    return status;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

ccl::status ccl_coll_build_topo_reduce_scatter_fill(ccl_sched* sched,
                                                    ccl_buffer send_buf,
                                                    ccl_buffer recv_buf,
                                                    size_t recv_count,
                                                    size_t offset,
                                                    size_t combined_recv_count,
                                                    bool inplace,
                                                    const ccl_datatype& dtype,
                                                    ccl::reduction op,
                                                    ccl_comm* comm) {
    LOG_DEBUG("build topo reduce_scatter, recv_count ", recv_count);
    if (recv_count == 0) {
        return ccl::status::success;
    }

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();
    ccl_comm* r2r_comm = comm->get_r2r_comm().get();

    const int comm_rank = comm->rank();
    const int comm_size = comm->size();
    const int pair_comm_size = pair_comm->size();
    const int even_comm_size = even_comm->size();
    const int node_comm_size = node_comm->size();
    const int r2r_comm_size = r2r_comm->size();

    const ccl::topo_manager& topo_manager = comm->get_topo_manager();
    const bool is_single_node = topo_manager.is_single_node;
    const bool is_single_card = topo_manager.is_single_card;
    const bool is_multi_card = (even_comm_size > 1);

    const size_t count = comm_size * recv_count;
    std::vector<ze_event_handle_t> wait_events;
    ze_event_handle_t out_event;

    std::vector<ze_handle_exchange_entry::mem_desc_t> in_buffers{
        { send_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0
        { recv_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 1
    };

    // TODO: fix - reduce_scatter pipeline uses xelink write which seems to fail with int8
    const bool use_reduce_scatter_pipeline =
        ccl::global_data::env().reduce_scatter_monolithic_pipeline_kernel && pair_comm_size > 1 &&
        dtype != ccl::datatype::int8 && ccl::global_data::env().enable_ze_bidir_algo;
    LOG_DEBUG("topo/reduce_scatter pipeline ", use_reduce_scatter_pipeline);

    // optimized non-fallback algorithm is currently supported for bidirectional case
    const bool use_non_fallback_algo = !ccl::global_data::env().reduce_scatter_fallback_algo &&
                                       ccl::global_data::env().enable_ze_bidir_algo;
    LOG_DEBUG("topo/reduce_scatter fallback algo ", !use_non_fallback_algo);

    LOG_DEBUG("topo/reduce_scatter inplace ", inplace);

    // temporary buffer is used for scaleout, fallback algorithm and inplace.
    // for non fallback version, temp buffer is not used for single card and single plane
    const bool use_tmp_buf = !is_single_node || !use_non_fallback_algo || !is_single_card ||
                             pair_comm_size > 1 || inplace;

    bool is_rs_write = !ccl::global_data::env().reduce_scatter_topo_read &&
                       !use_reduce_scatter_pipeline && use_non_fallback_algo &&
                       dtype != ccl::datatype::int8;
    LOG_DEBUG("topo/reduce_scatter is_rs_write:",
              is_rs_write,
              ccl::global_data::env().reduce_scatter_topo_read,
              use_reduce_scatter_pipeline,
              use_non_fallback_algo,
              dtype != ccl::datatype::int8);
    size_t base_count = count;
    size_t pair_comm_offset = 0;
    size_t pair_comm_offset_bytes = 0;

    if (ccl::global_data::env().enable_ze_bidir_algo) {
        base_count = count / pair_comm->size();
        pair_comm_offset = base_count * pair_comm->rank();
        pair_comm_offset_bytes = pair_comm_offset * dtype.size();

        if (pair_comm->rank() == pair_comm->size() - 1)
            base_count += count % pair_comm->size();
    }

    size_t main_block_count = base_count / even_comm_size;
    size_t block_count = main_block_count;
    if (even_comm->rank() == even_comm_size - 1) {
        block_count += base_count % even_comm_size;
    }

    // setup tmp buffer for write copy mode
    ccl_buffer tmp_write_buf;

    if (is_rs_write) {
        size_t tmp_buf_bytes = 0;
        tmp_buf_bytes = dtype.size() * ((even_comm_size - 1) * block_count);
        // workaround with dummy 1 byte to avoid allocation with 0 byte
        if (tmp_buf_bytes == 0) {
            tmp_buf_bytes = 1;
        }
        ccl::alloc_param alloc_param(
            tmp_buf_bytes, ccl::buffer_type::ze, ccl::buffer_place::device);
        tmp_write_buf = sched->alloc_buffer(alloc_param);
        LOG_DEBUG("topo/reduce_scatter: allocate temp write buffer");
    }

    size_t tmp_write_buf_idx = -1;
    if (is_rs_write) {
        in_buffers.push_back({ tmp_write_buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
        tmp_write_buf_idx = in_buffers.size() - 1;
    }
    // tmp_bufs is used to store the output of the initial mdfi reduce
    std::vector<ccl_buffer> tmp_bufs;
    size_t tmp_buf_idx_start = -1;
    ccl_buffer tmp_buf;
    // reduce_scatter pipeline entry require distinct temp buffers for each peer
    if (use_non_fallback_algo && use_reduce_scatter_pipeline) {
        ze_utils::alloc_tmp_bufs(
            sched, comm, tmp_bufs, in_buffers, tmp_buf_idx_start, count, dtype);
        if (!is_single_node) {
            // TODO: add device memory manager support or any mechanism that
            // allows to control the memory consumption once the pipleine chunking is implemented

            // allocate buffer to rearrange input send_buf
            size_t tmp_buf_bytes = count * dtype.size();
            ccl::alloc_param alloc_param(
                tmp_buf_bytes, ccl::buffer_type::ze, ccl::buffer_place::device);
            tmp_buf = sched->alloc_buffer(alloc_param);
            // scaleout rearranges send_buf into tmp_buf and uses this rearranged buf as input
            in_buffers[0] = { tmp_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }; // 0
        }
    }
    else if (use_tmp_buf) {
        // allocate buffer to rearrange input send_buf
        size_t tmp_buf_bytes = count * dtype.size();
        // single-node non-fallback algo only needs temp buffer
        // for a plane to do mdfi reduce and not for rearrranging
        if (use_non_fallback_algo && is_single_node) {
            tmp_buf_bytes /= pair_comm_size;
        }
        // allocate buffer to write output of intial
        // mdfi reduce in multi-node non-fallback.
        // for single-node since we are not rearranging,
        // we can use the already allocated tmp_buf_bytes.
        // i.e. first part of the buffer is used for rearragnemnt
        // and next part is used for mdfi reduce
        size_t tmp_buf_offset_after_reorder = 0;
        if (use_non_fallback_algo && !is_single_node) {
            tmp_buf_offset_after_reorder = tmp_buf_bytes;
            tmp_buf_bytes += (tmp_buf_bytes / pair_comm_size);
        }

        ccl::alloc_param alloc_param(
            tmp_buf_bytes, ccl::buffer_type::ze, ccl::buffer_place::device);
        tmp_buf = sched->alloc_buffer(alloc_param);

        const size_t tmp_buf_size_per_rank = recv_count * r2r_comm_size * dtype.size();
        if (use_non_fallback_algo && !is_single_node) {
            // scaleout rearranges send_buf into tmp_buf and uses this rearranged buf as input
            in_buffers[0] = { tmp_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }; // 0
        }

        // partition the tmp_buf to use for mdfi reduce
        tmp_bufs.push_back(tmp_buf + tmp_buf_offset_after_reorder);
        tmp_buf_idx_start = in_buffers.size();
        in_buffers.push_back({ tmp_bufs.front().get_ptr(), ccl::ze::ipc_mem_type::memory }); // 2
        for (int i = 0; i < even_comm_size - 1; i++) {
            tmp_bufs.push_back(tmp_bufs.back() + tmp_buf_size_per_rank);
        }
    }

    size_t ipc_event_count{};
    size_t max_ipc_event_count{ 6 };
    ze_event_pool_handle_t ipc_event_pool{};
    if (ccl::global_data::env().enable_ze_barrier) {
        ipc_event_pool = sched->get_memory().ipc_event_pool_manager.create(max_ipc_event_count);
        in_buffers.push_back({ static_cast<void*>(ipc_event_pool), ccl::ze::ipc_mem_type::pool });
    }

    int skip_rank = ccl_comm::invalid_rank;
    if (ccl::global_data::env().enable_kernel_1s_ipc_wa && is_single_card) {
        skip_rank = ccl::global_data::env().kernel_1s_lead;
    }

    sched->try_enable_ze_single_list();

    ccl::add_handle_exchange(sched,
                             node_comm,
                             wait_events,
                             out_event,
                             in_buffers,
                             skip_rank,
                             ipc_event_pool,
                             ipc_event_count++);

    CCL_THROW_IF_NOT(comm_size % 2 == 0, "unexpected comm_size ", comm_size);
    CCL_THROW_IF_NOT(node_comm_size % 2 == 0, "unexpected node_comm_size ", node_comm_size);

    if (!sched->is_deps_barrier() && sched->has_deps_entry()) {
        // Submit dummy ze_entry for earlier L0 submission of the workload
        // This has to be done after handle exchange to ensure that the IPC handles are ready
        entry_factory::create<ze_dummy_entry>(sched);

        // Dependencies output signal event has to be among wait events for comm_barrier
        wait_events.push_back(sched->get_related_deps_out_event());

        // Submit comm_barrier to ensure synchronization for early submitted entries
        ccl::add_comm_barrier(
            sched, node_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        clear_and_push_back(wait_events, out_event);
    }

    sched->group->register_chunk_start(sched);

    if (use_non_fallback_algo) {
        if (!is_single_node) {
            // rearrange data from send_buf to tmp_buf
            // this is essentially a transpose of input data visualized as
            // node_comm_size X r2r_comm_size to r2r_comm_size X node_comm_size
            std::vector<ze_event_handle_t> parallel_copy_events;
            for (int node_comm_idx = 0; node_comm_idx < node_comm_size; node_comm_idx++) {
                for (int r2r_comm_idx = 0; r2r_comm_idx < r2r_comm_size; r2r_comm_idx++) {
                    const size_t recv_bytes = recv_count * dtype.size();
                    const size_t dst_offset =
                        (node_comm_idx * r2r_comm_size + r2r_comm_idx) * recv_bytes;
                    // Send buffer contains data for ALL chunks, so we must account
                    // them into `src_offset` by using `combined_recv_count`.
                    //
                    // IMPORTANT: Implementation of the collective has to take into account
                    // that `tmp_buf` contains data ONLY for single chunk while `send_buf`
                    // has data for all chunks together.
                    const size_t src_offset = (r2r_comm_idx * node_comm_size + node_comm_idx) *
                                                  (combined_recv_count * dtype.size()) +
                                              offset;

                    copy_attr attr{};
                    attr.direction = copy_direction::d2d;
                    //TODO: make offset calculation more general and robust
                    auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                                      send_buf + src_offset,
                                                                      tmp_buf + dst_offset,
                                                                      recv_count,
                                                                      dtype,
                                                                      attr,
                                                                      wait_events);
                    parallel_copy_events.push_back(entry->entry_event);
                }
            }

            // make sure all ranks have finished rearranging send_buf
            ccl::add_comm_barrier(sched, node_comm, parallel_copy_events, out_event);
            clear_and_push_back(wait_events, out_event);
        }

        // ze_a2a_pipeline_read_write_entry only works for even_comm_size > 1 and
        // therefore we need to deal with even_comm_size == 1 separately
        if (even_comm_size == 1) {
            LOG_DEBUG("topo/scale_up/intra: use ze_onesided_reduce");
            // Count of reduction performed by a r2r comm member.
            // Each member of r2r comm is also performing reduction
            // for peers in the comm, so scaleout reduce_scatter
            // over r2r comm will have all data necessary for each
            // rank.
            const size_t pair_comm_count = recv_count * r2r_comm_size;
            // For single node, offset in send_buf is based on two things:
            //  1) Each of two ranks work on data for itself, i.e rank 0 - first half,
            //     rank 1 - second half
            //  2) Chunking adds additional offset and changes overall count of elements
            //     per rank, so we must use `combined_recv_count` instead of just `pair_comm_count`.
            //     In other words, first we must find a point where data for each rank starts
            //     and then we add chunk offset.
            //
            // In scaleout, the tmp_buf does not contain data from other chunks,
            // so we treat the send_buf as if there was no chunking.
            const size_t pair_comm_local_send_offset =
                is_single_node ? offset / dtype.size() +
                                     combined_recv_count * r2r_comm_size * pair_comm->rank()
                               : pair_comm_count * pair_comm->rank();

            // for single-node use send_buf and for multi-node use rearranged tmp_buf
            const ccl_buffer pair_comm_send_buf =
                is_single_node ? send_buf + pair_comm_local_send_offset * dtype.size()
                               : tmp_buf + pair_comm_local_send_offset * dtype.size();

            // for multi-node, write to tmp_buf that will be later used for scaleout
            auto entry = entry_factory::create<ze_onesided_reduce_entry>(
                sched,
                pair_comm_send_buf,
                !is_single_node ? *tmp_bufs.begin() : recv_buf + offset,
                pair_comm_count,
                dtype,
                op,
                pair_comm->rank(),
                pair_comm,
                wait_events,
                pair_comm_local_send_offset);
            clear_and_push_back(wait_events, entry->entry_event);
        }
        else if (pair_comm_size > 1) {
            LOG_DEBUG("topo/scale_up/intra: use ze_a2a_pipeline_read_write_entry");

            // For scaleout we calculate offsets in rearranged buffer, so the send offset
            // does not have to take other chunks into account. `combined_recv_count` is only
            // required if the `send_buf` contains data for more than one chunk.
            // Send buffer block count of elements for scaleout is equal
            // to `recv_count` of each rank times `r2r_comm_size`, so the reduction is performed
            // locally for data needed by ranks in r2r communicator
            const size_t send_buf_block_count =
                is_single_node ? combined_recv_count : count / pair_comm_size / even_comm_size;
            const size_t send_buf_inner_offset = is_single_node ? offset : 0;

            // allreduce and reduce divides whole data into two continous chunks,
            // whereas reduce_scatter divides them as even and odd chunks.
            // when using pipeline, write data to peer using xelink after mdfi reduce.
            // when not using pipeline, write data to local buffer after mdfi reduce.
            ze_a2a_pipeline_read_write_entry::attr attrs{ .use_continous_data = false,
                                                          .use_remote_target =
                                                              use_reduce_scatter_pipeline };
            auto entry = entry_factory::create<ze_a2a_pipeline_read_write_entry>(
                sched,
                comm,
                is_single_node ? send_buf : tmp_buf,
                tmp_bufs,
                tmp_buf_idx_start,
                count,
                send_buf_block_count,
                send_buf_inner_offset,
                dtype,
                op,
                wait_events,
                attrs);
            clear_and_push_back(wait_events, entry->entry_event);

            ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);
            clear_and_push_back(wait_events, out_event);
        }

        ccl_buffer out_tmp_buf = *tmp_bufs.begin();
        if (even_comm_size > 1 && use_reduce_scatter_pipeline) {
            LOG_DEBUG("topo/scale_up/intra: use ze_a2a_pipeline_reduce_entry");
            // reduce from remote even_comm peer buffers or from local buffer
            ccl_buffer dst_recv_buf = is_single_node ? recv_buf + offset : out_tmp_buf;
            auto entry = entry_factory::create<ze_a2a_pipeline_reduce_entry>(
                sched, comm, dst_recv_buf, tmp_bufs, count, dtype, op, wait_events);
            clear_and_push_back(wait_events, entry->entry_event);
        }
        else if (even_comm_size > 1) {
            const size_t tmp_buf_count_per_rank = recv_count * r2r_comm_size;
            const size_t tmp_buf_size_per_rank = tmp_buf_count_per_rank * dtype.size();

            // perform xelink read and followed by reduce
            ccl_buffer src_send_buf = is_single_node ? send_buf : tmp_buf;
            size_t send_buf_idx = 0;
            // when both tiles are used, we need to read from temporary
            // which contains the result of mdfi reduce
            if (pair_comm_size > 1) {
                src_send_buf = tmp_bufs.front();
                send_buf_idx = tmp_buf_idx_start;
            }
            std::vector<size_t> block_counts(even_comm_size, recv_count * r2r_comm_size);

            if (!is_rs_write) {
                LOG_DEBUG("topo/scale_up/intra: use ze_a2a_reduce_scatter_entry");
                out_tmp_buf = src_send_buf + even_comm->rank() * tmp_buf_size_per_rank;
                auto entry = entry_factory::create<ze_a2a_reduce_scatter_entry>(
                    sched,
                    src_send_buf,
                    is_single_node ? recv_buf : out_tmp_buf,
                    block_counts.data(),
                    dtype,
                    op,
                    even_comm,
                    wait_events,
                    send_buf_idx,
                    0); // pair_comm_offset
                clear_and_push_back(wait_events, entry->entry_event);
            }
            else {
                // copy using write
                LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_write_copy_entry");
                reduce_scatter_args rs_args = { even_comm, std::move(block_counts), dtype, op };
                reduce_scatter_bufs rs_bufs = { src_send_buf,
                                                is_single_node ? recv_buf : out_tmp_buf,
                                                tmp_write_buf,
                                                tmp_write_buf_idx,
                                                0 }; //pair_comm_offset

                auto copy_entry = entry_factory::create<ze_a2a_reduce_scatter_write_copy_entry>(
                    sched, rs_args, rs_bufs, wait_events);
                clear_and_push_back(wait_events, copy_entry->entry_event);
                ccl::add_comm_barrier(
                    sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
                clear_and_push_back(wait_events, out_event);

                LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_write_kernel_entry");
                // local reduction
                auto kernel_entry = entry_factory::create<ze_a2a_reduce_scatter_write_kernel_entry>(
                    sched, rs_args, rs_bufs, wait_events);
                clear_and_push_back(wait_events, kernel_entry->entry_event);
                ccl::add_comm_barrier(
                    sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
                clear_and_push_back(wait_events, out_event);
            }
        }

        if (!is_single_node) {
            ccl_coll_param coll_param{ false };
            coll_param.ctype = ccl_coll_reduce_scatter;
            coll_param.send_buf = out_tmp_buf;
            coll_param.recv_buf = recv_buf + offset;
            coll_param.count = recv_count;
            coll_param.dtype = dtype;
            coll_param.reduction = op;
            coll_param.comm = r2r_comm;

            ccl::add_scaleout(sched, coll_param, is_single_node, wait_events, out_event);
            if (out_event) {
                clear_and_push_back(wait_events, out_event);
            }
        }

        if (recv_count != combined_recv_count) {
            // This barrier is required to avoid race condition
            // across ranks trying to read from the XeLink buffers
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);
        }

        sched->group->register_chunk_end(sched, out_event);

        return ccl::status::success;
    }

    // note: start section common with allreduce topo
    // the following section is based on the allreduce topo implementation
    // it uses steps from allreduce topo to collect allreduce results on tmp buffer

    if (!ccl::global_data::env().enable_ze_bidir_algo &&
        pair_comm->rank() != ccl::global_data::env().kernel_1s_lead) {
        ccl::add_comm_barrier(
            sched, pair_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        ipc_event_pool_manager::check_ipc_event_count(
            sched->coll_param.ctype, ipc_event_count, max_ipc_event_count);
        return ccl::status::success;
    }
    const ccl_buffer tmp_recv_buf = tmp_bufs.front();
    const size_t tmp_recv_buf_idx = tmp_buf_idx_start;
    size_t even_comm_offset_bytes = main_block_count * even_comm->rank() * dtype.size();
    ccl_buffer pair_comm_send_buf = send_buf + pair_comm_offset_bytes;
    ccl_buffer pair_comm_recv_buf = tmp_recv_buf + pair_comm_offset_bytes;
    ccl_buffer even_comm_recv_buf = tmp_recv_buf + pair_comm_offset_bytes + even_comm_offset_bytes;

    LOG_DEBUG("rank: ",
              pair_comm->rank(),
              ", count: ",
              base_count,
              ", pair_comm_offset: ",
              pair_comm_offset);
    if (is_single_card) {
        LOG_DEBUG("topo/scale_up/intra: use ze_onesided_allreduce");
        auto entry = entry_factory::create<ze_onesided_allreduce_entry>(sched,
                                                                        pair_comm_send_buf,
                                                                        pair_comm_recv_buf,
                                                                        base_count,
                                                                        dtype,
                                                                        op,
                                                                        pair_comm,
                                                                        wait_events,
                                                                        pair_comm_offset);
        clear_and_push_back(wait_events, entry->entry_event);
    }
    else {
        LOG_DEBUG("topo/scale_up/intra: use ze_onesided_reduce");
        auto entry = entry_factory::create<ze_onesided_reduce_entry>(sched,
                                                                     pair_comm_send_buf,
                                                                     pair_comm_recv_buf,
                                                                     base_count,
                                                                     dtype,
                                                                     op,
                                                                     pair_comm->rank(),
                                                                     pair_comm,
                                                                     wait_events,
                                                                     pair_comm_offset);
        clear_and_push_back(wait_events, entry->entry_event);
    }
    sched->add_barrier();

    bool is_read_allgatherv = ccl::global_data::env().allgatherv_topo_read;
    if (is_multi_card) {
        ccl::add_comm_barrier(
            sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        clear_and_push_back(wait_events, out_event);
        // cannot use ze_a2a_allreduce_entry with allgatherv_read
        // since comm_barrier is not available inside the entry
        if (is_single_node && !is_read_allgatherv) {
            LOG_DEBUG("topo/scale_up/intra: use ze_a2a_allreduce_entry");
            auto entry = entry_factory::create<ze_a2a_allreduce_entry>(sched,
                                                                       pair_comm_recv_buf,
                                                                       pair_comm_recv_buf,
                                                                       base_count,
                                                                       dtype,
                                                                       op,
                                                                       even_comm,
                                                                       wait_events,
                                                                       tmp_recv_buf_idx,
                                                                       tmp_recv_buf_idx,
                                                                       pair_comm_offset);
            clear_and_push_back(wait_events, entry->entry_event);
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);
        }
        else {
            LOG_DEBUG("topo/scale_up/inter: use ze_a2a_reduce_scatter_entry");
            std::vector<size_t> block_counts(even_comm->size(), main_block_count);
            block_counts.back() += base_count % even_comm_size;
            auto entry = entry_factory::create<ze_a2a_reduce_scatter_entry>(sched,
                                                                            pair_comm_recv_buf,
                                                                            even_comm_recv_buf,
                                                                            block_counts.data(),
                                                                            dtype,
                                                                            op,
                                                                            even_comm,
                                                                            wait_events,
                                                                            tmp_recv_buf_idx,
                                                                            pair_comm_offset);
            clear_and_push_back(wait_events, entry->entry_event);
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);
        }
    }

    ccl_coll_param coll_param{ false };
    coll_param.ctype = ccl_coll_allreduce;
    coll_param.send_buf = even_comm_recv_buf;
    coll_param.recv_buf = even_comm_recv_buf;
    coll_param.count = block_count;
    coll_param.dtype = dtype;
    coll_param.reduction = op;
    coll_param.comm = r2r_comm;

    ccl::add_scaleout(sched, coll_param, is_single_node, wait_events, out_event);
    if (out_event) {
        clear_and_push_back(wait_events, out_event);
    }

    if (is_multi_card && (!is_single_node || is_read_allgatherv)) {
        LOG_DEBUG("topo/scale_up/inter: use ze_a2a_allgatherv");
        // for multinode with allgatherv_read, use a comm_barrier to make sure all
        // r2r scaleout within even_comm has finished so that remote reads are valid
        if (!is_single_node && is_read_allgatherv) {
            ccl::add_comm_barrier(
                sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
            clear_and_push_back(wait_events, out_event);
        }

        std::vector<size_t> recv_counts(even_comm_size, main_block_count);
        recv_counts.back() += base_count % even_comm_size;
        // allgatherv entry takes an array of recv bufs with one buffer for each rank
        // rather than a single large recv buf that needs to be divided
        std::vector<ccl_buffer> recv_bufs;
        for (int i = 0; i < even_comm_size; i++) {
            recv_bufs.push_back(pair_comm_recv_buf + i * main_block_count * dtype.size());
        }
        auto entry = entry_factory::create<ze_a2a_allgatherv_entry>(sched,
                                                                    recv_bufs[even_comm->rank()],
                                                                    block_count,
                                                                    recv_bufs,
                                                                    recv_counts,
                                                                    dtype,
                                                                    even_comm,
                                                                    wait_events,
                                                                    tmp_recv_buf_idx,
                                                                    pair_comm_offset);
        clear_and_push_back(wait_events, entry->entry_event);
        ccl::add_comm_barrier(
            sched, even_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
        clear_and_push_back(wait_events, out_event);
    }

    if (!is_single_card && pair_comm->size() > 1) {
        LOG_DEBUG("topo/scale_up/intra: use ze_onesided_bcast");
        int peer_rank = (pair_comm->rank() + 1) % pair_comm->size();
        auto entry = entry_factory::create<ze_copy_entry>(sched,
                                                          tmp_recv_buf,
                                                          ccl_buffer(),
                                                          base_count,
                                                          dtype,
                                                          copy_attr(peer_rank,
                                                                    tmp_recv_buf_idx,
                                                                    copy_direction::t2t,
                                                                    false, /*pt2pt_op*/
                                                                    pair_comm,
                                                                    pair_comm_offset,
                                                                    pair_comm_offset),
                                                          wait_events);
        clear_and_push_back(wait_events, entry->entry_event);
        sched->add_barrier();
    }

    ccl::add_comm_barrier(
        sched, pair_comm, wait_events, out_event, ipc_event_pool, ipc_event_count++);
    clear_and_push_back(wait_events, out_event);

    ipc_event_pool_manager::check_ipc_event_count(
        sched->coll_param.ctype, ipc_event_count, max_ipc_event_count);

    // note: end section common with allreduce topo

    sched->add_barrier();
    ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);
    clear_and_push_back(wait_events, out_event);

    // copy part of temp buffer to recv buffer
    copy_attr attr{};
    attr.direction = copy_direction::d2d;
    size_t allred_offset_bytes = recv_count * comm->rank() * dtype.size();
    ccl_buffer allred_recv_buf = tmp_recv_buf + allred_offset_bytes;

    auto entry_copy = entry_factory::create<ze_copy_entry>(
        sched, allred_recv_buf, recv_buf, recv_count, dtype, attr, wait_events);

    clear_and_push_back(wait_events, entry_copy->entry_event);
    sched->add_barrier();
    ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);

    return ccl::status::success;
}

ccl::status ccl_coll_build_topo_reduce_scatter(ccl_sched* sched,
                                               ccl_buffer send_buf,
                                               ccl_buffer recv_buf,
                                               size_t count,
                                               const ccl_datatype& dtype,
                                               ccl::reduction op,
                                               ccl_comm* comm) {
    ccl_comm* pair_comm = comm->get_pair_comm().get();
    const int pair_comm_size = pair_comm->size();
    const int comm_rank = comm->rank();
    const int comm_size = comm->size();

    const bool use_reduce_scatter_pipeline =
        ccl::global_data::env().reduce_scatter_monolithic_pipeline_kernel && pair_comm_size > 1 &&
        dtype != ccl::datatype::int8 && ccl::global_data::env().enable_ze_bidir_algo;

    const bool use_non_fallback_algo = !ccl::global_data::env().reduce_scatter_fallback_algo &&
                                       ccl::global_data::env().enable_ze_bidir_algo;

    // Currently chunking is only enabled for the default `reduce_scatter` algorithm
    const bool chunking_enabled = use_reduce_scatter_pipeline && use_non_fallback_algo;

    // Checking in-placeness semantics before the chunking
    const bool inplace = ccl::is_reduce_scatter_inplace(
        send_buf.get_ptr(), recv_buf.get_ptr(), count, dtype.size(), comm_rank, comm_size);

    // TODO: enable when MLSL-3066 is fixed
    if (ccl::global_data::env().reduce_scatter_fallback_algo && inplace) {
        CCL_THROW("CCL_REDUCE_SCATTER_FALLBACK_ALGO=1 is incompatible with in-place semantics.");
    }

    return ccl_build_topo_uniform_buff_size_op(
        sched,
        send_buf,
        recv_buf,
        count,
        dtype.size(),
        ccl::global_data::env().reduce_scatter_pipe_chunk_count,
        "REDUCE_SCATTER",
        ccl::global_data::get().metrics_profiler->reduce_scatter_pipe,
        comm,
        [dtype, op, comm, inplace](ccl_sched* sched,
                                   ccl_buffer send_buf,
                                   ccl_buffer recv_buf,
                                   size_t count,
                                   size_t offset,
                                   size_t combined_recv_count) -> ccl::status {
            return ccl_coll_build_topo_reduce_scatter_fill(sched,
                                                           send_buf,
                                                           recv_buf,
                                                           count,
                                                           offset,
                                                           combined_recv_count,
                                                           inplace,
                                                           dtype,
                                                           op,
                                                           comm);
        },
        chunking_enabled ? ccl_chunking_mode::ccl_offset_explicit
                         : ccl_chunking_mode::ccl_pipeline_none,
        ccl_coll_type::ccl_coll_reduce_scatter);
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

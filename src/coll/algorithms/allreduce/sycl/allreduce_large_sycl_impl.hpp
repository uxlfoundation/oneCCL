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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_kernels.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_impl.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_large_sycl_impl.hpp"

template <typename T, int N, int vec_size>
ccl::event allreduce_large_read_write_ipc(const void *send_buf,
                                          void *recv_buf,
                                          size_t count,
                                          ccl::datatype dtype,
                                          ccl::reduction reduction,
                                          ccl_comm *comm,
                                          ccl_stream *global_stream,
                                          sycl_ptrs_type &sycl_ptrs,
                                          const ccl::vector_class<ccl::event> &deps,
                                          const bool is_single_plane) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    const bool is_use_tmp = ccl::global_data::env().sycl_allreduce_tmp_buf;
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int node_comm_size = node_comm->size();

    CCL_THROW_IF_NOT(
        node_comm->size() == N,
        "SYCL allreduce read write algo implemented for single plane or single GPU cases");

    // align kernel iteration count on kernel_mem_align
    const int rank_align_count = ccl::global_data::env().kernel_mem_align / dsize;
    const size_t align_count = N * rank_align_count;
    const size_t rem_count = count % align_count;
    const size_t pack_count = count - rem_count;

    // algorithm chunk calculation
    // the last rank also calculates the unaligned remaining counts
    const int rank = node_comm->rank();
    const size_t count_rank = pack_count / N;
    const size_t offset_rank = rank * count_rank * dsize;
    const size_t count_use = rank == N - 1 ? count_rank + rem_count : count_rank;

    std::array<void *, MAX_NODE_RANKS> src_ptrs, dst_ptrs;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event, barrier_event;

    for (int idx = 0; idx < N; idx++) {
        if (idx == rank) {
            src_ptrs[idx] = (char *)send_buf + offset_rank;
            dst_ptrs[idx] = (char *)recv_buf + offset_rank;
        }
        else {
            src_ptrs[idx] = (is_single_plane ? (char *)sycl_ptrs.xelink_ptrs_rd[idx]
                                             : (char *)sycl_ptrs.mdfi_ptr_rd) +
                            offset_rank;
            dst_ptrs[idx] = (is_single_plane ? (char *)sycl_ptrs.xelink_ptrs_wr[idx]
                                             : (char *)sycl_ptrs.mdfi_ptr_wr) +
                            offset_rank;
        }
    }

    barrier_event = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);
    work_event = q.submit([=](sycl::handler &h) {
        h.depends_on(barrier_event);

        const int work_group_size = 16;
        const int sub_group_size = 16;

        ccl_kernel_barrier_data dummy_kbd;
        ccl_comm_barrier_data dummy_cbd = node_comm->barrier_data();
        const size_t kernel_threads = count_use / vec_size + count_use % vec_size;
        const size_t kernel_size =
            ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

        h.parallel_for(
            sycl::nd_range<1>(kernel_size, work_group_size),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sub_group_size)]] {
                reduce_sum<T, N, vec_size, 1, 0, 0, 0>(nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       src_ptrs,
                                                       dst_ptrs,
                                                       dummy_kbd,
                                                       dummy_cbd,
                                                       count_use,
                                                       it);
            });
    });
    barrier_event = invoke_barrier(node_comm, q, { work_event }, is_cpu_barrier);

    // do the average reduction separately after sum
    if (reduction == ccl::reduction::avg) {
        // set dependencies
        std::vector<sycl::event> avg_deps_evs;
        avg_deps_evs.push_back(barrier_event);

        LOG_DEBUG("allreduce_large_ipc calculate average on counts: ",
                  count,
                  ", ranks: ",
                  node_comm_size);
        barrier_event = sycl_average(q, recv_buf, count, node_comm_size, dtype, avg_deps_evs);
    }

    return ccl::event::create_from_native(barrier_event);
}

template <typename T, int N, int vec_size>
ccl::event allreduce_large_read_write_tmp(const void *send_buf,
                                          void *recv_buf,
                                          size_t count,
                                          ccl::datatype dtype,
                                          ccl::reduction reduction,
                                          ccl_comm *comm,
                                          ccl_stream *global_stream,
                                          sycl_ptrs_type &sycl_ptrs,
                                          const ccl::vector_class<ccl::event> &deps,
                                          const bool is_single_plane) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int node_comm_size = node_comm->size();
    const int rank = node_comm->rank();
    CCL_THROW_IF_NOT(
        node_comm->size() == N,
        "SYCL allreduce read write algo implemented for single plane or single GPU cases");

    std::array<void *, MAX_NODE_RANKS> l_src_ptrs, l_dst_ptrs;
    void *src_tmp_ptr = get_tmp_buf(1, comm);
    void *dst_tmp_ptr = get_tmp_buf(2, comm);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    std::vector<sycl::event> memcpy_events;
    sycl::event work_event, barrier_event, memcpy_event;

    // chunk size is calculated originally as a tmp buffer divided by the number of ranks
    // aligned on the kernel_mem_align, here compute the full buffer size for the algo
    const size_t chunk_size = get_tmp_buf_size_per_rank() * node_comm_size;
    const size_t count_bytes = count * dsize;
    const size_t rem_chunk_size = count_bytes % chunk_size;
    const size_t num_chunks = count_bytes / chunk_size + (rem_chunk_size != 0);

    for (size_t nc = 0; nc < num_chunks; nc++) {
        // chunking calculation
        const size_t chunk_offset = nc * chunk_size;
        const size_t data_count =
            ((nc < count_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;

        // align kernel iteration count on kernel_mem_align
        const int rank_align_count = ccl::global_data::env().kernel_mem_align / dsize;
        const size_t align_count = N * rank_align_count;
        const size_t rem_count = data_count % align_count;
        const size_t pack_count = data_count - rem_count;

        // kernel working data size and buffers offset calculation
        const size_t count_rank = pack_count / N;
        const size_t offset_rank = rank * count_rank * dsize;
        // current iteration read/reduce/write data size
        const size_t count_use = rank == N - 1 ? count_rank + rem_count : count_rank;

        for (int idx = 0; idx < N; idx++) {
            if (idx == rank) {
                l_src_ptrs[idx] = (char *)src_tmp_ptr + offset_rank;
                l_dst_ptrs[idx] = (char *)dst_tmp_ptr + offset_rank;
            }
            else {
                l_src_ptrs[idx] = (is_single_plane ? (char *)sycl_ptrs.xelink_ptrs_rd[idx]
                                                   : (char *)sycl_ptrs.mdfi_ptr_rd) +
                                  offset_rank;
                l_dst_ptrs[idx] = (is_single_plane ? (char *)sycl_ptrs.xelink_ptrs_wr[idx]
                                                   : (char *)sycl_ptrs.mdfi_ptr_wr) +
                                  offset_rank;
            }
        }

        // copy from send_buf to the temp buffer
        memcpy_event = q.submit([=](sycl::handler &h) {
            if (nc == 0) {
                h.depends_on(dep_events);
            }
            else {
                h.depends_on(barrier_event);
            }
            h.memcpy(src_tmp_ptr, (char *)send_buf + chunk_offset, data_count * dsize);
        });
        memcpy_events.push_back(memcpy_event);

        barrier_event = invoke_barrier(node_comm, q, memcpy_events, is_cpu_barrier);
        memcpy_events.clear();

        work_event = q.submit([=](sycl::handler &h) {
            h.depends_on(barrier_event);

            const int work_group_size = 16;
            const int sub_group_size = 16;

            ccl_kernel_barrier_data dummy_kbd;
            ccl_comm_barrier_data dummy_cbd = node_comm->barrier_data();
            const size_t kernel_threads = count_use / vec_size + count_use % vec_size;
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

            h.parallel_for(
                sycl::nd_range<1>(kernel_size, work_group_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sub_group_size)]] {
                    reduce_sum<T, N, vec_size, 1, 0, 0, 0>(nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           l_src_ptrs,
                                                           l_dst_ptrs,
                                                           dummy_kbd,
                                                           dummy_cbd,
                                                           count_use,
                                                           it);
                });
        });
        barrier_event = invoke_barrier(node_comm, q, { work_event }, is_cpu_barrier);

        // copy reduction result from temp buffer to recv_buffer
        memcpy_event = q.submit([=](sycl::handler &h) {
            h.depends_on(barrier_event);
            h.memcpy((char *)recv_buf + chunk_offset, dst_tmp_ptr, data_count * dsize);
        });
        if (nc < num_chunks - 1) {
            memcpy_events.push_back(memcpy_event);
        }
    }

    // do the average reduction separately after sum
    if (reduction == ccl::reduction::avg) {
        // set dependencies
        std::vector<sycl::event> avg_deps_evs;
        avg_deps_evs.push_back(memcpy_event);

        LOG_DEBUG("allreduce_large_tmp calculate average on counts: ",
                  count,
                  ", ranks: ",
                  node_comm_size);
        memcpy_event = sycl_average(q, recv_buf, count, node_comm_size, dtype, avg_deps_evs);
    }

    return ccl::event::create_from_native(memcpy_event);
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP, bool use_full_vector>
ccl::event allreduce_large_impl(const void *send_buf,
                                void *recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm *comm,
                                ccl_stream *global_stream,
                                sycl_ptrs_type &sycl_ptrs,
                                const ccl::vector_class<ccl::event> &deps) {
    constexpr int N = NE;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();
    sycl::queue q_use = q;
    const bool is_use_tmp = ccl::global_data::env().sycl_allreduce_tmp_buf;
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    const int pair_comm_size = pair_comm->size();
    const int even_comm_size = even_comm->size();
    const int node_comm_size = node_comm->size();

    const bool is_multi_tile = pair_comm_size > 1;
    const bool is_multi_gpu = even_comm_size > 1;
    const bool is_single_plane = !is_multi_tile && is_multi_gpu;
    const bool is_single_gpu = is_multi_tile && !is_multi_gpu;
    const bool is_use_rw_opt = is_single_gpu || is_single_plane;

    constexpr int pipeline_size = 2;
    constexpr int vec_size = get_num_elements<T, 8, use_full_vector>();
    const size_t work_group_size = 16;

    // single plane/gpu case optimizations
    if (is_use_rw_opt) {
        if (is_use_tmp && node_comm_size == 2) {
            // For more than 2 ranks, the performance of RW algo with TMP buffer
            // is slightly worse than pipelined implementation
            if (is_single_gpu) {
                return allreduce_large_read_write_tmp<T, NP, vec_size>(send_buf,
                                                                       recv_buf,
                                                                       count,
                                                                       dtype,
                                                                       reduction,
                                                                       comm,
                                                                       global_stream,
                                                                       sycl_ptrs,
                                                                       deps,
                                                                       false);
            }
            if (is_single_plane) {
                return allreduce_large_read_write_tmp<T, NE, vec_size>(send_buf,
                                                                       recv_buf,
                                                                       count,
                                                                       dtype,
                                                                       reduction,
                                                                       comm,
                                                                       global_stream,
                                                                       sycl_ptrs,
                                                                       deps,
                                                                       true);
            }
        }
        if (!is_use_tmp) {
            if (is_single_gpu) {
                return allreduce_large_read_write_ipc<T, NP, vec_size>(send_buf,
                                                                       recv_buf,
                                                                       count,
                                                                       dtype,
                                                                       reduction,
                                                                       comm,
                                                                       global_stream,
                                                                       sycl_ptrs,
                                                                       deps,
                                                                       false);
            }
            if (is_single_plane) {
                return allreduce_large_read_write_ipc<T, NE, vec_size>(send_buf,
                                                                       recv_buf,
                                                                       count,
                                                                       dtype,
                                                                       reduction,
                                                                       comm,
                                                                       global_stream,
                                                                       sycl_ptrs,
                                                                       deps,
                                                                       true);
            }
        }
    }

    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }

    std::array<void *, MAX_GPUS> l_mdfi_send_ptrs, l_xelink_work_wr_ptrs, l_send_ptrs,
        l_send_ptrs_rem;
    std::array<void *, MAX_GPUS> l_mdfi_recv_ptrs, l_xelink_work_rd_ptrs, l_recv_ptrs;
    std::array<void *, MAX_GPUS> l_mdfi_recv_ptrs_prev, l_xelink_work_rd_ptrs_prev,
        l_recv_ptrs_prev;
    std::array<void *, MAX_GPUS> l_send_cp_src_ptrs, l_send_cp_dst_ptrs, l_send_cp_src_ptrs_next,
        l_send_cp_dst_ptrs_next;
    std::array<void *, MAX_GPUS> l_recv_cp_src_ptrs, l_recv_cp_dst_ptrs, l_recv_cp_src_ptrs_prev,
        l_recv_cp_src_ptrs_prev_2, l_recv_cp_dst_ptrs_prev, l_recv_cp_dst_ptrs_prev_2;
    std::array<void *, MAX_NODE_RANKS> l_work_ptrs, l_work_ptrs_prev, l_node_send_ptrs;
    void *l_recv_ptr = nullptr, *l_reduce_sum_dst = nullptr, *l_reduce_sum_dst_prev = nullptr;

    const int rank_align_count = ccl::global_data::env().kernel_mem_align / dsize;
    const size_t align_count = node_comm_size * rank_align_count;
    const size_t rem_count = count % align_count;
    const size_t pack_count = count - rem_count;
    const size_t recv_count = pack_count / node_comm_size;
    const size_t recv_bytes = recv_count * dsize;

    const size_t chunk_size = get_tmp_buf_size_per_rank();
    const size_t rem_chunk_size = recv_bytes % chunk_size;
    const size_t num_chunks = recv_bytes / chunk_size + (rem_chunk_size != 0 || rem_count != 0);
    const size_t pipeline_offset = chunk_size * even_comm_size;

    void *tmp_bufs_send[pipeline_size] = { get_tmp_buf(1, comm),
                                           (char *)get_tmp_buf(1, comm) + pipeline_offset };
    void *tmp_bufs_recv[pipeline_size] = { get_tmp_buf(2, comm),
                                           (char *)get_tmp_buf(2, comm) + pipeline_offset };
    std::array<void *, MAX_GPUS> xelink_work_bufs_wr[pipeline_size],
        xelink_work_bufs_rd[pipeline_size], work_bufs[pipeline_size];

    xelink_work_bufs_rd[0] = get_remote_even_tmp_buf(0, comm);
    void *work_buf = get_tmp_buf(0, comm);
    for (int i = 0; i < even_comm_size; i++) {
        work_bufs[0][i] = (char *)work_buf + chunk_size * i;
        work_bufs[1][i] = (char *)(work_bufs[0][i]) + pipeline_offset;
        xelink_work_bufs_rd[1][i] = (char *)xelink_work_bufs_rd[0][i] + pipeline_offset;
        xelink_work_bufs_wr[0][i] =
            (char *)xelink_work_bufs_rd[0][i] + chunk_size * even_comm->rank();
        xelink_work_bufs_wr[1][i] = (char *)xelink_work_bufs_wr[0][i] + pipeline_offset;
    }

    //TODO : update the algorithm to not use node_send_ptrs
    if (rem_count != 0) {
        l_send_ptrs_rem[0] = get_tmp_buf(2, comm);
        l_recv_ptr = (char *)recv_buf + pack_count * dsize;
        l_node_send_ptrs = get_remote_node_tmp_buf(2, comm);
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;
    for (size_t nc = 0; nc < num_chunks; nc++) {
        const int pipeline_index = nc % pipeline_size;
        const int pipeline_index_next = (nc + 1) % pipeline_size;

        const size_t chunk_offset = nc * chunk_size;
        const size_t data_count =
            ((nc < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;
        const size_t data_count_next =
            ((nc + 1 < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;
        const size_t data_count_prev = chunk_size / dsize;
        const size_t data_count_prev_2 = chunk_size / dsize;
        for (int i = 0; i < even_comm_size; i++) {
            int global_rank = comm->is_multi_thread_instance()
                                  ? i * pair_comm_size + pair_comm->rank()
                                  : even_comm->get_node_rank(i);
            // TODO: is there a better way to find the pair_neighbor global rank
            int global_rank_neighbor = (global_rank / pair_comm_size) * pair_comm_size;
            if (global_rank % pair_comm_size == 0) {
                global_rank_neighbor = global_rank_neighbor + 1;
            }

            // offset is direct offset within send_buf and
            // offset_tmp is offset within tmp_buf where data is copied from send_buf
            const size_t offset = global_rank * recv_bytes + chunk_offset;
            const size_t offset_neigh = global_rank_neighbor * recv_bytes + chunk_offset;
            const size_t offset_tmp = i * chunk_size;
            const size_t mdfi_offset_tmp = pipeline_index * pipeline_offset + offset_tmp;
            l_send_cp_src_ptrs[i] = (char *)send_buf + offset_neigh;
            l_send_cp_dst_ptrs[i] = (char *)tmp_bufs_send[pipeline_index] + offset_tmp;

            l_send_cp_src_ptrs_next[i] = (char *)l_send_cp_src_ptrs[i] + chunk_size;
            l_send_cp_dst_ptrs_next[i] = (char *)tmp_bufs_send[pipeline_index_next] + offset_tmp;

            // read_reduce_write
            l_mdfi_send_ptrs[i] =
                (char *)sycl_ptrs.mdfi_ptr_rd + (is_use_tmp ? mdfi_offset_tmp : offset);
            l_send_ptrs[i] = (char *)send_buf + offset;
            l_xelink_work_wr_ptrs[i] = (char *)xelink_work_bufs_wr[pipeline_index][i];

            // reduce_sum
            l_work_ptrs[i] = (char *)work_bufs[pipeline_index][i];

            // read_write
            l_xelink_work_rd_ptrs[i] = (char *)xelink_work_bufs_rd[pipeline_index][i];
            l_recv_ptrs[i] = (char *)recv_buf + offset;
            l_mdfi_recv_ptrs[i] =
                (char *)sycl_ptrs.mdfi_ptr_wr + (is_use_tmp ? mdfi_offset_tmp : offset);

            l_recv_cp_src_ptrs[i] = (char *)tmp_bufs_recv[pipeline_index] + offset_tmp;
            l_recv_cp_dst_ptrs[i] = (char *)recv_buf + offset_neigh;
        }

        // reduce_sum
        l_reduce_sum_dst = is_use_tmp
                               ? l_work_ptrs[0]
                               : (char *)recv_buf + node_comm->rank() * recv_bytes + chunk_offset;

        bool is_deps_added = false;
        // pipeline prologue
        // this data copy can also be done from the main kernel as first step with a guard of nc == 0
        if (is_use_tmp && nc == 0 && is_multi_tile) {
            is_deps_added = true;

            work_event = q_use.submit([=](sycl::handler &h) {
                const size_t kernel_threads_curr = data_count / vec_size + data_count % vec_size;
                const size_t kernel_threads_rem = rem_count / vec_size + rem_count % vec_size;
                const size_t kernel_threads = std::max({ kernel_threads_curr, kernel_threads_rem });
                const size_t kernel_size =
                    ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
                h.depends_on(dep_events);

                std::array<void *, MAX_GPUS> l_send_buf_pack_ptr;
                l_send_buf_pack_ptr[0] = (char *)send_buf + pack_count * dsize;
                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, work_group_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                        // copy first chunk from send buf to tmp buf
                        copy_data<T, N, vec_size>(
                            l_send_cp_dst_ptrs, l_send_cp_src_ptrs, data_count, it);
                        // copy the tail unaligned data from send buf to tmp buf's next offset
                        copy_data<T, 1, vec_size>(
                            l_send_ptrs_rem, l_send_buf_pack_ptr, rem_count, it);
                    });
            });
        }
        else if (rem_count != 0 && nc == 0) {
            is_deps_added = true;
            work_event = q_use.submit([=](sycl::handler &h) {
                h.depends_on(dep_events);
                h.memcpy(
                    l_send_ptrs_rem[0], (char *)send_buf + pack_count * dsize, rem_count * dsize);
            });
        }

        std::vector<sycl::event> barrier_deps;
        if (nc == 0 && !is_deps_added) {
            is_deps_added = true;
            barrier_deps = dep_events;
        }
        else {
            barrier_deps.push_back(work_event);
        }
        work_event = invoke_barrier(node_comm, q_use, barrier_deps, is_cpu_barrier);
        if (comm->is_multi_thread_instance() == true) {
            pthread_barrier_wait(
                &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        }
        work_event = q_use.submit([=](sycl::handler &h) {
            const size_t kernel_threads_curr = data_count / vec_size + data_count % vec_size;
            const size_t kernel_threads_prev =
                nc > 0 ? data_count_prev / vec_size + data_count_prev % vec_size : 0;
            const size_t kernel_threads_next =
                is_use_tmp && nc < num_chunks - 1
                    ? data_count_next / vec_size + data_count_next % vec_size
                    : 0;
            const size_t kernel_threads_rem =
                nc == 0 && rem_count != 0 ? rem_count / vec_size + rem_count % vec_size : 0;
            const size_t kernel_threads = std::max({ kernel_threads_curr,
                                                     kernel_threads_prev,
                                                     kernel_threads_next,
                                                     kernel_threads_rem });
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

            h.depends_on(work_event);

            ccl_kernel_barrier_data dummy_kbd;
            ccl_comm_barrier_data dummy_cbd = node_comm->barrier_data();

            h.parallel_for(
                sycl::nd_range<1>(kernel_size, work_group_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                    read_reduce_write<T, N, vec_size>(l_mdfi_send_ptrs,
                                                      l_send_ptrs,
                                                      l_xelink_work_wr_ptrs,
                                                      is_multi_tile,
                                                      data_count,
                                                      it);

                    // <typename T, int N, int vec_size, int use_block, int use_local_barrier, int use_global_barrier, int read_all>
                    if (nc > 0) {
                        reduce_sum<T, N, vec_size, 1, 0, 0>(nullptr,
                                                            l_reduce_sum_dst_prev,
                                                            nullptr,
                                                            l_work_ptrs_prev,
                                                            l_work_ptrs_prev,
                                                            dummy_kbd,
                                                            dummy_cbd,
                                                            data_count_prev,
                                                            it);
                    }

                    if (is_use_tmp && nc < num_chunks - 1 && is_multi_tile) {
                        copy_data<T, N, vec_size>(
                            l_send_cp_dst_ptrs_next, l_send_cp_src_ptrs_next, data_count_next, it);
                    }

                    if (nc == 0 && rem_count != 0) {
                        // TODO: change this node_comm allreduce to reduce on last rank
                        // and then as part of read_write kernel, perform the allgather
                        // so as to make it work with scaleout
                        reduce_sum<T, N * NP, vec_size, 1, 0, 0>(nullptr,
                                                                 l_recv_ptr,
                                                                 nullptr,
                                                                 l_node_send_ptrs,
                                                                 l_node_send_ptrs,
                                                                 dummy_kbd,
                                                                 dummy_cbd,
                                                                 rem_count,
                                                                 it);
                    }
                });
        });
        // when tmp_buf used, perform chunked allgatherv
        if (is_use_tmp && nc > 0) {
            const size_t kernel_threads_prev =
                data_count_prev / vec_size + data_count_prev % vec_size;
            const size_t kernel_threads_prev_2 =
                nc > 1 && is_multi_tile
                    ? data_count_prev_2 / vec_size + data_count_prev_2 % vec_size
                    : 0;
            const size_t kernel_threads = std::max(kernel_threads_prev, kernel_threads_prev_2);
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

            work_event = invoke_barrier(node_comm, q_use, { work_event }, is_cpu_barrier);

            work_event = q_use.submit([=](sycl::handler &h) {
                h.depends_on(work_event);

                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, work_group_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                        read_write<T, N, vec_size>(l_xelink_work_rd_ptrs_prev,
                                                   l_recv_ptrs_prev,
                                                   l_mdfi_recv_ptrs_prev,
                                                   is_multi_tile,
                                                   data_count_prev,
                                                   it);

                        if (nc > 1 && is_multi_tile) {
                            copy_data<T, N, vec_size>(l_recv_cp_dst_ptrs_prev_2,
                                                      l_recv_cp_src_ptrs,
                                                      data_count_prev_2,
                                                      it);
                        }
                    });
            });
        }
        // save prev pointers to be used in next iteration
        for (int i = 0; i < even_comm_size; i++) {
            l_work_ptrs_prev[i] = l_work_ptrs[i];
            l_xelink_work_rd_ptrs_prev[i] = l_xelink_work_rd_ptrs[i];
            l_recv_ptrs_prev[i] = l_recv_ptrs[i];
            l_mdfi_recv_ptrs_prev[i] = l_mdfi_recv_ptrs[i];

            l_recv_cp_src_ptrs_prev_2[i] = l_recv_cp_src_ptrs_prev[i];
            l_recv_cp_src_ptrs_prev[i] = l_recv_cp_src_ptrs[i];

            l_recv_cp_dst_ptrs_prev_2[i] = l_recv_cp_dst_ptrs_prev[i];
            l_recv_cp_dst_ptrs_prev[i] = l_recv_cp_dst_ptrs[i];
        }
        l_reduce_sum_dst_prev = l_reduce_sum_dst;
        // pipeline epilogue
        // this reduction can also be done from the main kernel as last step with a guard of nc == num_chunks - 1
        if (nc == num_chunks - 1) {
            work_event = invoke_barrier(node_comm, q_use, { work_event }, is_cpu_barrier);

            const size_t kernel_threads = data_count / vec_size + data_count % vec_size;
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

            const size_t kernel_threads_prev =
                nc > 0 && is_use_tmp && is_multi_tile
                    ? data_count_prev / vec_size + data_count_prev % vec_size
                    : 0;
            const size_t kernel_threads_rw = std::max({ kernel_threads_prev, kernel_threads });
            const size_t kernel_size_rw =
                ((kernel_threads_rw + work_group_size - 1) / work_group_size) * work_group_size;

            ccl_kernel_barrier_data dummy_kbd;
            ccl_comm_barrier_data dummy_cbd = node_comm->barrier_data();

            work_event = q_use.submit([=](sycl::handler &h) {
                h.depends_on(work_event);

                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, work_group_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                        reduce_sum<T, N, vec_size, 1, 0, 0>(nullptr,
                                                            l_reduce_sum_dst,
                                                            nullptr,
                                                            l_work_ptrs,
                                                            l_work_ptrs,
                                                            dummy_kbd,
                                                            dummy_cbd,
                                                            data_count,
                                                            it);
                    });
            });
            // when tmp_buf used, perform chunked allgatherv
            if (is_use_tmp) {
                work_event = invoke_barrier(node_comm, q_use, { work_event }, is_cpu_barrier);

                work_event = q_use.submit([=](sycl::handler &h) {
                    h.depends_on(work_event);

                    h.parallel_for(
                        sycl::nd_range<1>(kernel_size_rw, work_group_size),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                            read_write<T, N, vec_size>(l_xelink_work_rd_ptrs,
                                                       l_recv_ptrs,
                                                       l_mdfi_recv_ptrs,
                                                       is_multi_tile,
                                                       data_count,
                                                       it);
                            if (nc > 0 && is_multi_tile) {
                                copy_data<T, N, vec_size>(l_recv_cp_dst_ptrs_prev_2,
                                                          l_recv_cp_src_ptrs_prev_2,
                                                          data_count_prev,
                                                          it);
                            }
                        });
                });

                if (is_multi_tile) {
                    work_event = invoke_barrier(node_comm, q_use, { work_event }, is_cpu_barrier);
                    work_event = q_use.submit([=](sycl::handler &h) {
                        h.depends_on(work_event);

                        h.parallel_for(
                            sycl::nd_range<1>(kernel_size, work_group_size),
                            [=](sycl::nd_item<1> it)
                                [[sycl::reqd_sub_group_size(work_group_size)]] {
                                    copy_data<T, N, vec_size>(
                                        l_recv_cp_dst_ptrs, l_recv_cp_src_ptrs, data_count, it);
                                });
                    });
                }
            }
        }
    }
    // when tmp_buf is not used, perform a non-chunked single-kernel allgatherv
    if (!is_use_tmp) {
        std::array<void *, MAX_GPUS> l_peer_even_ptrs, l_local_ptrs, l_peer_pair_ptrs;
        for (int i = 0; i < even_comm->size(); i++) {
            // offsets for read_write kernel
            const int global_rank = even_comm->get_node_rank(i);
            const size_t offset_bytes = recv_bytes * global_rank;
            l_peer_even_ptrs[i] = (char *)sycl_ptrs.xelink_ptrs_wr[i] + offset_bytes;
            l_local_ptrs[i] = (char *)recv_buf + offset_bytes;
            l_peer_pair_ptrs[i] = (char *)sycl_ptrs.mdfi_ptr_wr + offset_bytes;
        }

        work_event = invoke_barrier(node_comm, q, { work_event }, is_cpu_barrier);
        const size_t kernel_threads = recv_count / vec_size + recv_count % vec_size;
        const size_t kernel_size =
            ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

        work_event = q.submit([=](sycl::handler &h) {
            h.depends_on(work_event);

            h.parallel_for(
                sycl::nd_range<1>(kernel_size, work_group_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                    read_write<T, N, vec_size>(l_peer_even_ptrs,
                                               l_local_ptrs,
                                               l_peer_pair_ptrs,
                                               is_multi_tile,
                                               recv_count,
                                               it);
                });
        });
        work_event = invoke_barrier(node_comm, q, { work_event }, is_cpu_barrier);
    }

    // do the average reduction separately after sum
    if (reduction == ccl::reduction::avg) {
        // set dependencies
        std::vector<sycl::event> avg_deps_evs;
        avg_deps_evs.push_back(work_event);

        LOG_DEBUG(
            "allreduce_large calculate average on counts: ", count, ", ranks: ", node_comm_size);
        work_event = sycl_average(q, recv_buf, count, node_comm_size, dtype, avg_deps_evs);
    }

    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }
    return ccl::event::create_from_native(work_event);
}

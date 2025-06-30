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
#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_kernels.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

template <typename T, int N>
void inline read_write_kernel(std::array<void*, MAX_GPUS> even_ptrs,
                              std::array<void*, MAX_GPUS> local_ptrs,
                              std::array<void*, MAX_GPUS> pair_ptrs,
                              const bool is_multi_tile,
                              const size_t idx) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        const T val = ((T*)even_ptrs[i])[idx];
        if (is_multi_tile) {
            ((T*)pair_ptrs[i])[idx] = val;
        }
        ((T*)local_ptrs[i])[idx] = val;
    }
}

template <typename T, int N, int vec_size>
void inline read_write(std::array<void*, MAX_GPUS> even_ptrs,
                       std::array<void*, MAX_GPUS> local_ptrs,
                       std::array<void*, MAX_GPUS> pair_ptrs,
                       const bool is_multi_tile,
                       const size_t count,
                       const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    const size_t packed_count = count / vec_size;

    sycl::sub_group sg = it.get_sub_group();
    const size_t sgSize = sg.get_local_range()[0];

    int base = (idx / sgSize) * sgSize * vec_size;
    const long rem_elem_count = count - base;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        read_write_kernel<AT, N>(even_ptrs, local_ptrs, pair_ptrs, is_multi_tile, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            read_write_kernel<T, N>(even_ptrs, local_ptrs, pair_ptrs, is_multi_tile, new_idx);
        }
    }
}

template <typename T>
ccl::event allgatherv_large_impl_ipc_ce(const void* send_buf,
                                        size_t send_count,
                                        void* recv_buf,
                                        const ccl::vector_class<size_t>& recv_counts,
                                        const ccl::vector_class<size_t>& offsets,
                                        ccl::datatype dtype,
                                        ccl_comm* comm,
                                        ccl_stream* global_stream,
                                        sycl_ptrs_type& sycl_ptrs,
                                        const ccl::vector_class<ccl::event>& deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();
    bool is_recording = use_recording_path(q);

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    // TODO: should we remove local copy for inplace
    bool is_multi_tile = pair_comm->size() > 1;
    bool is_xelink_read = ccl::global_data::env().allgatherv_topo_read;
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    // get the link engine queues for xelink communication
    // use computer engine or main copy engine for local copy
    std::vector<sycl::queue> q_lce;

    for (int i = 0, j = 0; i < even_comm->size(); i++) {
        bool is_local = (i == even_comm->rank());

        // use link engines 2, 4, 6 ...
        if (!is_local) {
            j++;
        }
        int lce_idx = (j * 2) % get_num_lce();

        // use main copy engine for local copy and link engine for xelink copy
        q_lce.emplace_back((is_local ? get_mce_queue(q) : get_lce_queue(q, lce_idx)));

        // use compute engine copy for local copy if kernel barrier is used
        if (is_local && !is_cpu_barrier) {
            q_lce[i] = q;
        }
    }

    sycl::event barrier_event2;
    if (is_xelink_read) {
        LOG_DEBUG("allgatherv large copy engine read");
        // TODO: can we change it to even_comm
        sycl::event barrier_event1 = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

        std::vector<sycl::event> cp_events(even_comm->size());
        for (int i = 0; i < even_comm->size(); i++) {
            int r = (i + even_comm->rank()) % even_comm->size();
            // TODO: make sure that get_node_rank() (or get_global_rank()) return the ABSOLUTE (i.e. MPI_COMM_WORLD) rank in the node
            const int global_rank = even_comm->get_node_rank(r);
            const size_t offset_bytes = !offsets.empty() ? offsets[global_rank] : send_count * global_rank * dsize;

            void* src = (char*)sycl_ptrs.xelink_ptrs_rd[r];
            void* local = (char*)recv_buf + offset_bytes;
            void* dst = (char*)sycl_ptrs.mdfi_ptr_wr + offset_bytes;

            sycl::event e1 = q_lce[r].submit([=](sycl::handler& h) {
                h.depends_on(barrier_event1);
                h.memcpy(local, src, dsize * send_count);
            });

            if (is_multi_tile) {
                cp_events[r] = get_mce_queue(q).submit([=](sycl::handler& h) {
                    h.depends_on(e1);
                    h.memcpy(dst, local, dsize * send_count);
                });
            }
            else {
                cp_events[r] = e1;
            }
        }

        // TODO: we can remove this barrier when single tile is used
        barrier_event2 = invoke_barrier(pair_comm, q, cp_events, is_cpu_barrier);
    }
    else {
        LOG_DEBUG("allgatherv large copy engine write");
        const int my_global_rank = node_comm->rank();
        const size_t my_offset_bytes = send_count * my_global_rank * dsize;

        // TODO: can we delete this barrier
        sycl::event barrier_event0 = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

        std::vector<sycl::event> cp_events1(even_comm->size());
        for (int i = 0; i < even_comm->size(); i++) {
            int r = (i + even_comm->rank()) % even_comm->size();
            void* xelink_tgt = (char*)sycl_ptrs.xelink_ptrs_wr[r] + my_offset_bytes;

            cp_events1[r] = q_lce[r].submit([=](sycl::handler& h) {
                h.depends_on(barrier_event0);
                h.memcpy(xelink_tgt, send_buf, dsize * send_count);
            });
        }
        sycl::event barrier_event1 = invoke_barrier(even_comm, q, cp_events1, is_cpu_barrier);

        if (is_multi_tile) {
            std::vector<sycl::event> cp_events2(even_comm->size());
            for (int i = 0; i < even_comm->size(); i++) {
                const int global_rank = even_comm->get_node_rank(i);
                const size_t offset_bytes =
                    !offsets.empty() ? offsets[global_rank] : send_count * global_rank * dsize;

                void* src = (char*)recv_buf + offset_bytes;
                void* dst = (char*)sycl_ptrs.mdfi_ptr_wr + offset_bytes;

                cp_events2[i] = get_mce_queue(q).submit([=](sycl::handler& h) {
                    h.depends_on(barrier_event1);
                    h.memcpy(dst, src, dsize * send_count);
                });
            }
            barrier_event2 = invoke_barrier(pair_comm, q, cp_events2, is_cpu_barrier);
        }
        else {
            barrier_event2 = barrier_event1;
        }
    }
    return ccl::event::create_from_native(barrier_event2);
}

template <typename T, int N, int vec_size_use>
ccl::event allgatherv_large_impl_ipc(const void* send_buf,
                                     size_t send_count,
                                     void* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     const ccl::vector_class<size_t>& offsets,
                                     ccl::datatype dtype,
                                     ccl_comm* comm,
                                     ccl_stream* global_stream,
                                     sycl_ptrs_type& sycl_ptrs,
                                     const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("allgatherv large kernel no tmp buffer");
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    bool is_recording = use_recording_path(q);

    std::array<void*, MAX_GPUS> local_peer_even_ptrs, local_local_ptrs, local_peer_pair_ptrs;
    for (int i = 0; i < even_comm->size(); i++) {
        // offsets for read_write kernel
        const int global_rank = even_comm->get_node_rank(i);
        const size_t offset_bytes = !offsets.empty() ? offsets[global_rank] : send_count * global_rank * dsize;
        local_peer_even_ptrs[i] = (char*)sycl_ptrs.xelink_ptrs_rd[i];
        local_local_ptrs[i] = (char*)recv_buf + offset_bytes;
        local_peer_pair_ptrs[i] = (char*)sycl_ptrs.mdfi_ptr_wr + offset_bytes;
    }

    sycl::event barrier_event1 = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    constexpr int work_group_size = 16;

    constexpr int vec_size = vec_size_use;
    const bool is_multi_tile = pair_comm->size() > 1;
    const size_t kernel_threads = send_count / vec_size + send_count % vec_size;
    const size_t kernel_size = ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

    sycl::event kernel_event = q.submit([=](sycl::handler& h) {
        h.depends_on(barrier_event1);
        h.parallel_for(
            sycl::nd_range<1>(kernel_size, work_group_size),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                read_write<T, N, vec_size>(
                    local_peer_even_ptrs, local_local_ptrs, local_peer_pair_ptrs, is_multi_tile, send_count, it);
            });
    });

    sycl::event barrier_event2 = invoke_barrier(node_comm, q, { kernel_event }, is_cpu_barrier);
    return ccl::event::create_from_native(barrier_event2);
}

template <typename T, int N, int vec_size_use>
ccl::event allgatherv_large_impl_tmp(const void* send_buf,
                                     size_t send_count,
                                     void* recv_buf,
                                     const ccl::vector_class<size_t>& recv_counts,
                                     const ccl::vector_class<size_t>& offsets,
                                     ccl::datatype dtype,
                                     ccl_comm* comm,
                                     ccl_stream* global_stream,
                                     sycl_ptrs_type& sycl_ptrs,
                                     const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("allgatherv large kernel with tmp buffer");
    sycl::queue q = global_stream->get_native_stream();
    bool is_recording = use_recording_path(q);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    static sycl::queue q_worker(q.get_device());
    static sycl::queue q_copy = get_mce_queue(q);

    sycl::queue q_use = q_worker;

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    const int pair_comm_size = pair_comm->size();
    const int even_comm_size = even_comm->size();
    const int comm_rank = node_comm->rank();

    // use kernel for local pipeline copies of next and prev buffers,
    // by default we using main copy engine using memcpy
    const bool use_kernel_copy = ccl::global_data::env().sycl_kernel_copy;

    constexpr int pipeline_size = 2;

    const size_t chunk_size = get_tmp_buf_size_per_rank();
    void* tmp_bufs[pipeline_size] = { get_tmp_buf(0, comm), get_tmp_buf(1, comm) };
    const size_t pipeline_offset = (char*)tmp_bufs[1] - (char*)tmp_bufs[0];

    std::array<void*, MAX_GPUS> local_peer_even_ptrs, local_local_ptrs, local_peer_pair_ptrs;
    std::array<void*, MAX_GPUS> recv_buf_dst_ptrs, tmp_buf_src_ptrs;
    std::array<void*, MAX_GPUS> recv_buf_dst_ptrs_prev, tmp_buf_src_ptrs_prev;
    std::array<void*, MAX_GPUS> tmp_send_buf_next, my_send_buf_next;

    const size_t chunk_count = chunk_size / dsize;
    const size_t num_chunks = send_count / chunk_count + (send_count % chunk_count != 0);
    const bool is_multi_tile = pair_comm_size > 1;

    std::vector<sycl::event> work_events;
    sycl::event output_event;
    for (size_t nc = 0; nc < num_chunks; nc++) {
        // setup pointers

        // alternate between tmp buffers since we use a pipeline of size 2
        // i.e. copy previous output from one tmp_buffer when allgatherv
        // is operating on the second tmp_buffer
        const int pipeline_index = nc % pipeline_size;
        const int pipeline_index_other = (nc + 1) % pipeline_size;
        const size_t pipeline_offset_use = pipeline_index * pipeline_offset;
        void* tmp_buf_use = tmp_bufs[pipeline_index];
        void* tmp_buf_other = tmp_bufs[pipeline_index_other];

        // offset on send buffer
        const size_t my_offset_count_send = chunk_count * nc;
        // offset on recv buffer
        const size_t my_offset_count = send_count * comm_rank + my_offset_count_send;
        // offset on tmp buffer
        const size_t my_offset_count_tmp = chunk_count * comm_rank;

        void* my_send_buf = (char*)send_buf + my_offset_count_send * dsize;
        void* tmp_send_buf = (char*)tmp_buf_use + my_offset_count_tmp * dsize;

        my_send_buf_next[0] = (char*)my_send_buf + chunk_count * dsize;
        tmp_send_buf_next[0] = (char*)tmp_buf_other + my_offset_count_tmp * dsize;

        for (int i = 0; i < even_comm_size; i++) {
            // offsets for read_write kernel
            int global_rank = comm->is_multi_thread_instance() ? i * pair_comm->size() + pair_comm->rank()
                                                               : even_comm->get_node_rank(i);
            const size_t offset_bytes = !offsets.empty() ? offsets[global_rank] + chunk_count * nc * dsize
                                                         : (send_count * global_rank + chunk_count * nc) * dsize;
            const size_t offset_bytes_tmp = chunk_count * global_rank * dsize;

            // xelink and mdfi ptrs are the tmp buffers in the other ranks
            local_peer_even_ptrs[i] = (char*)sycl_ptrs.xelink_ptrs_rd[i] + offset_bytes_tmp + pipeline_offset_use;
            local_local_ptrs[i] = (char*)recv_buf + offset_bytes;
            local_peer_pair_ptrs[i] = (char*)sycl_ptrs.mdfi_ptr_wr + offset_bytes_tmp + pipeline_offset_use;

            // offsets for copy kernel
            // TODO: is there a better way to find the pair_neighbor global rank
            int global_rank_neighbor = (global_rank / pair_comm_size) * pair_comm_size;
            if (global_rank % pair_comm_size == 0) {
                global_rank_neighbor = global_rank_neighbor + 1;
            }
            const size_t offset_bytes_c = !offsets.empty()
                                              ? offsets[global_rank_neighbor] + chunk_count * nc * dsize
                                              : (send_count * global_rank_neighbor + chunk_count * nc) * dsize;
            const size_t offset_bytes_c_tmp = chunk_count * global_rank_neighbor * dsize;
            recv_buf_dst_ptrs[i] = (char*)recv_buf + offset_bytes_c;
            tmp_buf_src_ptrs[i] = (char*)tmp_buf_use + offset_bytes_c_tmp;

            recv_buf_dst_ptrs_prev[i] = (char*)recv_buf_dst_ptrs[i] - chunk_count * dsize;
            // offset of prev tmp buffer is same but use tmp_buf_other instead of tmp_buf_use
            tmp_buf_src_ptrs_prev[i] = (char*)tmp_buf_other + offset_bytes_c_tmp;
        }

        // start the collective

        // if send_count is not a multiple of chunk_count, then last chunk will contain only remainder data
        const size_t data_count = (nc < send_count / chunk_count) ? chunk_count : send_count % chunk_count;
        const size_t data_count_next =
            (nc + 1 < send_count / chunk_count) ? chunk_count : send_count % chunk_count;
        // prev exists only if there is atleast 2 chunks
        const size_t data_count_prev = chunk_count;

        // TODO: should we move this outside the loop ?
        // pipeline prologue - copy first chunk from send_buf to tmp_buf using in-order queue
        if (nc == 0) {
            std::vector<sycl::event> dep_events = get_sycl_events(deps);
            sycl::event e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(tmp_send_buf, my_send_buf, dsize * data_count);
            });
            work_events.push_back(e);
        }

        sycl::event barrier_event1 = invoke_barrier(node_comm, q, work_events, is_cpu_barrier);
        work_events.clear();

        constexpr int work_group_size = 16;

        constexpr int vec_size = vec_size_use;
        const size_t kernel_threads_curr = data_count / vec_size + data_count % vec_size;
        const size_t kernel_threads_next =
            use_kernel_copy && nc < num_chunks - 1 ? data_count_next / vec_size + data_count_next % vec_size : 0;
        const size_t kernel_threads_prev =
            use_kernel_copy && nc > 0 ? data_count_prev / vec_size + data_count_prev % vec_size : 0;
        const size_t kernel_threads = std::max({ kernel_threads_curr, kernel_threads_next, kernel_threads_prev });
        const size_t kernel_size = ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

        sycl::event kernel_event = q.submit([=](sycl::handler& h) {
            h.depends_on(barrier_event1);
            h.parallel_for(
                sycl::nd_range<1>(kernel_size, work_group_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                    read_write<T, N, vec_size>(local_peer_even_ptrs,
                                               local_local_ptrs,
                                               local_peer_pair_ptrs,
                                               is_multi_tile,
                                               data_count,
                                               it);
                    // copy next input chunk
                    if (use_kernel_copy && nc < num_chunks - 1) {
                        copy_data<T, 1, vec_size>(tmp_send_buf_next, my_send_buf_next, data_count_next, it);
                    }
                    // copy prev output chunk
                    if (use_kernel_copy && nc > 0 && is_multi_tile) {
                        copy_data<T, N, vec_size>(
                            recv_buf_dst_ptrs_prev, tmp_buf_src_ptrs_prev, data_count_prev, it);
                    }
                });
        });
        work_events.push_back(kernel_event);

        std::vector<sycl::event> copy_events;
        // copy next input chunk
        if (!use_kernel_copy && nc < num_chunks - 1) {
            copy_data(dsize,
                      1,
                      tmp_send_buf_next,
                      my_send_buf_next,
                      data_count_next,
                      q_copy,
                      { barrier_event1 },
                      copy_events);
        }
        // copy prev output chunk
        if (!use_kernel_copy && nc > 0 && is_multi_tile) {
            // for last iteration, if read_write kernel is small, then use
            // compute engine for copying since it is faster than copy engine
            // and there is very less overlap with read_write since it is small
            const size_t small_size_threshold = ccl::global_data::env().sycl_allgatherv_small_threshold;

            //TODO: should we use single kernel copy when q_use is used
            sycl::queue q_copy_use = (nc == num_chunks - 1 && data_count < small_size_threshold) ? q_use : q_copy;
            copy_data(dsize,
                      N,
                      recv_buf_dst_ptrs_prev,
                      tmp_buf_src_ptrs_prev,
                      data_count_prev,
                      q_copy_use,
                      { barrier_event1 },
                      copy_events);
        }

        // WA: directly connecting the output event of q_copy to gpu kernels
        // cause failure when MPI binding is used - I_MPI_PIN_PROCESSOR_LIST
        if (!copy_events.empty()) {
            sycl::event e = q_use.submit([=](sycl::handler& h) {
                h.depends_on(copy_events);
                h.host_task([]() {});
            });
            work_events.push_back(e);
        }

        // TODO: move this outside of the looop
        // pipeline epilogue - copy the final output chunk from tmp_buffer to recv_buffer
        if (nc == num_chunks - 1 && is_multi_tile) {
            sycl::event barrier_event2;
            barrier_event2 = invoke_barrier(node_comm, q, work_events, is_cpu_barrier);
            work_events.clear();

            // TODO: find when to use single kernel copy vs memcpys
            constexpr bool use_single_kernel_copy = true;
            // use a single kernel to copy from tmp_buffer to recv_buffer
            if (use_single_kernel_copy) {
                output_event = q.submit([=](sycl::handler& h) {
                    h.depends_on(barrier_event2);

                    // vec_size of 1 is too slow for local copy
                    constexpr int vec_size = vec_size_use;
                    const size_t kernel_threads = data_count / vec_size + data_count % vec_size;
                    const size_t kernel_size =
                        ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
                    h.parallel_for(
                        sycl::nd_range<1>(kernel_size, work_group_size),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(work_group_size)]] {
                            copy_data<T, N, vec_size>(recv_buf_dst_ptrs, tmp_buf_src_ptrs, data_count, it);
                        });
                });
            }
            // use memcpys to copy from tmp_buffer to recv_buffer
            else {
                for (int i = 0; i < even_comm_size; i++) {
                    sycl::event e = q_use.submit([=](sycl::handler& h) {
                        h.depends_on(barrier_event2);
                        h.memcpy(recv_buf_dst_ptrs[i], tmp_buf_src_ptrs[i], dsize * data_count);
                    });
                    work_events.push_back(e);
                }
                output_event = submit_wait_on_events(q, work_events);
            }
        }
        else if (nc == num_chunks - 1) {
            sycl::event barrier_event2;
            barrier_event2 = invoke_barrier(node_comm, q, work_events, is_cpu_barrier);
            work_events.clear();
            output_event = submit_wait_on_events(q, { barrier_event2 });
        }
        else {
            output_event = submit_wait_on_events(q, work_events);
        }
    } // nc

    return ccl::event::create_from_native(output_event);
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP, bool use_full_vector>
ccl::event allgatherv_large_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const ccl::vector_class<size_t>& recv_counts,
                                 const ccl::vector_class<size_t>& offsets,
                                 ccl::datatype dtype,
                                 ccl_comm* comm,
                                 ccl_stream* global_stream,
                                 sycl_ptrs_type& sycl_ptrs,
                                 const ccl::vector_class<ccl::event>& deps,
                                 bool use_tmp) {
    constexpr int N = NE;
    // for 2 byte types with odd count, use 4 byte vectors instead of 8 bytes
    constexpr int vec_size_use = get_num_elements<T, 8, use_full_vector>();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    const bool is_tmp_used = use_tmp;

    ccl::event e;
    // TODO: copy engines currently does not support tmp buf
    if (ccl::global_data::env().sycl_copy_engine) {
        e = allgatherv_large_impl_ipc_ce<T>(
            send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, sycl_ptrs, deps);
    }
    else if (!is_tmp_used) {
        e = allgatherv_large_impl_ipc<T, N, vec_size_use>(
            send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, sycl_ptrs, deps);
    }
    else {
        e = allgatherv_large_impl_tmp<T, N, vec_size_use>(
            send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, sycl_ptrs, deps);
    }
    return e;
}

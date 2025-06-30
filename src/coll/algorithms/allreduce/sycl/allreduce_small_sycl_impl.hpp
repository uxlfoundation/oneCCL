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
#include "coll/algorithms/allreduce/sycl/allreduce_small_sycl_rw_kernel.hpp"

template <typename T, int N, int read_all>
inline void reduce_kernel(void *recv,
                          std::array<void *, MAX_NODE_RANKS> in,
                          std::array<void *, MAX_NODE_RANKS> out,
                          size_t idx) {
    T tmp_arr[N];
    // copy from remote to local array
    T sum = ((T *)in[0])[idx];
#pragma unroll
    for (int i = 1; i < N; i++) {
        tmp_arr[i] = ((T *)in[i])[idx];
    }

    // reduce from local array
    for (int i = 1; i < N; i++) {
        sum += tmp_arr[i];
    }

    // write to local recv buffer
    if (read_all) {
        ((T *)recv)[idx] = sum;
    }
    // write back to remote tmp buffers
    else {
#pragma unroll
        for (int i = 0; i < N; i++) {
            ((T *)out[i])[idx] = sum;
        }
    }
}

template <typename T,
          int N,
          int vec_size,
          int use_block,
          int use_local_barrier,
          int use_global_barrier,
          int read_all = 1,
          int M = 1,
          typename AT = sycl::vec<T, vec_size>>
void inline reduce_sum(const void *send,
                       void *recv,
                       void *tmp,
                       std::array<void *, MAX_NODE_RANKS> in,
                       std::array<void *, MAX_NODE_RANKS> out,
                       ccl_kernel_barrier_data kernel_barrier_data,
                       const ccl_comm_barrier_data comm_barrier_data,
                       const size_t count,
                       const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();

    const size_t packed_count = count / vec_size;

    if (use_local_barrier) {
        // copy data from send buffer to tmp buffer
        if (use_block && idx < packed_count) {
            using MAT = sycl::marray<AT, M>;
            ((MAT *)tmp)[idx] = ((MAT *)send)[idx];
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_count;
            if (new_idx < count) {
                using MT = sycl::marray<T, M>;
                ((MT *)tmp)[new_idx] = ((MT *)send)[new_idx];
            }
        }

        // local barrier within gpu
        kernel_barrier(kernel_barrier_data.get_sync_ptr(), it);
    }

    if (use_global_barrier) {
        // global communication barrier across ranks
        comm_barrier(comm_barrier_data, it);
    }

    // reset local barrier counter
    if (use_local_barrier && idx == 0) {
        kernel_barrier_data.reset_sync_data();
    }

    if (use_block && idx < packed_count) {
        reduce_kernel<AT, N, read_all>(recv, in, out, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            reduce_kernel<T, N, read_all>(recv, in, out, new_idx);
        }
    }
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP>
ccl::event allreduce_small_impl(const void *send_buf,
                                void *recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm *comm,
                                ccl_stream *global_stream,
                                const ccl::vector_class<ccl::event> &deps) {
    constexpr int N = NE * NP;
    sycl::queue q = global_stream->get_native_stream();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    size_t hw_threads = get_total_threads(q);
    bool is_recording = use_recording_path(q);

    // create sections as data size increases and increase the vector size
    // used in each thread so as to reduce the number of threads and this
    // reduces the overhead of atomic increment in local sync
    // and help to contain number of sycl thread <= hw threads.
    // for data sizes not contained within the sections, divide single kernel
    // currently a maximum of 2048 threads are used in each sections.
    const size_t sec_0 = std::min(2048ul, hw_threads);
    const size_t sec_1 = sec_0 * 8; // 8 byte vectors
    const size_t sec_2 = sec_1 * 2; // 16 byte vectors

    // use full vector (>= 8 bytes) if buffers are 4 byte aligned
    // we dont have to take the count for calculating alignment,
    // since we are not dividing the data among ranks
    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, 0, 0);
    const bool use_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    const bool use_kernel_sync = ccl::global_data::env().sycl_kernel_sync;

    auto [local_tmp_buf, remote_ptrs] =
        is_recording ? node_comm->get_all_tmp_bufs_gpu(true) : node_comm->get_all_tmp_bufs(true);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;
    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }
    // VS : vec_size, SGS : sub_group_size, LB : use_local_barrier, GB: use_global_barrier
    auto reduce_sum_invoke =
        [=, &q]<int VS, int SGS, int LB, int GB, typename AT = sycl::vec<T, VS>>(
            std::vector<sycl::event> l_dep_events) {
        constexpr int use_block = 1;
        constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
        const size_t kernel_threads = count / vec_size + count % vec_size;
        const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;

        // total number of hw threads is a multiple of sub_group
        CCL_THROW_IF_NOT(hw_threads % SGS == 0);
        // if synchronization is there, make sure sycl threads can be contained within hw threads
        if (LB == 1 || GB == 1) {
            if (kernel_size > hw_threads) {
                CCL_THROW("sycl threads : ",
                          kernel_size,
                          " > hw threads : ",
                          hw_threads,
                          " is not allowed in allreduce small for count :",
                          count);
            }
        }

        ccl_kernel_barrier_data kernel_barrier_data = get_kernel_barrier_data(comm).inc_slot();
        // if global barrier is not used, then do not increment the barrier counter
        ccl_comm_barrier_data comm_barrier_data =
            GB ? node_comm->barrier_inc() : node_comm->barrier_data();

        sycl::event local_event = q.submit([=](sycl::handler &h) {
            h.depends_on(l_dep_events);
            h.parallel_for(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    reduce_sum<T, N, VS, use_block, LB, GB, 1, 1, AT>(send_buf,
                                                                      recv_buf,
                                                                      local_tmp_buf,
                                                                      remote_ptrs,
                                                                      remote_ptrs,
                                                                      kernel_barrier_data,
                                                                      comm_barrier_data,
                                                                      count,
                                                                      it);
                });
        });
        return local_event;
    };

    // three phases of allreduce: local copy to tmp, comm_barrier, gather
    // VS : vec_size
    auto memcpy_reduce_sum = [=, &q]<int VS>() {
        sycl::event memcpy_event = q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.memcpy(local_tmp_buf, send_buf, dsize * count);
        });

        sycl::event barrier_event = invoke_barrier(node_comm, q, { memcpy_event }, use_cpu_barrier);

        sycl::event local_event =
            reduce_sum_invoke.template operator()<VS, 32, 0, 0>({ barrier_event });

        if (is_recording) {
            local_event = invoke_barrier(node_comm, q, { local_event }, use_cpu_barrier);
        }
        return local_event;
    };
    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }
    // run the three phases of collective as separate kernels.
    // when cpu barrier is enabled we cannot use single gpu kernel
    // since control has to go to cpu and perform the barrier.
    // also when user asks to remove the synchronization within kernel
    // run them as separate kernels since single kernel algorithm
    // will require the threads to synchronize between phases
    if (use_cpu_barrier || !use_kernel_sync || is_recording) {
        // TODO: use cpu_barrier option with read_write kernel
        if (use_full_vector) {
            constexpr int vec_size = get_num_elements<T, 8>();
            kernel_event = memcpy_reduce_sum.template operator()<vec_size>();
        }
        else {
            // for unaligned data use vector size of 1
            kernel_event = memcpy_reduce_sum.template operator()<1>();
        }
    }
    else if (!use_full_vector) {
        // if sycl threads are not more than hw threads we can
        // use single kernel that executes barrier internally
        if (count <= sec_0) {
            // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
            kernel_event = reduce_sum_invoke.template operator()<1, 32, 1, 1>(dep_events);
        }
        else {
            kernel_event = memcpy_reduce_sum.template operator()<1>();
        }
    }
    else if (count * dsize <= sec_0) {
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<1, 16, 1, 1>(dep_events);
    }
    else if (count * dsize <= sec_1) {
        constexpr int vec_size = get_num_elements<T, 8>();
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 16, 1, 1>(dep_events);
    }
    else if (count * dsize <= sec_2) {
        constexpr int vec_size = get_num_elements<T, 16>();
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = reduce_sum_invoke.template operator()<vec_size, 16, 1, 1>(dep_events);
    }
    else {
        if (comm_size <= 2) {
            if (count * dsize <= sec_2 * 8) {
                constexpr int vec_size = get_num_elements<T, 16>();
                // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
                using AT = sycl::marray<sycl::vec<T, vec_size>, 8>;
                kernel_event =
                    reduce_sum_invoke.template operator()<vec_size * 8, 32, 1, 1, AT>(dep_events);
            }
            // using marray of size more than 128 is slower than splitting kernels
            // use read_all kernel for 2 ranks since read_all with 2 ranks does not increase data
            else {
                constexpr int vec_size = get_num_elements<T, 8>();
                kernel_event = memcpy_reduce_sum.template operator()<vec_size>();
            }
        }
        // use read_write kernel for more than 2 ranks, since read_all will increase data too much (* vs +)
        // e.g. read_all with 12 ranks has 12*12 data reads whereas read_write has 12 read + 12 write
        else {
            constexpr int vec_size = get_num_elements<T, 16>();
            const int rank_align_count = ccl::global_data::env().kernel_mem_align / dsize;
            const size_t align_count = comm_size * rank_align_count;
            const size_t count_rem = count % align_count;
            const size_t count_per_rank_tmp = (count - count_rem) / comm_size;
            const size_t offset = count_per_rank_tmp * comm_rank;
            // if count is not divisible by comm_size,
            // then last rank has the remaining data
            const size_t count_per_rank =
                count_per_rank_tmp + (comm_rank == comm_size - 1 ? count_rem : 0);

            constexpr int vec_size_cp = vec_size * N;
            const int wg_size = 16, sg_size = 16;
            const size_t kernel_threads_cp = count / vec_size_cp + count % vec_size_cp;
            const size_t kernel_threads_red = count_per_rank / vec_size + count_per_rank % vec_size;
            const size_t kernel_threads = std::max(kernel_threads_cp, kernel_threads_red);
            const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;

            // use separate kernels if there are more sycl threads than hw threads
            // to avoid deadlocking in sync barriers from single kernels
            if (kernel_size > hw_threads) {
                constexpr int vec_size = get_num_elements<T, 8>();
                kernel_event = memcpy_reduce_sum.template operator()<vec_size>();
                return ccl::event::create_from_native(kernel_event);
            }

            std::array<void *, MAX_NODE_RANKS> remote_ptrs_rank;
            for (int i = 0; i < comm_size; i++) {
                remote_ptrs_rank[i] = (char *)(remote_ptrs[i]) + offset * dsize;
            }

            ccl_kernel_barrier_data kernel_barrier_data = get_kernel_barrier_data(comm).inc_slot();
            ccl_comm_barrier_data comm_barrier_data = node_comm->barrier_inc();

            kernel_event = q.submit([=](sycl::handler &h) {
                h.depends_on(dep_events);
                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, wg_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                        // <vec_size, use_block, use_local_barrier, use_global_barrier, read_all, multiplier>
                        reduce_sum_general<T, N, vec_size, 1, 1, 1, 0, N>(send_buf,
                                                                          recv_buf,
                                                                          local_tmp_buf,
                                                                          remote_ptrs_rank,
                                                                          remote_ptrs_rank,
                                                                          kernel_barrier_data,
                                                                          comm_barrier_data,
                                                                          count,
                                                                          count_per_rank,
                                                                          it);
                    });
            });

            ccl_comm_barrier_data comm_barrier_data_next = node_comm->barrier_inc();
            kernel_event = q.submit([=](sycl::handler &h) {
                h.depends_on(kernel_event);
                h.parallel_for(
                    sycl::nd_range<1>(kernel_size, wg_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                        // global communication barrier across ranks
                        comm_barrier(comm_barrier_data_next, it);

                        // <vec_size, multiplier>
                        copy_data<T, vec_size, N>(recv_buf, local_tmp_buf, count, it);
                    });
            });
        }
    }

    // do the average reduction separately after sum
    if (reduction == ccl::reduction::avg) {
        // set dependencies
        std::vector<sycl::event> avg_deps_evs;
        avg_deps_evs.push_back(kernel_event);

        LOG_DEBUG("allreduce_small calculate average on counts: ", count, ", ranks: ", comm_size);
        kernel_event = sycl_average(q, recv_buf, count, comm_size, dtype, avg_deps_evs);
    }

    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }
    return ccl::event::create_from_native(kernel_event);
}

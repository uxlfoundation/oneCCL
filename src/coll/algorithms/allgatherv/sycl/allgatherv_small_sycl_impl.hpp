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
inline void gather_kernel(std::array<void*, MAX_NODE_RANKS> out,
                          std::array<void*, MAX_NODE_RANKS> in,
                          size_t idx) {
    T tmp_arr[N];

#pragma unroll
    // copy from remote to local array
    for (int i = 0; i < N; i++) {
        tmp_arr[i] = ((T*)in[i])[idx];
    }

#pragma unroll
    // copy from local array to output
    for (int i = 0; i < N; i++) {
        ((T*)out[i])[idx] = tmp_arr[i];
    }
}

template <typename T, int N, int vec_size, int use_block, int use_local_barrier, int use_global_barrier>
void inline gather(const void* send,
                   void* tmp,
                   std::array<void*, MAX_NODE_RANKS> out,
                   std::array<void*, MAX_NODE_RANKS> in,
                   ccl_kernel_barrier_data kernel_barrier_data,
                   const ccl_comm_barrier_data comm_barrier_data,
                   const size_t copy_count,
                   const size_t main_count,
                   const size_t last_count,
                   const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    const size_t packed_copy_count = copy_count / vec_size;
    const size_t packed_main_count = main_count / vec_size;
    const size_t packed_last_count = last_count / vec_size;

    if (use_local_barrier) {
        // copy data from send buffer to tmp buffer
        if (use_block && idx < packed_copy_count) {
            ((AT*)tmp)[idx] = ((AT*)send)[idx];
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_copy_count;
            if (new_idx < copy_count) {
                ((T*)tmp)[new_idx] = ((T*)send)[new_idx];
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

    // gather kernel
    if (main_count == last_count) {
        // default case, all counts are equal
        if (use_block && idx < packed_main_count) {
            gather_kernel<AT, N>(out, in, idx);
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_main_count;
            if (new_idx < main_count) {
                gather_kernel<T, N>(out, in, new_idx);
            }
        }
    }
    else {
        // special case, the last block is bigger.
        // restrict the compilation for 1 rank case,
        // it is handled ouside of the kernels
        if constexpr (N > 1) {
            // first N - 1 blocks
            if (use_block && idx < packed_main_count) {
                gather_kernel<AT, N - 1>(out, in, idx);
            }
            else {
                const size_t new_idx = idx + (vec_size - 1) * packed_main_count;
                if (new_idx < main_count) {
                    gather_kernel<T, N - 1>(out, in, new_idx);
                }
            }
        }
        // last block
        if (use_block && idx < packed_last_count) {
            ((AT*)out[N - 1])[idx] = ((AT*)in[N - 1])[idx];
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_last_count;
            if (new_idx < last_count) {
                ((T*)out[N - 1])[new_idx] = ((T*)in[N - 1])[new_idx];
            }
        }
    }
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP>
ccl::event allgatherv_small_impl(const void* send_buf,
                                 size_t send_count,
                                 void* recv_buf,
                                 const ccl::vector_class<size_t>& recv_counts,
                                 const ccl::vector_class<size_t>& offsets,
                                 ccl::datatype dtype,
                                 ccl_comm* comm,
                                 ccl_stream* global_stream,
                                 const ccl::vector_class<ccl::event>& deps) {
    constexpr int N = NE * NP;
    sycl::queue q = global_stream->get_native_stream();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    // TODO: which communicator to use for scaleout
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    // allgatherv kernels currently do not support variable counts.
    // however, the last rank may have a different count, if allgather is called
    // from allreduce collective for example.
    // small allgaterv kernel handles this special case.
    // use first element as a base count for checking hw sections.
    const size_t count = recv_counts.front();
    const size_t last_count = recv_counts.back();

    size_t hw_threads = get_total_threads(q);

    // create sections as data size increases and increase the vector size
    // used in each thread so as to reduce the number of threads and this
    // reduces the overhead of atomic increment in local sync
    // and help to contain number of sycl thread <= hw threads.
    // for data sizes not contained within the sections, divide single kernel
    // into multiple kernels to avoid deadlocks when sycl thread > hw threads.
    // currently a maximum of 2048 threads are used in each sections.
    const size_t sec_0 = std::min(2048ul, hw_threads);
    const size_t sec_1 = sec_0 * 8; // 8 byte vectors
    const size_t sec_2 = sec_1 * 2; // 16 byte vectors
    const size_t sec_3 = sec_2 * 2; // 32 byte vectors

    // use full vector (>= 8 bytes) if buffers and data size are 4 byte aligned
    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, count, dsize);
    const bool use_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    const bool use_kernel_sync = ccl::global_data::env().sycl_kernel_sync;
    const bool is_recording = use_recording_path(q);

    auto [local_tmp_buf, remote_ptrs] =
        is_recording ? node_comm->get_all_tmp_bufs_gpu(true) : node_comm->get_all_tmp_bufs(true);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;

    // VS : vec_size, SGS : sub_group_size, LB : use_local_barrier, GB : use_global_barrier
    auto gather_invoke = [=, &q]<int VS, int SGS, int LB, int GB>(std::vector<sycl::event> sycl_deps) {
        constexpr int use_block = 1;
        constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
        const size_t kernel_threads_copy = send_count / vec_size + send_count % vec_size;
        const size_t kernel_threads_main = count / vec_size + count % vec_size;
        const size_t kernel_threads_last = last_count / vec_size + last_count % vec_size;
        const size_t kernel_threads = std::max({ kernel_threads_copy, kernel_threads_main, kernel_threads_last });
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
                          " is not allowed in allgatherv small for count :",
                          count);
            }
        }

        std::array<void*, MAX_NODE_RANKS> out_ptrs;
        // TODO: is it better to only pass recv_buf to the kernel and do this calculation there
        for (int i = 0; i < comm_size; i++) {
            out_ptrs[i] = !offsets.empty() ? (char*)recv_buf + offsets[i] : (char*)recv_buf + i * count * dsize;
        }

        ccl_kernel_barrier_data kernel_barrier_data = get_kernel_barrier_data(comm).inc_slot();
        // if global barrier is not used, then do not increment the barrier counter
        ccl_comm_barrier_data comm_barrier_data = GB ? node_comm->barrier_inc() : node_comm->barrier_data();

        sycl::event local_event = q.submit([=](sycl::handler& h) {
            h.depends_on(sycl_deps);
            h.parallel_for(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    gather<T, N, VS, use_block, LB, GB>(send_buf,
                                                        local_tmp_buf,
                                                        out_ptrs,
                                                        remote_ptrs,
                                                        kernel_barrier_data,
                                                        comm_barrier_data,
                                                        send_count,
                                                        count,
                                                        last_count,
                                                        it);
                });
        });
        return local_event;
    };

    // three phases of allgather: local copy to tmp, comm_barrier, gather
    // VS : vec_size
    auto memcpy_gather = [=, &q]<int VS>() {
        sycl::event memcpy_event = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(local_tmp_buf, send_buf, dsize * send_count);
        });

        sycl::event barrier_event = invoke_barrier(node_comm, q, { memcpy_event }, use_cpu_barrier);

        sycl::event local_event = gather_invoke.template operator()<VS, 32, 0, 0>({ barrier_event });
        if (is_recording) {
            local_event = invoke_barrier(node_comm, q, { local_event }, use_cpu_barrier);
        }
        return local_event;
    };

    // run the three phases of collective as separate kernels.
    // when cpu barrier is enabled we cannot use single gpu kernel
    // since control has to go to cpu and perform the barrier.
    // also when user asks to remove the synchronization within kernel
    // run them as separate kernels since single kernel algorithm
    // will require the threads to synchronize between phases
    if (use_cpu_barrier || !use_kernel_sync || is_recording) {
        if (use_full_vector) {
            constexpr int vec_size = get_num_elements<T, 8>();
            kernel_event = memcpy_gather.template operator()<vec_size>();
        }
        else {
            // for unaligned data use vector size of 1
            kernel_event = memcpy_gather.template operator()<1>();
        }
    }
    // for unaligned data use vector size of 1
    else if (!use_full_vector) {
        // if sycl threads are not more than hw threads we can
        // use single kernel that executes barrier internally
        if (count <= sec_0) {
            // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
            kernel_event = gather_invoke.template operator()<1, 32, 1, 1>(dep_events);
        }
        else {
            kernel_event = memcpy_gather.template operator()<1>();
        }
    }
    else if (count * dsize <= sec_0) {
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = gather_invoke.template operator()<1, 32, 1, 1>(dep_events);
    }
    else if (count * dsize <= sec_1) {
        constexpr int vec_size = get_num_elements<T, 8>();
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = gather_invoke.template operator()<vec_size, 32, 1, 1>(dep_events);
    }
    else if (count * dsize <= sec_2) {
        constexpr int vec_size = get_num_elements<T, 16>();
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = gather_invoke.template operator()<vec_size, 32, 1, 1>(dep_events);
    }
    else if (count * dsize <= sec_3) {
        constexpr int vec_size = get_num_elements<T, 32>();
        // <vec_size, sub_group_size, use_local_barrier, use_global_barrier>
        kernel_event = gather_invoke.template operator()<vec_size, 16, 1, 1>(dep_events);
    }
    else {
        constexpr int vec_size = get_num_elements<T, 8>();
        kernel_event = memcpy_gather.template operator()<vec_size>();
    }

    return ccl::event::create_from_native(kernel_event);
}

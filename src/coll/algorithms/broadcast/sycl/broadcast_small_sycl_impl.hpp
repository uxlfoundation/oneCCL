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

template <typename T, int N>
inline void broadcast_kernel(const void* in, std::array<void*, MAX_NODE_RANKS> out, size_t idx) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        ((T*)out[i])[idx] = ((T*)in)[idx];
    }
}

template <typename T, int N, int vec_size, int use_block>
void inline broadcast(const void* in,
                      std::array<void*, MAX_NODE_RANKS> out,
                      const size_t count,
                      const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    const size_t packed_count = count / vec_size;

    if (use_block && idx < packed_count) {
        broadcast_kernel<AT, N>(in, out, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            broadcast_kernel<T, N>(in, out, new_idx);
        }
    }
}

// NE is the number of ranks in even_comm and
// NP is the number of ranks in pair_comm
template <typename T, int NE, int NP>
ccl::event broadcast_small_impl(const void* send_buf,
                                void* recv_buf,
                                size_t count,
                                ccl::datatype dtype,
                                int root,
                                ccl_comm* comm,
                                ccl_stream* global_stream,
                                const ccl::vector_class<ccl::event>& deps) {
    constexpr int N = NE * NP;
    sycl::queue q = global_stream->get_native_stream();
    bool is_recording = use_recording_path(q);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int comm_size = node_comm->size();
    const int comm_rank = node_comm->rank();

    // use full vector (>= 8 bytes) if buffers and data size are 4 byte aligned
    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, count, dsize);
    const bool use_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    const bool inplace = send_buf == recv_buf;

    auto [local_tmp_buf, remote_ptrs] =
        is_recording ? node_comm->get_all_tmp_bufs_gpu(true) : node_comm->get_all_tmp_bufs(true);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event kernel_event;
    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }
    // VS : vec_size, SGS : sub_group_size
    auto broadcast_invoke = [=, &q]<int VS, int SGS>(std::vector<sycl::event>& sycl_deps) {
        constexpr int use_block = 1;
        constexpr int vec_size = VS, wg_size = SGS, sg_size = SGS;
        const size_t kernel_threads = count / vec_size + count % vec_size;
        const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;

        sycl::event local_event = q.submit([=](sycl::handler& h) {
            h.depends_on(sycl_deps);
            h.parallel_for(
                sycl::nd_range<1>(kernel_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    broadcast<T, N, VS, use_block>(send_buf, remote_ptrs, count, it);
                });
        });
        return local_event;
    };

    // VS : vec_size
    auto broadcast_memcpy = [=, &q]<int VS>(std::vector<sycl::event> sycl_deps) {
        if (comm_rank == root) {
            sycl::event local_event = broadcast_invoke.template operator()<VS, 32>(sycl_deps);
            sycl_deps.clear();
            sycl_deps.push_back(std::move(local_event));
        }
        sycl::event barrier_event = invoke_barrier(node_comm, q, sycl_deps, use_cpu_barrier);
        sycl::event end_event{};
        if (comm_rank != root || !inplace) {
            sycl::event copy_event = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, local_tmp_buf, dsize * count);
            });
            end_event = copy_event;
        }
        else {
            end_event = barrier_event;
        }
        if (is_recording) {
            end_event = invoke_barrier(node_comm, q, { end_event }, use_cpu_barrier);
        }

        return end_event;
    };

    if (use_full_vector) {
        constexpr int vec_size = get_num_elements<T, 8>();
        kernel_event = broadcast_memcpy.template operator()<vec_size>(dep_events);
    }
    else {
        // for unaligned data use vector size of 1
        kernel_event = broadcast_memcpy.template operator()<1>(dep_events);
    }
    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(
            &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }
    return ccl::event::create_from_native(kernel_event);
}

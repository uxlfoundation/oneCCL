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
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_ring.hpp"

#define USE_PARALLEL_FOR 1

template <typename T>
void inline copy2_kernel(void* dst_ptr1, void* dst_ptr2, const void* src_ptr, size_t idx) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__) && defined(CCL_SYCL_ENABLE_ARCB)
    message_t v;
    if (sizeof(T) == 16) {
        lscLoad<16, CacheCtrl::L1UC_L3C>(v, ((message_t*)src_ptr) + idx);
        lscStore<16, CacheCtrl::L1WB_L3WB>(((message_t*)dst_ptr1) + idx, v);
        lscStore<16, CacheCtrl::L1WB_L3WB>(((message_t*)dst_ptr2) + idx, v);
    }
    else {
        ((T*)dst_ptr1)[idx] = ((T*)src_ptr)[idx];
        ((T*)dst_ptr2)[idx] = ((T*)src_ptr)[idx];
    }
#else
    ((T*)dst_ptr1)[idx] = ((T*)src_ptr)[idx];
    ((T*)dst_ptr2)[idx] = ((T*)src_ptr)[idx];
#endif
}

template <typename T, int vec_size>
void inline copy2(void* dst_ptr1,
                  void* dst_ptr2,
                  const void* src_ptr,
                  const size_t count,
                  const size_t offset,
                  const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id() + offset;
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        copy2_kernel<AT>(dst_ptr1, dst_ptr2, src_ptr, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            copy2_kernel<T>(dst_ptr1, dst_ptr2, src_ptr, new_idx);
        }
    }
}

// load and store to two destinations
template <typename T, int vec_size>
#if !USE_PARALLEL_FOR
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
#endif
void inline copy_2dst_wrap(void* dst_ptr1, void* dst_ptr2, const void* src_ptr, const size_t count) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const size_t req_threads = count / vec_size + count % vec_size;

#pragma unroll
    for (size_t offset = 0; offset < req_threads; offset += num_threads) {
        copy2<T, vec_size>(dst_ptr1, dst_ptr2, src_ptr, count, offset, it);
    }
}

template <typename T, int vec_size>
struct Copy2Kernel {
    Copy2Kernel(void* dst_ptr1, void* dst_ptr2, const void* src_ptr, const size_t count, ccl_comm* node_comm)
            : dst_ptr1(dst_ptr1),
              dst_ptr2(dst_ptr2),
              src_ptr(src_ptr),
              count(count),
              flag_data(node_comm->flag_data()) {}

    void operator()(sycl::nd_item<1> it) const {
        copy_2dst_wrap<T, vec_size>(dst_ptr1, dst_ptr2, src_ptr, count);
        p2p_barrier(flag_data, it, false /* use_subgroups */, true /* use_root_sync */);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    void* dst_ptr1;
    void* dst_ptr2;
    const void* src_ptr;
    size_t count;
    ccl_comm_flag_data flag_data;
};

template <typename T>
sycl::event allgatherv_large_su_ring_chunking_no_ipc(sycl::queue& q,
                                                     const void* send_buf,
                                                     size_t send_count,
                                                     void* recv_buf,
                                                     const ccl::vector_class<size_t>& recv_counts,
                                                     const ccl::vector_class<size_t>& offsets,
                                                     ccl::datatype dtype,
                                                     ccl_comm* comm,
                                                     ccl_stream* global_stream,
                                                     bool in_place,
                                                     sycl_ptrs_type& sycl_ptrs,
                                                     int& slot) {
    sycl::event e;
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    int world = node_comm->size();
    int rank = node_comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    size_t send_size = send_count * dsize;

    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    constexpr int use_full_vector = true;
    sycl::device device = q.get_device();
    // Query the maximum work-group size
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 && work_group_size <= max_work_group_size);
    constexpr int vec_size = get_num_elements<T, 16, use_full_vector>();
    const size_t kernel_threads = send_count / vec_size + send_count % vec_size;
    const size_t kernel_size = ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range(kernel_size, work_group_size);

    const bool use_memcpy = ccl::global_data::env().sycl_copy_engine;

    int right = (rank + 1) % world;

    const int pipeline_size = world;
    size_t chunk_size = ccl::global_data::env().sycl_tmp_buf_size / pipeline_size;
    std::array<void*, ARC_MAX_NUM> work_bufs;
    std::array<void*, ARC_MAX_NUM> remote_work_bufs;
    work_bufs[0] = get_tmp_buf(0, comm);
    remote_work_bufs[0] = get_remote_node_tmp_buf(0, comm)[right];
    for (int i = 1; i < pipeline_size; i++) {
        work_bufs[i] = (char*)work_bufs[i - 1] + chunk_size;
        remote_work_bufs[i] = (char*)remote_work_bufs[i - 1] + chunk_size;
    }

    int s = rank;
    int iter = 0;
    // first iteration
    void* src = (void*)send_buf;
    void* remote_dst = remote_work_bufs[slot];
    void* local_dst = (char*)recv_buf + offsets[s];
    if (use_memcpy) {
        if (in_place) {
            q.memcpy(remote_dst, send_buf, send_size);
        }
        else {
            // on BMG, this two memcpy can not overlap
            q.memcpy(local_dst, send_buf, send_size);
            q.memcpy(remote_dst, send_buf, send_size);
        }
    }
    else {
        if (in_place) {
            q.parallel_for(
                nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    copy_kernel_wrap<T, vec_size>(remote_dst, src, send_count);
                });
        }
        else {
            q.parallel_for(
                nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    copy_2dst_wrap<T, vec_size>(local_dst, remote_dst, src, send_count);
                });
        }
    }
    s = (s + world - 1) % world;
    iter++;
    invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);

    while (iter < world - 1) {
        int next_slot = (slot + 1) % pipeline_size;
        src = work_bufs[slot];
        local_dst = (char*)recv_buf + offsets[s];
        remote_dst = remote_work_bufs[next_slot];

        if (use_memcpy) {
            q.memcpy(local_dst, src, send_size);
            q.memcpy(remote_dst, src, send_size);
        }
        else {
            q.parallel_for(
                nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    copy_2dst_wrap<T, vec_size>(local_dst, remote_dst, src, send_count);
                });
        }
        invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);

        s = (s + world - 1) % world;
        iter++;
        slot = next_slot;
    }
    // last
    {
        src = work_bufs[slot];
        local_dst = (char*)recv_buf + offsets[s];
        if (use_memcpy) {
            q.memcpy(local_dst, src, send_size);
        }
        else {
            q.parallel_for(
                nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    copy_kernel_wrap<T, vec_size>(local_dst, src, send_count);
                });
        }
        slot = (slot + 1) % pipeline_size;
    }

    return e;
}

// CE only
template <typename T>
sycl::event allgatherv_large_su_ring_chunking_single_kernel(sycl::queue& q,
                                                            const void* send_buf,
                                                            size_t send_count,
                                                            void* recv_buf,
                                                            const ccl::vector_class<size_t>& recv_counts,
                                                            const ccl::vector_class<size_t>& offsets,
                                                            ccl::datatype dtype,
                                                            ccl_comm* comm,
                                                            ccl_stream* global_stream,
                                                            bool in_place,
                                                            sycl_ptrs_type& sycl_ptrs) {
    sycl::event e;
    int world = comm->size();
    int rank = comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    size_t send_size = send_count * dsize;

    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    constexpr int use_full_vector = true;
    constexpr int vec_size = get_num_elements<T, 16, use_full_vector>();

    sycl::device device = q.get_device();
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 && work_group_size <= max_work_group_size);
    const size_t sg_size = 16;

    const size_t max_threads = ccl::global_data::get().ze_data->devices[0].total_threads * 4;
    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    if (!num_threads)
        num_threads = max_threads;
    num_threads = std::min(num_threads, max_threads);
    size_t kernel_threads = send_count / vec_size + send_count % vec_size;
    kernel_threads = std::min(kernel_threads, num_threads);
    const size_t kernel_size = ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range(kernel_size, work_group_size);

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    constexpr bool use_root_sync = true;
    bool is_recording = use_recording_path(q);

    ccl_comm_flag_data flag_data = node_comm->flag_data();

    int right = (rank + 1) % world;
    T* peer_recv = (T*)(sycl_ptrs.node_ptrs_wr[right]);
    int s = rank;
    int iter = 0;
    while (iter < world - 1) {
        char* src = (char*)recv_buf + offsets[s];
        char* dst = (char*)peer_recv + offsets[s];

        if (in_place || iter > 0) {
            CopyKernel<T, vec_size> cp_kernel(dst, src, send_count, node_comm.get());
            q.parallel_for(nd_range, cp_kernel);
        }
        else {
            // for out-of-place and the first iteration, copy from send buf
            // to both recv_buf and right neighbor's recv_buf
            void* dst2 = (char*)recv_buf + offsets[rank];
            const void* src = send_buf;
            Copy2Kernel<T, vec_size> cp2_kernel(dst, dst2, src, send_count, node_comm.get());
            q.parallel_for(nd_range, cp2_kernel);
        }

        s = (s + world - 1) % world;
        iter++;
    }

    return e;
}

template <typename T>
sycl::event allgatherv_large_su_ring_chunking(sycl::queue& q,
                                              const void* send_buf,
                                              size_t send_count,
                                              void* recv_buf,
                                              const ccl::vector_class<size_t>& recv_counts,
                                              const ccl::vector_class<size_t>& offsets,
                                              ccl::datatype dtype,
                                              ccl_comm* comm,
                                              ccl_stream* global_stream,
                                              bool in_place,
                                              sycl_ptrs_type& sycl_ptrs) {
    sycl::event e;
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    int world = node_comm->size();
    int rank = node_comm->rank();

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    size_t send_size = send_count * dsize;

    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    constexpr int use_full_vector = true;
    sycl::device device = q.get_device();
    // Query the maximum work-group size
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 && work_group_size <= max_work_group_size);
    constexpr int vec_size = get_num_elements<T, 16, use_full_vector>();
    const size_t kernel_threads = send_count / vec_size + send_count % vec_size;
    const size_t kernel_size = ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range(kernel_size, work_group_size);

    bool is_recording = use_recording_path(q);
    const bool use_memcpy = ccl::global_data::env().sycl_copy_engine;

    int right = (rank + 1) % world;
    T* peer_recv = (T*)(sycl_ptrs.node_ptrs_wr[right]);
    int s = rank;
    int iter = 0;
    while (iter < world - 1) {
        char* src = (char*)recv_buf + offsets[s];
        char* dst = (char*)peer_recv + offsets[s];

        if (use_memcpy) {
            if (in_place || iter > 0) {
                q.memcpy(dst, src, send_size);
            }
            else {
                // on BMG, this two memcpy can not overlap
                q.memcpy((char*)recv_buf + offsets[rank], send_buf, send_size);
                q.memcpy(dst, send_buf, send_size);
            }
        }
        else {
            if (in_place || iter > 0) {
                q.parallel_for(
                    nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        copy_kernel_wrap<T, vec_size>(dst, src, send_count);
                    });
            }
            else {
                // for out-of-place, in the first iteration, copy from send buf
                // to both recv_buf and right neighbor's recv_buf
                void* dst2 = (char*)recv_buf + offsets[rank];
                const void* src = send_buf;
                q.parallel_for(
                    nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        copy_2dst_wrap<T, vec_size>(dst, dst2, src, send_count);
                    });
            }
        }
        invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);

        s = (s + world - 1) % world;
        iter++;
    }

    return e;
}

template <typename T>
sycl::event allgatherv_large_su_ring(sycl::queue& q,
                                     const void* send_buf,
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
    LOG_DEBUG("allgatherv large ring kernel buffer send_count:", send_count);
    sycl::event e;
    int world = comm->size();
    int rank = comm->rank();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    int node_size = node_comm->size();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();

    bool in_place = ccl::is_allgatherv_inplace(
        send_buf, send_count, recv_buf, recv_counts.data(), offsets.data(), dsize, rank, world);
    // when use_tmp is true, large temp buffer is used
    size_t chunk_size = use_tmp ? ccl::global_data::env().sycl_tmp_buf_size / world
                                : ccl::global_data::env().sycl_allgatherv_chunking_threshold;
    size_t max_pack_count;
    if (chunk_size == 0 || send_count * ccl_dtype.size() <= chunk_size) {
        max_pack_count = send_count;
    }
    else {
        max_pack_count = chunk_size;
        int typesize = std::max(4, (int)ccl_dtype.size());
        max_pack_count = max_pack_count / typesize * typesize;
        max_pack_count = max_pack_count / dsize;
        CCL_ASSERT(max_pack_count > 0);
    }

    int nchunks = (send_count + max_pack_count - 1) / max_pack_count;

    const bool use_single_kernel = ccl::global_data::env().sycl_simple_single_kernel;
    const bool use_memcpy = ccl::global_data::env().sycl_copy_engine;

    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    size_t send_offset = 0;
    int slot = 0;
    for (int iter = 0; iter < nchunks; iter++) {
        size_t pack_count = (iter < nchunks - 1) ? max_pack_count : send_count - send_offset;
        std::vector<size_t> scaleup_counts(node_size, pack_count);
        std::vector<size_t> scaleup_offsets(node_size);
        for (int r = 0; r < node_size; r++) {
            scaleup_offsets[r] = (offsets.empty() ? r * send_count * dsize : offsets[r]) + send_offset * dsize;
        }
        if (use_tmp) {
            e = allgatherv_large_su_ring_chunking_no_ipc<T>(q,
                                                            (T*)send_buf + send_offset,
                                                            pack_count,
                                                            recv_buf,
                                                            scaleup_counts,
                                                            scaleup_offsets,
                                                            dtype,
                                                            node_comm.get(),
                                                            global_stream,
                                                            in_place,
                                                            sycl_ptrs,
                                                            slot);
        }
        else if (use_single_kernel && !use_memcpy) {
            e = allgatherv_large_su_ring_chunking_single_kernel<T>(q,
                                                                   (T*)send_buf + send_offset,
                                                                   pack_count,
                                                                   recv_buf,
                                                                   scaleup_counts,
                                                                   scaleup_offsets,
                                                                   dtype,
                                                                   node_comm.get(),
                                                                   global_stream,
                                                                   in_place,
                                                                   sycl_ptrs);
        }
        else {
            e = allgatherv_large_su_ring_chunking<T>(q,
                                                     (T*)send_buf + send_offset,
                                                     pack_count,
                                                     recv_buf,
                                                     scaleup_counts,
                                                     scaleup_offsets,
                                                     dtype,
                                                     node_comm.get(),
                                                     global_stream,
                                                     in_place,
                                                     sycl_ptrs);
        }
        send_offset += pack_count;
    } // for

    {
        std::vector<sycl::event> evts;
        e = submit_wait_on_events(q, evts);
    }

    return e;
}

template <typename T>
sycl::event allgatherv_large_su_a2a(sycl::queue& q,
                                    const void* send_buf,
                                    size_t send_count,
                                    void* recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    const ccl::vector_class<size_t>& offsets,
                                    ccl::datatype dtype,
                                    ccl_comm* comm,
                                    ccl_stream* global_stream,
                                    sycl_ptrs_type& sycl_ptrs,
                                    const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("allgatherv_large_su_a2a");
    sycl::event kernel_event;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    bool is_recording = use_recording_path(q);

    sycl::event barrier_event1 = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    int rank = comm->rank();
    const int N = comm->size();
    for (int i = 0; i < N; i++) {
        // scatter the ranks and limit the amount to copy
        int peer = (rank + i) % N;
        // limit amount of write due to crash in KMD (read timeout error)
        const size_t max_chunk = 512 * 1024 * 1024;
        size_t left = send_count * dsize;
        size_t offset = 0;
        while (left > 0) {
            size_t chunk = left > max_chunk ? max_chunk : left;
            kernel_event = q.submit([=](sycl::handler& h) {
                h.depends_on(barrier_event1);
                h.memcpy(((char*)sycl_ptrs.node_ptrs_wr[peer] + rank * send_count * dsize) + offset,
                         (char*)send_buf + offset,
                         chunk);
            });
            left -= chunk;
            offset += chunk;
            // skip the barrier for the very last iterations
            if (i < N - 1 || left > 0)
                kernel_event = invoke_barrier(node_comm, q, { kernel_event }, is_cpu_barrier);
        }
    }

    kernel_event = invoke_barrier(node_comm, q, { kernel_event }, is_cpu_barrier);
    return kernel_event;
}

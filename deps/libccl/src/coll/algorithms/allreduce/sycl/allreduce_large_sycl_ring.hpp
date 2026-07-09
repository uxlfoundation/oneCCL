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

#pragma once
#include "oneapi/ccl.hpp"
#include "common/global/global.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/utils/sycl_barrier.hpp"
#include "coll/algorithms/utils/sycl_copy.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_ring.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_large_sycl_ring.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_sycl.hpp"
#include "coll/algorithms/utils/transmit/ring_transmit.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#define USE_PARALLEL_FOR 1

constexpr bool use_root_sync = true;

struct coll_params {
    ccl_comm_flag_data flag_data = { -1, -1 };
    ccl_comm_barrier_data barrier_data = { -1, -1 };
    void *peer_recv;
    //std::array<void *, pipeline_size> work_bufs, remote_work_bufs;
};

template <typename T, int vec_size>
#if !USE_PARALLEL_FOR
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
#endif
void inline allgather_wrap(const void *send_buf,
                           void *recv_buf,
                           size_t count,
                           const coll_params params) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const int comm_rank = params.flag_data.rank();
    const int comm_size = params.flag_data.size();
    const size_t count_per_rank = count / comm_size;

    p2p_barrier(params.flag_data, it, false, use_root_sync);

    int s = comm_rank;
    for (int i = 1; i < comm_size - 1; i++) {
        s = (s - 1 + comm_size) % comm_size;
        const size_t offset_count = count_per_rank * s;
        T *src = (T *)recv_buf + offset_count;
        T *dst = (T *)params.peer_recv + offset_count;

        copy_kernel_wrap<T, vec_size, vec_size>(dst, src, count_per_rank, true);
        // barrier
        p2p_barrier(params.flag_data, it, false, use_root_sync);
    }
}

template <typename T, int vec_size>
struct AllgatherPhaseKernel {
    AllgatherPhaseKernel(const void *send_buf,
                         void *recv_buf,
                         size_t count,
                         const coll_params params)
            : send_buf(send_buf),
              recv_buf(recv_buf),
              count(count),
              params(params) {}

    void operator()(sycl::nd_item<1> pos) const {
        allgather_wrap<T, vec_size>(send_buf, recv_buf, count, params);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    const void *send_buf;
    void *recv_buf;
    size_t count;
    const coll_params params;
};

//constexpr int pipeline_size = ccl_large_tmp_bufs::buf_count;

struct rs_coll_params {
    ccl_comm_flag_data flag_data = { -1, -1 };
    ccl_comm_barrier_data barrier_data = { -1, -1 };
    void *peer_recv;
    size_t chunk_size, num_chunks;
    std::array<void *, pipeline_size> work_bufs, remote_work_bufs;
};

template <typename T, int vec_size>
#if !USE_PARALLEL_FOR
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
#endif
void inline reduce_scatter_wrap(const void *send_buf,
                                void *recv_buf,
                                size_t count,
                                ccl_reduction_data reduction,
                                const rs_coll_params params) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const int comm_rank = params.flag_data.rank();
    const int comm_size = params.flag_data.size();
    int dsize = sizeof(T);
    const size_t count_per_rank = count / comm_size;
    const size_t recv_count = count_per_rank;
    const size_t recv_bytes = recv_count * dsize;
    const size_t rem_chunk_size = recv_bytes % params.chunk_size;
    const size_t last_block_count = count_per_rank + count % comm_size;

    const size_t offset_count = count_per_rank * comm_rank;
    void *recv_buf_rank_offset = (T *)recv_buf + offset_count;
    void *peer_recv_buf_rank_offset = (T *)params.peer_recv + offset_count;
    const size_t last_recv_bytes = last_block_count * dsize;

    for (size_t nc = 0; nc < params.num_chunks; nc++) {
        const size_t chunk_offset = nc * params.chunk_size;
        const size_t data_count =
            ((nc < recv_bytes / params.chunk_size) ? params.chunk_size : rem_chunk_size) / dsize;

        // starting indexes
        int s = (comm_rank - 1 + comm_size) % comm_size;
        int r = (s - 1 + comm_size) % comm_size;

        for (int i = 0; i < comm_size - 1; i++) {
            void *out1 = (i == comm_size - 2) ? (char *)recv_buf_rank_offset + chunk_offset
                                              : params.work_bufs[i % pipeline_size];
            void *in1 = (char *)send_buf + r * recv_bytes + chunk_offset;
            void *in2 = params.work_bufs[i % pipeline_size];
            void *out2 = (i == comm_size - 2) ? (char *)peer_recv_buf_rank_offset + chunk_offset
                                              : params.remote_work_bufs[(i + 1) % pipeline_size];

            // copy to remote
            if (i == 0) {
                void *src = (char *)send_buf + s * recv_bytes + chunk_offset;
                void *dst = params.remote_work_bufs[i % pipeline_size];
                copy_kernel_wrap<T, vec_size, vec_size>(dst, src, data_count, 1);
            }

            // barrier
            p2p_barrier(params.flag_data, it, false, true /* use_root_sync */);
            // reduce and copy to remote
            reduce_copy_kernel_wrap<T, vec_size, vec_size>(out1, in1, in2, out2, data_count, reduction, 1);

            s = r;
            r = (s - 1 + comm_size) % comm_size;
        }
    }
}

template <typename T, int vec_size>
struct ReduceScatterPhaseKernel {
    ReduceScatterPhaseKernel(const void *send_buf,
                             void *recv_buf,
                             size_t count,
                             ccl_reduction_data reduction,
                             const rs_coll_params params)
            : send_buf(send_buf),
              recv_buf(recv_buf),
              count(count),
              reduction(reduction),
              params(params) {}

    void operator()(sycl::nd_item<1> pos) const {
        reduce_scatter_wrap<T, vec_size>(send_buf, recv_buf, count, reduction, params);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    const void *send_buf;
    void *recv_buf;
    size_t count;
    ccl_reduction_data reduction;
    const rs_coll_params params;
};

// CE only, no BLT
template <typename T>
void allreduce_large_su_ring_write_single_kernel(const void *send_buf,
                                                 void *recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 ccl::reduction reduction,
                                                 ccl_comm *comm,
                                                 ccl_stream *global_stream,
                                                 sycl_ptrs_type &sycl_ptrs,
                                                 const ccl::vector_class<ccl::event> &deps) {
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    assert(!is_cpu_barrier);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int N = node_comm->size();
    const int rank = node_comm->rank();
    const int peer_rank = (rank + 1) % N;
    const size_t count_per_rank = count / N;

    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, count_per_rank, dsize);
    constexpr int full_vec_size = get_num_elements<T, 16, true>();
    constexpr int partial_vec_size = get_num_elements<T, 16, false>();
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;

    const size_t last_block_count = count_per_rank + count % N;
    T *peer_send = (T *)(sycl_ptrs.node_ptrs_rd[peer_rank]);
    T *peer_recv = (T *)(sycl_ptrs.node_ptrs_wr[peer_rank]);

    // Query the maximum work-group size
    sycl::device device = q.get_device();
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 && work_group_size <= max_work_group_size);
    const size_t sub_group_size = 16;

    const size_t max_threads =
        ccl::global_data::get().ze_data->devices[0].total_threads * 4; // large GRF mode
    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    if (!num_threads)
        num_threads = max_threads;
    num_threads = std::min(num_threads, max_threads);
    size_t kernel_threads = count_per_rank / vec_size + count_per_rank % vec_size;
    kernel_threads = std::min(kernel_threads, num_threads);
    const size_t kernel_size =
        ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range(kernel_size, work_group_size);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    std::array<void *, pipeline_size> l_remote_work_ptrs, l_work_ptrs;

    const size_t offset_count = count_per_rank * rank;
    const size_t recv_count = count_per_rank;
    const size_t recv_bytes = recv_count * dsize;
    const size_t last_recv_bytes = last_block_count * dsize;

    size_t chunk_size = get_tmp_buf_size_per_rank();
    if (chunk_size > last_recv_bytes)
        chunk_size = last_recv_bytes;
    const size_t rem_chunk_size = recv_bytes % chunk_size;
    const size_t num_chunks = recv_bytes / chunk_size + (rem_chunk_size != 0);

    std::array<void *, pipeline_size> work_bufs;
    std::array<void *, pipeline_size> remote_work_bufs;
    for (int i = 0; i < pipeline_size; i++) {
        work_bufs[i] = get_tmp_buf(i, comm);
        remote_work_bufs[i] = get_remote_node_tmp_buf(i, comm)[peer_rank];
    }

    invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    size_t chunk_data_count = chunk_size / dsize;
    size_t chunk_kernel_threads =
        std::min(chunk_data_count / vec_size + chunk_data_count % vec_size, num_threads);
    const size_t chunk_kernel_size =
        ((chunk_kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range_l(kernel_size, work_group_size);

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    ccl_comm_flag_data flag_data = node_comm->flag_data();

    rs_coll_params rs_params;
    rs_params.flag_data = flag_data;
    rs_params.barrier_data = node_comm->barrier_data();
    rs_params.peer_recv = peer_recv;
    rs_params.chunk_size = chunk_size;
    rs_params.num_chunks = num_chunks;
    rs_params.work_bufs = work_bufs;
    rs_params.remote_work_bufs = remote_work_bufs;
    if (use_full_vector) {
        ReduceScatterPhaseKernel<T, full_vec_size> rs_kernel(
            send_buf, recv_buf, count, reduction_op, rs_params);
        q.parallel_for(nd_range_l, rs_kernel);
    }
    else {
        ReduceScatterPhaseKernel<T, partial_vec_size> rs_kernel(
            send_buf, recv_buf, count, reduction_op, rs_params);
        q.parallel_for(nd_range_l, rs_kernel);
    }

    // allgather
    coll_params params;
    params.flag_data = node_comm->flag_data();
    params.barrier_data = node_comm->barrier_data();
    params.peer_recv = peer_recv;
    if (use_full_vector) {
        AllgatherPhaseKernel<T, full_vec_size> ag_kernel(send_buf, recv_buf, count, params);
        q.parallel_for(nd_range, ag_kernel);
    }
    else {
        AllgatherPhaseKernel<T, partial_vec_size> ag_kernel(send_buf, recv_buf, count, params);
        q.parallel_for(nd_range, ag_kernel);
    }
}

// support memcpy
template <typename T>
void allreduce_large_su_ring_write_multi_kernel(const void *send_buf,
                                                void *recv_buf,
                                                size_t count,
                                                ccl::datatype dtype,
                                                ccl::reduction reduction,
                                                ccl_comm *comm,
                                                ccl_stream *global_stream,
                                                sycl_ptrs_type &sycl_ptrs,
                                                const ccl::vector_class<ccl::event> &deps) {
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    assert(!is_cpu_barrier);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int N = node_comm->size();
    const int rank = node_comm->rank();
    const int peer_rank = (rank + 1) % N;
    const size_t count_per_rank = count / N;

    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, count_per_rank, dsize);
    constexpr int full_vec_size = get_num_elements<T, 16, true>();
    constexpr int partial_vec_size = get_num_elements<T, 16, false>();
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;

    /*
    static auto exe_bndl_cr =
        syclexp::get_kernel_bundle<allgather_wrap<T, vec_size>, sycl::bundle_state::executable>(
            q.get_context());
    static sycl::kernel ag_func_cr =
        exe_bndl_cr.template ext_oneapi_get_kernel<allgather_wrap<T, vec_size>>();
*/

    T *peer_recv = (T *)(sycl_ptrs.node_ptrs_wr[peer_rank]);

    const bool use_memcpy = ccl::global_data::env().sycl_copy_engine;
    size_t num_threads = ccl::global_data::env().sycl_num_threads;

    sycl::device device = q.get_device();
    // Query the maximum work-group size
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 && work_group_size <= max_work_group_size);
    const size_t sub_group_size = 16;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    const int pipeline_size = std::max(N, ccl_large_tmp_bufs::buf_count);

    const size_t my_offset_count = count_per_rank * rank;
    const size_t recv_count = count_per_rank;
    const size_t recv_bytes = recv_count * dsize;
    void *recv_buf_rank_offset = (T *)recv_buf + my_offset_count;
    void *peer_recv_buf_rank_offset = (T *)peer_recv + my_offset_count;
    const size_t chunk_size = ccl::global_data::env().sycl_tmp_buf_size / pipeline_size /
                              sizeof(message_t) * sizeof(message_t);
    const size_t chunk_count = chunk_size / dsize;
    const size_t rem_chunk_count = recv_count % chunk_count;
    const size_t num_chunks = recv_count / chunk_count + (rem_chunk_count != 0);

    std::array<void *, ARC_MAX_NUM> work_bufs;
    std::array<void *, ARC_MAX_NUM> remote_work_bufs;
    work_bufs[0] = get_tmp_buf(0, comm);
    remote_work_bufs[0] = get_remote_node_tmp_buf(0, comm)[peer_rank];
    for (int i = 1; i < pipeline_size; i++) {
        work_bufs[i] = (char *)work_bufs[i - 1] + chunk_size;
        remote_work_bufs[i] = (char *)remote_work_bufs[i - 1] + chunk_size;
    }

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    int slot = 0;
    for (size_t nc = 0; nc < num_chunks; nc++) {
        const size_t chunk_offset = nc * chunk_count * dsize;
        const size_t data_count = (nc < recv_count / chunk_count) ? chunk_count : rem_chunk_count;

        // starting indexes
        int s = (rank + N - 1) % N;
        int r = (s + N - 1) % N;

        for (int i = 0; i < N - 1; i++) {
            int next_slot = (slot + 1) % pipeline_size;
            void *out1 =
                (i == N - 2) ? (char *)recv_buf_rank_offset + chunk_offset : work_bufs[slot];
            void *in1 = (char *)send_buf + r * recv_bytes + chunk_offset;
            void *in2 = work_bufs[slot];
            void *out2 = (i == N - 2) ? (char *)peer_recv_buf_rank_offset + chunk_offset
                                      : remote_work_bufs[next_slot];

            size_t kernel_threads = data_count / vec_size + data_count % vec_size;
            if (num_threads) {
                kernel_threads = std::min(kernel_threads, num_threads);
            }
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
            sycl::nd_range<1> nd_range_l(kernel_size, work_group_size);

            // copy to remote
            if (i == 0) {
                void *src = (char *)send_buf + s * recv_bytes + chunk_offset;
                void *dst = remote_work_bufs[slot];
                if (use_memcpy) {
                    q.memcpy(dst, src, data_count * dsize);
                }
                else {
                    q.parallel_for(
                        nd_range_l, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                            copy_kernel_wrap<T, full_vec_size, partial_vec_size>(dst, src, data_count, use_full_vector);
                        });
                }
            }

            // barrier
            invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);

            // local reduce + remote write
            if (use_memcpy) {
                q.parallel_for(
                    nd_range_l, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        reduce_kernel_wrap<T, full_vec_size, partial_vec_size>(out1, in1, in2, data_count, reduction_op, use_full_vector);
                    });
                q.memcpy(out2, out1, data_count * dsize);
            }
            else {
                if (use_full_vector) {
                    ReduceCopyKernel<T, full_vec_size> rc_kernel(
                        out1, in1, in2, out2, data_count, reduction_op);
                    q.parallel_for(nd_range_l, rc_kernel);
		}
                else {
                    ReduceCopyKernel<T, partial_vec_size> rc_kernel(
                        out1, in1, in2, out2, data_count, reduction_op);
                    q.parallel_for(nd_range_l, rc_kernel);
                }
            }

            s = r;
            r = (s + N - 1) % N;
            slot = next_slot;
        }
    }

    invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);

    // allgather
    size_t kernel_threads = count_per_rank / vec_size + count_per_rank % vec_size;
    if (num_threads) {
        kernel_threads = std::min(kernel_threads, num_threads);
    }
    const size_t kernel_size =
        ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range(kernel_size, work_group_size);

    int s = rank;
    for (int i = 1; i < N - 1; i++) {
        s = (s - 1 + N) % N;
        const size_t offset_count = count_per_rank * s;
        T *src = (T *)recv_buf + offset_count;
        T *dst = peer_recv + offset_count;
        if (use_memcpy) {
            q.memcpy(dst, src, count_per_rank * dsize);
        }
        else {
            q.parallel_for(
                nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    copy_kernel_wrap<T, full_vec_size, partial_vec_size>(dst, src, count_per_rank, use_full_vector);
                });
        }
        invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);
    }
}

// no IPC
template <typename T>
sycl::event allreduce_large_su_ring_write_no_ipc(const void *send_buf,
                                                 void *recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 ccl::reduction reduction,
                                                 ccl_comm *comm,
                                                 ccl_stream *global_stream,
                                                 const ccl::vector_class<ccl::event> &deps) {
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    assert(!is_cpu_barrier);

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int N = node_comm->size();
    const int rank = node_comm->rank();
    const int peer_rank = (rank + 1) % N;
    const size_t count_per_rank = count / N;

    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, count_per_rank, dsize);
    constexpr int full_vec_size = get_num_elements<T, 16, true>();
    constexpr int partial_vec_size = get_num_elements<T, 16, false>();
    int vec_size = use_full_vector ? full_vec_size : partial_vec_size;

    const bool use_memcpy = ccl::global_data::env().sycl_copy_engine;
    size_t num_threads = ccl::global_data::env().sycl_num_threads;

    sycl::device device = q.get_device();
    // Query the maximum work-group size
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 && work_group_size <= max_work_group_size);
    const size_t sub_group_size = 16;

    size_t kernel_threads = count_per_rank / vec_size + count_per_rank % vec_size;
    if (num_threads) {
        kernel_threads = std::min(kernel_threads, num_threads);
    }
    const size_t kernel_size =
        ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range(kernel_size, work_group_size);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    const int pipeline_size = N;
    const size_t chunk_size = ccl::global_data::env().sycl_tmp_buf_size / pipeline_size;

    const size_t my_offset_count = count_per_rank * rank;
    const size_t recv_count = count_per_rank;
    const size_t recv_bytes = recv_count * dsize;
    void *recv_buf_rank_offset = (T *)recv_buf + my_offset_count;
    const size_t rem_chunk_size = recv_bytes % chunk_size;
    const size_t num_chunks = recv_bytes / chunk_size + (rem_chunk_size != 0);

    std::array<void *, ARC_MAX_NUM> work_bufs;
    std::array<void *, ARC_MAX_NUM> remote_work_bufs;
    work_bufs[0] = get_tmp_buf(0, comm);
    remote_work_bufs[0] = get_remote_node_tmp_buf(0, comm)[peer_rank];
    for (int i = 1; i < pipeline_size; i++) {
        work_bufs[i] = (char *)work_bufs[i - 1] + chunk_size;
        remote_work_bufs[i] = (char *)remote_work_bufs[i - 1] + chunk_size;
    }

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    int slot = 0;
    for (size_t nc = 0; nc < num_chunks; nc++) {
        const size_t chunk_offset = nc * chunk_size;
        const size_t data_count =
            ((nc < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;

        // starting indexes
        int s = (rank - 1 + N) % N;
        int r = (s - 1 + N) % N;

        for (int i = 0; i < N - 1; i++) {
            int next_slot = (slot + 1) % pipeline_size;
            void *out1 =
                (i == N - 2) ? (char *)recv_buf_rank_offset + chunk_offset : work_bufs[slot];
            void *in1 = (char *)send_buf + r * recv_bytes + chunk_offset;
            void *in2 = work_bufs[slot];
            void *out2 = remote_work_bufs[next_slot];

            const size_t kernel_threads = data_count / vec_size + data_count % vec_size;
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
            sycl::nd_range<1> nd_range_l(kernel_size, work_group_size);

            // copy to remote
            if (i == 0) {
                void *src = (char *)send_buf + s * recv_bytes + chunk_offset;
                void *dst = remote_work_bufs[slot];
                if (use_memcpy) {
                    q.memcpy(dst, src, data_count * dsize);
                }
                else {
                    q.parallel_for(
                        nd_range_l, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                            copy_kernel_wrap<T, full_vec_size, partial_vec_size>(dst, src, data_count, use_full_vector);
                        });
                }
            }

            // barrier
            invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);

            // local reduce + remote write
            if (use_memcpy) {
                q.parallel_for(
                    nd_range_l, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        reduce_kernel_wrap<T, full_vec_size, partial_vec_size>(out1, in1, in2, data_count, reduction_op, use_full_vector);
                    });
                q.memcpy(out2, out1, data_count * dsize);
            }
            else {
                q.parallel_for(
                    nd_range_l, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        reduce_copy_kernel_wrap<T, full_vec_size, partial_vec_size>(
                            out1, in1, in2, out2, data_count, reduction_op, use_full_vector);
                    });
            }

            s = r;
            r = (s - 1 + N) % N;
            slot = next_slot;
        }

        // allgather
        invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);

        s = rank;
        for (int i = 1; i < N - 1; i++) {
            s = (s - 1 + N) % N;
            int next_slot = (slot + 1) % pipeline_size;
            const size_t offset_count = count_per_rank * s;
            void *src = work_bufs[slot];
            void *local_dst = (char *)recv_buf + s * recv_bytes + chunk_offset;
            void *remote_dst = remote_work_bufs[next_slot];
            if (use_memcpy) {
                q.memcpy(local_dst, src, data_count * dsize);
                q.memcpy(remote_dst, src, data_count * dsize);
            }
            else {
                q.parallel_for(
                    nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        copy_2dst_wrap<T, full_vec_size, partial_vec_size>(local_dst, remote_dst, src, data_count, use_full_vector);
                    });
            }
            invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);
            slot = next_slot;
        }
        // last step
        {
            s = (s - 1 + N) % N;
            void *src = work_bufs[slot];
            void *local_dst = (char *)recv_buf + s * recv_bytes + chunk_offset;
            if (use_memcpy) {
                q.memcpy(local_dst, src, data_count * dsize);
            }
            else {
                q.parallel_for(
                    nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        copy_kernel_wrap<T, full_vec_size, partial_vec_size>(local_dst, src, data_count, use_full_vector);
                    });
            }
        }
    }

    {
        std::vector<sycl::event> evts;
        work_event = submit_wait_on_events(q, evts);
    }

    if (reduction == ccl::reduction::avg) {
        std::vector<sycl::event> evs;
        evs.push_back(work_event);
        work_event = sycl_average(q, recv_buf, count_per_rank * N, N, dtype, evs);
    }

    if (count % N) {
        size_t offset = count_per_rank * N;
        bool done;
        ccl::event e = allreduce_ll<RingTransmit>((char *)send_buf + offset * dsize,
                                                  (char *)recv_buf + offset * dsize,
                                                  count % N,
                                                  dtype,
                                                  reduction,
                                                  comm,
                                                  global_stream,
                                                  done);
        assert(done);
        work_event = e.get_native();
    }

    return work_event;
}

// requires IPC
template <typename T>
sycl::event allreduce_large_su_ring(const void *send_buf,
                                    void *recv_buf,
                                    size_t count,
                                    ccl::datatype dtype,
                                    ccl::reduction reduction,
                                    ccl_comm *comm,
                                    ccl_stream *global_stream,
                                    sycl_ptrs_type &sycl_ptrs,
                                    const ccl::vector_class<ccl::event> &deps) {
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int N = node_comm->size();
    const int rank = node_comm->rank();
    const int peer_rank = (rank + 1) % N;
    const size_t count_per_rank = count / N;

    const bool use_full_vector = can_use_full_vector(send_buf, recv_buf, count_per_rank, dsize);
    constexpr int full_vec_size = get_num_elements<T, 16, true>();
    constexpr int partial_vec_size = get_num_elements<T, 16, false>();
    int vec_size = use_full_vector ? full_vec_size : partial_vec_size;

#if !USE_PARALLEL_FOR
    static auto exe_bndl_cp =
        syclexp::get_kernel_bundle<copy_kernel_wrap<T, full_vec_size, partial_vec_size>, sycl::bundle_state::executable>(
            q.get_context());
    static sycl::kernel k_func_cp =
        exe_bndl_cp.template ext_oneapi_get_kernel<copy_kernel_wrap<T, full_vec_size, partial_vec_size>>();

    static auto exe_bndl_rs =
        syclexp::get_kernel_bundle<reduce_kernel_wrap<T, full_vec_size, partial_vec_size>, sycl::bundle_state::executable>(
            q.get_context());
    static sycl::kernel k_func_rs =
        exe_bndl_rs.template ext_oneapi_get_kernel<reduce_kernel_wrap<T, full_vec_size, partial_vec_size>>();

    static auto exe_bndl_rc =
        syclexp::get_kernel_bundle<reduce_copy_kernel_wrap<T, full_vec_size, partial_vec_size>,
                                   sycl::bundle_state::executable>(q.get_context());
    static sycl::kernel k_func_rc =
        exe_bndl_rc.template ext_oneapi_get_kernel<reduce_copy_kernel_wrap<T, full_vec_size, partial_vec_size>>();
#endif

    T *peer_send = (T *)(sycl_ptrs.node_ptrs_rd[peer_rank]);
    T *peer_recv = (T *)(sycl_ptrs.node_ptrs_wr[peer_rank]);

    const bool use_memcpy = ccl::global_data::env().sycl_copy_engine;
    const bool use_single_kernel = ccl::global_data::env().sycl_simple_single_kernel;
    const bool read_algo = ccl::global_data::env().sycl_allreduce_simple_read;

    // Query the maximum work-group size
    sycl::device device = q.get_device();
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t work_group_size = ccl::global_data::env().sycl_work_group_size;
    assert(work_group_size > 0 && work_group_size <= max_work_group_size);
    const size_t sub_group_size = 16;
    const size_t kernel_threads = count_per_rank / vec_size + count_per_rank % vec_size;
    const size_t kernel_size =
        ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
    sycl::nd_range<1> nd_range(kernel_size, work_group_size);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    if (read_algo) {
        ccl_reduction_data reduction_op = make_reduction_operation(reduction);

        invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

        // reduce_scatter
        int s = (rank + 2) % N;
        for (int i = 0; i < N - 1; i++) {
            const size_t offset_count = count_per_rank * s;
            T *in1 = ((i == 0) ? peer_send : peer_recv) + offset_count;
            T *in2 = (T *)send_buf + offset_count;
            T *out = (T *)recv_buf + offset_count;
#if USE_PARALLEL_FOR
            q.parallel_for(
                nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    reduce_kernel_wrap<T, full_vec_size, partial_vec_size>(out, in1, in2, count_per_rank, reduction_op, use_full_vector);
                });
#else
            syclexp::nd_launch(q, nd_range, k_func_rs, out, in1, in2, count_per_rank);
#endif
            invoke_barrier(node_comm, q, {}, is_cpu_barrier);
            s = (s + 1) % N;
        }

        // allgather
        s = (rank + 1) % N;
        for (int i = 0; i < N - 1; i++) {
            const size_t offset_count = count_per_rank * s;
            T *src = peer_recv + offset_count;
            T *dst = (T *)recv_buf + offset_count;
            if (use_memcpy) {
                q.memcpy(dst, src, count_per_rank * dsize);
            }
            else {
#if USE_PARALLEL_FOR
                q.parallel_for(
                    nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                        copy_kernel_wrap<T, full_vec_size, partial_vec_size>(dst, src, count_per_rank, use_full_vector);
                    });
#else
                syclexp::nd_launch(q, nd_range, k_func_cp, dst, src, count_per_rank);
#endif
            }
            invoke_barrier(node_comm, q, {}, is_cpu_barrier);
            s = (s + 1) % N;
        }
    }
    else {
        if (!use_memcpy && use_single_kernel) {
            allreduce_large_su_ring_write_single_kernel<T>(
                send_buf, recv_buf, count, dtype, reduction, comm, global_stream, sycl_ptrs, deps);
        }
        else {
            allreduce_large_su_ring_write_multi_kernel<T>(
                send_buf, recv_buf, count, dtype, reduction, comm, global_stream, sycl_ptrs, deps);
        }
    }

    {
        std::vector<sycl::event> evts;
        work_event = submit_wait_on_events(q, evts);
    }

    if (reduction == ccl::reduction::avg) {
        std::vector<sycl::event> evs;
        evs.push_back(work_event);
        work_event = sycl_average(q, recv_buf, count_per_rank * N, N, dtype, evs);
    }

    if (count % N) {
        size_t offset = count_per_rank * N;
        bool done;
        ccl::event e = allreduce_ll<RingTransmit>((char *)send_buf + offset * dsize,
                                                  (char *)recv_buf + offset * dsize,
                                                  count % N,
                                                  dtype,
                                                  reduction,
                                                  comm,
                                                  global_stream,
                                                  done);
        assert(done);
        work_event = e.get_native();
    }

    return work_event;
}

#ifndef CCL_NO_EXTERN_TEMPLATE_INST
// Suppress local instantiation in TUs that only call this impl via
// invoke_collective. Instantiations live in the sibling inst/ .cpp files.

#define TEMPLATE_ALLREDUCE_LARGE_SU_RING(DT) \
    template sycl::event allreduce_large_su_ring<DT>(const void *send_buf, \
                                                     void *recv_buf, \
                                                     size_t count, \
                                                     ccl::datatype dtype, \
                                                     ccl::reduction reduction, \
                                                     ccl_comm *comm, \
                                                     ccl_stream *global_stream, \
                                                     sycl_ptrs_type &sycl_ptrs, \
                                                     const ccl::vector_class<ccl::event> &deps)

#define TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(DT) \
    template sycl::event allreduce_large_su_ring_write_no_ipc<DT>( \
        const void *send_buf, \
        void *recv_buf, \
        size_t count, \
        ccl::datatype dtype, \
        ccl::reduction reduction, \
        ccl_comm *comm, \
        ccl_stream *global_stream, \
        const ccl::vector_class<ccl::event> &deps)

extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(short);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(short);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(int8_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(int8_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(uint8_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(uint8_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(int);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(int);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(uint32_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(uint32_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(int64_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(int64_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(uint64_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(uint64_t);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(float);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(float);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(double);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(double);
#ifdef CCL_SYCL_VEC_SUPPORT_FP16
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(sycl::half);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(sycl::half);
#endif
#ifdef CCL_SYCL_VEC_SUPPORT_BF16
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING(sycl::ext::oneapi::bfloat16);
extern TEMPLATE_ALLREDUCE_LARGE_SU_RING_WRITE_NO_IPC(sycl::ext::oneapi::bfloat16);
#endif
#endif // !CCL_NO_EXTERN_TEMPLATE_INST

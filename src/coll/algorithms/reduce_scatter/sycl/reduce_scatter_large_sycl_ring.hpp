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
//#include "coll/algorithms/utils/sycl_ll256.hpp"
#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"

#define RS_USE_PARALLEL_FOR 1

using message_t = sycl::vec<uint32_t, 4>;

template <typename T>
void inline copy_kernel(void *dst_ptr, void *src_ptr, size_t idx) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__) && defined(CCL_SYCL_ENABLE_ARCB)
    message_t v;
    if (sizeof(T) == 16) {
        lscLoad<16, CacheCtrl::L1UC_L3C>(v, ((message_t *)src_ptr) + idx);
        lscStore<16, CacheCtrl::L1WB_L3WB>(((message_t *)dst_ptr) + idx, v);
    }
    else {
        ((T *)dst_ptr)[idx] = ((T *)src_ptr)[idx];
    }
#else
    ((T *)dst_ptr)[idx] = ((T *)src_ptr)[idx];
#endif
}

template <typename T, int vec_size>
void inline copy(void *dst_ptr,
                 void *src_ptr,
                 const size_t count,
                 const size_t offset,
                 const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id() + offset;
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        copy_kernel<AT>(dst_ptr, src_ptr, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            copy_kernel<T>(dst_ptr, src_ptr, new_idx);
        }
    }
}

template <typename T>
inline void reduce_kernel(void *out, void *in1, void *in2, size_t idx) {
    ((T *)out)[idx] = ((T *)in1)[idx] + ((T *)in2)[idx];
}

template <typename T, int vec_size>
void inline reduce(void *out,
                   void *in1,
                   void *in2,
                   const size_t count,
                   const size_t offset,
                   const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id() + offset;
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        reduce_kernel<AT>(out, in1, in2, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            reduce_kernel<T>(out, in1, in2, new_idx);
        }
    }
}

template <typename T>
inline void reduce_copy_kernel(void *out1, void *in1, void *in2, void *out2, size_t idx) {
    reduce_kernel<T>(out1, in1, in2, idx);
    copy_kernel<T>(out2, out1, idx);
}

template <typename T, int vec_size>
void inline reduce_copy(void *out1,
                        void *in1,
                        void *in2,
                        void *out2,
                        const size_t count,
                        const size_t offset,
                        const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id() + offset;
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        reduce_copy_kernel<AT>(out1, in1, in2, out2, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            reduce_copy_kernel<T>(out1, in1, in2, out2, new_idx);
        }
    }
}

template <typename T, int vec_size>
//SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void inline copy_kernel_wrap(void *dst_ptr, void *src_ptr, const size_t count) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const size_t req_threads = count / vec_size + count % vec_size;

#pragma unroll
    for (size_t offset = 0; offset < req_threads; offset += num_threads) {
        copy<T, vec_size>(dst_ptr, src_ptr, count, offset, it);
    }
}

template <typename T, int vec_size>
struct CopyKernel {
    CopyKernel(void *dst_ptr, void *src_ptr, const size_t count, ccl_comm *node_comm)
            : dst_ptr(dst_ptr),
              src_ptr(src_ptr),
              count(count),
              flag_data(node_comm->flag_data()) {}

    void operator()(sycl::nd_item<1> it) const {
        copy_kernel_wrap<T, vec_size>(dst_ptr, src_ptr, count);
        p2p_barrier(flag_data, it, false /* use_subgroups */, true /* use_root_sync */);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    void *dst_ptr;
    void *src_ptr;
    size_t count;
    ccl_comm_flag_data flag_data;
};

template <typename T, int vec_size>
//SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void inline reduce_kernel_wrap(void *out, void *in1, void *in2, const size_t count) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const size_t req_threads = count / vec_size + count % vec_size;

#pragma unroll
    for (size_t offset = 0; offset < req_threads; offset += num_threads) {
        reduce<T, vec_size>(out, in1, in2, count, offset, it);
    }
}

template <typename T, int vec_size>
struct ReduceKernel {
    ReduceKernel(void *out, void *in1, void *in2, const size_t count)
            : out(out),
              in1(in1),
              in2(in2),
              count(count) {}

    void operator()(sycl::nd_item<1> pos) const {
        reduce_kernel_wrap<T, vec_size>(out, in1, in2, count);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16> };
    }

    void *out;
    void *in1;
    void *in2;
    size_t count;
};

template <typename T, int vec_size>
//SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void inline reduce_copy_kernel_wrap(void *out1, void *in1, void *in2, void *out2, const size_t count) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const size_t req_threads = count / vec_size + count % vec_size;

#pragma unroll
    for (size_t offset = 0; offset < req_threads; offset += num_threads) {
        reduce_copy<T, vec_size>(out1, in1, in2, out2, count, offset, it);
    }
}

template <typename T, int vec_size>
struct ReduceCopyKernel {
    ReduceCopyKernel(void *out1, void *in1, void *in2, void *out2, const size_t count)
            : out1(out1),
              in1(in1),
              in2(in2),
              out2(out2),
              count(count) {}

    void operator()(sycl::nd_item<1> pos) const {
        reduce_copy_kernel_wrap<T, vec_size>(out1, in1, in2, out2, count);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16> };
    }

    void *out1;
    void *in1;
    void *in2;
    void *out2;
    size_t count;
};

template <typename T>
sycl::event reduce_scatter_large_su_ring_multi_kernels(const void *send_buf,
                                                       void *recv_buf,
                                                       size_t recv_count,
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

    constexpr int use_full_vector = true;
    constexpr int vec_size = get_num_elements<T, 16, use_full_vector>();

#if !RS_USE_PARALLEL_FOR
    static auto exe_bndl_cp =
        syclexp::get_kernel_bundle<copy_kernel_wrap<T, vec_size>, sycl::bundle_state::executable>(q.get_context());
    static sycl::kernel k_func_cp = exe_bndl_cp.template ext_oneapi_get_kernel<copy_kernel_wrap<T, vec_size>>();

    static auto exe_bndl_rs =
        syclexp::get_kernel_bundle<reduce_kernel_wrap<T, vec_size>, sycl::bundle_state::executable>(
            q.get_context());
    static sycl::kernel k_func_rs = exe_bndl_rs.template ext_oneapi_get_kernel<reduce_kernel_wrap<T, vec_size>>();

    static auto exe_bndl_rc =
        syclexp::get_kernel_bundle<reduce_copy_kernel_wrap<T, vec_size>, sycl::bundle_state::executable>(
            q.get_context());
    static sycl::kernel k_func_rc =
        exe_bndl_rc.template ext_oneapi_get_kernel<reduce_copy_kernel_wrap<T, vec_size>>();
#endif

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int N = node_comm->size();
    const int rank = node_comm->rank();
    const int dest_rank = (rank + 1) % N;

    const int pipeline_size = N;
    const size_t chunk_size = ccl::global_data::env().sycl_tmp_buf_size / pipeline_size;

    const size_t recv_bytes = recv_count * dsize;
    const size_t rem_chunk_size = recv_bytes % chunk_size;
    const size_t num_chunks = recv_bytes / chunk_size + (rem_chunk_size != 0);

    std::array<void *, ARC_MAX_NUM> work_bufs;
    std::array<void *, ARC_MAX_NUM> remote_work_bufs;
    work_bufs[0] = get_tmp_buf(0, comm);
    remote_work_bufs[0] = get_remote_node_tmp_buf(0, comm)[dest_rank];
    for (int i = 1; i < pipeline_size; i++) {
        work_bufs[i] = (char *)work_bufs[i - 1] + chunk_size;
        remote_work_bufs[i] = (char *)remote_work_bufs[i - 1] + chunk_size;
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    const bool use_memcpy = ccl::global_data::env().sycl_copy_engine;
    sycl::device device = q.get_device();
    // Query the maximum work-group size
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t wg_size = ccl::global_data::env().sycl_work_group_size;
    assert(wg_size > 0 && wg_size <= max_work_group_size);
    const size_t sg_size = 16;

    int slot = 0;
    for (size_t nc = 0; nc < num_chunks; nc++) {
        const size_t chunk_offset = nc * chunk_size;
        const size_t data_count = ((nc < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;

        // starting indexes
        int s = (rank - 1 + N) % N;
        int r = (s - 1 + N) % N;

        for (int i = 0; i < N - 1; i++) {
            int next_slot = (slot + 1) % pipeline_size;
            void *out1 = (i == N - 2) ? (char *)recv_buf + chunk_offset : work_bufs[slot];
            void *in1 = (char *)send_buf + r * recv_bytes + chunk_offset;
            void *in2 = work_bufs[slot];
            void *out2 = remote_work_bufs[next_slot];

            const size_t kernel_threads = data_count / vec_size + data_count % vec_size;
            const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
            sycl::nd_range<1> nd_range(kernel_size, wg_size);

            // copy to remote
            if (i == 0) {
                void *src = (char *)send_buf + s * recv_bytes + chunk_offset;
                void *dst = remote_work_bufs[slot];
                if (use_memcpy) {
                    q.memcpy(dst, src, data_count * sizeof(T));
                }
                else {
#if RS_USE_PARALLEL_FOR
                    q.parallel_for(
                        nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                            copy_kernel_wrap<T, vec_size>(dst, src, data_count);
                        });
#else
                    syclexp::nd_launch(q, nd_range, k_func_cp, dst, src, data_count);
#endif
                }
            }

            // barrier
            invoke_p2p_barrier(node_comm, q, {}, is_cpu_barrier);

            // local reduce
            if (i < N - 2) {
                sycl::nd_range<1> nd_range(kernel_size, wg_size);
                if (use_memcpy) {
#if RS_USE_PARALLEL_FOR
                    q.parallel_for(nd_range, [=](sycl::nd_item<1> it) {
                        reduce_kernel_wrap<T, vec_size>(out1, in1, in2, data_count);
                    });
#else
                    syclexp::nd_launch(q, nd_range, k_func_rs, out1, in1, in2, data_count);
#endif
                    q.memcpy(out2, out1, data_count * sizeof(T));
                }
                else {
#if RS_USE_PARALLEL_FOR
                    q.parallel_for(
                        nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                            reduce_copy_kernel_wrap<T, vec_size>(out1, in1, in2, out2, data_count);
                        });
#else
                    syclexp::nd_launch(q, nd_range, k_func_rc, out1, in1, in2, out2, data_count);
#endif
                }
            }
            else {
#if RS_USE_PARALLEL_FOR
                q.parallel_for(nd_range, [=](sycl::nd_item<1> it) {
                    reduce_kernel_wrap<T, vec_size>(out1, in1, in2, data_count);
                });
#else
                syclexp::nd_launch(q, nd_range, k_func_rs, out1, in1, in2, data_count);
#endif
            }

            s = r;
            r = (s - 1 + N) % N;
            slot = next_slot;
        }
    }

    if (reduction == ccl::reduction::avg) {
        std::vector<sycl::event> evs;
        evs.push_back(work_event);
        work_event = sycl_average(q, recv_buf, recv_count, comm->size(), dtype, evs);
    }
    else {
        std::vector<sycl::event> evts;
        work_event = submit_wait_on_events(q, evts);
    }

    return work_event;
}

constexpr int pipeline_size = ccl_large_tmp_bufs::buf_count;

struct reduce_scatter_coll_params {
    ccl_comm_flag_data flag = { -1, -1 };
    ccl_comm_barrier_data barrier_data = { -1, -1 };
    size_t chunk_size;
    std::array<void *, pipeline_size> work_bufs, remote_work_bufs;
};

template <typename T, int vec_size>
//SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void inline reduce_scatter_kernel_wrap(const void *send_buf,
                                       void *recv_buf,
                                       size_t recv_count,
                                       ccl::reduction reduction,
                                       const reduce_scatter_coll_params params) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);

    const int comm_rank = params.flag.rank();
    const int comm_size = params.flag.size();

    const int dsize = sizeof(T);
    const size_t recv_bytes = recv_count * dsize;
    const size_t chunk_size = params.chunk_size;
    const size_t rem_chunk_size = recv_bytes % chunk_size;
    const size_t num_chunks = recv_bytes / chunk_size + (rem_chunk_size != 0);
    std::array<void *, pipeline_size> work_bufs, remote_work_bufs;
    work_bufs = params.work_bufs;
    remote_work_bufs = params.remote_work_bufs;

    // use_root_sync is true
    comm_barrier<true>(params.barrier_data, it, true, true /* gpu_counter_increase */);

    for (size_t nc = 0; nc < num_chunks; nc++) {
        const size_t chunk_offset = nc * chunk_size;
        const size_t count = ((nc < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;
        const size_t req_threads = count / vec_size + count % vec_size;

        // starting indexes
        int s = (comm_rank - 1 + comm_size) % comm_size;
        int r = (s - 1 + comm_size) % comm_size;

        for (int i = 0; i < comm_size - 1; i++) {
            void *src =
                (i == 0) ? (char *)send_buf + s * recv_bytes + chunk_offset : work_bufs[(i - 1) % pipeline_size];
            void *dst = remote_work_bufs[i % pipeline_size];
            void *out1 = (i == comm_size - 2) ? (char *)recv_buf + chunk_offset : work_bufs[i % pipeline_size];
            void *in1 = (char *)send_buf + r * recv_bytes + chunk_offset;
            void *in2 = work_bufs[i % pipeline_size];
            void *out2 = remote_work_bufs[(i + 1) % pipeline_size];

            if (i == 0) {
                void *src = (char *)send_buf + s * recv_bytes + chunk_offset;
                void *dst = remote_work_bufs[i % pipeline_size];
#pragma unroll
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    copy<T, vec_size>(dst, src, count, offset, it);
                }
            }

            p2p_barrier(params.flag, it, false, true);

            if (i < comm_size - 2) {
#pragma unroll
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    reduce<T, vec_size>(out1, in1, in2, count, offset, it);
                    copy<T, vec_size>(out2, out1, count, offset, it);
                }
            }
            else {
#pragma unroll
                for (size_t offset = 0; offset < req_threads; offset += num_threads) {
                    reduce<T, vec_size>(out1, in1, in2, count, offset, it);
                }
            }

            s = r;
            r = (s - 1 + comm_size) % comm_size;
        }
    }
}

template <typename T, int vec_size>
struct ReduceScatterKernel {
    ReduceScatterKernel(const void *send_buf,
                        void *recv_buf,
                        size_t recv_count,
                        ccl::reduction reduction,
                        const reduce_scatter_coll_params params)
            : send_buf(send_buf),
              recv_buf(recv_buf),
              recv_count(recv_count),
              reduction(reduction),
              params(params) {}

    void operator()(sycl::nd_item<1> pos) const {
        reduce_scatter_kernel_wrap<T, vec_size>(send_buf, recv_buf, recv_count, reduction, params);
    }

    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    const void *send_buf;
    void *recv_buf;
    size_t recv_count;
    ccl::reduction reduction;
    const reduce_scatter_coll_params params;
};

template <typename T>
sycl::event reduce_scatter_large_su_ring_single_kernel(const void *send_buf,
                                                       void *recv_buf,
                                                       size_t recv_count,
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

    constexpr int use_full_vector = true;
    constexpr int vec_size = get_num_elements<T, 16, use_full_vector>();

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    const int N = node_comm->size();
    const int rank = node_comm->rank();
    const int dest_rank = (rank + 1) % N;

    std::array<void *, pipeline_size> l_remote_work_ptrs, l_work_ptrs;

    const size_t recv_bytes = recv_count * dsize;
    const size_t chunk_size = get_tmp_buf_size_per_rank();

    std::array<void *, pipeline_size> work_bufs;
    std::array<void *, pipeline_size> remote_work_bufs;
    for (int i = 0; i < pipeline_size; i++) {
        work_bufs[i] = get_tmp_buf(i, comm);
        remote_work_bufs[i] = get_remote_node_tmp_buf(i, comm)[dest_rank];
    }

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    sycl::device device = q.get_device();
    const size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    const size_t wg_size = ccl::global_data::env().sycl_work_group_size;
    assert(wg_size > 0 && wg_size <= max_work_group_size);
    const size_t sg_size = 16;

    constexpr bool use_root_sync = true;
    const size_t max_threads = ccl::global_data::get().ze_data->devices[0].total_threads * 4; // large GRF mode
    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    if (!num_threads)
        num_threads = max_threads;
    num_threads = std::min(num_threads, max_threads);

    const size_t data_count = std::min(chunk_size / dsize, recv_count);
    const size_t kernel_threads_actual = data_count / vec_size + data_count % vec_size;
    const size_t kernel_threads = std::min(kernel_threads_actual, num_threads);
    const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
    sycl::nd_range<1> nd_range(kernel_size, wg_size);

    reduce_scatter_coll_params params;
    params.flag = node_comm->flag_data();
    params.barrier_data = node_comm->barrier_data();
    params.work_bufs = work_bufs;
    params.remote_work_bufs = remote_work_bufs;
    params.chunk_size = chunk_size;

    ReduceScatterKernel<T, vec_size> rs_kernel(send_buf, recv_buf, recv_count, reduction, params);
    q.parallel_for(nd_range, rs_kernel);

    if (reduction == ccl::reduction::avg) {
        std::vector<sycl::event> evs;
        evs.push_back(work_event);
        work_event = sycl_average(q, recv_buf, recv_count, comm->size(), dtype, evs);
    }
    else {
        std::vector<sycl::event> evts;
        work_event = submit_wait_on_events(q, evts);
    }

    return work_event;
}

template <typename T>
sycl::event reduce_scatter_large_su_ring(const void *send_buf,
                                         void *recv_buf,
                                         size_t recv_count,
                                         ccl::datatype dtype,
                                         ccl::reduction reduction,
                                         ccl_comm *comm,
                                         ccl_stream *global_stream,
                                         sycl_ptrs_type &sycl_ptrs,
                                         const ccl::vector_class<ccl::event> &deps) {
    const bool use_memcpy = ccl::global_data::env().sycl_copy_engine;
    const bool use_single_kernel = ccl::global_data::env().sycl_simple_single_kernel;

    if (use_memcpy || !use_single_kernel) {
        return reduce_scatter_large_su_ring_multi_kernels<T>(
            send_buf, recv_buf, recv_count, dtype, reduction, comm, global_stream, sycl_ptrs, deps);
    }
    else {
        return reduce_scatter_large_su_ring_single_kernel<T>(
            send_buf, recv_buf, recv_count, dtype, reduction, comm, global_stream, sycl_ptrs, deps);
    }
}

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
#include "coll/algorithms/utils/sycl_barrier.hpp"
#include "coll/algorithms/utils/sycl_copy.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

template <typename T, int vec_size, int GPUS>
class oneccl_reduce_scatter_large_read {};
template <typename T, bool use_full_vector, int vec_size, int GPUS>
class oneccl_reduce_scatter_large_prologue {};
template <typename T, bool use_full_vector, int vec_size, int GPUS>
class oneccl_reduce_scatter_large_main {};
template <typename T, bool use_full_vector, int vec_size, int GPUS>
class oneccl_reduce_scatter_large_epilogue {};

template <typename T, int vec_size>
sycl::event reduce_scatter_large_read_invoke(std::array<void *, MAX_NODE_RANKS> send_bufs,
                                             void *recv_buf,
                                             size_t count,
                                             const ccl_reduction_data &reduction,
                                             std::shared_ptr<ccl_comm> comm,
                                             sycl::queue &q,
                                             std::vector<sycl::event> deps,
                                             size_t offset = 0) {
    const int node_comm_size = comm->size();
    std::array<void *, MAX_NODE_RANKS> src_ptrs;
    for (int i = 0; i < node_comm_size; i++) {
        src_ptrs[i] = (char *)send_bufs[i] + offset;
    }
    void *recv_ptr = (char *)recv_buf + offset;
    sycl::event work_event = q.submit([=](sycl::handler &h) {
        h.depends_on(deps);
        const int work_group_size = 16;
        const int sub_group_size = 16;
        ccl_kernel_barrier_data dummy_kbd;
        ccl_comm_barrier_data dummy_cbd = comm->barrier_data();
        const size_t kernel_threads = count / vec_size + count % vec_size;
        const size_t kernel_size = ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;
        static_for_each_gpu(
            [&]<int GPUS>() {
                h.parallel_for<oneccl_reduce_scatter_large_read<T, vec_size, GPUS>>(
                    sycl::nd_range<1>(kernel_size, work_group_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sub_group_size)]] {
                        reduce_base<T, GPUS, vec_size, 1, 0, 0>(nullptr,
                                                                recv_ptr,
                                                                nullptr,
                                                                src_ptrs,
                                                                src_ptrs,
                                                                dummy_kbd,
                                                                dummy_cbd,
                                                                reduction,
                                                                count,
                                                                it);
                    });
            },
            node_comm_size);
    });
    return work_event;
}

// find how much offset the pointers have with alignment
// returns the offset if all pointers have same offset or else returns 0
inline size_t get_alignment_offset(std::array<void *, MAX_NODE_RANKS> ptrs, int ranks, int alignment) {
    const size_t align_offset = reinterpret_cast<uintptr_t>(ptrs[0]) % alignment;
    for (int i = 1; i < ranks; i++) {
        if (reinterpret_cast<uintptr_t>(ptrs[i]) % alignment != align_offset) {
            return 0;
        }
    }
    return align_offset;
}

template <typename T, int vec_size>
ccl::event reduce_scatter_large_read_ipc(const void *send_buf,
                                         void *recv_buf,
                                         size_t recv_count,
                                         ccl::datatype dtype,
                                         ccl::reduction reduction,
                                         ccl_comm *comm,
                                         ccl_stream *global_stream,
                                         sycl_ptrs_type &sycl_ptrs,
                                         const ccl::vector_class<ccl::event> &deps) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    const int pair_comm_size = pair_comm->size();
    const int node_comm_size = node_comm->size();

    CCL_THROW_IF_NOT(node_comm_size == pair_comm_size,
                     "SYCL reduce_scatter read algo is only implemented for single GPU case");

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);

    const int rank = node_comm->rank();
    const size_t offset_rank = rank * recv_count * dsize;
    std::array<void *, MAX_NODE_RANKS> src_ptrs;
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event barrier_event, dep_event, work_event;
    for (int idx = 0; idx < pair_comm_size; idx++) {
        if (idx == rank) {
            src_ptrs[idx] = (char *)send_buf + offset_rank;
        }
        else {
            src_ptrs[idx] = (char *)sycl_ptrs.mdfi_ptr_rd + offset_rank;
        }
    }

    // currently peeling is supported if all ranks has same alignment offset
    const size_t align_offset =
        get_alignment_offset(src_ptrs, pair_comm_size, ccl::global_data::env().sycl_kernels_line_size);
    size_t offset_byte = 0, offset_count = 0;
    if (align_offset != 0) {
        offset_byte = ccl::utils::get_aligned_offset_byte(
            src_ptrs[0], recv_count * dsize, ccl::global_data::env().sycl_kernels_line_size);
        offset_count = offset_byte / dsize;
    }

    barrier_event = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    // perform kernel on the peeled first unaligned small chunk
    if (offset_byte != 0) {
        dep_event = reduce_scatter_large_read_invoke<T, 1>(
            src_ptrs, recv_buf, offset_count, reduction_op, node_comm, q, { barrier_event });
    }
    else {
        dep_event = barrier_event;
    }

    // perform kernel on the aligned large chunk
    work_event = reduce_scatter_large_read_invoke<T, vec_size>(
        src_ptrs, recv_buf, recv_count - offset_count, reduction_op, node_comm, q, { dep_event }, offset_byte);

    work_event = invoke_barrier(node_comm, q, { work_event }, is_cpu_barrier);

    // do the average reduction separately after sum
    if (reduction == ccl::reduction::avg) {
        // set dependencies
        std::vector<sycl::event> avg_deps_evs;
        avg_deps_evs.push_back(work_event);

        LOG_DEBUG("reduce_scatter_large_read_ipc calculate average on counts: ",
                  recv_count,
                  ", ranks: ",
                  node_comm_size);
        work_event = sycl_average(q, recv_buf, recv_count, node_comm_size, dtype, avg_deps_evs);
    }

    return ccl::event::create_from_native(work_event);
}

//TODO : currently full vector size (8 bytes) is not used for non 4 byte aligned data,
// check whether we can copy data to a tmp_buffer and use 8 byte vectors for reduce kernel.

template <typename T, bool use_full_vector>
ccl::event reduce_scatter_large_impl(const void *send_buf,
                                     void *recv_buf,
                                     size_t recv_count,
                                     ccl::datatype dtype,
                                     ccl::reduction reduction,
                                     ccl_comm *comm,
                                     ccl_stream *global_stream,
                                     sycl_ptrs_type &sycl_ptrs,
                                     const ccl::vector_class<ccl::event> &deps,
                                     const bool use_tmp) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    sycl::queue q = global_stream->get_native_stream();
    sycl::queue q_use = q;

    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();
    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();

    const int pair_comm_size = pair_comm->size();
    const int even_comm_size = even_comm->size();
    const int node_comm_size = node_comm->size();

    constexpr int pipeline_size = 2;

    const bool is_multi_tile = pair_comm_size > 1;
    const bool is_multi_gpu = even_comm_size > 1;
    const bool is_single_gpu = even_comm_size == 1;

    const bool is_tmp_used = use_tmp;

    if (is_single_gpu && !is_tmp_used) {
        constexpr int vec_size = get_num_elements<T, 8, use_full_vector>();
        return reduce_scatter_large_read_ipc<T, vec_size>(
            send_buf, recv_buf, recv_count, dtype, reduction, comm, global_stream, sycl_ptrs, deps);
    }

    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(&ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);
    const bool reduce_has_pre_operation = ccl_reduction_type_storage::is_custom(reduction);

    std::array<void *, MAX_GPUS> l_mdfi_send_ptrs, l_xelink_work_ptrs, l_send_ptrs;
    std::array<void *, MAX_GPUS> l_cp_src_ptrs, l_cp_dst_ptrs, l_cp_src_ptrs_next, l_cp_dst_ptrs_next;
    std::array<void *, MAX_NODE_RANKS> l_work_ptrs, l_work_ptrs_prev;
    void *l_recv_ptr = recv_buf, *l_recv_ptr_prev = recv_buf;

    const size_t recv_bytes = recv_count * dsize;
    const size_t chunk_size = get_tmp_buf_size_per_rank();
    const size_t rem_chunk_size = recv_bytes % chunk_size;
    const size_t num_chunks = recv_bytes / chunk_size + (rem_chunk_size != 0);

    // 0 index is used for tmp work buffer and
    // 1 index is used to copy input data
    void *work_buf = get_tmp_buf(0, comm);
    std::array<void *, MAX_GPUS> work_bufs[pipeline_size];
    std::array<void *, MAX_GPUS> xelink_work_bufs[pipeline_size];
    xelink_work_bufs[0] = get_remote_even_tmp_buf(0, comm);
    const size_t pipeline_offset = chunk_size * even_comm_size;
    for (int i = 0; i < even_comm_size; i++) {
        work_bufs[0][i] = (char *)work_buf + chunk_size * i;
        work_bufs[1][i] = (char *)(work_bufs[0][i]) + pipeline_offset;
        xelink_work_bufs[0][i] = (char *)(xelink_work_bufs[0][i]) + chunk_size * even_comm->rank();
        xelink_work_bufs[1][i] = (char *)(xelink_work_bufs[0][i]) + pipeline_offset;
    }

    void *tmp_bufs[pipeline_size];
    tmp_bufs[0] = get_tmp_buf(1, comm);
    tmp_bufs[1] = (char *)(tmp_bufs[0]) + pipeline_offset;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    sycl::event work_event;

    for (size_t nc = 0; nc < num_chunks; nc++) {
        const int pipeline_index = nc % pipeline_size;
        const int pipeline_index_next = (nc + 1) % pipeline_size;

        const size_t chunk_offset = nc * chunk_size;
        const size_t data_count = ((nc < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;
        const size_t data_count_next = ((nc + 1 < recv_bytes / chunk_size) ? chunk_size : rem_chunk_size) / dsize;
        const size_t data_count_prev = chunk_size / dsize;

        for (int i = 0; i < even_comm_size; i++) {
            int global_rank = comm->is_multi_thread_instance() ? i * pair_comm->size() + pair_comm->rank()
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

            l_cp_src_ptrs[i] = (char *)send_buf + offset_neigh;
            l_cp_dst_ptrs[i] = (char *)tmp_bufs[pipeline_index] + offset_tmp;

            l_cp_src_ptrs_next[i] = (char *)l_cp_src_ptrs[i] + chunk_size;
            l_cp_dst_ptrs_next[i] = (char *)tmp_bufs[pipeline_index_next] + offset_tmp;

            l_mdfi_send_ptrs[i] = (char *)sycl_ptrs.mdfi_ptr_rd + (is_tmp_used ? mdfi_offset_tmp : offset);
            l_send_ptrs[i] = (char *)send_buf + offset;
            // for single gpu, we dont need to write to a tmp buffer and reduce,
            // instead we can directly write to the recv buffer
            l_xelink_work_ptrs[i] =
                is_multi_gpu ? (char *)xelink_work_bufs[pipeline_index][i] : (char *)recv_buf + chunk_offset;

            l_work_ptrs[i] = (char *)work_bufs[pipeline_index][i];
        }
        l_recv_ptr = (char *)recv_buf + chunk_offset;

        // for 2 byte types with odd count or non 4 byte aligned data, use 4 byte vectors instead of 8 bytes
        constexpr int vec_size = get_num_elements<T, 8, use_full_vector>();
        const size_t work_group_size = 16;
        const size_t sub_group_size = 16;

        // pipeline prologue
        // this data copy can also be done from the main kernel as first step with a guard of nc == 0
        bool is_deps_added = false;
        if (is_tmp_used && nc == 0 && is_multi_tile) {
            constexpr int vec_size_cp = use_full_vector ? vec_size : 1;
            const size_t work_group_size_cp = use_full_vector ? work_group_size : 32;
            const size_t sub_group_size_cp = use_full_vector ? sub_group_size : 32;

            is_deps_added = true;
            work_event = q_use.submit([=](sycl::handler &h) {
                const size_t kernel_threads = data_count / vec_size_cp + data_count % vec_size_cp;
                const size_t kernel_size =
                    ((kernel_threads + work_group_size_cp - 1) / work_group_size_cp) * work_group_size_cp;
                h.depends_on(dep_events);
                static_for_each_gpu(
                    [&]<int GPUS>() {
                        h.parallel_for<
                            oneccl_reduce_scatter_large_prologue<T, use_full_vector, vec_size_cp, GPUS>>(
                            sycl::nd_range<1>(kernel_size, work_group_size_cp),
                            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sub_group_size_cp)]] {
                                // copy first chunk from send buf to tmp buf
                                if (!reduce_has_pre_operation) {
                                    copy_data<T, GPUS, vec_size_cp>(l_cp_dst_ptrs, l_cp_src_ptrs, data_count, it);
                                }
                                else {
                                    // apply user defined pre-operation on each copied data element
                                    copy_and_modify_data<T, vec_size_cp>(l_cp_dst_ptrs,
                                                                         l_cp_src_ptrs,
                                                                         even_comm_size,
                                                                         data_count,
                                                                         reduction_op,
                                                                         it);
                                }
                            });
                    },
                    even_comm_size);
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
            pthread_barrier_wait(&ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        }

        work_event = q_use.submit([=](sycl::handler &h) {
            const size_t kernel_threads_curr = data_count / vec_size + data_count % vec_size;
            const size_t kernel_threads_prev =
                nc > 0 ? data_count_prev / vec_size + data_count_prev % vec_size : 0;
            const size_t kernel_threads_next = is_tmp_used && nc < num_chunks - 1 && is_multi_tile
                                                   ? data_count_next / vec_size + data_count_next % vec_size
                                                   : 0;
            const size_t kernel_threads =
                std::max({ kernel_threads_curr, kernel_threads_prev, kernel_threads_next });
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

            h.depends_on(work_event);

            ccl_kernel_barrier_data dummy_kbd;
            ccl_comm_barrier_data dummy_cbd = node_comm->barrier_data();

            static_for_each_gpu(
                [&]<int GPUS>() {
                    h.parallel_for<oneccl_reduce_scatter_large_main<T, use_full_vector, vec_size, GPUS>>(
                        sycl::nd_range<1>(kernel_size, work_group_size),
                        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sub_group_size)]] {
                            read_reduce_write<T, GPUS, vec_size>(l_mdfi_send_ptrs,
                                                                 l_send_ptrs,
                                                                 l_xelink_work_ptrs,
                                                                 reduction_op,
                                                                 is_multi_tile,
                                                                 data_count,
                                                                 it);
                            if (nc > 0 && is_multi_gpu) {
                                reduce_base<T, GPUS, vec_size, 1, 0, 0>(nullptr,
                                                                        l_recv_ptr_prev,
                                                                        nullptr,
                                                                        l_work_ptrs_prev,
                                                                        l_work_ptrs_prev,
                                                                        dummy_kbd,
                                                                        dummy_cbd,
                                                                        reduction_op,
                                                                        data_count_prev,
                                                                        it);
                            }
                            if (is_tmp_used && nc < num_chunks - 1 && is_multi_tile) {
                                // copy next chunk from send buf to tmp buf
                                if (!reduce_has_pre_operation) {
                                    copy_data<T, GPUS, vec_size>(
                                        l_cp_dst_ptrs_next, l_cp_src_ptrs_next, data_count_next, it);
                                }
                                else {
                                    // apply user defined pre-operation on each copied data element
                                    copy_and_modify_data<T, vec_size>(l_cp_dst_ptrs_next,
                                                                      l_cp_src_ptrs_next,
                                                                      even_comm_size,
                                                                      data_count_next,
                                                                      reduction_op,
                                                                      it);
                                }
                            }
                        });
                },
                even_comm_size);
        });

        // save prev pointers to be used in next iteration
        for (int i = 0; i < even_comm_size; i++) {
            l_work_ptrs_prev[i] = l_work_ptrs[i];
        }
        l_recv_ptr_prev = l_recv_ptr;

        // pipeline epilogue
        // this reduction can also be done from the main kernel as last step with a guard of nc == num_chunks - 1
        if (nc == num_chunks - 1 && is_multi_gpu) {
            work_event = invoke_barrier(node_comm, q_use, { work_event }, is_cpu_barrier);

            const size_t kernel_threads = data_count / vec_size + data_count % vec_size;
            const size_t kernel_size =
                ((kernel_threads + work_group_size - 1) / work_group_size) * work_group_size;

            ccl_kernel_barrier_data dummy_kbd;
            ccl_comm_barrier_data dummy_cbd = node_comm->barrier_data();

            work_event = q_use.submit([=](sycl::handler &h) {
                h.depends_on(work_event);
                static_for_each_gpu(
                    [&]<int GPUS>() {
                        h.parallel_for<oneccl_reduce_scatter_large_epilogue<T, use_full_vector, vec_size, GPUS>>(
                            sycl::nd_range<1>(kernel_size, work_group_size),
                            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sub_group_size)]] {
                                reduce_base<T, GPUS, vec_size, 1, 0, 0>(nullptr,
                                                                        l_recv_ptr,
                                                                        nullptr,
                                                                        l_work_ptrs,
                                                                        l_work_ptrs,
                                                                        dummy_kbd,
                                                                        dummy_cbd,
                                                                        reduction_op,
                                                                        data_count,
                                                                        it);
                            });
                    },
                    even_comm_size);
            });
        }
    }

    if (!is_multi_gpu) {
        work_event = invoke_barrier(node_comm, q_use, { work_event }, is_cpu_barrier);
    }

    // do the average reduction separately after sum
    if (reduction == ccl::reduction::avg) {
        // set dependencies
        std::vector<sycl::event> avg_deps_evs;
        avg_deps_evs.push_back(work_event);

        LOG_DEBUG("reduce_scatter_large calculate average on counts: ", recv_count, ", ranks: ", node_comm_size);
        work_event = sycl_average(q, recv_buf, recv_count, node_comm_size, dtype, avg_deps_evs);
    }

    if (comm->is_multi_thread_instance() == true) {
        pthread_barrier_wait(&ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
    }

    return ccl::event::create_from_native(work_event);
}

#ifndef CCL_NO_EXTERN_TEMPLATE_INST
// Suppress local instantiation in TUs that only call this impl via
// invoke_collective. Instantiations live in the sibling inst/ .cpp files.

#define TEMPLATE_REDUCE_SCATTER_LARGE(DT, B) \
    template ccl::event reduce_scatter_large_impl<DT, B>(const void *send_buf, \
                                                         void *recv_buf, \
                                                         size_t recv_count, \
                                                         ccl::datatype dtype, \
                                                         ccl::reduction reduction, \
                                                         ccl_comm *comm, \
                                                         ccl_stream *global_stream, \
                                                         sycl_ptrs_type &sycl_ptrs, \
                                                         const ccl::vector_class<ccl::event> &deps, \
                                                         const bool use_tmp)

extern TEMPLATE_REDUCE_SCATTER_LARGE(short, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(short, false);
extern TEMPLATE_REDUCE_SCATTER_LARGE(int8_t, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(int8_t, false);
extern TEMPLATE_REDUCE_SCATTER_LARGE(uint8_t, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(uint8_t, false);
extern TEMPLATE_REDUCE_SCATTER_LARGE(int, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(int, false);
extern TEMPLATE_REDUCE_SCATTER_LARGE(uint32_t, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(uint32_t, false);
extern TEMPLATE_REDUCE_SCATTER_LARGE(int64_t, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(int64_t, false);
extern TEMPLATE_REDUCE_SCATTER_LARGE(uint64_t, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(uint64_t, false);
extern TEMPLATE_REDUCE_SCATTER_LARGE(float, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(float, false);
extern TEMPLATE_REDUCE_SCATTER_LARGE(double, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(double, false);
#ifdef CCL_SYCL_VEC_SUPPORT_FP16
extern TEMPLATE_REDUCE_SCATTER_LARGE(sycl::half, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(sycl::half, false);
#endif
#ifdef CCL_SYCL_VEC_SUPPORT_BF16
extern TEMPLATE_REDUCE_SCATTER_LARGE(sycl::ext::oneapi::bfloat16, true);
extern TEMPLATE_REDUCE_SCATTER_LARGE(sycl::ext::oneapi::bfloat16, false);
#endif
#endif // !CCL_NO_EXTERN_TEMPLATE_INST

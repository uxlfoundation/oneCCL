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
#include <cstdint>
#include <algorithm>
#include <mutex>
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_ring.hpp"
#include "coll/algorithms/alltoallv/sycl/alltoallv_sycl.hpp"

template <typename DataType, size_t vec_size>
class oneccl_alltoallv_write_aligned {};
template <typename DataType, size_t vec_size>
class oneccl_alltoallv_read_aligned {};
template <typename DataType, size_t vec_size, bool is_multi_thread>
class oneccl_alltoallv_write_peel_front {};
template <typename DataType, size_t vec_size, bool is_multi_thread>
class oneccl_alltoallv_write_aligned_main {};
template <typename DataType, size_t vec_size, bool is_multi_thread>
class oneccl_alltoallv_write_peel_back {};
template <typename DataType, size_t vec_size, bool is_multi_thread>
class oneccl_alltoallv_read_peel_front {};
template <typename DataType, size_t vec_size, bool is_multi_thread>
class oneccl_alltoallv_read_aligned_main {};
template <typename DataType, size_t vec_size, bool is_multi_thread>
class oneccl_alltoallv_read_peel_back {};

static size_t get_alignment_bytes(size_t min_alignment_bytes) {
    size_t alignment_bytes = ccl::global_data::env().kernel_mem_align;

    // keep track of the 'state' via once_flag
    static std::once_flag min_alignment_warning_flag;
    if (alignment_bytes < min_alignment_bytes) {
        std::call_once(min_alignment_warning_flag, [&]() {
            LOG_WARN("requested alignment [",
                     alignment_bytes,
                     "] bytes is lower than minimal alignment required by the selected algorithm [",
                     min_alignment_bytes,
                     "]; further execution might cause alignment errors or crashes; "
                     "this warning will not be printed again");
        });
    }

    static std::once_flag mvec_alignment_warning_flag;
    if (alignment_bytes % min_alignment_bytes != 0) {
        std::call_once(mvec_alignment_warning_flag, [&]() {
            LOG_WARN("requested alignment [",
                     alignment_bytes,
                     "] bytes is not a multiple of alignment required by the selected algorithm [",
                     min_alignment_bytes,
                     "]; further execution might cause alignment errors or crashes; "
                     "this warning will not be printed again");
        });
    }

    static std::once_flag suggested_alignment_warning_flag;
    constexpr size_t validated_perf_alignment = 128;
    if (alignment_bytes % validated_perf_alignment != 0) {
        std::call_once(suggested_alignment_warning_flag, [&]() {
            LOG_WARN("requested alignment [",
                     alignment_bytes,
                     "] bytes is not a multiple of the suggested alignment [",
                     validated_perf_alignment,
                     "]; perf might be suboptimal; "
                     "this warning will not be printed again");
        });
    }

    return alignment_bytes;
}

template <typename DataType>
void calculate_peel_counts(const std::array<DataType *, MAX_NODE_RANKS> &buf_to_peel,
                           size_t *peel_front_count,
                           size_t alignment_bytes,
                           const std::vector<size_t> &counts,
                           const size_t comm_size) {
    CCL_THROW_IF_NOT(alignment_bytes != 0, "alignment cannot be 0");
    CCL_THROW_IF_NOT((alignment_bytes % sizeof(DataType)) == 0,
                     "alignment must be a multiple of DataType size");

    for (size_t i = 0; i < comm_size; ++i) {
        // calculate aligned pointers
        uintptr_t ptr_val = reinterpret_cast<uintptr_t>(buf_to_peel[i]);
        // the first vector is always peeled
        // so that all buffers have the same aligned loop iterations
        // this simplifies the kernel and gives good performance
        DataType *ptr_aligned =
            reinterpret_cast<DataType *>((ptr_val / alignment_bytes + 1) * alignment_bytes);

        CCL_THROW_IF_NOT(ptr_aligned > buf_to_peel[i],
                         "aligned pointer cannot be lower than the buffer start");

        peel_front_count[i] = ptr_aligned - buf_to_peel[i];

        LOG_TRACE("ptr_aligned: ",
                  ptr_aligned,
                  ", buf_to_peel[i]: ",
                  buf_to_peel[i],
                  ", peel_front_count[i]: ",
                  peel_front_count[i]);

        // handle corner case - less elements to transfer than to peel
        if (peel_front_count[i] > counts[i]) {
            peel_front_count[i] = counts[i];
        }
    }
}

template <typename DataType>
DataType divide_round_up(DataType x, DataType y) {
    return (x + y - 1) / y;
}

template <typename DataType>
bool check_all_aligned(size_t *peel_front_count,
                       size_t max_peel_loop_count,
                       DataType *local_buffer,
                       size_t alignment_bytes,
                       size_t max_count,
                       size_t vec_size_bytes,
                       const std::vector<size_t> &counts,
                       const size_t comm_size) {
    // local buffer needs to meet alignment for sycl_vec
    bool local_buf_vec_aligned = (reinterpret_cast<uintptr_t>(local_buffer) % vec_size_bytes) == 0;
    // end needs to be aligned on vec_size_bytes to produce no remainder
    bool is_count_alignment_mul = (max_count * sizeof(DataType) % vec_size_bytes) == 0;
    // initial value, to be modified
    bool are_all_ranks_aligned = is_count_alignment_mul && local_buf_vec_aligned;

    for (uint32_t i = 0; i < comm_size; ++i) {
        LOG_TRACE("peel_front_count[", i, "] = ", peel_front_count[i]);
        // additional precondition check
        CCL_THROW_IF_NOT(peel_front_count[i] <= max_peel_loop_count,
                         "peeling more elements than the upper limit");
        size_t peel_front_bytes = peel_front_count[i] * sizeof(DataType);
        bool is_rank_aligned = (peel_front_bytes == alignment_bytes);
        // each per-rank count must be a whole number of vectors, otherwise the
        // aligned fast-path kernels would overrun this rank's slice
        bool is_rank_count_vec_mul = (counts[i] * sizeof(DataType) % vec_size_bytes) == 0;
        are_all_ranks_aligned = (are_all_ranks_aligned && is_rank_aligned && is_rank_count_vec_mul);
    }
    LOG_TRACE("max_peel_loop_count: ",
              max_peel_loop_count,
              ", local_buf_vec_aligned:",
              local_buf_vec_aligned,
              ", is_count_alignment_mul",
              is_count_alignment_mul,
              ", is_aligned: ",
              are_all_ranks_aligned);

    return are_all_ranks_aligned;
}

// use copy engine and rank scatter
template <size_t vec_size, typename DataType>
std::vector<sycl::event> alltoallv_vec_write_aligned_with_ce(
    sycl::queue &queue,
    const std::shared_ptr<ccl_comm> node_comm,
    const std::array<DataType *, MAX_NODE_RANKS> &send_bufs,
    const std::vector<size_t> &send_counts,
    const std::vector<size_t> &send_offsets,
    std::array<DataType *, MAX_NODE_RANKS> &recv_write_ptrs,
    const size_t rank,
    const size_t comm_size,
    sycl::event &dep) {
    sycl::event e;
    const bool use_barrier = ccl::global_data::env().sycl_alltoall_use_barrier;
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;
    for (uint32_t i = 0; i < comm_size; i++) {
        // scatter the ranks
        uint32_t peer = (rank + i) % comm_size;
        e = queue.submit([=](sycl::handler &h) {
            h.depends_on(dep);
            h.memcpy(recv_write_ptrs[peer],
                     (char *)send_bufs[rank] + send_offsets[peer],
                     send_counts[peer] * sizeof(DataType));
        });
        if (use_barrier)
            invoke_p2p_barrier(node_comm, queue, {}, is_cpu_barrier);
    }
    return { e };
}

template <size_t vec_size, typename DataType>
std::vector<sycl::event> alltoallv_vec_write_aligned_with_eu(
    sycl::queue &queue,
    const std::shared_ptr<ccl_comm> node_comm,
    const std::array<DataType *, MAX_NODE_RANKS> &send_bufs,
    const std::vector<size_t> &send_counts,
    const std::vector<size_t> &send_offsets,
    std::array<DataType *, MAX_NODE_RANKS> &recv_write_ptrs,
    const size_t max_count,
    const size_t rank,
    const size_t comm_size,
    sycl::event &dep) {
    sycl::event e;

    constexpr int use_full_vector = true;
    constexpr int new_vec_size = get_num_elements<DataType, 16, use_full_vector>();
    sycl::device device = queue.get_device();
    const size_t wg_size = ccl::global_data::env().sycl_work_group_size;

    size_t kernel_threads = (max_count + new_vec_size - 1) / new_vec_size;
    size_t num_threads = ccl::global_data::env().sycl_num_threads;
    if (num_threads)
        kernel_threads = std::min(kernel_threads, num_threads);
    const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
    sycl::nd_range<1> nd_range(kernel_size, wg_size);
    const size_t sg_size = 16;

    DataType *send_ptr = send_bufs[rank];
    DataType *recv_buf_ptrs[MAX_NODE_RANKS];
    size_t offset[MAX_NODE_RANKS];
    size_t count[MAX_NODE_RANKS];
    for (uint32_t i = 0; i < comm_size; i++) {
        recv_buf_ptrs[i] = recv_write_ptrs[i];
        offset[i] = send_offsets[i] / sizeof(DataType);
        count[i] = send_counts[i];
    }

    e = queue.submit([=](sycl::handler &h) {
        h.depends_on(dep);
        h.parallel_for(nd_range, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
            for (uint32_t i = 0; i < comm_size; i++) {
                uint32_t peer = (rank + i) % comm_size;
                copy_kernel_wrap<DataType, new_vec_size, new_vec_size>(
                    recv_buf_ptrs[peer], send_ptr + offset[peer], count[peer], 1);
            }
        });
    });
    return { e };
}

template <size_t vec_size, typename DataType>
std::vector<sycl::event> alltoallv_vec_write_aligned(
    sycl::queue &queue,
    const std::shared_ptr<ccl_comm> node_comm,
    const std::array<DataType *, MAX_NODE_RANKS> &send_bufs,
    const std::vector<size_t> &send_counts,
    const std::vector<size_t> &send_offsets,
    std::array<DataType *, MAX_NODE_RANKS> &recv_write_ptrs,
    const size_t max_count,
    const size_t rank,
    const size_t comm_size,
    sycl::event &dep) {
    CCL_THROW_IF_NOT(vec_size > 0, "vec_size has to be a positive value");

    bool is_arc = is_arc_card(ccl::global_data::get().ze_data->devices[0].family);
    // for BMG on PCIe systems
    if (is_arc) {
        return alltoallv_vec_write_aligned_with_eu<vec_size, DataType>(queue,
                                                                       node_comm,
                                                                       send_bufs,
                                                                       send_counts,
                                                                       send_offsets,
                                                                       recv_write_ptrs,
                                                                       max_count,
                                                                       rank,
                                                                       comm_size,
                                                                       dep);
    }
    // copy vector contents into plain C-arrays so the kernel captures
    // device-copyable types (std::vector is not device copyable)
    size_t counts[MAX_NODE_RANKS];
    size_t offsets[MAX_NODE_RANKS];
    for (size_t i = 0; i < comm_size; ++i) {
        counts[i] = send_counts[i];
        offsets[i] = send_offsets[i] / sizeof(DataType);
    }

    sycl::event e = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep);
        // start and end are aligned, the internals are a multiple of sycl_vec size
        // therefore, there is no remainder to handle
        cgh.parallel_for<oneccl_alltoallv_write_aligned<DataType, vec_size>>(
            sycl::range<1>(max_count / vec_size), [=](sycl::id<1> idx) {
                size_t start_idx = idx * vec_size;
                for (uint32_t i = 0; i < comm_size; ++i) {
                    uint32_t peer = (rank + i) % comm_size;
                    if (start_idx < counts[peer]) {
                        sycl::vec<DataType, vec_size> data =
                            *static_cast<sycl::vec<DataType, vec_size> *>(
                                (void *)(send_bufs[rank] + offsets[peer] + start_idx));
                        *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                            &recv_write_ptrs[peer][start_idx]) = data;
                    }
                }
            });
    });
    return { e };
}

template <size_t vec_size, typename DataType>
std::vector<sycl::event> alltoallv_vec_write(
    sycl::queue &queue,
    const std::shared_ptr<ccl_comm> node_comm,
    const std::array<DataType *, MAX_NODE_RANKS> &send_bufs,
    const std::vector<size_t> &send_counts,
    const std::vector<size_t> &send_offsets,
    std::array<DataType *, MAX_NODE_RANKS> &recv_write_ptrs,
    const size_t rank,
    const size_t comm_size,
    sycl::event &dep,
    const alltoallv_ctx &options,
    bool is_multi_thread = false) {
    CCL_THROW_IF_NOT(vec_size > 0, "vec_size has to be a positive value");
    constexpr size_t vec_size_bytes = vec_size * sizeof(DataType);

    bool is_arc = is_arc_card(ccl::global_data::get().ze_data->devices[0].family);
    if (is_arc) {
        const bool use_memcpy = ccl::global_data::env().sycl_alltoall_copy_engine;
        // copy engine can handle alignment internally
        if (use_memcpy) {
            return alltoallv_vec_write_aligned_with_ce<vec_size, DataType>(queue,
                                                                           node_comm,
                                                                           send_bufs,
                                                                           send_counts,
                                                                           send_offsets,
                                                                           recv_write_ptrs,
                                                                           rank,
                                                                           comm_size,
                                                                           dep);
        }
    }

    size_t peel_front_count[MAX_NODE_RANKS];
    std::array<DataType *, MAX_NODE_RANKS> peel_offsets{};

    for (uint32_t i = 0; i < comm_size; ++i) {
        peel_offsets[i] = recv_write_ptrs[i];
    }

    size_t alignment_bytes = get_alignment_bytes(vec_size_bytes);

    calculate_peel_counts<DataType>(
        peel_offsets, peel_front_count, alignment_bytes, send_counts, comm_size);

    size_t max_peel_loop_vec_count = divide_round_up(alignment_bytes, vec_size_bytes);
    size_t max_peel_loop_count = max_peel_loop_vec_count * vec_size;

    bool are_all_ranks_aligned = check_all_aligned(peel_front_count,
                                                   max_peel_loop_count,
                                                   send_bufs[rank],
                                                   alignment_bytes,
                                                   options.max_count,
                                                   vec_size_bytes,
                                                   send_counts,
                                                   comm_size);
    if (are_all_ranks_aligned) {
        // aligned data, no reason to peel
        // fallback to simpler implementation for performance reasons
        LOG_DEBUG("alltoallv_vec_write: all ranks are aligned, using aligned implementation");
        return alltoallv_vec_write_aligned<vec_size, DataType>(queue,
                                                               node_comm,
                                                               send_bufs,
                                                               send_counts,
                                                               send_offsets,
                                                               recv_write_ptrs,
                                                               options.max_count,
                                                               rank,
                                                               comm_size,
                                                               dep);
    }

    LOG_DEBUG("alltoallv_vec_write: not all ranks are aligned, using unaligned implementation");
    // collect per-rank aligned counts to handle variable sizes
    size_t aligned_count_per_rank[MAX_NODE_RANKS];
    size_t end_count_per_rank[MAX_NODE_RANKS];
    size_t max_aligned_loop_count = 0;
    size_t max_end_loop_count = 0;

    for (size_t i = 0; i < comm_size; ++i) {
        size_t remaining =
            (send_counts[i] > peel_front_count[i]) ? send_counts[i] - peel_front_count[i] : 0;
        aligned_count_per_rank[i] = remaining / vec_size;
        end_count_per_rank[i] = remaining % vec_size;
        max_aligned_loop_count = std::max(max_aligned_loop_count, aligned_count_per_rank[i]);
        max_end_loop_count = std::max(max_end_loop_count, end_count_per_rank[i]);
    }

    // plain C-arrays for SYCL kernel to capture
    DataType *send_ptr = send_bufs[rank];
    DataType *recv_ptrs[MAX_NODE_RANKS];
    size_t offsets[MAX_NODE_RANKS];

    for (size_t i = 0; i < comm_size; ++i) {
        recv_ptrs[i] = recv_write_ptrs[i];
        offsets[i] = send_offsets[i] / sizeof(DataType);
    }

    if (is_multi_thread) {
        // WA: Use original queue in multi-threaded mode to preserve dependency chains.
        // Out-of-order queues break SYCL event dependencies across threads since events
        // are tied to their creation queue. Cross-queue depends_on() can cause hangs.
        std::vector<sycl::event> events;

        // Only submit kernels if they have work to do
        if (max_peel_loop_count > 0) {
            events.push_back(queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_write_peel_front<DataType, vec_size, true>>(
                    sycl::range<1>(max_peel_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            // peel front: element-by-element copy into unaligned destination
                            if (idx < peel_front_count[buffer_index]) {
                                DataType data = (send_ptr + offsets[buffer_index])[idx];
                                recv_ptrs[buffer_index][idx] = data;
                            }
                        }
                    });
            }));
        }

        if (max_aligned_loop_count > 0) {
            events.push_back(queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_write_aligned_main<DataType, vec_size, true>>(
                    sycl::range<1>(max_aligned_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < aligned_count_per_rank[buffer_index]) {
                                size_t start_idx = idx * vec_size + peel_front_count[buffer_index];
                                sycl::vec<DataType, vec_size> data;
#pragma unroll
                                for (size_t i = 0; i < vec_size; ++i) {
                                    data[i] = (send_ptr + offsets[buffer_index])[start_idx + i];
                                }
                                *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                                    &recv_ptrs[buffer_index][start_idx]) = data;
                            }
                        }
                    });
            }));
        }

        if (max_end_loop_count > 0) {
            events.push_back(queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_write_peel_back<DataType, vec_size, true>>(
                    sycl::range<1>(max_end_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < end_count_per_rank[buffer_index]) {
                                size_t start_idx = peel_front_count[buffer_index] +
                                                   aligned_count_per_rank[buffer_index] * vec_size +
                                                   idx;
                                DataType data = (send_ptr + offsets[buffer_index])[start_idx];
                                recv_ptrs[buffer_index][start_idx] = data;
                            }
                        }
                    });
            }));
        }

        return events;
    }
    else {
        // add submissions to out-of-order queue, for performance reasons
        sycl::queue out_of_order_q(queue.get_context(), queue.get_device());
        std::vector<sycl::event> result;

        if (max_peel_loop_count > 0) {
            result.push_back(out_of_order_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_write_peel_front<DataType, vec_size, false>>(
                    sycl::range<1>(max_peel_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < peel_front_count[buffer_index]) {
                                DataType data = (send_ptr + offsets[buffer_index])[idx];
                                recv_ptrs[buffer_index][idx] = data;
                            }
                        }
                    });
            }));
        }

        if (max_aligned_loop_count > 0) {
            result.push_back(out_of_order_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_write_aligned_main<DataType, vec_size, false>>(
                    sycl::range<1>(max_aligned_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < aligned_count_per_rank[buffer_index]) {
                                size_t start_idx = idx * vec_size + peel_front_count[buffer_index];
                                sycl::vec<DataType, vec_size> data;
#pragma unroll
                                for (size_t i = 0; i < vec_size; ++i) {
                                    data[i] = (send_ptr + offsets[buffer_index])[start_idx + i];
                                }
                                *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                                    &recv_ptrs[buffer_index][start_idx]) = data;
                            }
                        }
                    });
            }));
        }

        if (max_end_loop_count > 0) {
            result.push_back(out_of_order_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_write_peel_back<DataType, vec_size, false>>(
                    sycl::range<1>(max_end_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < end_count_per_rank[buffer_index]) {
                                size_t start_idx = peel_front_count[buffer_index] +
                                                   aligned_count_per_rank[buffer_index] * vec_size +
                                                   idx;
                                DataType data = (send_ptr + offsets[buffer_index])[start_idx];
                                recv_ptrs[buffer_index][start_idx] = data;
                            }
                        }
                    });
            }));
        }

        return result;
    }
}

// aligned fast-path read: vectorized loads from the remote source and
// vectorized stores into the local recv buffer (used when every rank's
// remote read source and local recv slice are already aligned)
template <size_t vec_size, typename DataType>
std::vector<sycl::event> alltoallv_vec_read_aligned(
    sycl::queue &queue,
    const std::array<DataType *, MAX_NODE_RANKS> &send_read_ptrs,
    std::array<DataType *, MAX_NODE_RANKS> &recv_bufs,
    const std::vector<size_t> &recv_counts,
    const std::vector<size_t> &recv_offsets,
    const size_t max_count,
    const size_t rank,
    const size_t comm_size,
    sycl::event &dep) {
    CCL_THROW_IF_NOT(vec_size > 0, "vec_size has to be a positive value");

    DataType *recv_ptr = recv_bufs[rank];
    DataType *read_ptrs[MAX_NODE_RANKS];
    size_t counts[MAX_NODE_RANKS];
    size_t offsets[MAX_NODE_RANKS];
    for (size_t i = 0; i < comm_size; ++i) {
        read_ptrs[i] = send_read_ptrs[i];
        counts[i] = recv_counts[i];
        offsets[i] = recv_offsets[i] / sizeof(DataType);
    }

    sycl::event e = queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep);
        cgh.parallel_for<oneccl_alltoallv_read_aligned<DataType, vec_size>>(
            sycl::range<1>(max_count / vec_size), [=](sycl::id<1> idx) {
                size_t start_idx = idx * vec_size;
                for (size_t peer = 0; peer < comm_size; ++peer) {
                    if (start_idx < counts[peer]) {
                        sycl::vec<DataType, vec_size> data =
                            *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                                &read_ptrs[peer][start_idx]);
                        *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                            &(recv_ptr + offsets[peer])[start_idx]) = data;
                    }
                }
            });
    });
    return { e };
}

template <size_t vec_size, typename DataType>
std::vector<sycl::event> alltoallv_vec_read(
    sycl::queue &queue,
    const std::array<DataType *, MAX_NODE_RANKS> &send_read_ptrs,
    std::array<DataType *, MAX_NODE_RANKS> &recv_bufs,
    const std::vector<size_t> &recv_counts,
    const std::vector<size_t> &recv_offsets,
    const size_t rank,
    const size_t comm_size,
    sycl::event &dep,
    const alltoallv_ctx &options,
    bool is_multi_thread = false) {
    CCL_THROW_IF_NOT(vec_size > 0, "vec_size has to be a positive value");
    constexpr size_t vec_size_bytes = vec_size * sizeof(DataType);

    size_t peel_front_count[MAX_NODE_RANKS];
    std::array<DataType *, MAX_NODE_RANKS> peel_offsets{};

    for (size_t i = 0; i < comm_size; ++i) {
        peel_offsets[i] = send_read_ptrs[i];
    }

    size_t alignment_bytes = get_alignment_bytes(vec_size_bytes);

    calculate_peel_counts<DataType>(
        peel_offsets, peel_front_count, alignment_bytes, recv_counts, comm_size);

    size_t max_peel_loop_vec_count = divide_round_up(alignment_bytes, vec_size_bytes);
    size_t max_peel_loop_count = max_peel_loop_vec_count * vec_size;

    bool are_all_ranks_aligned = check_all_aligned(peel_front_count,
                                                   max_peel_loop_count,
                                                   recv_bufs[rank],
                                                   alignment_bytes,
                                                   options.max_count,
                                                   vec_size_bytes,
                                                   recv_counts,
                                                   comm_size);
    if (are_all_ranks_aligned) {
        // aligned data, no reason to peel
        // fallback to simpler implementation for performance reasons
        LOG_DEBUG("alltoallv_vec_read: all ranks are aligned, using aligned implementation");
        return alltoallv_vec_read_aligned<vec_size, DataType>(queue,
                                                              send_read_ptrs,
                                                              recv_bufs,
                                                              recv_counts,
                                                              recv_offsets,
                                                              options.max_count,
                                                              rank,
                                                              comm_size,
                                                              dep);
    }

    LOG_DEBUG("alltoallv_vec_read: not all ranks are aligned, using unaligned implementation");
    // collect per-rank aligned counts to handle variable sizes
    size_t aligned_count_per_rank[MAX_NODE_RANKS];
    size_t end_count_per_rank[MAX_NODE_RANKS];
    size_t max_aligned_loop_count = 0;
    size_t max_end_loop_count = 0;

    for (size_t i = 0; i < comm_size; ++i) {
        size_t remaining =
            (recv_counts[i] > peel_front_count[i]) ? recv_counts[i] - peel_front_count[i] : 0;
        aligned_count_per_rank[i] = remaining / vec_size;
        end_count_per_rank[i] = remaining % vec_size;
        max_aligned_loop_count = std::max(max_aligned_loop_count, aligned_count_per_rank[i]);
        max_end_loop_count = std::max(max_end_loop_count, end_count_per_rank[i]);
    }

    DataType *recv_ptr = recv_bufs[rank];
    DataType *read_ptrs[MAX_NODE_RANKS];
    size_t offsets[MAX_NODE_RANKS];

    for (size_t i = 0; i < comm_size; ++i) {
        read_ptrs[i] = send_read_ptrs[i];
        offsets[i] = recv_offsets[i] / sizeof(DataType);
    }

    if (is_multi_thread) {
        // WA: Use original queue in multi-threaded mode to preserve dependency chains.
        // Out-of-order queues break SYCL event dependencies across threads since events
        // are tied to their creation queue. Cross-queue depends_on() can cause hangs.
        std::vector<sycl::event> events;

        // Only submit kernels if they have work to do
        if (max_peel_loop_count > 0) {
            events.push_back(queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_read_peel_front<DataType, vec_size, true>>(
                    sycl::range<1>(max_peel_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < peel_front_count[buffer_index]) {
                                DataType data = read_ptrs[buffer_index][idx];
                                (recv_ptr + offsets[buffer_index])[idx] = data;
                            }
                        }
                    });
            }));
        }

        if (max_aligned_loop_count > 0) {
            events.push_back(queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_read_aligned_main<DataType, vec_size, true>>(
                    sycl::range<1>(max_aligned_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < aligned_count_per_rank[buffer_index]) {
                                size_t start_idx = idx * vec_size + peel_front_count[buffer_index];
                                sycl::vec<DataType, vec_size> data =
                                    *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                                        &read_ptrs[buffer_index][start_idx]);
#pragma unroll
                                for (size_t i = 0; i < vec_size; ++i) {
                                    (recv_ptr + offsets[buffer_index])[start_idx + i] = data[i];
                                }
                            }
                        }
                    });
            }));
        }

        if (max_end_loop_count > 0) {
            events.push_back(queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_read_peel_back<DataType, vec_size, true>>(
                    sycl::range<1>(max_end_loop_count), [=](sycl::id<1> idx) {
#pragma unroll
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < end_count_per_rank[buffer_index]) {
                                size_t start_idx = peel_front_count[buffer_index] +
                                                   aligned_count_per_rank[buffer_index] * vec_size +
                                                   idx;
                                DataType data = read_ptrs[buffer_index][start_idx];
                                (recv_ptr + offsets[buffer_index])[start_idx] = data;
                            }
                        }
                    });
            }));
        }

        return events;
    }
    else {
        // add submissions to out-of-order queue, for performance reasons
        sycl::queue out_of_order_q(queue.get_context(), queue.get_device());
        std::vector<sycl::event> result;

        if (max_peel_loop_count > 0) {
            result.push_back(out_of_order_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_read_peel_front<DataType, vec_size, false>>(
                    sycl::range<1>(max_peel_loop_count), [=](sycl::id<1> idx) {
#pragma unroll
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < peel_front_count[buffer_index]) {
                                DataType data = read_ptrs[buffer_index][idx];
                                (recv_ptr + offsets[buffer_index])[idx] = data;
                            }
                        }
                    });
            }));
        }

        if (max_aligned_loop_count > 0) {
            result.push_back(out_of_order_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_read_aligned_main<DataType, vec_size, false>>(
                    sycl::range<1>(max_aligned_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < aligned_count_per_rank[buffer_index]) {
                                size_t start_idx = idx * vec_size + peel_front_count[buffer_index];
                                sycl::vec<DataType, vec_size> data =
                                    *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                                        &read_ptrs[buffer_index][start_idx]);
#pragma unroll
                                for (size_t i = 0; i < vec_size; ++i) {
                                    (recv_ptr + offsets[buffer_index])[start_idx + i] = data[i];
                                }
                            }
                        }
                    });
            }));
        }

        if (max_end_loop_count > 0) {
            result.push_back(out_of_order_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep);
                cgh.parallel_for<oneccl_alltoallv_read_peel_back<DataType, vec_size, false>>(
                    sycl::range<1>(max_end_loop_count), [=](sycl::id<1> idx) {
                        for (size_t buffer_index = 0; buffer_index < comm_size; ++buffer_index) {
                            if (idx < end_count_per_rank[buffer_index]) {
                                size_t start_idx = peel_front_count[buffer_index] +
                                                   aligned_count_per_rank[buffer_index] * vec_size +
                                                   idx;
                                DataType data = read_ptrs[buffer_index][start_idx];
                                (recv_ptr + offsets[buffer_index])[start_idx] = data;
                            }
                        }
                    });
            }));
        }

        return result;
    }
}

template <size_t vec_size, typename DataType>
std::vector<sycl::event> alltoallv_large_vec_size_impl(
    sycl::queue &queue,
    const std::shared_ptr<ccl_comm> node_comm,
    const std::array<DataType *, MAX_NODE_RANKS> &send_bufs,
    const std::vector<size_t> &send_counts,
    const std::vector<size_t> &send_offsets,
    const std::array<DataType *, MAX_NODE_RANKS>
        &send_read_ptrs, // remote: peer's send_buf + peer's send_displs[rank]
    std::array<DataType *, MAX_NODE_RANKS> &recv_bufs,
    const std::vector<size_t> &recv_counts,
    const std::vector<size_t> &recv_offsets,
    std::array<DataType *, MAX_NODE_RANKS>
        &recv_write_ptrs, // remote: peer's recv_buf + peer's recv_displs[rank]
    const size_t rank,
    const size_t comm_size,
    sycl::event &dep,
    const alltoallv_ctx &options) {
    switch (ccl::global_data::env().sycl_alltoall_protocol) {
        case ccl_sycl_alltoall_protocol::read: {
            return alltoallv_vec_read<vec_size>(queue,
                                                send_read_ptrs,
                                                recv_bufs,
                                                recv_counts,
                                                recv_offsets,
                                                rank,
                                                comm_size,
                                                dep,
                                                options);
        }
        case ccl_sycl_alltoall_protocol::write: {
            return alltoallv_vec_write<vec_size>(queue,
                                                 node_comm,
                                                 send_bufs,
                                                 send_counts,
                                                 send_offsets,
                                                 recv_write_ptrs,
                                                 rank,
                                                 comm_size,
                                                 dep,
                                                 options);
        }
        default: {
            CCL_THROW("unknown alltoallv protocol type");
            break;
        }
    }
    CCL_THROW("unknown alltoallv protocol type");
}

template <typename T>
ccl::event alltoallv_large_impl(const void *send_buf,
                                const std::vector<size_t> &send_counts,
                                const std::vector<size_t> &send_displs,
                                void *recv_buf,
                                const std::vector<size_t> &recv_counts,
                                const std::vector<size_t> &recv_displs,
                                ccl::datatype dtype,
                                ccl_comm *comm,
                                ccl_stream *global_stream,
                                const ccl::vector_class<ccl::event> &deps,
                                const alltoallv_ctx &options) {
    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    auto node_comm = comm->get_node_comm();
    const size_t rank = node_comm->rank();
    const size_t comm_size = node_comm->size();
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    sycl::queue q = global_stream->get_native_stream();

    std::vector<void *> ptrs{ (void *)send_buf, (T *)recv_buf }; // index 0 and 1

    // ptrs indices start at 2 + i
    // recv_buf offsets for write
    for (size_t i = 0; i < comm_size; ++i)
        ptrs.push_back((char *)recv_buf + recv_displs[i]);

    // ptrs indices start at 2 + comm_size + i
    // send_buf offsets for read
    for (size_t i = 0; i < comm_size; ++i)
        ptrs.push_back((char *)send_buf + send_displs[i]);

    std::array<T *, MAX_NODE_RANKS> send_pointers;
    std::array<T *, MAX_NODE_RANKS> recv_pointers;
    std::array<T *, MAX_NODE_RANKS> send_read_ptrs;
    std::array<T *, MAX_NODE_RANKS> recv_write_ptrs;

    auto &g = ccl::global_data::get().shared_data;

    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("alltoallv_large running do_ipc_exchangeExt for ptrs");
        do_ipc_exchangeExt(
            comm, g->hash_table, global_stream, ptrs, comm->global_current_id);
        send_pointers = get_ipc_ptrsExt<T, MAX_NODE_RANKS>(*g.get(),
                                                           node_comm,
                                                           g->hash_table,
                                                           0,
                                                           0,
                                                           (void *)send_buf,
                                                           comm->global_current_id,
                                                           comm->get_even_comm(),
                                                           comm->get_pair_comm());
        recv_pointers = get_ipc_ptrsExt<T, MAX_NODE_RANKS>(*g.get(),
                                                           node_comm,
                                                           g->hash_table,
                                                           0,
                                                           1,
                                                           (void *)recv_buf,
                                                           comm->global_current_id,
                                                           comm->get_even_comm(),
                                                           comm->get_pair_comm());

        // recv_write_ptrs[peer] == peer_recv_buf + peer_recv_displs[rank]
        recv_write_ptrs = get_ipc_ptrsExt<T, MAX_NODE_RANKS>(
            *g.get(), node_comm, g->hash_table, 0, 2 + rank,
            (char *)recv_buf + recv_displs[rank],
            comm->global_current_id, comm->get_even_comm(), comm->get_pair_comm());

        // send_read_ptrs[peer] == peer_send_buf + peer_send_displs[rank]
        send_read_ptrs = get_ipc_ptrsExt<T, MAX_NODE_RANKS>(
            *g.get(), node_comm, g->hash_table, 0, 2 + comm_size + rank,
            (char *)send_buf + send_displs[rank],
            comm->global_current_id, comm->get_even_comm(), comm->get_pair_comm());
        
        if (comm->is_multi_thread_instance()) {
            pthread_barrier_wait(
                ccl::global_data::get().shared_data->get_barrier(comm->global_current_id));
        }
    }
    else {
        LOG_DEBUG("alltoallv_large running get_registered_ptrs for send and recv buffers");
        auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

        {
            auto [sched, exchange_entry] = do_ipc_exchange(comm, global_stream, ptrs);
            send_pointers = get_ipc_ptrs<T, MAX_NODE_RANKS>(node_comm, 0, (void *)send_buf, sched);
            recv_pointers = get_ipc_ptrs<T, MAX_NODE_RANKS>(node_comm, 1, (void *)recv_buf, sched);

            // recv_write_ptrs[peer] == peer_recv_buf + peer_recv_displs[rank]
            recv_write_ptrs = get_ipc_ptrs<T, MAX_NODE_RANKS>(
                node_comm, 2 + rank, (char *)recv_buf + recv_displs[rank], sched);

            // send_read_ptrs[peer] == peer_send_buf + peer_send_displs[rank]
            send_read_ptrs = get_ipc_ptrs<T, MAX_NODE_RANKS>(
                node_comm, 2 + comm_size + rank, (char *)send_buf + send_displs[rank], sched);
        }
    }
    auto dep = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    constexpr size_t vec_size =
        std::max(static_cast<size_t>(8 / sizeof(T)), static_cast<size_t>(1));

    std::vector<sycl::event> kernel_events =
        alltoallv_large_vec_size_impl<vec_size>(q,
                                                node_comm,
                                                send_pointers,
                                                send_counts,
                                                send_displs,
                                                send_read_ptrs,
                                                recv_pointers,
                                                recv_counts,
                                                recv_displs,
                                                recv_write_ptrs,
                                                rank,
                                                comm_size,
                                                dep,
                                                options);

    sycl::event barrier_event2 = invoke_barrier(node_comm, q, kernel_events, is_cpu_barrier);

    return ccl::event::create_from_native(barrier_event2);
}
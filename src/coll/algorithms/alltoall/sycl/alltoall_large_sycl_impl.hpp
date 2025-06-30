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
#include <cstdint>
#include <algorithm>
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"

static size_t get_alignment_bytes(size_t min_alignment_bytes) {
    size_t alignment_bytes = ccl::global_data::env().kernel_mem_align;

    static bool min_alignment_warning_printed = false;
    if (alignment_bytes < min_alignment_bytes && !min_alignment_warning_printed) {
        LOG_WARN("requested alignment [",
                 alignment_bytes,
                 "] bytes is lower than minimal alignment required by the selected algorithm [",
                 min_alignment_bytes,
                 "]; further execution might cause alignment errors or crashes; "
                 "this warning will not be printed again");
        min_alignment_warning_printed = true;
    }

    static bool mvec_alignment_warning_printed = false;
    if (alignment_bytes % min_alignment_bytes != 0 && !mvec_alignment_warning_printed) {
        LOG_WARN("requested alignment [",
                 alignment_bytes,
                 "] bytes is not a multiple of alignment required by the selected algorithm [",
                 min_alignment_bytes,
                 "]; further execution might cause alignment errors or crashes; "
                 "this warning will not be printed again");
        mvec_alignment_warning_printed = true;
    }

    static bool suggested_alignment_warning_printed = false;
    constexpr size_t validated_perf_alignment = 128;
    if (alignment_bytes % validated_perf_alignment != 0 && !suggested_alignment_warning_printed) {
        LOG_WARN("requested alignment [",
                 alignment_bytes,
                 "] bytes is not a multiple of the suggested alignment [",
                 validated_perf_alignment,
                 "]; perf might be suboptimal; "
                 "this warning will not be printed again");
        suggested_alignment_warning_printed = true;
    }

    return alignment_bytes;
}

template <typename DataType>
DataType divide_round_up(DataType x, DataType y) {
    return (x + y - 1) / y;
}

template <typename DataType, size_t N_RANKS>
std::vector<sycl::event> alltoall_memcpy_write(sycl::queue &queue,
                                               const std::array<DataType *, N_RANKS> &send_bufs,
                                               std::array<DataType *, N_RANKS> &recv_bufs,
                                               size_t per_rank_count,
                                               size_t rank,
                                               sycl::event &dep) {
    std::vector<sycl::event> events;
    events.reserve(recv_bufs.size());

    DataType *send_buf = send_bufs[rank];

    for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
        DataType *recv_buf = recv_bufs[buffer_index];

        size_t recv_offset = rank * per_rank_count;
        size_t send_offset = buffer_index * per_rank_count;

        events.push_back(queue.memcpy(
            recv_buf + recv_offset, send_buf + send_offset, per_rank_count * sizeof(DataType)));
    }

    return events;
}

template <typename DataType, size_t N_RANKS>
std::vector<sycl::event> alltoall_memcpy_read(sycl::queue &queue,
                                              const std::array<DataType *, N_RANKS> &send_bufs,
                                              std::array<DataType *, N_RANKS> &recv_bufs,
                                              size_t per_rank_count,
                                              size_t rank,
                                              sycl::event &dep) {
    std::vector<sycl::event> events;
    events.reserve(send_bufs.size());
    DataType *recv_buf = recv_bufs[rank];

    for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
        DataType *send_buf = send_bufs[buffer_index];

        size_t recv_offset = buffer_index * per_rank_count;
        size_t send_offset = rank * per_rank_count;

        events.push_back(queue.memcpy(
            recv_buf + recv_offset, send_buf + send_offset, per_rank_count * sizeof(DataType)));
    }

    return events;
}

template <size_t vec_size, typename DataType, size_t N_RANKS>
std::vector<sycl::event> alltoall_vec_write_aligned(
    sycl::queue &queue,
    const std::array<DataType *, N_RANKS> &send_bufs,
    std::array<DataType *, N_RANKS> &recv_bufs,
    size_t per_rank_count,
    size_t rank,
    sycl::event &dep) {
    CCL_THROW_IF_NOT(vec_size > 0, "vec_size has to be a positive value");

    return { queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep);
        // start and end are properly aligned, the internals are a multiple of sycl_vec size
        // therefore, there is no remainder to handle
        cgh.parallel_for(sycl::range<1>(per_rank_count / vec_size), [=](sycl::id<1> idx) {
            size_t start_idx = idx * vec_size;
#pragma unroll
            for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
                sycl::vec<DataType, vec_size> data = *static_cast<sycl::vec<DataType, vec_size> *>(
                    (void *)&send_bufs[rank][buffer_index * per_rank_count + start_idx]);
                *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                    &recv_bufs[buffer_index][rank * per_rank_count + start_idx]) = data;
            }
        });
    }) };
}

template <size_t vec_size, typename DataType, size_t N_RANKS>
std::vector<sycl::event> alltoall_vec_read_aligned(sycl::queue &queue,
                                                   const std::array<DataType *, N_RANKS> &send_bufs,
                                                   std::array<DataType *, N_RANKS> &recv_bufs,
                                                   size_t per_rank_count,
                                                   size_t rank,
                                                   sycl::event &dep) {
    CCL_THROW_IF_NOT(vec_size > 0, "vec_size has to be a positive value");

    return { queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dep);
        // start and end are properly aligned, the internals are a multiple of sycl_vec size
        // therefore, there is no remainder to handle
        cgh.parallel_for(sycl::range<1>(per_rank_count / vec_size), [=](sycl::id<1> idx) {
            size_t start_idx = idx * vec_size;
#pragma unroll
            for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
                sycl::vec<DataType, vec_size> data = *static_cast<sycl::vec<DataType, vec_size> *>(
                    (void *)&send_bufs[buffer_index][rank * per_rank_count + start_idx]);

                *static_cast<sycl::vec<DataType, vec_size> *>(
                    (void *)&recv_bufs[rank][buffer_index * per_rank_count + start_idx]) = data;
            }
        });
    }) };
}

template <typename DataType, size_t N_RANKS>
void calculate_peel_counts(const std::array<DataType *, N_RANKS> &buf_to_peel,
                           std::array<size_t, N_RANKS> &peel_front_count,
                           size_t alignment_bytes,
                           size_t per_rank_count) {
    CCL_THROW_IF_NOT(alignment_bytes != 0, "alignment cannot be 0");

    size_t alignment_elements = alignment_bytes / sizeof(DataType);
    CCL_THROW_IF_NOT((alignment_bytes % sizeof(DataType)) == 0,
                     "alignment must be a multiple of DataType size");

    for (size_t i = 0; i < N_RANKS; ++i) {
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
        if (peel_front_count[i] > per_rank_count) {
            peel_front_count[i] = per_rank_count;
        }
    }
}

template <typename DataType, size_t N_RANKS>
bool check_all_aligned(std::array<size_t, N_RANKS> &peel_front_count,
                       size_t max_peel_loop_count,
                       DataType *local_buffer,
                       size_t alignment_bytes,
                       size_t per_rank_count,
                       size_t vec_size_bytes) {
    // local buffer needs to meet alignment for sycl_vec
    bool local_buf_vec_aligned = (reinterpret_cast<uintptr_t>(local_buffer) % vec_size_bytes) == 0;
    // end needs to be aligned on vec_size_bytes to produce no remainder
    bool is_count_alignment_mul = (per_rank_count * sizeof(DataType) % vec_size_bytes) == 0;
    // initial value, to be modified
    bool are_all_ranks_aligned = is_count_alignment_mul && local_buf_vec_aligned;

    for (size_t i = 0; i < N_RANKS; ++i) {
        LOG_TRACE("peel_front_count[", i, "] = ", peel_front_count[i]);
        // additional precondition check
        CCL_THROW_IF_NOT(peel_front_count[i] <= max_peel_loop_count,
                         "peeling more elements than the upper limit");
        size_t peel_front_bytes = peel_front_count[i] * sizeof(DataType);
        bool is_rank_aligned = (peel_front_bytes == alignment_bytes);
        are_all_ranks_aligned = (are_all_ranks_aligned && is_rank_aligned);
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

template <size_t vec_size, typename DataType, size_t N_RANKS>
std::vector<sycl::event> alltoall_vec_write(sycl::queue &queue,
                                            const std::array<DataType *, N_RANKS> &send_bufs,
                                            std::array<DataType *, N_RANKS> &recv_bufs,
                                            size_t per_rank_count,
                                            size_t rank,
                                            sycl::event &dep) {
    CCL_THROW_IF_NOT(vec_size > 0, "vec_size has to be a positive value");
    constexpr size_t vec_size_bytes = vec_size * sizeof(DataType);

    std::array<size_t, N_RANKS> peel_front_count{};
    std::array<DataType *, N_RANKS> peel_offsets{};

    for (size_t i = 0; i < N_RANKS; ++i) {
        peel_offsets[i] = &recv_bufs[i][rank * per_rank_count];
    }

    size_t alignment_bytes = get_alignment_bytes(vec_size_bytes);

    calculate_peel_counts<DataType, N_RANKS>(
        peel_offsets, peel_front_count, alignment_bytes, per_rank_count);

    size_t max_peel_loop_vec_count = divide_round_up(alignment_bytes, vec_size_bytes);
    size_t max_peel_loop_count = max_peel_loop_vec_count * vec_size;

    bool are_all_ranks_aligned = check_all_aligned(peel_front_count,
                                                   max_peel_loop_count,
                                                   send_bufs[rank],
                                                   alignment_bytes,
                                                   per_rank_count,
                                                   vec_size_bytes);
    if (are_all_ranks_aligned) {
        // aligned data, no reason to peel
        // fallback to simpler implementation for performance reasons
        return alltoall_vec_write_aligned<vec_size, DataType, N_RANKS>(
            queue, send_bufs, recv_bufs, per_rank_count, rank, dep);
    }

    size_t aligned_loop_count = per_rank_count / vec_size;
    // subtract max number that can be processed in the prefix loop
    size_t to_subtract_from_aligned_loop = max_peel_loop_vec_count;
    aligned_loop_count = (aligned_loop_count >= to_subtract_from_aligned_loop)
                             ? aligned_loop_count - to_subtract_from_aligned_loop
                             : 0;
    // the remainder must process all that is left
    // the peel front loop should process at least 1 element
    // but we are ignoring this fact here and we assume it might process 0
    size_t end_loop_count = per_rank_count - aligned_loop_count * vec_size;

    // add submissions to out-of-order queue, for performance reasons
    sycl::queue out_of_order_q(queue.get_context(), queue.get_device());

    return {
        out_of_order_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep);
            // start and end are properly aligned, the internals are a multiple of sycl_vec size
            // therefore, there is no remainder to handle
            cgh.parallel_for(sycl::range<1>(max_peel_loop_count), [=](sycl::id<1> idx) {
#pragma unroll
                for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
                    // peel front
                    size_t start_send = buffer_index * per_rank_count;
                    size_t start_recv = rank * per_rank_count;

                    // both accesses unaligned, copy element-by-element
                    if (idx < peel_front_count[buffer_index]) {
                        DataType data = send_bufs[rank][start_send + idx];
                        recv_bufs[buffer_index][start_recv + idx] = data;
                    }
                }
            });
        }),
        out_of_order_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep);
            // start and end are properly aligned, the internals are a multiple of sycl_vec size
            // therefore, there is no remainder to handle
            cgh.parallel_for(sycl::range<1>(aligned_loop_count), [=](sycl::id<1> idx) {
#pragma unroll
                for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
                    // handle aligned data
                    size_t start_idx = idx * vec_size + peel_front_count[buffer_index];
                    size_t start_send_idx = buffer_index * per_rank_count + start_idx;
                    size_t recv_idx = rank * per_rank_count + start_idx;
                    // the read operation (local) might be unaligned
                    // read element-by-element into sycl::vec
                    sycl::vec<DataType, vec_size> data;
#pragma unroll
                    for (size_t i = 0; i < vec_size; ++i) {
                        data[i] = send_bufs[rank][start_send_idx + i];
                    }
                    // write operation (remote) is aligned
                    *(sycl::vec<DataType, vec_size> *)static_cast<void *>(
                        &recv_bufs[buffer_index][recv_idx]) = data;
                }
            });
        }),
        out_of_order_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep);
            // start and end are properly aligned, the internals are a multiple of sycl_vec size
            // therefore, there is no remainder to handle
            cgh.parallel_for(sycl::range<1>(end_loop_count), [=](sycl::id<1> idx) {
#pragma unroll
                for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
                    // peel back
                    size_t start_idx =
                        peel_front_count[buffer_index] + aligned_loop_count * vec_size;
                    size_t start_send = buffer_index * per_rank_count;
                    size_t start_recv = rank * per_rank_count;

                    size_t loop_idx = start_idx + idx;
                    // both accesses unaligned, copy element-by-element
                    if (loop_idx < per_rank_count) {
                        DataType data = send_bufs[rank][start_send + loop_idx];
                        recv_bufs[buffer_index][start_recv + loop_idx] = data;
                    }
                }
            });
        }),
    };
}

template <size_t vec_size, typename DataType, size_t N_RANKS>
std::vector<sycl::event> alltoall_vec_read(sycl::queue &queue,
                                           const std::array<DataType *, N_RANKS> &send_bufs,
                                           std::array<DataType *, N_RANKS> &recv_bufs,
                                           size_t per_rank_count,
                                           size_t rank,
                                           sycl::event &dep) {
    CCL_THROW_IF_NOT(vec_size > 0, "vec_size has to be a positive value");
    constexpr size_t vec_size_bytes = vec_size * sizeof(DataType);

    std::array<size_t, N_RANKS> peel_front_count{};
    std::array<DataType *, N_RANKS> peel_offsets{};

    for (size_t i = 0; i < N_RANKS; ++i) {
        peel_offsets[i] = &send_bufs[i][rank * per_rank_count];
    }

    size_t alignment_bytes = get_alignment_bytes(vec_size_bytes);

    calculate_peel_counts<DataType, N_RANKS>(
        peel_offsets, peel_front_count, alignment_bytes, per_rank_count);

    size_t max_peel_loop_vec_count = divide_round_up(alignment_bytes, vec_size_bytes);
    size_t max_peel_loop_count = max_peel_loop_vec_count * vec_size;

    bool are_all_ranks_aligned = check_all_aligned(peel_front_count,
                                                   max_peel_loop_count,
                                                   recv_bufs[rank],
                                                   alignment_bytes,
                                                   per_rank_count,
                                                   vec_size_bytes);
    if (are_all_ranks_aligned) {
        // aligned data, no reason to peel
        // fallback to simpler implementation for performance reasons
        return alltoall_vec_read_aligned<vec_size, DataType, N_RANKS>(
            queue, send_bufs, recv_bufs, per_rank_count, rank, dep);
    }

    size_t aligned_loop_count = per_rank_count / vec_size;
    // subtract max number that can be processed in the prefix loop
    size_t to_subtract_from_aligned_loop = max_peel_loop_vec_count;
    aligned_loop_count = (aligned_loop_count >= to_subtract_from_aligned_loop)
                             ? aligned_loop_count - to_subtract_from_aligned_loop
                             : 0;
    // the remainder must process all that is left
    // the peel front loop should process at least 1 element
    // but we are ignoring this fact here and we assume it might process 0
    size_t end_loop_count = per_rank_count - aligned_loop_count * vec_size;

    // add submissions to out-of-order queue, for performance reasons
    sycl::queue out_of_order_q(queue.get_context(), queue.get_device());

    return {
        out_of_order_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep);
            // start and end are properly aligned, the internals are a multiple of sycl_vec size
            // therefore, there is no remainder to handle
            cgh.parallel_for(sycl::range<1>(max_peel_loop_count), [=](sycl::id<1> idx) {
#pragma unroll
                for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
                    // peel front
                    size_t start_recv = buffer_index * per_rank_count;
                    size_t start_send = rank * per_rank_count;

                    // both accesses unaligned, copy element-by-element
                    if (idx < peel_front_count[buffer_index]) {
                        DataType data = send_bufs[buffer_index][start_send + idx];
                        recv_bufs[rank][start_recv + idx] = data;
                    }
                }
            });
        }),
        out_of_order_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep);
            // start and end are properly aligned, the internals are a multiple of sycl_vec size
            // therefore, there is no remainder to handle
            cgh.parallel_for(sycl::range<1>(aligned_loop_count), [=](sycl::id<1> idx) {
#pragma unroll
                for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
                    // handle aligned data
                    size_t start_idx = idx * vec_size + peel_front_count[buffer_index];
                    size_t start_recv_idx = buffer_index * per_rank_count + start_idx;
                    size_t send_idx = rank * per_rank_count + start_idx;

                    // the read operation (remote) is aligned
                    sycl::vec<DataType, vec_size> data =
                        *static_cast<sycl::vec<DataType, vec_size> *>(
                            (void *)&send_bufs[buffer_index][send_idx]);
                // write operation (local) might be unaligned
                // write element-by-element from sycl::vec
#pragma unroll
                    for (size_t i = 0; i < vec_size; ++i) {
                        recv_bufs[rank][start_recv_idx + i] = data[i];
                    }
                }
            });
        }),
        out_of_order_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep);
            // start and end are properly aligned, the internals are a multiple of sycl_vec size
            // therefore, there is no remainder to handle
            cgh.parallel_for(sycl::range<1>(end_loop_count), [=](sycl::id<1> idx) {
#pragma unroll
                for (size_t buffer_index = 0; buffer_index < N_RANKS; ++buffer_index) {
                    // peel back
                    size_t start_idx =
                        peel_front_count[buffer_index] + aligned_loop_count * vec_size;
                    size_t start_recv = buffer_index * per_rank_count;
                    size_t start_send = rank * per_rank_count;

                    size_t loop_idx = start_idx + idx;
                    // both accesses unaligned, copy element-by-element
                    if (loop_idx < per_rank_count) {
                        DataType data = send_bufs[buffer_index][start_send + loop_idx];
                        recv_bufs[rank][start_recv + loop_idx] = data;
                    }
                }
            });
        }),
    };
}

template <size_t vec_size, typename DataType, size_t N_RANKS>
std::vector<sycl::event> alltoall_large_vec_size_impl(
    sycl::queue &queue,
    const std::array<DataType *, N_RANKS> &send_bufs,
    std::array<DataType *, N_RANKS> &recv_bufs,
    size_t count,
    size_t rank,
    sycl::event &dep) {
    switch (ccl::global_data::env().sycl_alltoall_protocol) {
        case ccl_sycl_alltoall_protocol::read: {
            return alltoall_vec_read<vec_size>(queue, send_bufs, recv_bufs, count, rank, dep);
        }
        case ccl_sycl_alltoall_protocol::write: {
            return alltoall_vec_write<vec_size>(queue, send_bufs, recv_bufs, count, rank, dep);
        }
        default: {
            CCL_THROW("unknown alltoall protocol type");
            break;
        }
    }
    CCL_THROW("unknown alltoall protocol type");
}

template <typename T, size_t N_RANKS>
ccl::event alltoall_large_impl(const void *send_buf,
                               void *recv_buf,
                               size_t count,
                               ccl::datatype dtype,
                               ccl_comm *comm,
                               ccl_stream *global_stream,
                               const ccl::vector_class<ccl::event> &deps) {
    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    auto node_comm = comm->get_node_comm();
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    sycl::queue q = global_stream->get_native_stream();

    std::vector<void *> ptrs{ (void *)send_buf, (T *)recv_buf }; // index 0 and 1
    size_t rank = node_comm->rank();

    std::array<T *, N_RANKS> send_pointers;
    std::array<T *, N_RANKS> recv_pointers;

    if (comm->is_multi_thread_instance() == true) {
        ccl::global_data::get().shared_data->do_ipc_exchangeExt(
            comm,
            ccl::global_data::get().shared_data->hash_table,
            global_stream,
            ptrs,
            comm->global_current_id);
        send_pointers = ccl::global_data::get().shared_data->get_ipc_ptrsExt<T, N_RANKS>(
            node_comm,
            ccl::global_data::get().shared_data->hash_table,
            0,
            0,
            (void *)send_buf,
            comm->global_current_id,
            0,
            comm->get_even_comm(),
            comm->get_pair_comm());
        recv_pointers = ccl::global_data::get().shared_data->get_ipc_ptrsExt<T, N_RANKS>(
            node_comm,
            ccl::global_data::get().shared_data->hash_table,
            0,
            1,
            (void *)recv_buf,
            comm->global_current_id,
            0,
            comm->get_even_comm(),
            comm->get_pair_comm());
        if (comm->is_multi_thread_instance()) {
            pthread_barrier_wait(
                &ccl::global_data::get().shared_data->barrier_waits[comm->global_current_id]);
        }
    }
    else {
        auto [sched, exchange_entry] = do_ipc_exchange(comm, global_stream, ptrs);
        send_pointers = get_ipc_ptrs<T, N_RANKS>(node_comm, 0, (void *)send_buf, sched);
        recv_pointers = get_ipc_ptrs<T, N_RANKS>(node_comm, 1, (void *)recv_buf, sched);
    }
    auto dep = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    constexpr size_t vec_size =
        std::max(static_cast<size_t>(8 / sizeof(T)), static_cast<size_t>(1));

    std::vector<sycl::event> kernel_events =
        alltoall_large_vec_size_impl<vec_size>(q, send_pointers, recv_pointers, count, rank, dep);

    sycl::event barrier_event2 = invoke_barrier(node_comm, q, kernel_events, is_cpu_barrier);

    return ccl::event::create_from_native(barrier_event2);
}

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

#include "coll/algorithms/utils/sycl_coll_base.hpp"

#define MAX_RANK            16
#define SIMD_COMPUTE_MAX    256
#define SIMD_COMPUTE        (SIMD_COMPUTE_MAX / sizeof(data_type))
#define SIMD_SYNC           32
#define UNROLL_SIZE         1
#define BUFFER_COUNT        2
#define SYNC_BYTE           (SIMD_SYNC * sizeof(int) * 2)
#define MAX_COUNT           (32 * 1024 * 1024 / sizeof(data_type))
#define ALIGNMENT_BYTE      256
#define EU_COUNT_PER_RANK   512
#define THREAD_COUNT_PER_EU 8
#define HW_THREAD_COUNT     (EU_COUNT_PER_RANK * THREAD_COUNT_PER_EU)
#define RANKS_PER_GPU       2

template <typename data_type, size_t align>
void local_copy(int idx,
                const void *send_buf,
                int myoffset,
                uint32_t size,
                int threads_already_processed,
                void *temp_buffer[],
                uint32_t temp_rank,
                int size_per_buffer_kernel,
                int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    int abs_offset_in_chunk = idx + threads_already_processed;
    int read_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;

    //do copy from input buffer to temp buffer.
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE> buffer;
#pragma unroll
    for (int i = 0; i < UNROLL_SIZE; i++) {
        buffer.template select<SIMD_COMPUTE, 1>(i * SIMD_COMPUTE) = block_load<data_type, SIMD_COMPUTE>(
            (data_type *)send_buf + myoffset + read_offset + i * SIMD_COMPUTE, properties{ alignment<align> });
    }

    // write to my rank's temp buffer's first chunk
    data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
    local_temp_ptr += size_per_buffer_kernel * buffer_index_kernel;
    local_temp_ptr += idx * SIMD_COMPUTE * UNROLL_SIZE;
#pragma unroll
    for (int i = 0; i < UNROLL_SIZE; i++) {
        block_store<data_type, SIMD_COMPUTE>(local_temp_ptr + i * SIMD_COMPUTE,
                                             buffer.template select<SIMD_COMPUTE, 1>(i * SIMD_COMPUTE),
                                             properties{ alignment<align> });
    }
    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void read_write(int *even_ranks,
                int my_rank_index,
                int idx,
                const void *send_buf,
                int myoffset,
                void *out_buffer,
                uint32_t size,
                int threads_already_processed,
                void *temp_buffer[],
                uint32_t temp_rank,
                int outer_iter,
                int size_per_buffer_kernel,
                int buffer_index_kernel,
                int chunk_size) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    // read from xelinks of all odd/even ranks
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> xe_buffer;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
        int rr = even_ranks[r];
        data_type *temp_ptr = (data_type *)temp_buffer[rr];
        temp_ptr += size_per_buffer_kernel * buffer_index_kernel;
        temp_ptr += idx * SIMD_COMPUTE * UNROLL_SIZE;
#pragma unroll
        for (int i = 0; i < UNROLL_SIZE; i++) {
            xe_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE) =
                block_load<data_type, SIMD_COMPUTE>(temp_ptr + i * SIMD_COMPUTE, properties{ alignment<align> });
        }
    }

    // write to mdfi buffer
    data_type *mdfi_buffer = (data_type *)temp_buffer[temp_rank ^ 1];
    mdfi_buffer += size_per_buffer_kernel * buffer_index_kernel;
    mdfi_buffer += chunk_size + idx * SIMD_COMPUTE * UNROLL_SIZE;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                mdfi_buffer + r * chunk_size + i * SIMD_COMPUTE,
                xe_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE),
                properties{ alignment<align> });
        }
    }

    // write to output buffer
    data_type *out_ptr = (data_type *)out_buffer;
    int abs_offset_in_chunk = idx + threads_already_processed;
    int write_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    bool in_place = send_buf == (data_type *)out_buffer + temp_rank * size;
    if (write_offset + SIMD_COMPUTE * UNROLL_SIZE <= size) {
        //#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            if ((int)r == my_rank_index && in_place) {
                continue;
            }
            int rr = even_ranks[r];
#pragma unroll
            for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
                block_store<data_type, SIMD_COMPUTE>(
                    out_ptr + rr * size + write_offset + i * SIMD_COMPUTE,
                    xe_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE),
                    properties{ alignment<align> });
            }
        }
    }
    else {
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            if ((int)r == my_rank_index && in_place) {
                continue;
            }
            int rr = even_ranks[r];
#if 0
            int count = (size - write_offset + SIMD_COMPUTE - 1) / SIMD_COMPUTE;
            for (int i = 0; i < count; i++)
            {
                lsc_block_store<data_type, SIMD_COMPUTE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                    (out_ptr + rr * size + write_offset + i * SIMD_COMPUTE, xe_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE));
            }
#else
            int count = size - write_offset;
            for (int i = 0; i < count; i++) {
                out_ptr[rr * size + write_offset + i] = xe_buffer[r * SIMD_COMPUTE * UNROLL_SIZE + i];
            }
#endif
        }
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void write_output(int *even_ranks,
                  int my_rank_index,
                  int idx,
                  void *out_buffer,
                  uint32_t size,
                  int threads_already_processed,
                  void *temp_buffer[],
                  uint32_t temp_rank,
                  int outer_iter,
                  int size_per_buffer_kernel,
                  int buffer_index_kernel,
                  int chunk_size) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> buffer;
    data_type *read_ptr = (data_type *)temp_buffer[temp_rank];
    read_ptr += size_per_buffer_kernel * buffer_index_kernel;
    read_ptr += chunk_size + idx * SIMD_COMPUTE * UNROLL_SIZE;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
#pragma unroll
        for (int i = 0; i < UNROLL_SIZE; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE) =
                block_load<data_type, SIMD_COMPUTE>(read_ptr + i * SIMD_COMPUTE, properties{ alignment<align> });
        }
        read_ptr += chunk_size;
    }

    data_type *write_ptr = (data_type *)out_buffer;
    int abs_offset_in_chunk = idx + threads_already_processed;
    int write_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    if (write_offset + SIMD_COMPUTE * UNROLL_SIZE <= size) {
        //#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            int rr = even_ranks[r] ^ 1;
            //sycl::_V1::ext::oneapi::experimental::printf("write_output [%d] write to %d\n", temp_rank, rr);
#pragma unroll
            for (int i = 0; i < UNROLL_SIZE; i++) {
                block_store<data_type, SIMD_COMPUTE>(
                    write_ptr + rr * size + write_offset + i * SIMD_COMPUTE,
                    buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE),
                    properties{ alignment<align> });
            }
        }
    }
    else {
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            int rr = even_ranks[r] ^ 1;
#if 0
            int count = (size - write_offset + SIMD_COMPUTE - 1) / SIMD_COMPUTE;
            for (int i = 0; i < count; i++)
            {
                lsc_block_store<data_type, SIMD_COMPUTE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                    (write_ptr + rr * size + write_offset + i * SIMD_COMPUTE, buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE));
            }
#else
            int count = size - write_offset;
            for (int i = 0; i < count; i++) {
                write_ptr[rr * size + write_offset + i] = buffer[r * SIMD_COMPUTE * UNROLL_SIZE + i];
            }
#endif
        }
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void nocopy_read_write(int *even_ranks,
                       int my_rank_index,
                       int idx,
                       void **send_bufs,
                       void **out_buffers,
                       uint32_t size,
                       uint32_t temp_rank) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    int offset = idx * SIMD_COMPUTE * UNROLL_SIZE;

    //sycl::ext::intel::esimd::properties props{alignment<sizeof(T)>};
    //sycl::ext::oneapi::experimental::detail::empty_properties_t  PropertyListT;
    //constexpr size_t Alignment = sycl::ext::oneapi::experimental::detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(data_type));
    //constexpr size_t Alignment = 2;
    //sycl::_V1::ext::intel::esimd::properties props{cache_hint_L1<cached>};
    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::flushl3, lsc_scope::gpus>();
    // read from xelinks of all odd/even ranks
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> in_buffer;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
        int r0 = (my_rank_index + r) % (TEMP_WORLD / 2);
        int rr = even_ranks[r0];
        data_type *read_ptr = (data_type *)send_bufs[rr];
        read_ptr += offset;
#pragma unroll
        for (int i = 0; i < UNROLL_SIZE; i++) {
            in_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE) =
                block_load<data_type, SIMD_COMPUTE>(read_ptr + i * SIMD_COMPUTE, properties{ alignment<align> });
        }
    }

    // write to mdfi buffer and output buffer
    data_type *mdfi_buffer = (data_type *)out_buffers[temp_rank ^ 1];
    data_type *out_ptr = (data_type *)out_buffers[temp_rank];
    int rank_in_place = send_bufs[temp_rank] == ((data_type *)out_buffers[temp_rank] + temp_rank * size);
    int mdfi_in_place =
        send_bufs[temp_rank ^ 1] == ((data_type *)out_buffers[temp_rank ^ 1] + (temp_rank ^ 1) * size);
    if (offset + SIMD_COMPUTE * UNROLL_SIZE <= size) {
        //#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            uint32_t r0 = (my_rank_index + r) % (TEMP_WORLD / 2);
            uint32_t rr = even_ranks[r0];
#pragma unroll
            for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
                if (rr != (temp_rank ^ 1) || !mdfi_in_place) {
                    block_store<data_type, SIMD_COMPUTE>(mdfi_buffer + rr * size + offset + i * SIMD_COMPUTE,
                                                         in_buffer.template select<SIMD_COMPUTE, 1>(
                                                             r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE),
                                                         properties{ alignment<align> });
                }
                if (rr != temp_rank || !rank_in_place) {
                    block_store<data_type, SIMD_COMPUTE>(out_ptr + rr * size + offset + i * SIMD_COMPUTE,
                                                         in_buffer.template select<SIMD_COMPUTE, 1>(
                                                             r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE),
                                                         properties{ alignment<align> });
                }
            }
        }
    }
    else {
        int vc_count = (size - offset) / SIMD_COMPUTE;
        int count = size - offset - SIMD_COMPUTE * vc_count;
        //sycl::_V1::ext::oneapi::experimental::printf("offset [%d] size %d vc_count:%d count:%d \n", offset, size, vc_count, count);
        //#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            uint32_t r0 = (my_rank_index + r) % (TEMP_WORLD / 2);
            uint32_t rr = even_ranks[r0];
#if 0
            int count = (size - offset + SIMD_COMPUTE - 1) / SIMD_COMPUTE;
            for (int i = 0; i < count; i++)
            {
                lsc_block_store<data_type, SIMD_COMPUTE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                    (mdfi_buffer + rr * size + offset + i * SIMD_COMPUTE, in_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE));
                lsc_block_store<data_type, SIMD_COMPUTE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                    (out_ptr + rr * size + offset + i * SIMD_COMPUTE, in_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE));
            }
#else
            for (int i = 0; i < vc_count; i++) {
                if (rr != (temp_rank ^ 1) || !mdfi_in_place) {
                    block_store<data_type, SIMD_COMPUTE>(mdfi_buffer + rr * size + offset + i * SIMD_COMPUTE,
                                                         in_buffer.template select<SIMD_COMPUTE, 1>(
                                                             r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE),
                                                         properties{ alignment<align> });
                }
                if (rr != temp_rank || !rank_in_place) {
                    block_store<data_type, SIMD_COMPUTE>(out_ptr + rr * size + offset + i * SIMD_COMPUTE,
                                                         in_buffer.template select<SIMD_COMPUTE, 1>(
                                                             r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE),
                                                         properties{ alignment<align> });
                }
            }
            for (int i = 0; i < count; i++) {
                if (rr != (temp_rank ^ 1) || !mdfi_in_place) {
                    mdfi_buffer[rr * size + offset + vc_count * SIMD_COMPUTE + i] =
                        in_buffer[r * SIMD_COMPUTE * UNROLL_SIZE + vc_count * SIMD_COMPUTE + i];
                }
                if (rr != temp_rank || !rank_in_place) {
                    out_ptr[rr * size + offset + vc_count * SIMD_COMPUTE + i] =
                        in_buffer[r * SIMD_COMPUTE * UNROLL_SIZE + vc_count * SIMD_COMPUTE + i];
                }
            }
#endif
        }
    }
    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::flushl3, lsc_scope::system>();
}

//constexpr sycl::specialization_id<uint32_t> temp_world_const;

template <typename dtype, size_t align>
class AllgatherMediumKernel_local_copy;
template <typename dtype, size_t align>
class AllgatherMediumKernel_read_write;
template <typename dtype, size_t align>
class AllgatherMediumKernel_write_output;

template <typename dtype>
class AllgatherMediumKernel_nocopy_read_write;

template <typename dtype>
class AllgathervMediumKernel_GlobalSync;
template <typename dtype>
class AllgathervMediumKernel_LocalSync;

template <typename data_type, uint32_t max_rank = MAX_RANK>
class sycl_allgatherv_medium : public sycl_coll_base<data_type> {
public:
    sycl_allgatherv_medium() : sycl_coll_base<data_type>() {
        buffer_index = 0;
        size_per_buffer = 0;
    }

    void init(sycl::queue &queue, ccl_comm *comm_in, ccl_stream *stream, uint32_t rank_in, uint32_t world_in) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        rank = rank_in;
        world = world_in;
        max_count_per_rank = (MAX_COUNT + SIMD_COMPUTE * UNROLL_SIZE - 1) / (SIMD_COMPUTE * UNROLL_SIZE) *
                             SIMD_COMPUTE * UNROLL_SIZE;
        data_size_per_buffer = max_count_per_rank * (world / 2 + 1);
        size_per_buffer = data_size_per_buffer * sizeof(data_type) + SYNC_BYTE;

        void *local_triple_buffer = sycl::malloc_device(size_per_buffer * BUFFER_COUNT, queue);
        auto e = queue.memset(local_triple_buffer, 0, size_per_buffer * BUFFER_COUNT);
        e.wait();

        this->exchange_peer_ipc_mem(queue,
                                    comm_in,
                                    stream,
                                    local_triple_buffer,
                                    NULL,
                                    rank,
                                    world,
                                    data_size_per_buffer * sizeof(data_type),
                                    (void **)buffers,
                                    (void **)sync_buffer,
                                    offsets,
                                    ipc_handle,
                                    NULL,
                                    NULL /* mmap_buffers */,
                                    false /* to_cache */);
        this->initialized = true;

        global_stream = stream;
        comm = comm_in->get_node_comm().get();
        even_comm = comm_in->get_even_comm().get();
    }

    ccl::event allgatherv(sycl::queue &queue,
                          const void *send_buf,
                          size_t send_count,
                          void *recv_buf,
                          const ccl::vector_class<size_t> &recv_counts,
                          bool &done) {
        done = true;

        // check local alignment
        size_t is_aligned =
            (size_t)send_buf % 4 == 0 && (size_t)recv_buf % 4 == 0 && (send_count * sizeof(data_type)) % 4 == 0;

        if (ccl::global_data::env().sycl_allgatherv_tmp_buf) {
            if (is_aligned) {
                return allgatherv_copy<4>(queue, send_buf, send_count, recv_buf, recv_counts);
            }
            else {
                return allgatherv_copy<2>(queue, send_buf, send_count, recv_buf, recv_counts);
            }
        }
        else {
            return allgatherv_nocopy(queue, send_buf, send_count, recv_buf, recv_counts);
        }
    }

private:
    template <size_t align>
    ccl::event allgatherv_copy(sycl::queue &queue,
                               const void *send_buf,
                               size_t send_count,
                               void *recv_buf,
                               const ccl::vector_class<size_t> &recv_counts) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        assert(this->initialized == true);

        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }

        uint32_t myoffset = 0;
        if (send_buf == recv_buf)
            myoffset = send_count * temp_rank;

        int even_ranks[max_rank];
        int myrank;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank)
                myrank = i;
            //printf("even rank %d: %d neighbor: %d\n", i, even_ranks[i], even_ranks[i] ^ 1);
        }

        int chunk_size = max_count_per_rank;
        int buffer_index_kernel = buffer_index;

        int size_per_buffer_kernel __attribute__((unused)) = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel __attribute__((unused)) =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));

        int threads_already_processed = 0;
        //uint32_t total_threads_needed_sync __attribute__((unused)) = 1;
        int outerloop_iter_count = (send_count + max_count_per_rank - 1) / max_count_per_rank;

        for (int outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++) {
            uint32_t threads_needed_per_chunk;
            uint32_t total_threads_needed __attribute__((unused));
            if ((outer_iter + 1) * max_count_per_rank < (int)send_count) {
                threads_needed_per_chunk = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);
            }
            else {
                uint32_t leftover = send_count - outer_iter * max_count_per_rank;
                threads_needed_per_chunk =
                    (leftover + SIMD_COMPUTE * UNROLL_SIZE - 1) / (SIMD_COMPUTE * UNROLL_SIZE);
            }
            int wg_size __attribute__((unused)) = 1;
            total_threads_needed = threads_needed_per_chunk;

            int innerloop_iter_count __attribute__((unused)) =
                (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;

            uint32_t persist_threads_needed = total_threads_needed;
            if (persist_threads_needed > HW_THREAD_COUNT)
                persist_threads_needed = HW_THREAD_COUNT;

            // FIRST KERNEL
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class AllgatherMediumKernel_local_copy<data_type, align>>(
                    sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                    {
                    uint32_t idx = idx2.get_global_id();
                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                        int index = idx + inner_iter * HW_THREAD_COUNT;
                        if ((uint32_t)index >= total_threads_needed)
                            break;

                        local_copy<data_type, align>(index,
                                                     send_buf,
                                                     myoffset,
                                                     send_count,
                                                     threads_already_processed,
                                                     (void **)temp_buffer,
                                                     temp_rank,
                                                     size_per_buffer_kernel,
                                                     buffer_index_kernel);
                    }
                    });
            });

            e = global_sync(
                queue, temp_rank, temp_world, size_per_buffer_for_sync_kernel * buffer_index_kernel, 1, 0);

            // SECOND KERNEL
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class AllgatherMediumKernel_read_write<data_type, align>>(
                    sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                    {
                    //ESIMD kernel
                    uint32_t idx = idx2.get_global_id();
                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                        int index = idx + inner_iter * HW_THREAD_COUNT;
                        if ((uint32_t)index >= total_threads_needed)
                            break;

                        auto read_write_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                            read_write<w, data_type, a>((int *)even_ranks,
                                                        myrank,
                                                        index,
                                                        send_buf,
                                                        myoffset,
                                                        recv_buf,
                                                        send_count,
                                                        threads_already_processed,
                                                        (void **)temp_buffer,
                                                        temp_rank,
                                                        outer_iter,
                                                        size_per_buffer_kernel,
                                                        buffer_index_kernel,
                                                        chunk_size);
                        };
                        invoke_esimd_function<align>(read_write_lambda, temp_world);
                    }
                });
            });

            e = global_sync(
                queue, temp_rank, temp_world, size_per_buffer_for_sync_kernel * buffer_index_kernel, 2, 1);

            // THIRD KERNEL
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class AllgatherMediumKernel_write_output<data_type, align>>(
                    sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                    {
                    uint32_t idx = idx2.get_global_id();
                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                        int index = idx + inner_iter * HW_THREAD_COUNT;
                        if ((uint32_t)index >= total_threads_needed)
                            break;

                        auto write_output_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                            write_output<w, data_type, a>((int *)even_ranks,
                                                          myrank,
                                                          index,
                                                          recv_buf,
                                                          send_count,
                                                          threads_already_processed,
                                                          (void **)temp_buffer,
                                                          temp_rank,
                                                          outer_iter,
                                                          size_per_buffer_kernel,
                                                          buffer_index_kernel,
                                                          chunk_size);
                        };
                        invoke_esimd_function<align>(write_output_lambda, temp_world);
                    }
                });
            });
            //e.wait();

            threads_already_processed += total_threads_needed;
            buffer_index++;
            buffer_index %= BUFFER_COUNT;
            buffer_index_kernel = buffer_index;
        } // end of for outer_iter

        return ccl::event::create_from_native(e);
    }

    ccl::event allgatherv_nocopy(sycl::queue &queue,
                                 const void *send_buf,
                                 size_t send_count,
                                 void *recv_buf,
                                 const ccl::vector_class<size_t> &recv_counts) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        assert(this->initialized == true);

        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }

        int even_ranks[max_rank];
        int myrank;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank)
                myrank = i;
            //printf("[%d] even rank %d: %d neighbor: %d\n", temp_rank, i, even_ranks[i], even_ranks[i] ^ 1);
        }

        void *in_buffers[max_rank];
        void *out_buffers[max_rank];
        LOG_DEBUG("No-copy kernel calling exchange_peer_ipc_mem");
        //printf("[%d] before exchange: %p %p \n", rank, send_buf, recv_buf);
        this->exchange_peer_ipc_mem(queue,
                                    comm,
                                    global_stream,
                                    (void **)send_buf,
                                    recv_buf,
                                    rank,
                                    world,
                                    0,
                                    (void **)in_buffers,
                                    NULL,
                                    NULL,
                                    NULL,
                                    (void **)out_buffers);
        int align4 = all_aligned((void **)in_buffers, temp_world, send_count, sizeof(data_type), 4) &&
                     all_aligned((void **)out_buffers, temp_world, send_count, sizeof(data_type), 4);

        int buffer_index_kernel __attribute__((unused)) = buffer_index;
        int size_per_buffer_kernel __attribute__((unused)) = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel __attribute__((unused)) =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));

        //int threads_already_processed = 0;
        uint32_t total_threads_needed_sync __attribute__((unused)) = 1;
        //outerloop_iter_count = (send_count + max_count_per_rank - 1) / max_count_per_rank;

        uint32_t threads_needed_per_chunk;
        uint32_t total_threads_needed __attribute__((unused));
        threads_needed_per_chunk = (send_count + SIMD_COMPUTE * UNROLL_SIZE - 1) / (SIMD_COMPUTE * UNROLL_SIZE);
        int wg_size __attribute__((unused)) = 1;
        total_threads_needed = threads_needed_per_chunk;

        int innerloop_iter_count __attribute__((unused)) =
            (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;

        //        uint32_t total_threads_dispatched = (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        //        uint32_t total_wg_count = total_threads_dispatched / wg_size;

        uint32_t persist_threads_needed = total_threads_needed;
        if (persist_threads_needed > HW_THREAD_COUNT)
            persist_threads_needed = HW_THREAD_COUNT;

            //printf("innerloop_iter_count: %d total_threads_needed:%d persist_threads_needed:%d \n", innerloop_iter_count, total_threads_needed, persist_threads_needed);

#if 0
        persist_threads_needed = (persist_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = persist_threads_needed / wg_size;
#endif

        // a GPU barrier to make sure all ranks are ready
        e = global_sync(queue, temp_rank, temp_world, size_per_buffer_for_sync_kernel * buffer_index_kernel, 0, 0);

        e = queue.submit([&](sycl::handler &cgh) {
            // Set the coefficient of the convolution as constant.
            // This will build a specific kernel the coefficient available as literals.
            //            cgh.set_specialization_constant<temp_world_const>(temp_world);
            cgh.parallel_for<class AllgatherMediumKernel_nocopy_read_write<data_type>>(
                sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                {
                uint32_t idx = idx2.get_global_id();

                //uint32_t temp_world_kernel = h.get_specialization_constant<temp_world_const>();

                //ESIMD kernel
                for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                    int index = idx + inner_iter * HW_THREAD_COUNT;
                    if ((uint32_t)index >= total_threads_needed)
                        break;

                    auto nocopy_read_write_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                        nocopy_read_write<w, data_type, a>((int *)even_ranks,
                                                           myrank,
                                                           index,
                                                           (void **)in_buffers,
                                                           (void **)out_buffers,
                                                           send_count,
                                                           temp_rank);
                    };
                    if (align4)
                        invoke_esimd_function<4>(nocopy_read_write_lambda, temp_world);
                    else
                        invoke_esimd_function<2>(nocopy_read_write_lambda, temp_world);
                }
            });
        });

#if 1
        e = global_sync(queue, temp_rank, temp_world, size_per_buffer_for_sync_kernel * buffer_index_kernel, 1, 1);
#else
        // sync two tiles of a same GPU before exiting the call
        e = local_sync(queue, temp_rank, temp_world, size_per_buffer_for_sync_kernel * buffer_index_kernel, 1, 1);
#endif

        //threads_already_processed += total_threads_needed;
        buffer_index++;
        buffer_index %= BUFFER_COUNT;
        //buffer_index_kernel = buffer_index;

        return ccl::event::create_from_native(e);
    }

    //sync all the ranks here before consuming the results.
    // offset = size_per_buffer_for_sync_kernel * buffer_index_kernel
    sycl::event global_sync(sycl::queue &queue,
                            int temp_rank,
                            uint32_t temp_world,
                            int offset,
                            int index,
                            int reset) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        void *temp_sync_buffer[max_rank];
        for (uint32_t i = 0; i < temp_world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }
        sycl::event e;
        uint32_t total_threads_needed_sync = 1;
        int wg_size = 1;
        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class AllgathervMediumKernel_GlobalSync<data_type>>(
                sycl::nd_range<1>({ total_threads_needed_sync }, wg_size), [=](sycl::nd_item<1> idx) SYCL_ESIMD_KERNEL
                {
                //ESIMD kernel
                simd<ushort, SIMD_SYNC> ramp;
#pragma unroll
                for (uint32_t i = 0; i < SIMD_SYNC; i++) {
                    ramp[i] = i * sizeof(int);
                }

                //since other ranks might still be doing local_sum, we need to sync ranks here.
                //After the sync is done, the second half of hte temp buffer will be replaced with new sum val.
                simd_mask<SIMD_SYNC> pred;
                simd<int, SIMD_SYNC> status0;
                pred = false;
                pred[index] = true;

                //sync .
                for (uint32_t i = 0; i < temp_world; i++) {
                    int *sync_ptr = (int *)temp_sync_buffer[i] + offset;
                    ////never true. Used to force dependecy with prev kernel
                    //if (total_threads_needed_sync == 0x7fffffff)
                    //    sync_ptr = temp_buffer[0];
                    lsc_atomic_update<atomic_op::inc,
                                      int,
                                      SIMD_SYNC,
                                      lsc_data_size::default_size,
                                      cache_hint::none,
                                      cache_hint::none>(sync_ptr, ramp, pred);
                }

                //wait for all the local TG to sync. Then sync the other remote GPUs
                int *sync_ptr = (int *)temp_sync_buffer[temp_rank] + offset;
                status0 = lsc_atomic_update<atomic_op::load,
                                            int,
                                            SIMD_SYNC,
                                            lsc_data_size::default_size,
                                            cache_hint::none,
                                            cache_hint::none>(sync_ptr, ramp, pred);
                while (status0[index] != temp_world) {
                    status0 = lsc_atomic_update<atomic_op::load,
                                                int,
                                                SIMD_SYNC,
                                                lsc_data_size::default_size,
                                                cache_hint::none,
                                                cache_hint::none>(sync_ptr, ramp, pred);
                }

                if (reset) {
                    //init the atomic counter to 0 for the next run
                    status0 = 0;
                    pred = true;
                    lsc_atomic_update<atomic_op::store,
                                      int,
                                      SIMD_SYNC,
                                      lsc_data_size::default_size,
                                      cache_hint::none,
                                      cache_hint::none>(
                        sync_ptr, ramp, status0, pred); //initialize the counter for the next run
                }
                });//parallel_for
        }); //submit()
        return e;
    }

    // sync tiles in a GPU
    sycl::event local_sync(sycl::queue queue,
                           int temp_rank,
                           uint32_t temp_world,
                           int offset,
                           int index,
                           int reset) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }
        sycl::event e;
        uint32_t total_threads_needed_sync = 1;
        int wg_size = 1;

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class AllgathervMediumKernel_LocalSync<data_type>>(
                sycl::nd_range<1>({ total_threads_needed_sync }, wg_size), [=](sycl::item<1> idx) SYCL_ESIMD_KERNEL
                {
                //ESIMD kernel
                simd<ushort, SIMD_SYNC> ramp;
#pragma unroll
                for (uint32_t i = 0; i < SIMD_SYNC; i++) {
                    ramp[i] = i * sizeof(int);
                }

                //sync only the rank pair within the same gpu.
                simd_mask<SIMD_SYNC> pred;
                simd<int, SIMD_SYNC> status0;
                pred = false;
                pred[index] = true;

                //sync .
                int *sync_ptr = (int *)temp_sync_buffer[temp_rank ^ 1] + offset;
                lsc_atomic_update<atomic_op::inc,
                                  int,
                                  SIMD_SYNC,
                                  lsc_data_size::default_size,
                                  cache_hint::none,
                                  cache_hint::none>(sync_ptr, ramp, pred);
                sync_ptr = (int *)temp_sync_buffer[temp_rank] + offset;
                lsc_atomic_update<atomic_op::inc,
                                  int,
                                  SIMD_SYNC,
                                  lsc_data_size::default_size,
                                  cache_hint::none,
                                  cache_hint::none>(sync_ptr, ramp, pred);

                //wait for all the local TG to sync. Then sync the other remote GPUs
                status0 = lsc_atomic_update<atomic_op::load,
                                            int,
                                            SIMD_SYNC,
                                            lsc_data_size::default_size,
                                            cache_hint::none,
                                            cache_hint::none>(sync_ptr, ramp, pred);
                while (status0[index] != RANKS_PER_GPU) {
                    status0 = lsc_atomic_update<atomic_op::load,
                                                int,
                                                SIMD_SYNC,
                                                lsc_data_size::default_size,
                                                cache_hint::none,
                                                cache_hint::none>(sync_ptr, ramp, pred);
                }
                if (reset) {
                    //init the atomic counter to 0 for the next run
                    status0 = 0;
                    pred = true;
                    lsc_atomic_update<atomic_op::store,
                                      int,
                                      SIMD_SYNC,
                                      lsc_data_size::default_size,
                                      cache_hint::none,
                                      cache_hint::none>(
                        sync_ptr, ramp, status0, pred); //initialize the counter for the next run
                }
                });//parallel_for
        }); //submit()
        return e;
    }

    void release(sycl::queue &queue) {
        // Clean up, close/put ipc handles, free memory, etc.
        auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
        for (int i = 0; i < world; i++) {
            if (i != rank) {
                ZE_CALL(zeMemCloseIpcHandle, (l0_ctx, (char *)buffers[i] - offsets[i]));
            }
        }

        sycl::free(buffers[rank], queue);
        this->initialized = false;
    }

private:
    void *buffers[max_rank];
    void *sync_buffer[max_rank];
    size_t offsets[max_rank];
    ze_ipc_mem_handle_t ipc_handle[max_rank];
    int rank{ ccl::utils::invalid_rank }, world{ ccl::utils::invalid_err_code };
    int buffer_index{ ccl::utils::invalid_err_code };
    int size_per_buffer{ ccl::utils::invalid_bytes_value };
    int max_count_per_rank{ ccl::utils::initial_count_value };
    int data_size_per_buffer{ ccl::utils::invalid_bytes_value };
    ccl_stream *global_stream{};
    ccl_comm *comm{};
    ccl_comm *even_comm{};
};

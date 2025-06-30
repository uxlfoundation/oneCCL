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
#define SYNC_BYTE           (SIMD_SYNC * sizeof(int) * 2)
#define ALIGNMENT_BYTE      256
#define MAX_COUNT           (16 * 1024 * 1024 / sizeof(data_type))
#define EU_COUNT_PER_RANK   512
#define THREAD_COUNT_PER_EU 8
#define HW_THREAD_COUNT     (EU_COUNT_PER_RANK * THREAD_COUNT_PER_EU)
#define RANKS_PER_GPU       2
#define NO_KERNEL           0
#define FIRST_KERNEL        1
#define SECOND_KERNEL       2
#define THIRD_KERNEL        4

#define NOCOPY_KERNEL_NUM   2
#define NOCOPY_BUFFER_COUNT NOCOPY_KERNEL_NUM
#define NOCOPY_LAST_KERNEL  SECOND_KERNEL

#define COPY_KERNEL_NUM   3
#define COPY_BUFFER_COUNT COPY_KERNEL_NUM
#define COPY_LAST_KERNEL  THIRD_KERNEL

#define RUN_SECOND_KERNEL \
    if (sw_pipeline_kernel_state[ii] & FIRST_KERNEL) { \
        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) { \
            int index = idx + inner_iter * HW_THREAD_COUNT; \
            if ((uint32_t)index >= total_threads_needed) \
                break; \
            auto nocopy_reduce_read_write_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL { \
                nocopy_reduce_read_write<w, data_type, a>((int *)even_ranks, \
                                                          my_rank_index, \
                                                          index, \
                                                          (void **)in_buffers, \
                                                          out_buffer, \
                                                          recv_size, \
                                                          threads_already_processed[ii], \
                                                          (void **)temp_buffer, \
                                                          temp_rank, \
                                                          outer_iter, \
                                                          size_per_buffer_kernel, \
                                                          ii, \
                                                          threads_needed_per_chunk); \
            }; \
            if (align4) \
                invoke_esimd_function<4>(nocopy_reduce_read_write_lambda, temp_world); \
            else \
                invoke_esimd_function<2>(nocopy_reduce_read_write_lambda, temp_world); \
        } \
    } // end of if SECOND_KERNEL

#define RUN_THIRD_KERNEL \
    if (sw_pipeline_kernel_state[ii] & SECOND_KERNEL) { \
        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) { \
            int index = idx + inner_iter * HW_THREAD_COUNT; \
            if ((uint32_t)index >= total_threads_needed) \
                break; \
            auto all_sum_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL { \
                all_sum<w, data_type, a>(index, \
                                         send_buf, \
                                         out_buffer, \
                                         recv_size, \
                                         threads_already_processed[ii], \
                                         (void **)temp_buffer, \
                                         temp_rank, \
                                         outer_iter, \
                                         size_per_buffer_kernel, \
                                         ii, \
                                         threads_needed_per_chunk); \
            }; \
            if (align4) \
                invoke_esimd_function<4>(all_sum_lambda, temp_world); \
            else \
                invoke_esimd_function<2>(all_sum_lambda, temp_world); \
        } \
    } // end of if THIRD_KERNEL

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void nocopy_reduce_read_write(int *even_ranks,
                              int my_rank_index,
                              int idx,
                              void **in_buffers,
                              void *out_buffer,
                              size_t recv_size,
                              int threads_already_processed,
                              void *temp_buffer[],
                              uint32_t temp_rank,
                              int outer_iter,
                              int size_per_buffer_kernel,
                              int buffer_index_kernel,
                              uint32_t threads_needed_per_chunk) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    size_t chunk_size = threads_needed_per_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    size_t abs_offset_in_chunk = idx + threads_already_processed;
    size_t read_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    if (read_offset >= recv_size)
        return;
    data_type *mdfi_ptr = (data_type *)in_buffers[temp_rank ^ 1];
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> mdfi_buffer;
    data_type *local_ptr = (data_type *)in_buffers[temp_rank];
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> local_buffer;
    if (read_offset + UNROLL_SIZE * SIMD_COMPUTE <= recv_size) {
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            int rr = even_ranks[r]; // even rank copies odd chunks
#pragma unroll
            for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
                mdfi_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i) =
                    block_load<data_type, SIMD_COMPUTE>(mdfi_ptr + rr * recv_size + read_offset + i * SIMD_COMPUTE,
                                                        properties{ alignment<align> });
                local_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i) =
                    block_load<data_type, SIMD_COMPUTE>(
                        local_ptr + rr * recv_size + read_offset + i * SIMD_COMPUTE,
                        properties{ alignment<align> });
            }
        }
    }
    else {
        int count = recv_size - read_offset;
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            int rr = even_ranks[r]; // even rank copies odd chunks
            for (uint32_t i = 0; i < count; i++) {
                mdfi_buffer[r * SIMD_COMPUTE * UNROLL_SIZE + i] = mdfi_ptr[rr * recv_size + read_offset + i];
                local_buffer[r * SIMD_COMPUTE * UNROLL_SIZE + i] = local_ptr[rr * recv_size + read_offset + i];
            }
        }
    }

    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> sum;
    if (even_ranks[0] == 0) {
#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            sum.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE) =
                local_buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE) +
                mdfi_buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE);
        }
    }
    else {
#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            sum.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE) =
                mdfi_buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE) +
                local_buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE);
        }
    }

    //store the result to the buffer
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
        int rr = even_ranks[r];
        data_type *write_ptr = (data_type *)temp_buffer[rr];
        write_ptr += size_per_buffer_kernel * buffer_index_kernel;
        size_t out_offset = (temp_rank / 2) * chunk_size + idx * SIMD_COMPUTE * UNROLL_SIZE;
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            //save the all sum in the second half of the temp slot
            block_store<data_type, SIMD_COMPUTE>(
                write_ptr + out_offset + i * SIMD_COMPUTE,
                sum.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void local_copy(int *even_ranks,
                int idx,
                const void *in_buffer,
                size_t recv_size,
                int threads_already_processed,
                void *temp_buffer[],
                uint32_t temp_rank,
                int outer_iter,
                int size_per_buffer_kernel,
                int buffer_index_kernel,
                uint32_t threads_needed_per_chunk) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    size_t chunk_size = threads_needed_per_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    size_t abs_offset_in_chunk = idx + threads_already_processed;
    // even rank copies odd chunks
    size_t read_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;

    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> buffer;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
        int rr = even_ranks[r] ^ 1; // even rank copies odd chunks
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)in_buffer + rr * recv_size + read_offset + i * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
    }

    // write to myrank's second half of the temp buffer
    data_type *ptr = (data_type *)temp_buffer[temp_rank];
    ptr += size_per_buffer_kernel * buffer_index_kernel;
    ptr += chunk_size * TEMP_WORLD / 2 + idx * SIMD_COMPUTE * UNROLL_SIZE;
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                ptr + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + i * SIMD_COMPUTE),
                properties{ alignment<align> });
        }
        ptr += chunk_size;
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void reduce_read_write(int *even_ranks,
                       int my_rank_index,
                       int idx,
                       const void *in_buffer,
                       void *out_buffer,
                       size_t recv_size,
                       int threads_already_processed,
                       void *temp_buffer[],
                       uint32_t temp_rank,
                       int outer_iter,
                       int size_per_buffer_kernel,
                       int buffer_index_kernel,
                       uint32_t threads_needed_per_chunk,
                       int in_place) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    size_t chunk_size = threads_needed_per_chunk * SIMD_COMPUTE * UNROLL_SIZE;

    data_type *mdfi_ptr = (data_type *)temp_buffer[temp_rank ^ 1];
    mdfi_ptr += size_per_buffer_kernel * buffer_index_kernel;
    int mdfi_offset = chunk_size * TEMP_WORLD / 2 + idx * SIMD_COMPUTE * UNROLL_SIZE;
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> mdfi_buffer;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            mdfi_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>((data_type *)mdfi_ptr + mdfi_offset + i * SIMD_COMPUTE,
                                                    properties{ alignment<align> });
        }
        mdfi_ptr += chunk_size;
    }

    size_t abs_offset_in_chunk = idx + threads_already_processed;
    size_t read_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> local_buffer;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
        int rr = even_ranks[r];
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            local_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)in_buffer + rr * recv_size + read_offset + i * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
    }

    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> sum;
    if (even_ranks[0] == 0) {
#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            sum.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE) =
                local_buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE) +
                mdfi_buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE);
        }
    }
    else {
#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            sum.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE) =
                mdfi_buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE) +
                local_buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE);
        }
    }

    //store the result to the first half of the buffer
    if (TEMP_WORLD > 2 || in_place) {
        //#pragma unroll
        for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
            int rr = even_ranks[r];
            data_type *write_ptr = (data_type *)temp_buffer[rr];
            write_ptr += size_per_buffer_kernel * buffer_index_kernel;
            int out_offset = (temp_rank / 2) * chunk_size + idx * SIMD_COMPUTE * UNROLL_SIZE;
#pragma unroll
            for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
                //save the all sum in the second half of the temp slot
                block_store<data_type, SIMD_COMPUTE>(
                    write_ptr + out_offset + i * SIMD_COMPUTE,
                    sum.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i),
                    properties{ alignment<align> });
            }
        }
    }
    else {
        // directly write to output
        data_type *write_ptr = (data_type *)out_buffer;
        write_ptr += (idx + threads_already_processed) * SIMD_COMPUTE * UNROLL_SIZE;
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            //save the all sum in the second half of the temp slot
            block_store<data_type, SIMD_COMPUTE>(write_ptr + i * SIMD_COMPUTE,
                                                 sum.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                                                 properties{ alignment<align> });
        }
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void all_sum(int idx,
             const void *in_buffer,
             void *out_buffer,
             size_t recv_size,
             int threads_already_processed,
             void *temp_buffer[],
             uint32_t temp_rank,
             int outer_iter,
             int size_per_buffer_kernel,
             int buffer_index_kernel,
             uint32_t threads_needed_per_chunk) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    size_t chunk_size = threads_needed_per_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    //read the input data
    data_type *ptr = (data_type *)temp_buffer[temp_rank];
    ptr += size_per_buffer_kernel * buffer_index_kernel;
    size_t read_offset = idx * SIMD_COMPUTE * UNROLL_SIZE;
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> buffer;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * UNROLL_SIZE * r + i * SIMD_COMPUTE) =
                lsc_block_load<data_type,
                               SIMD_COMPUTE,
                               lsc_data_size::default_size,
                               cache_hint::uncached,
                               cache_hint::cached>(ptr + read_offset + i * SIMD_COMPUTE);
        }
        ptr += chunk_size;
    }
    simd<data_type, SIMD_COMPUTE *UNROLL_SIZE> sum = 0;
#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
#if 0
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            sum.template select<SIMD_COMPUTE, 1>(i * SIMD_COMPUTE) += buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i);
        }
#else
        sum = sum + buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE);
#endif
    }

    //store the result
    data_type *write_ptr = (data_type *)out_buffer;
    size_t write_offset = (idx + threads_already_processed) * SIMD_COMPUTE * UNROLL_SIZE;
    if (write_offset + SIMD_COMPUTE * UNROLL_SIZE <= recv_size) {
        write_ptr += write_offset;
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            lsc_block_store<data_type,
                            SIMD_COMPUTE,
                            lsc_data_size::default_size,
                            cache_hint::uncached,
                            cache_hint::write_back> //save the all sum in the second half of the temp slot.
                (write_ptr + i * SIMD_COMPUTE, sum.template select<SIMD_COMPUTE, 1>(i * SIMD_COMPUTE));
        }
    }
    else {
        for (uint32_t i = write_offset; i < recv_size; i++)
            *(write_ptr + i) = sum[i - write_offset];
    }
}

template <typename dtype, size_t align>
class ReduceScatterLargeKernel;

template <typename dtype>
class ReduceScatterLargeNoCopyKernel;

template <typename dtype>
class ReduceScatterLargeKernel_GlobalSync;
template <typename dtype>
class ReduceScatterLargeKernel_LocalSync;

template <typename data_type, uint32_t max_rank = MAX_RANK, uint32_t max_buffer = 1024 /*KB*/>
class sycl_reduce_scatter_large : public sycl_coll_base<data_type> {
public:
    sycl_reduce_scatter_large() : sycl_coll_base<data_type>() {
        size_per_buffer = 0;
        buffer_index = 0;
    }

    void init(sycl::queue &queue, ccl_comm *comm_in, ccl_stream *stream, uint32_t rank_in, uint32_t world_in) {
        //using namespace __ESIMD_NS;
        //using namespace __ESIMD_ENS;

        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.
        size_t alloc_size;
        if (ccl::global_data::env().sycl_reduce_scatter_tmp_buf) {
            max_count_per_rank = (MAX_COUNT + SIMD_COMPUTE * UNROLL_SIZE - 1) / (SIMD_COMPUTE * UNROLL_SIZE) *
                                 SIMD_COMPUTE * UNROLL_SIZE;
            data_size_per_buffer = max_count_per_rank * world;
            size_per_buffer = data_size_per_buffer * sizeof(data_type) + SYNC_BYTE;
            alloc_size = size_per_buffer * COPY_BUFFER_COUNT;
        }
        else { // use half of the size
            max_count_per_rank = (MAX_COUNT + SIMD_COMPUTE * UNROLL_SIZE - 1) / (SIMD_COMPUTE * UNROLL_SIZE) *
                                 SIMD_COMPUTE * UNROLL_SIZE;
            data_size_per_buffer = max_count_per_rank * world / 2;
            size_per_buffer = data_size_per_buffer * sizeof(data_type) + SYNC_BYTE;
            alloc_size = size_per_buffer * NOCOPY_BUFFER_COUNT;
        }
        void *local_buffer = sycl::malloc_device(alloc_size, queue);
        auto e = queue.memset(local_buffer, 0, alloc_size);
        e.wait();

        // XXX: gain access to remote pointers
        this->exchange_peer_ipc_mem(queue,
                                    comm_in,
                                    stream,
                                    local_buffer,
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
        this->comm = comm_in;
        even_comm = comm_in->get_even_comm().get();
    }

    ccl::event reduce_scatter(sycl::queue &queue,
                              const void *send_buf,
                              void *out_buffer,
                              ccl::datatype dtype,
                              size_t recv_size,
                              ccl::reduction reduction,
                              const ccl::vector_class<ccl::event> &deps,
                              bool &done) {
        sycl::event e;
        // check local alignment
        if ((recv_size * sizeof(data_type)) % 4) {
            done = false;
            return ccl::event::create_from_native(e);
        }
        size_t is_aligned =
            (size_t)send_buf % 4 == 0 && (size_t)out_buffer % 4 == 0 && (recv_size * sizeof(data_type)) % 4 == 0;

        if (ccl::global_data::env().sycl_reduce_scatter_tmp_buf) {
            if (is_aligned)
                e = reduce_scatter_copy<4>(queue, send_buf, out_buffer, recv_size, deps, done);
            else
                e = reduce_scatter_copy<2>(queue, send_buf, out_buffer, recv_size, deps, done);
        }
        else {
            e = reduce_scatter_nocopy(queue, send_buf, out_buffer, recv_size, deps, done);
        }

        if (reduction == ccl::reduction::avg) {
            std::vector<sycl::event> evs;
            evs.push_back(e);
            e = sycl_average(queue, out_buffer, recv_size, world, dtype, evs);
        }

        return ccl::event::create_from_native(e);
    }

private:
    template <size_t align>
    sycl::event reduce_scatter_copy(sycl::queue &queue,
                                    const void *send_buf,
                                    void *out_buffer,
                                    size_t recv_size,
                                    const ccl::vector_class<ccl::event> &deps,
                                    bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        assert(this->initialized == true);
        done = true;
        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }

        if (recv_size / (SIMD_COMPUTE * UNROLL_SIZE) < temp_world) {
            done = false;
            return e;
        }

        int in_place = ((char *)send_buf + rank * recv_size * sizeof(data_type) == out_buffer);
        int even_ranks[max_rank];
        int my_rank_index = -1;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank)
                my_rank_index = i;
        }
        int size_per_buffer_kernel __attribute__((unused)) = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel __attribute__((unused)) =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int sync_reset_counter = 0;
        int buffer_index_kernel_for_sync = buffer_index;
        int outer_iter;

        //this is the outerloop count that requires full hw thread count.
        //This doesnt include the outloop iteration that only needs partial thread count
        int outerloop_iter_count = recv_size / max_count_per_rank;

        //uint32_t total_threads_needed_sync = 1;
        int wg_size __attribute__((unused)) = 1;
        int start, end;

        int sw_pipeline_insert_index = 0;
        int sw_pipeline_insert_counter = 0;
        int sw_pipeline_kernel_state[COPY_KERNEL_NUM];
        int threads_already_processed[COPY_KERNEL_NUM];
        for (int i = 0; i < COPY_KERNEL_NUM; i++) {
            threads_already_processed[i] = 0;
            sw_pipeline_kernel_state[i] = NO_KERNEL;
        }

        int first_iter = 1;
        uint32_t threads_needed_per_chunk = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);
        for (int iter = 0; iter < 2; iter++) {
            uint32_t total_threads_needed;
            if (iter == 1) //if second iteration, then handle the partial usage of the temp buffer
            {
                //if there is little more left to compute, then finish them
                if (outerloop_iter_count * max_count_per_rank < recv_size) {
                    start = outerloop_iter_count;
                    end = start + 1;
                    uint32_t leftover = recv_size - outerloop_iter_count * max_count_per_rank;
                    total_threads_needed =
                        (leftover + SIMD_COMPUTE * UNROLL_SIZE - 1) / (SIMD_COMPUTE * UNROLL_SIZE);
                }
                else {
                    break;
                }
            }
            else {
                start = 0;
                end = outerloop_iter_count;
                total_threads_needed = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);

                if (end == 0)
                    continue; //there is nothing to do when end is 0 so check the next iter.
            }

            int innerloop_iter_count = (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;
            uint32_t persist_threads_needed = total_threads_needed;
            if (persist_threads_needed > HW_THREAD_COUNT)
                persist_threads_needed = HW_THREAD_COUNT;

            for (outer_iter = start; outer_iter < end + COPY_KERNEL_NUM - 1; outer_iter++) {
                //if more outer_iter remaining since there is more new processing to do, then insert them to the SW pipeline.
                //During the sw pipeline tail, there is nothing to dispatch.
                if (outer_iter < end) {
                    sw_pipeline_kernel_state[sw_pipeline_insert_index] = FIRST_KERNEL;
                    threads_already_processed[sw_pipeline_insert_index] = sw_pipeline_insert_counter;
                    sw_pipeline_insert_index++;
                    if (sw_pipeline_insert_index >= COPY_KERNEL_NUM) {
                        sw_pipeline_insert_index =
                            0; //By the time the index wraps arounds, the kernel that was in this slot previously has already completed.
                    }
                    sw_pipeline_insert_counter += threads_needed_per_chunk;
                }

                if (!first_iter) {
                    //sync all the ranks within the single GPU.
                    e = global_sync(queue,
                                    temp_rank,
                                    temp_world,
                                    size_per_buffer_for_sync_kernel * buffer_index_kernel_for_sync,
                                    4,
                                    1);
                    sync_reset_counter++;
                    buffer_index_kernel_for_sync++;
                    buffer_index_kernel_for_sync %= COPY_BUFFER_COUNT;
                }
                first_iter = 0;

                e = queue.submit([&](sycl::handler &cgh) {
                        cgh.parallel_for<class ReduceScatterLargeKernel<data_type, align>>(
                            sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                            {
                            uint32_t idx = idx2.get_global_id();
                            //check if there is any kernel in the SW pipelines. If yes, execute them.
                            //to optimize, the order of loop i=0,1,2,.. can be shuffled so that different
                            //ranks can do different kernels at particular time. The purpose is to better
                            //balance the HW resource usage in the PVC node.
                            for (int ii = 0; ii < COPY_KERNEL_NUM; ii++) {
                                //wrap the SW pipeline index so that it is [0, KERNEL_NUM - 1]. Used instead of the expensive modulo.
                                //sycl::_V1::ext::oneapi::experimental::printf("Kernel rank%d  %d - %x  innerloop_iter_count: %d\n", temp_rank, ii, sw_pipeline_kernel_state[ii], innerloop_iter_count);

                                if (sw_pipeline_kernel_state[ii] & FIRST_KERNEL) {
                                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                                        int index = idx + inner_iter * HW_THREAD_COUNT;
                                        if ((uint32_t)index >= total_threads_needed)
                                            break;
                                        auto local_copy_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                            local_copy<w, data_type, a>((int *)even_ranks,
                                                                        index,
                                                                        send_buf,
                                                                        recv_size,
                                                                        threads_already_processed[ii],
                                                                        (void **)temp_buffer,
                                                                        temp_rank,
                                                                        outer_iter,
                                                                        size_per_buffer_kernel,
                                                                        ii,
                                                                        threads_needed_per_chunk);
                                        };
                                        invoke_esimd_function<align>(local_copy_lambda, temp_world);
                                    }
                                } // end of if FIRST_KERNEL
                                if (sw_pipeline_kernel_state[ii] & SECOND_KERNEL) {
                                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                                        int index = idx + inner_iter * HW_THREAD_COUNT;
                                        if ((uint32_t)index >= total_threads_needed)
                                            break;
                                        auto reduce_read_write_lamda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                            reduce_read_write<w, data_type, a>((int *)even_ranks,
                                                                               my_rank_index,
                                                                               index,
                                                                               send_buf,
                                                                               out_buffer,
                                                                               recv_size,
                                                                               threads_already_processed[ii],
                                                                               (void **)temp_buffer,
                                                                               temp_rank,
                                                                               outer_iter,
                                                                               size_per_buffer_kernel,
                                                                               ii,
                                                                               threads_needed_per_chunk,
                                                                               in_place);
                                        };
                                        invoke_esimd_function<align>(reduce_read_write_lamda, temp_world);
                                    }
                                } // end of if SECOND_KERNEL
                                if ((sw_pipeline_kernel_state[ii] & THIRD_KERNEL) &&
                                    (temp_world > 2 || in_place)) {
                                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                                        int index = idx + inner_iter * HW_THREAD_COUNT;
                                        if ((uint32_t)index >= total_threads_needed)
                                            break;
                                        auto all_sum_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                            all_sum<w, data_type, a>(index,
                                                                     send_buf,
                                                                     out_buffer,
                                                                     recv_size,
                                                                     threads_already_processed[ii],
                                                                     (void **)temp_buffer,
                                                                     temp_rank,
                                                                     outer_iter,
                                                                     size_per_buffer_kernel,
                                                                     ii,
                                                                     threads_needed_per_chunk);
                                        };
                                        invoke_esimd_function<align>(all_sum_lambda, temp_world);
                                    }
                                } // end of if THIRD_KERNEL
                            } // end of for
                         });//parallel_for
                }); //submit()
                //e.wait();

                //update the sw pipeline process state so that next kernel will be processed in next round
                for (int i = 0; i < COPY_KERNEL_NUM; i++) {
                    if (sw_pipeline_kernel_state[i] & COPY_LAST_KERNEL)
                        sw_pipeline_kernel_state[i] =
                            0; //remove the kernel from the sw pipeline if it is fifth kernel. Everything is already executed.
                    else
                        sw_pipeline_kernel_state[i] <<= 1;
                }
            } // end of outer_iter
        } // end of for iter = 2

        buffer_index += sync_reset_counter;
        buffer_index %= COPY_BUFFER_COUNT;

        return e;
    }

    sycl::event reduce_scatter_nocopy(sycl::queue &queue,
                                      const void *send_buf,
                                      void *out_buffer,
                                      size_t recv_size,
                                      const ccl::vector_class<ccl::event> &deps,
                                      bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        assert(this->initialized == true);
        done = true;
        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }

        if (recv_size / (SIMD_COMPUTE * UNROLL_SIZE) < temp_world) {
            done = false;
            return e;
        }

        int even_ranks[max_rank];
        int my_rank_index = -1;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank)
                my_rank_index = i;
        }
        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel = size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int outerloop_iter_count; //Since 16 elements in temp buffer is used to process 8 element output, the outer loop count must be doubled roughly.
        int sync_reset_counter = 0;
        int buffer_index_kernel_for_sync = buffer_index;
        int outer_iter;

        int max_elements_per_MAX_COUNT __attribute__((unused)) = (recv_size + SIMD_COMPUTE * UNROLL_SIZE - 1) /
                                                                 (SIMD_COMPUTE * UNROLL_SIZE) * SIMD_COMPUTE *
                                                                 UNROLL_SIZE;
        int max_threads_per_MAX_COUNT __attribute__((unused)) = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);

        //this is the outerloop count that requires full hw thread count.
        //This doesnt include the outloop iteration that only needs partial thread count
        outerloop_iter_count = recv_size / max_count_per_rank;

        int wg_size __attribute__((unused)) = 1;
        int start, end;

        //init the sw pipeline
        int sw_pipeline_insert_index = 0;
        int sw_pipeline_insert_counter = 0;
        int sw_pipeline_kernel_state[NOCOPY_KERNEL_NUM];
        int threads_already_processed[NOCOPY_KERNEL_NUM];
        for (int i = 0; i < NOCOPY_KERNEL_NUM; i++) {
            threads_already_processed[i] = 0;
            sw_pipeline_kernel_state[i] = NO_KERNEL;
        }

        //cpu_timer<1> ctimer;
        //ctimer.start(0);
        void *in_buffers[max_rank];
        this->exchange_peer_ipc_mem(queue,
                                    comm,
                                    global_stream,
                                    (void **)send_buf,
                                    NULL,
                                    rank,
                                    world,
                                    0,
                                    (void **)in_buffers,
                                    NULL,
                                    NULL,
                                    NULL,
                                    NULL);
        //ctimer.stop(0);
        //printf("exchange_peer_ipc_mem time: %fus\n", ctimer.get_us(0));
        // check remote alignment
        int align4 = all_aligned((void **)in_buffers, temp_world, recv_size, sizeof(data_type), 4);

        int first_iter = 1;
        uint32_t threads_needed_per_chunk = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);
        for (int iter = 0; iter < 2; iter++) {
            uint32_t total_threads_needed;
            if (iter == 1) //if second iteration, then handle the partial usage of the temp buffer
            {
                //if there is little more left to compute, then finish them
                if (outerloop_iter_count * max_count_per_rank < recv_size) {
                    start = outerloop_iter_count;
                    end = start + 1;
                    //total_threads_needed = (recv_size - start * max_elements_per_MAX_COUNT + SIMD_COMPUTE * temp_world - 1) / (SIMD_COMPUTE * temp_world);
                    uint32_t leftover = recv_size - outerloop_iter_count * max_count_per_rank;
                    total_threads_needed =
                        (leftover + SIMD_COMPUTE * UNROLL_SIZE - 1) / (SIMD_COMPUTE * UNROLL_SIZE);
                }
                else {
                    break;
                }
            }
            else {
                start = 0;
                end = outerloop_iter_count;
                //total_threads_needed = max_threads_per_MAX_COUNT;
                total_threads_needed = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);

                if (end == 0)
                    continue; //there is nothing to do when end is 0 so check the next iter.
            }

            int innerloop_iter_count = (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;
            uint32_t persist_threads_needed = total_threads_needed;
            if (persist_threads_needed > HW_THREAD_COUNT)
                persist_threads_needed = HW_THREAD_COUNT;

            for (outer_iter = start; outer_iter < end + NOCOPY_KERNEL_NUM - 1; outer_iter++) {
                //if more outer_iter remaining since there is more new processing to do, then insert them to the SW pipeline.
                //During the sw pipeline tail, there is nothing to dispatch.
                if (outer_iter < end) {
                    sw_pipeline_kernel_state[sw_pipeline_insert_index] = FIRST_KERNEL;
                    threads_already_processed[sw_pipeline_insert_index] = sw_pipeline_insert_counter;
                    sw_pipeline_insert_index++;
                    if (sw_pipeline_insert_index >= NOCOPY_KERNEL_NUM) {
                        sw_pipeline_insert_index =
                            0; //By the time the index wraps arounds, the kernel that was in this slot previously has already completed.
                    }
                    sw_pipeline_insert_counter += threads_needed_per_chunk;
                }

                if (first_iter) {
                    e = local_sync(queue,
                                   temp_rank,
                                   temp_world,
                                   size_per_buffer_for_sync_kernel * buffer_index_kernel_for_sync,
                                   0,
                                   0);
                }
                else {
                    e = global_sync(queue,
                                    temp_rank,
                                    temp_world,
                                    size_per_buffer_for_sync_kernel * buffer_index_kernel_for_sync,
                                    4,
                                    1);
                    sync_reset_counter++;
                    buffer_index_kernel_for_sync++;
                    buffer_index_kernel_for_sync %= NOCOPY_BUFFER_COUNT;
                }
                first_iter = 0;

                e = queue.submit([&](sycl::handler &cgh) {
                        cgh.parallel_for<class ReduceScatterLargeNoCopyKernel<data_type>>(
                            sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                            {
                            uint32_t idx = idx2.get_global_id();
                            //ESIMD kernel

                            //check if there is any kernel in the SW pipelines. If yes, execute them.
                            //to optimize, the order of loop i=0,1,2,.. can be shuffled so that different
                            // ranks can do different kernels at particular time.
                            // The purpose is to better balance the HW resource usage in the PVC node.
                            for (int ii = 0; ii < NOCOPY_KERNEL_NUM; ii++) {
                                RUN_SECOND_KERNEL
                                RUN_THIRD_KERNEL
                            } // end of for
                         });//parallel_for
                }); //submit()
                //e.wait();

                //update the sw pipeline process state so that next kernel will be processed in next round
                for (int i = 0; i < NOCOPY_KERNEL_NUM; i++) {
                    if (sw_pipeline_kernel_state[i] & NOCOPY_LAST_KERNEL)
                        sw_pipeline_kernel_state[i] =
                            0; //remove the kernel from the sw pipeline if it is fifth kernel. Everything is already executed.
                    else
                        sw_pipeline_kernel_state[i] <<= 1;
                }

                //std::cout << "rank" << temp_rank << " iter" << iter << " outer_iter" << outer_iter << " kernel1 done." << "\n";

            } // end of outer_iter
        } // end of for iter = 2

        buffer_index += sync_reset_counter;
        buffer_index %= NOCOPY_BUFFER_COUNT;

        return e;
    }

    //sync all the ranks here before consuming the results.
    // offset = size_per_buffer_for_sync_kernel * buffer_index_kernel
    sycl::event global_sync(sycl::queue queue,
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
            cgh.parallel_for<class ReduceScatterLargeKernel_GlobalSync<data_type>>(
                sycl::nd_range<1>({ total_threads_needed_sync }, wg_size), [=](sycl::nd_item<1> idx) SYCL_ESIMD_KERNEL
                {
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
            cgh.parallel_for<class ReduceScatterLargeKernel_LocalSync<data_type>>(
                sycl::nd_range<1>({ total_threads_needed_sync }, wg_size), [=](sycl::nd_item<1> idx) SYCL_ESIMD_KERNEL
                {
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
    size_t size_per_buffer{ 0 };
    int data_size_per_buffer{ ccl::utils::invalid_bytes_value };
    uint32_t max_count_per_rank{ 0 };
    int buffer_index{ ccl::utils::invalid_err_code };
    ccl_stream *global_stream{};
    ccl_comm *comm{};
    ccl_comm *even_comm{};
};

#define REDUCE_SCATTER_LARGE_API(TYPE) \
    void init_reduce_scatter_large_##TYPE(ccl::datatype dtype, \
                                          sycl::queue &queue, \
                                          ccl_comm *comm, \
                                          ccl_stream *stream, \
                                          uint32_t rank_in, \
                                          uint32_t world_in) { \
        if (!rs_large_##TYPE.inited()) { \
            LOG_INFO("invoking large reduce_scatter first time for datatype: ", dtype); \
            rs_large_##TYPE.init(queue, comm, stream, rank_in, world_in); \
        } \
    } \
\
    ccl::event run_reduce_scatter_large_##TYPE(ccl::datatype dtype, \
                                               sycl::queue &queue, \
                                               const void *send_buf, \
                                               void *recv_buf, \
                                               size_t recv_count, \
                                               ccl::reduction reduction, \
                                               const ccl::vector_class<ccl::event> &deps, \
                                               bool &done) { \
        return rs_large_##TYPE.reduce_scatter( \
            queue, send_buf, recv_buf, dtype, recv_count, reduction, deps, done); \
    }

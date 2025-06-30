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

#define MAX_RANK         16
#define SIMD_COMPUTE_MAX 256
#define SIMD_COMPUTE     (SIMD_COMPUTE_MAX / sizeof(data_type))
#define SIMD_SYNC        32
#define UNROLL_SIZE      1
#define BUFFER_COUNT     2
#define SYNC_BYTE        (SIMD_SYNC * sizeof(int) * 2)
#define ALIGNMENT_BYTE   256
#define NOCOPY_MAX_COUNT (256 * 1024 * 1024 / sizeof(data_type))
#define COPY_MAX_COUNT   (32 * 1024 * 1024 / sizeof(data_type))
//#define EU_COUNT_PER_RANK 448
#define EU_COUNT_PER_RANK   512
#define THREAD_COUNT_PER_EU 8
#define HW_THREAD_COUNT     (EU_COUNT_PER_RANK * THREAD_COUNT_PER_EU)
#define RANKS_PER_GPU       2

// nocopy version only use half of the buffer of the copy version
template <uint32_t TEMP_WORLD, typename data_type>
void nocopy_reduce_read_write(int *even_ranks,
                              int my_rank_index,
                              int idx,
                              void **in_buffers,
                              void *out_buffer,
                              uint32_t recv_size,
                              int threads_already_processed,
                              void *temp_buffer[],
                              uint32_t temp_rank,
                              int outer_iter,
                              int size_per_buffer_kernel,
                              int buffer_index_kernel,
                              uint32_t threads_needed_per_chunk) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    int chunk_size = threads_needed_per_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    int abs_offset_in_chunk = idx + threads_already_processed;
    int read_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;

    data_type *mdfi_ptr = (data_type *)in_buffers[temp_rank ^ 1];
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> mdfi_buffer;
    data_type *local_ptr = (data_type *)in_buffers[temp_rank];
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> local_buffer;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
        int rr = even_ranks[r]; // even rank copies odd chunks
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            mdfi_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i) =
                lsc_block_load<data_type,
                               SIMD_COMPUTE,
                               lsc_data_size::default_size,
                               cache_hint::uncached,
                               cache_hint::uncached>(mdfi_ptr + rr * recv_size + read_offset + i * SIMD_COMPUTE);
            local_buffer.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i) =
                lsc_block_load<data_type,
                               SIMD_COMPUTE,
                               lsc_data_size::default_size,
                               cache_hint::uncached,
                               cache_hint::uncached>(local_ptr + rr * recv_size + read_offset + i * SIMD_COMPUTE);
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
        //write_ptr += size_per_buffer_kernel * buffer_index_kernel;
        int out_offset = (temp_rank / 2) * chunk_size + idx * SIMD_COMPUTE * UNROLL_SIZE;
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            lsc_block_store<data_type,
                            SIMD_COMPUTE,
                            lsc_data_size::default_size,
                            cache_hint::uncached,
                            cache_hint::write_back> //save the all sum in the second half of the temp slot.
                (write_ptr + out_offset + i * SIMD_COMPUTE,
                 sum.template select<SIMD_COMPUTE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE + SIMD_COMPUTE * i));
        }
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void local_copy(int *even_ranks,
                int idx,
                const void *in_buffer,
                uint32_t recv_size,
                int threads_already_processed,
                void *temp_buffer[],
                uint32_t temp_rank,
                int outer_iter,
                int size_per_buffer_kernel,
                int buffer_index_kernel,
                uint32_t threads_needed_per_chunk) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    int chunk_size = threads_needed_per_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    int abs_offset_in_chunk = idx + threads_already_processed;
    // even rank copies odd chunks
    int read_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;

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
    //ptr += size_per_buffer_kernel * buffer_index_kernel;
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
                       uint32_t recv_size,
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

    int chunk_size = threads_needed_per_chunk * SIMD_COMPUTE * UNROLL_SIZE;

    data_type *mdfi_ptr = (data_type *)temp_buffer[temp_rank ^ 1];
    //mdfi_ptr += size_per_buffer_kernel * buffer_index_kernel;
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

    int abs_offset_in_chunk = idx + threads_already_processed;
    int read_offset = abs_offset_in_chunk * SIMD_COMPUTE * UNROLL_SIZE;
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
            //write_ptr += size_per_buffer_kernel * buffer_index_kernel;
            int out_offset = (temp_rank / 2) * chunk_size + idx * SIMD_COMPUTE * UNROLL_SIZE;
#pragma unroll
            for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
                // save the all sum in the second half of the temp slot
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
        size_t write_offset = (idx + threads_already_processed) * SIMD_COMPUTE * UNROLL_SIZE;
        if (write_offset + SIMD_COMPUTE * UNROLL_SIZE <= recv_size) {
            write_ptr += write_offset;
#pragma unroll
            for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
                //save the all sum in the second half of the temp slot
                block_store<data_type, SIMD_COMPUTE>(write_ptr + i * SIMD_COMPUTE,
                                                     sum.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                                                     properties{ alignment<align> });
            }
        }
        else {
            for (uint32_t i = write_offset; i < recv_size; i++)
                *(write_ptr + i) = sum[i - write_offset];
        }
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void all_sum(int idx,
             const void *in_buffer,
             void *out_buffer,
             uint32_t recv_size,
             int threads_already_processed,
             void *temp_buffer[],
             uint32_t temp_rank,
             int outer_iter,
             int size_per_buffer_kernel,
             int buffer_index_kernel,
             uint32_t threads_needed_per_chunk) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    int chunk_size = threads_needed_per_chunk * SIMD_COMPUTE * UNROLL_SIZE;
    //read the input data
    data_type *ptr = (data_type *)temp_buffer[temp_rank];
    //ptr += size_per_buffer_kernel * buffer_index_kernel;
    int read_offset = idx * SIMD_COMPUTE * UNROLL_SIZE;
    simd<data_type, SIMD_COMPUTE * UNROLL_SIZE * TEMP_WORLD / 2> buffer;
    //#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * UNROLL_SIZE * r + i * SIMD_COMPUTE) =
                block_load<data_type, SIMD_COMPUTE>(ptr + read_offset + i * SIMD_COMPUTE,
                                                    properties{ alignment<align> });
        }
        ptr += chunk_size;
    }
    simd<data_type, SIMD_COMPUTE *UNROLL_SIZE> sum = 0;
#pragma unroll
    for (uint32_t r = 0; r < TEMP_WORLD / 2; r++) {
        sum = sum + buffer.template select<SIMD_COMPUTE * UNROLL_SIZE, 1>(r * SIMD_COMPUTE * UNROLL_SIZE);
    }

    //store the result
    data_type *write_ptr = (data_type *)out_buffer;
    int write_offset = (idx + threads_already_processed) * SIMD_COMPUTE * UNROLL_SIZE;
    if (write_offset + SIMD_COMPUTE * UNROLL_SIZE <= recv_size) {
        write_ptr += write_offset;
#pragma unroll
        for (uint32_t i = 0; i < UNROLL_SIZE; i++) {
            //save the all sum in the second half of the temp slot
            block_store<data_type, SIMD_COMPUTE>(write_ptr + i * SIMD_COMPUTE,
                                                 sum.template select<SIMD_COMPUTE, 1>(i * SIMD_COMPUTE),
                                                 properties{ alignment<align> });
        }
    }
    else {
        for (uint32_t i = write_offset; i < recv_size; i++)
            *(write_ptr + i) = sum[i - write_offset];
    }
}

template <typename dtype, size_t align>
class ReduceScatterMediumKernel_local_copy;
template <typename dtype, size_t align>
class ReduceScatterMediumKernel_reduce_read_write;
template <typename dtype, size_t align>
class ReduceScatterMediumKernel_local_all_sum;

template <typename dtype>
class ReduceScatterMediumKernel_nocopy_reduce_read_write;
template <typename dtype>
class ReduceScatterMediumKernel_nocopy_local_all_sum;

template <typename dtype>
class ReduceScatterMediumKernel_GlobalSync;
template <typename dtype>
class ReduceScatterMediumKernel_LocalSync;

template <typename data_type, uint32_t max_rank = MAX_RANK, uint32_t max_buffer = 1024 /*KB*/>
class sycl_reduce_scatter_medium : public sycl_coll_base<data_type> {
public:
    sycl_reduce_scatter_medium() : sycl_coll_base<data_type>() {
        size_per_buffer = 0;
        buffer_index = 0;
        data_size_per_buffer = 0;
    }

    void init(sycl::queue &queue, ccl_comm *comm_in, ccl_stream *stream, uint32_t rank_in, uint32_t world_in) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.

        max_count_per_rank = (COPY_MAX_COUNT + SIMD_COMPUTE * UNROLL_SIZE - 1) / (SIMD_COMPUTE * UNROLL_SIZE) *
                             SIMD_COMPUTE * UNROLL_SIZE;
        data_size_per_buffer = max_count_per_rank * world;
        size_per_buffer = data_size_per_buffer * sizeof(data_type);
        size_per_buffer = (size_per_buffer + ALIGNMENT_BYTE - 1) / ALIGNMENT_BYTE * ALIGNMENT_BYTE;

        void *local_buffer = sycl::malloc_device(size_per_buffer + SYNC_BYTE * BUFFER_COUNT, queue);
        //printf("allocate temp buffer: max_count_per_rank:%d size_per_buffer:%ld %ld\n", max_count_per_rank, size_per_buffer, (size_t)size_per_buffer + SYNC_BYTE * BUFFER_COUNT);
        auto e = queue.memset(local_buffer, 0, size_per_buffer + SYNC_BYTE * BUFFER_COUNT);
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
        bool is_aligned =
            (size_t)send_buf % 4 == 0 && (size_t)out_buffer % 4 == 0 && (recv_size * sizeof(data_type)) % 4 == 0;

        if (is_aligned)
            e = reduce_scatter_copy<4>(queue, send_buf, out_buffer, recv_size, deps, done);
        else
            e = reduce_scatter_copy<2>(queue, send_buf, out_buffer, recv_size, deps, done);

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
        /*
        void* temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) 
        {
            temp_sync_buffer[i] = sync_buffer[i];
        }*/

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
            //printf("even rank %d: %d neighbor: %d\n", i, even_ranks[i], even_ranks[i] ^ 1);
        }
        int size_per_buffer_kernel __attribute__((unused)) = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel __attribute__((unused)) =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int buffer_index_kernel __attribute__((unused)) = buffer_index;
        int outerloop_iter_count; //Since 16 elements in temp buffer is used to process 8 element output, the outer loop count must be doubled roughly.
        int outer_iter;
        //todo:
        //5. prefetch in persistent threads?
        int max_elements_per_MAX_COUNT __attribute__((unused)) = (recv_size + SIMD_COMPUTE * UNROLL_SIZE - 1) /
                                                                 (SIMD_COMPUTE * UNROLL_SIZE) * SIMD_COMPUTE *
                                                                 UNROLL_SIZE;
        int max_threads_per_MAX_COUNT __attribute__((unused)) = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);
        //this is the outerloop count that requires full hw thread count.
        //This doesnt include the outloop iteration that only needs partial thread count
        int threads_already_processed __attribute__((unused)) = 0;
        outerloop_iter_count = (recv_size + max_count_per_rank - 1) / max_count_per_rank;
        uint32_t total_threads_needed_sync __attribute__((unused)) = 1;

        //printf("[%d] max_count_per_rank: %d max_threads_per_MAX_COUNT: %d max_elements_per_MAX_COUNT: %d outerloop_iter_count: %d\n",
        // temp_rank, max_count_per_rank, max_threads_per_MAX_COUNT, max_elements_per_MAX_COUNT, outerloop_iter_count);
        std::vector<sycl::event> dep_events = get_sycl_events(deps);

        for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++) {
            uint32_t threads_needed_per_chunk __attribute__((unused));
            uint32_t total_threads_needed __attribute__((unused));
            if ((outer_iter + 1) * max_count_per_rank < recv_size) {
                threads_needed_per_chunk = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);
            }
            else {
                uint32_t leftover = recv_size - outer_iter * max_count_per_rank;
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
                //printf("outer_iter=%d outerloop_iter_count: %d total_threads_needed: %d threads_needed_per_chunk:
                // %d innerloop_iter_count: %d persist_threads_needed: %d\n", outer_iter, outerloop_iter_count,
                // total_threads_needed, threads_needed_per_chunk, innerloop_iter_count, persist_threads_needed);

#define KERNEL_EXEC_MAP (1 + 2 + 4 + 8 + 16)

#if KERNEL_EXEC_MAP & 1
            // local copy half of the data to tmp buffer
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep_events);
                    cgh.parallel_for<class ReduceScatterMediumKernel_local_copy<data_type, align>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        //ESIMD kernel
                        uint32_t idx = idx2.get_global_id();
                        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= total_threads_needed)
                                break;

                            auto local_copy_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                local_copy<w, data_type, a>((int *)even_ranks,
                                                            index,
                                                            send_buf,
                                                            recv_size,
                                                            threads_already_processed,
                                                            (void **)temp_buffer,
                                                            temp_rank,
                                                            outer_iter,
                                                            size_per_buffer_kernel,
                                                            buffer_index_kernel,
                                                            threads_needed_per_chunk);
                            };
                            invoke_esimd_function<align>(local_copy_lambda, temp_world);
                        }
                        });//parallel_for
            }); //submit()
            //printf("kernel0\n");
            dep_events.clear();
            dep_events.push_back(std::move(e));
#endif
#if KERNEL_EXEC_MAP & 2
            //sync all the ranks within the single GPU.
            e = local_sync(queue, temp_rank, temp_world, SYNC_BYTE * buffer_index_kernel, 0, 0, dep_events);
            //printf("kernel1\n");
            dep_events.clear();
            dep_events.push_back(std::move(e));
#endif
#if KERNEL_EXEC_MAP & 4
            //local reduction kernel
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep_events);
                    cgh.parallel_for<class ReduceScatterMediumKernel_reduce_read_write<data_type, align>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        //ESIMD kernel
                        uint32_t idx = idx2.get_global_id();
                        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++)
                        //for (int inner_iter = 0; inner_iter < 1; inner_iter++)
                        {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= total_threads_needed)
                                break;
                            //                                int index = idx;

                            auto reduce_read_write_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                reduce_read_write<w, data_type, a>((int *)even_ranks,
                                                                   my_rank_index,
                                                                   index,
                                                                   send_buf,
                                                                   out_buffer,
                                                                   recv_size,
                                                                   threads_already_processed,
                                                                   (void **)temp_buffer,
                                                                   temp_rank,
                                                                   outer_iter,
                                                                   size_per_buffer_kernel,
                                                                   buffer_index_kernel,
                                                                   threads_needed_per_chunk,
                                                                   in_place);
                            };
                            invoke_esimd_function<align>(reduce_read_write_lambda, temp_world);
                        }
                        });//parallel_for
            }); //submit()
            //printf("kernel2\n");
            dep_events.clear();
            dep_events.push_back(std::move(e));
#endif
#if KERNEL_EXEC_MAP & 8
            //sync all the ranks here before consuming the results.
            e = global_sync(queue, temp_rank, temp_world, SYNC_BYTE * buffer_index_kernel, 1, 1, dep_events);
            //printf("kernel3\n");
            dep_events.clear();
            dep_events.push_back(std::move(e));
#endif
#if KERNEL_EXEC_MAP & 16
            if (temp_world > 2 || in_place) {
                int innerloop_local_sum_iter_count __attribute__((unused)) =
                    (threads_needed_per_chunk + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;
                uint32_t persist_local_sum_threads_needed = threads_needed_per_chunk;
                if (persist_local_sum_threads_needed > HW_THREAD_COUNT)
                    persist_local_sum_threads_needed = HW_THREAD_COUNT;
                //local reduction kernel
                e = queue.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(dep_events);
                    cgh.parallel_for<class ReduceScatterMediumKernel_local_all_sum<data_type, align>>(
                        sycl::nd_range<1>({ persist_local_sum_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        //ESIMD kernel
                        uint32_t idx = idx2.get_global_id();
                        for (int inner_iter = 0; inner_iter < innerloop_local_sum_iter_count; inner_iter++) {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= threads_needed_per_chunk)
                                break;

                            auto all_sum_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                all_sum<w, data_type, a>(index,
                                                         send_buf,
                                                         out_buffer,
                                                         recv_size,
                                                         threads_already_processed,
                                                         (void **)temp_buffer,
                                                         temp_rank,
                                                         outer_iter,
                                                         size_per_buffer_kernel,
                                                         buffer_index_kernel,
                                                         threads_needed_per_chunk);
                            };
                            invoke_esimd_function<align>(all_sum_lambda, temp_world);
                        }
                        });//parallel_for
                }); //submit()
                dep_events.clear();
                dep_events.push_back(std::move(e));
                //printf("kernel4\n");
            } // end if
#endif
            threads_already_processed += threads_needed_per_chunk;
            buffer_index++;
            buffer_index %= BUFFER_COUNT;
            buffer_index_kernel = buffer_index;
        } //for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++)

        e = dep_events.back();
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
        /*
        void* temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) 
        {
            temp_sync_buffer[i] = sync_buffer[i];
        }*/

        if (recv_size / (SIMD_COMPUTE * UNROLL_SIZE) < temp_world) {
            done = false;
            return e;
        }
#if 0
	if (recv_size > max_count_per_rank) {
            //printf("reduce-scatter medium fallback recv_size: %d max_count_per_rank: %d \n", recv_size, max_count_per_rank);
            done = false;
	    return ccl::event::create_from_native(e);
	}
#endif
        int even_ranks[max_rank];
        int my_rank_index = -1;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank)
                my_rank_index = i;
            //printf("even rank %d: %d neighbor: %d\n", i, even_ranks[i], even_ranks[i] ^ 1);
        }

        int size_per_buffer_kernel __attribute__((unused)) = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel __attribute__((unused)) =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int buffer_index_kernel __attribute__((unused)) = buffer_index;
        int outerloop_iter_count; //Since 16 elements in temp buffer is used to process 8 element output, the outer loop count must be doubled roughly.
        int outer_iter;
        //todo:
        //5. prefetch in persistent threads?
        int max_elements_per_MAX_COUNT __attribute__((unused)) = (recv_size + SIMD_COMPUTE * UNROLL_SIZE - 1) /
                                                                 (SIMD_COMPUTE * UNROLL_SIZE) * SIMD_COMPUTE *
                                                                 UNROLL_SIZE;
        int max_threads_per_MAX_COUNT __attribute__((unused)) = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);

        int threads_already_processed __attribute__((unused)) = 0;
        outerloop_iter_count =
            (recv_size + max_count_per_rank - 1) /
            max_count_per_rank; //this is the outerloop count that requires full hw thread count. This doesnt include the outloop iteration that only needs partial thread count
        uint32_t total_threads_needed_sync __attribute__((unused)) = 1;

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

        std::vector<sycl::event> dep_events = get_sycl_events(deps);

        //must sync tiles in the single GPU.
        e = local_sync(queue, temp_rank, temp_world, SYNC_BYTE * buffer_index_kernel, 0, 0, dep_events);
        dep_events.clear();
        dep_events.push_back(std::move(e));

        //printf("[%d] max_count_per_rank: %d max_threads_per_MAX_COUNT: %d max_elements_per_MAX_COUNT: %d outerloop_iter_count: %d\n",
        //temp_rank, max_count_per_rank, max_threads_per_MAX_COUNT, max_elements_per_MAX_COUNT, outerloop_iter_count);
        for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++) {
            uint32_t threads_needed_per_chunk __attribute__((unused));
            uint32_t total_threads_needed __attribute__((unused));
            if ((outer_iter + 1) * max_count_per_rank < recv_size) {
                threads_needed_per_chunk = max_count_per_rank / (SIMD_COMPUTE * UNROLL_SIZE);
            }
            else {
                uint32_t leftover = recv_size - outer_iter * max_count_per_rank;
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
            //printf("outer_iter=%d outerloop_iter_count: %d total_threads_needed: %d threads_needed_per_chunk: %d innerloop_iter_count:
            //%d persist_threads_needed: %d\n", outer_iter, outerloop_iter_count, total_threads_needed, threads_needed_per_chunk, innerloop_iter_count, persist_threads_needed);

            //local reduction kernel
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.depends_on(dep_events);
                    cgh.parallel_for<class ReduceScatterMediumKernel_nocopy_reduce_read_write<data_type>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        //ESIMD kernel
                        uint32_t idx = idx2.get_global_id();
                        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++)
                        //for (int inner_iter = 0; inner_iter < 1; inner_iter++)
                        {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= total_threads_needed)
                                break;

                            switch (temp_world) {
                                case 2:
                                    nocopy_reduce_read_write<2, data_type>((int *)even_ranks,
                                                                           my_rank_index,
                                                                           index,
                                                                           (void **)in_buffers,
                                                                           out_buffer,
                                                                           recv_size,
                                                                           threads_already_processed,
                                                                           (void **)temp_buffer,
                                                                           temp_rank,
                                                                           outer_iter,
                                                                           size_per_buffer_kernel,
                                                                           buffer_index_kernel,
                                                                           threads_needed_per_chunk);
                                    break;
                                case 4:
                                    nocopy_reduce_read_write<4, data_type>((int *)even_ranks,
                                                                           my_rank_index,
                                                                           index,
                                                                           (void **)in_buffers,
                                                                           out_buffer,
                                                                           recv_size,
                                                                           threads_already_processed,
                                                                           (void **)temp_buffer,
                                                                           temp_rank,
                                                                           outer_iter,
                                                                           size_per_buffer_kernel,
                                                                           buffer_index_kernel,
                                                                           threads_needed_per_chunk);
                                    break;
                                case 6:
                                    nocopy_reduce_read_write<6, data_type>((int *)even_ranks,
                                                                           my_rank_index,
                                                                           index,
                                                                           (void **)in_buffers,
                                                                           out_buffer,
                                                                           recv_size,
                                                                           threads_already_processed,
                                                                           (void **)temp_buffer,
                                                                           temp_rank,
                                                                           outer_iter,
                                                                           size_per_buffer_kernel,
                                                                           buffer_index_kernel,
                                                                           threads_needed_per_chunk);
                                    break;
                                case 8:
                                    nocopy_reduce_read_write<8, data_type>((int *)even_ranks,
                                                                           my_rank_index,
                                                                           index,
                                                                           (void **)in_buffers,
                                                                           out_buffer,
                                                                           recv_size,
                                                                           threads_already_processed,
                                                                           (void **)temp_buffer,
                                                                           temp_rank,
                                                                           outer_iter,
                                                                           size_per_buffer_kernel,
                                                                           buffer_index_kernel,
                                                                           threads_needed_per_chunk);
                                    break;
                                case 10:
                                    nocopy_reduce_read_write<10, data_type>((int *)even_ranks,
                                                                            my_rank_index,
                                                                            index,
                                                                            (void **)in_buffers,
                                                                            out_buffer,
                                                                            recv_size,
                                                                            threads_already_processed,
                                                                            (void **)temp_buffer,
                                                                            temp_rank,
                                                                            outer_iter,
                                                                            size_per_buffer_kernel,
                                                                            buffer_index_kernel,
                                                                            threads_needed_per_chunk);
                                    break;
                                case 12:
                                    nocopy_reduce_read_write<12, data_type>((int *)even_ranks,
                                                                            my_rank_index,
                                                                            index,
                                                                            (void **)in_buffers,
                                                                            out_buffer,
                                                                            recv_size,
                                                                            threads_already_processed,
                                                                            (void **)temp_buffer,
                                                                            temp_rank,
                                                                            outer_iter,
                                                                            size_per_buffer_kernel,
                                                                            buffer_index_kernel,
                                                                            threads_needed_per_chunk);
                                    break;
                                case 14:
                                    nocopy_reduce_read_write<14, data_type>((int *)even_ranks,
                                                                            my_rank_index,
                                                                            index,
                                                                            (void **)in_buffers,
                                                                            out_buffer,
                                                                            recv_size,
                                                                            threads_already_processed,
                                                                            (void **)temp_buffer,
                                                                            temp_rank,
                                                                            outer_iter,
                                                                            size_per_buffer_kernel,
                                                                            buffer_index_kernel,
                                                                            threads_needed_per_chunk);
                                    break;
                                case 16:
                                    nocopy_reduce_read_write<16, data_type>((int *)even_ranks,
                                                                            my_rank_index,
                                                                            index,
                                                                            (void **)in_buffers,
                                                                            out_buffer,
                                                                            recv_size,
                                                                            threads_already_processed,
                                                                            (void **)temp_buffer,
                                                                            temp_rank,
                                                                            outer_iter,
                                                                            size_per_buffer_kernel,
                                                                            buffer_index_kernel,
                                                                            threads_needed_per_chunk);
                                    break;
                                default: break;
                            }
                        }
                        });//parallel_for
            }); //submit()
            //printf("kernel2\n");
            dep_events.clear();
            dep_events.push_back(std::move(e));

            //sync all the ranks here before consuming the results.
            e = global_sync(queue, temp_rank, temp_world, SYNC_BYTE * buffer_index_kernel, 1, 1, dep_events);
            dep_events.clear();
            dep_events.push_back(std::move(e));
            //printf("kernel3\n");

            if (temp_world > 2) {
                int innerloop_local_sum_iter_count __attribute__((unused)) =
                    (threads_needed_per_chunk + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;
                uint32_t persist_local_sum_threads_needed = threads_needed_per_chunk;
                if (persist_local_sum_threads_needed > HW_THREAD_COUNT)
                    persist_local_sum_threads_needed = HW_THREAD_COUNT;
                //local reduction kernel
                e = queue.submit([&](sycl::handler &cgh) {
                    cgh.depends_on(dep_events);
                    cgh.parallel_for<class ReduceScatterMediumKernel_nocopy_local_all_sum<data_type>>(
                        sycl::nd_range<1>({ persist_local_sum_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        //ESIMD kernel
                        uint32_t idx = idx2.get_global_id();
                        for (int inner_iter = 0; inner_iter < innerloop_local_sum_iter_count; inner_iter++) {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= threads_needed_per_chunk)
                                break;

                            switch (temp_world) {
                                case 2:
                                    all_sum<2, data_type>(index,
                                                          send_buf,
                                                          out_buffer,
                                                          recv_size,
                                                          threads_already_processed,
                                                          (void **)temp_buffer,
                                                          temp_rank,
                                                          outer_iter,
                                                          size_per_buffer_kernel,
                                                          buffer_index_kernel,
                                                          threads_needed_per_chunk);
                                    break;
                                case 4:
                                    all_sum<4, data_type>(index,
                                                          send_buf,
                                                          out_buffer,
                                                          recv_size,
                                                          threads_already_processed,
                                                          (void **)temp_buffer,
                                                          temp_rank,
                                                          outer_iter,
                                                          size_per_buffer_kernel,
                                                          buffer_index_kernel,
                                                          threads_needed_per_chunk);
                                    break;
                                case 6:
                                    all_sum<6, data_type>(index,
                                                          send_buf,
                                                          out_buffer,
                                                          recv_size,
                                                          threads_already_processed,
                                                          (void **)temp_buffer,
                                                          temp_rank,
                                                          outer_iter,
                                                          size_per_buffer_kernel,
                                                          buffer_index_kernel,
                                                          threads_needed_per_chunk);
                                    break;
                                case 8:
                                    all_sum<8, data_type>(index,
                                                          send_buf,
                                                          out_buffer,
                                                          recv_size,
                                                          threads_already_processed,
                                                          (void **)temp_buffer,
                                                          temp_rank,
                                                          outer_iter,
                                                          size_per_buffer_kernel,
                                                          buffer_index_kernel,
                                                          threads_needed_per_chunk);
                                    break;
                                case 10:
                                    all_sum<10, data_type>(index,
                                                           send_buf,
                                                           out_buffer,
                                                           recv_size,
                                                           threads_already_processed,
                                                           (void **)temp_buffer,
                                                           temp_rank,
                                                           outer_iter,
                                                           size_per_buffer_kernel,
                                                           buffer_index_kernel,
                                                           threads_needed_per_chunk);
                                    break;
                                case 12:
                                    all_sum<12, data_type>(index,
                                                           send_buf,
                                                           out_buffer,
                                                           recv_size,
                                                           threads_already_processed,
                                                           (void **)temp_buffer,
                                                           temp_rank,
                                                           outer_iter,
                                                           size_per_buffer_kernel,
                                                           buffer_index_kernel,
                                                           threads_needed_per_chunk);
                                    break;
                                case 14:
                                    all_sum<14, data_type>(index,
                                                           send_buf,
                                                           out_buffer,
                                                           recv_size,
                                                           threads_already_processed,
                                                           (void **)temp_buffer,
                                                           temp_rank,
                                                           outer_iter,
                                                           size_per_buffer_kernel,
                                                           buffer_index_kernel,
                                                           threads_needed_per_chunk);
                                    break;
                                case 16:
                                    all_sum<16, data_type>(index,
                                                           send_buf,
                                                           out_buffer,
                                                           recv_size,
                                                           threads_already_processed,
                                                           (void **)temp_buffer,
                                                           temp_rank,
                                                           outer_iter,
                                                           size_per_buffer_kernel,
                                                           buffer_index_kernel,
                                                           threads_needed_per_chunk);
                                    break;
                                default: break;
                            }
                        }
                        });//parallel_for
                }); //submit()
                dep_events.clear();
                dep_events.push_back(std::move(e));
                //printf("kernel4\n");
            } // end if

            threads_already_processed += threads_needed_per_chunk;
            buffer_index++;
            buffer_index %= BUFFER_COUNT;
            buffer_index_kernel = buffer_index;
        } //for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++)

        e = dep_events.back();
        return e;
    }

    //sync all the ranks here before consuming the results.
    // offset = size_per_buffer_for_sync_kernel * buffer_index_kernel
    sycl::event global_sync(sycl::queue queue,
                            int temp_rank,
                            uint32_t temp_world,
                            int offset,
                            int index,
                            int reset,
                            const ccl::vector_class<sycl::event> &dep_events) {
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
            cgh.depends_on(dep_events);
            cgh.parallel_for<class ReduceScatterMediumKernel_GlobalSync<data_type>>(
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
                           int reset,
                           const ccl::vector_class<sycl::event> &dep_events) {
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
            cgh.depends_on(dep_events);
            cgh.parallel_for<class ReduceScatterMediumKernel_LocalSync<data_type>>(
                sycl::nd_range<1>({ total_threads_needed_sync }, wg_size), [=](sycl::nd_item<1> idx) SYCL_ESIMD_KERNEL
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
    size_t size_per_buffer{ 0 }; // todo align size_t or int for all algos
    int data_size_per_buffer{ ccl::utils::invalid_bytes_value };
    uint32_t max_count_per_rank{ 0 };
    int buffer_index{ ccl::utils::invalid_err_code };
    ccl_stream *global_stream{};
    ccl_comm *comm{};
    ccl_comm *even_comm{};
};

#define REDUCE_SCATTER_MEDIUM_API(TYPE) \
    void init_reduce_scatter_medium_##TYPE(ccl::datatype dtype, \
                                           sycl::queue &queue, \
                                           ccl_comm *comm, \
                                           ccl_stream *stream, \
                                           uint32_t rank_in, \
                                           uint32_t world_in) { \
        if (!rs_medium_##TYPE.inited()) { \
            LOG_INFO("invoking medium reduce_scatter first time for datatype: ", dtype); \
            rs_medium_##TYPE.init(queue, comm, stream, rank_in, world_in); \
        } \
    } \
\
    ccl::event run_reduce_scatter_medium_##TYPE(ccl::datatype dtype, \
                                                sycl::queue &queue, \
                                                const void *send_buf, \
                                                void *recv_buf, \
                                                size_t recv_count, \
                                                ccl::reductuin reduction, \
                                                const ccl::vector_class<ccl::event> &deps, \
                                                bool &done) { \
        return rs_medium_##TYPE.reduce_scatter( \
            queue, send_buf, recv_buf, dtype, recv_count, reduction, deps, done); \
    }

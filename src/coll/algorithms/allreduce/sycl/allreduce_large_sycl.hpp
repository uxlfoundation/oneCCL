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
#define INIT_SIZE        64
#define INIT_COUNT       1
#define SIMD_INIT        (INIT_SIZE * INIT_COUNT)
#define SIMD_COMPUTE_MAX 256
#define SIMD_COMPUTE     (SIMD_COMPUTE_MAX / sizeof(data_type))
#define SIMD_SYNC        32
#define SYNC_BYTE        (SIMD_SYNC * sizeof(int) * 2)
#define ALIGNMENT_BYTE   256
#define NOCOPY_MAX_SIZE  (128 * 1024 * 1024)
#define COPY_MAX_SIZE    (64 * 1024 * 1024)
//#define EU_COUNT_PER_RANK 448
#define EU_COUNT_PER_RANK   512
#define THREAD_COUNT_PER_EU 8
#define HW_THREAD_COUNT     (EU_COUNT_PER_RANK * THREAD_COUNT_PER_EU)
#define RANKS_PER_GPU       2
#define NO_KERNEL           0
#define FIRST_KERNEL        1
#define SECOND_KERNEL       2
#define THIRD_KERNEL        4
#define FOURTH_KERNEL       8
#define FIFTH_KERNEL        16

#define NOCOPY_KERNEL_NUM   3
#define NOCOPY_LAST_KERNEL  THIRD_KERNEL
#define NOCOPY_BUFFER_COUNT NOCOPY_KERNEL_NUM

#define COPY_KERNEL_NUM   5
#define COPY_LAST_KERNEL  FIFTH_KERNEL
#define COPY_BUFFER_COUNT COPY_KERNEL_NUM

extern void *allreduce_large_buffer;
extern void *allreduce_large_buffers[MAX_RANK];
extern void *allreduce_large_sync_buffer[MAX_RANK];
extern size_t allreduce_large_offsets[MAX_RANK];
extern ze_ipc_mem_handle_t allreduce_large_ipc_handle[MAX_RANK];
extern int allreduce_large_buffer_index;

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void load_input_to_temp_buffer(int idx,
                               const void *in_buffer,
                               uint32_t size,
                               int threads_already_processed,
                               void *temp_buffer[],
                               uint32_t temp_rank,
                               int size_per_buffer_kernel,
                               int buffer_index_kernel2) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    uint32_t read_offset = (idx + threads_already_processed) * SIMD_COMPUTE * TEMP_WORLD;
    simd<data_type, SIMD_COMPUTE *TEMP_WORLD> buffer = 0;

    if (read_offset + SIMD_COMPUTE * TEMP_WORLD > size) {
        int count = (size - read_offset + SIMD_COMPUTE - 1) / SIMD_COMPUTE;
        for (int i = 0; i < count; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)in_buffer + read_offset + i * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
    }
    else {
#pragma unroll
        for (unsigned int i = 0; i < TEMP_WORLD; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)in_buffer + read_offset + i * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
    }

    data_type *ptr = (data_type *)temp_buffer[temp_rank];
    ptr += size_per_buffer_kernel * buffer_index_kernel2;
    ptr += idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD; i++) {
        block_store<data_type, SIMD_COMPUTE>(
            ptr + i * SIMD_COMPUTE,
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
            properties{ alignment<align> });
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void local_sum_and_distribute_to_remote_ranks(int *even_ranks,
                                              int myrank,
                                              int idx,
                                              const void *in_buffer,
                                              uint32_t size,
                                              int threads_already_processed,
                                              void *temp_buffer[],
                                              uint32_t temp_rank,
                                              int size_per_buffer_kernel,
                                              int buffer_index_kernel2) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    int is_odd = (even_ranks[0] == 1);
    //read the input data
    data_type *ptr_even =
        (data_type *)temp_buffer[temp_rank & 0xfffffffe] + is_odd * SIMD_COMPUTE * TEMP_WORLD / 2;
    data_type *ptr_odd =
        (data_type *)temp_buffer[temp_rank | 1] + is_odd * SIMD_COMPUTE * TEMP_WORLD / 2;
    ptr_even +=
        idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 + size_per_buffer_kernel * buffer_index_kernel2;
    ptr_odd +=
        idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 + size_per_buffer_kernel * buffer_index_kernel2;
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD> buffer;
    uint32_t i;
#pragma unroll
    for (i = 0; i < TEMP_WORLD / 2; i++) {
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>((data_type *)ptr_even + i * SIMD_COMPUTE,
                                                properties{ alignment<align> });
    }
#pragma unroll
    for (i = TEMP_WORLD / 2; i < TEMP_WORLD; i++) {
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>(
                (data_type *)ptr_odd + (i - TEMP_WORLD / 2) * SIMD_COMPUTE,
                properties{ alignment<align> });
    }
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> sum;
    sum = buffer.template select<SIMD_COMPUTE * TEMP_WORLD / 2, 1>(0) +
          buffer.template select<SIMD_COMPUTE * TEMP_WORLD / 2, 1>(SIMD_COMPUTE * TEMP_WORLD / 2);

    //store the result in at (SIMD_COMPUTE * TEMP_WORLD) offset in remote ranks' temp buffers.
    //distribute to other ranks. But even(odd) rank goes to other even(odd) rank.
#pragma unroll
    for (i = 0; i < TEMP_WORLD / 2; i++) {
        data_type *ptr = (data_type *)temp_buffer[even_ranks[i]];
        ptr += idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 +
               size_per_buffer_kernel * buffer_index_kernel2 + TEMP_WORLD * SIMD_COMPUTE;
        block_store<data_type, SIMD_COMPUTE>(ptr + (temp_rank / 2) * SIMD_COMPUTE,
                                             sum.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                                             properties{ alignment<align> });
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void all_sum(int idx,
             const void *in_buffer,
             uint32_t size,
             int threads_already_processed,
             void *temp_buffer[],
             uint32_t temp_rank,
             int size_per_buffer_kernel,
             int buffer_index_kernel2) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    data_type *ptr = (data_type *)temp_buffer[temp_rank];
    int read_offset =
        idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 +
        SIMD_COMPUTE *
            TEMP_WORLD; //points to second half of the temp slot since that's where the data is from other ranks.
    ptr += read_offset + size_per_buffer_kernel * buffer_index_kernel2;
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> buffer;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>((data_type *)ptr + i * SIMD_COMPUTE,
                                                properties{ alignment<align> });
    }
    simd<data_type, SIMD_COMPUTE> sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        sum = sum + buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i);
    }
    //store the result back to same location, only one SIMD_COMPUTE
    // save the all sum in the second half of the temp slot.
    block_store<data_type, SIMD_COMPUTE>(ptr, sum, properties{ alignment<align> });
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void gather_from_remote_and_dist_to_rank_pair(int *even_ranks,
                                              int idx,
                                              void *out_buffer,
                                              uint32_t size,
                                              int threads_already_processed,
                                              void *temp_buffer[],
                                              uint32_t temp_rank,
                                              int size_per_buffer_kernel,
                                              int buffer_index_kernel2) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> buffer;

#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        //read the values
        data_type *read_ptr = (data_type *)temp_buffer[even_ranks[i]];
        read_ptr += size_per_buffer_kernel * buffer_index_kernel2;
        read_ptr += idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 +
                    SIMD_COMPUTE * TEMP_WORLD; //get the sum from the second half of temp slot
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>(read_ptr, properties{ alignment<align> });
    }

    //write the data to the pair of ranks within the same gpu
    //gather in the first half of the slot
    data_type *mdfi_ptr = (data_type *)temp_buffer[temp_rank ^ 1];
    mdfi_ptr += size_per_buffer_kernel * buffer_index_kernel2;
    mdfi_ptr += idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        // save the results in the first half of temp slot
        block_store<data_type, SIMD_COMPUTE>(
            mdfi_ptr + i * SIMD_COMPUTE,
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
            properties{ alignment<align> });
    }

    int is_odd = (even_ranks[0] == 1);
    data_type *out_ptr = (data_type *)out_buffer;
    uint32_t write_offset = (idx + threads_already_processed) * SIMD_COMPUTE * TEMP_WORLD +
                            is_odd * SIMD_COMPUTE * TEMP_WORLD / 2;
    if (write_offset + SIMD_COMPUTE * TEMP_WORLD / 2 <= size) {
#pragma unroll
        for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                out_ptr + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
    }
    else if (write_offset < size) {
        int vec_count = (size - write_offset) / SIMD_COMPUTE;
        for (int i = 0; i < vec_count; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                out_ptr + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
        int count = size - write_offset - vec_count * SIMD_COMPUTE;
        for (int i = 0; i < count; i++) {
            out_ptr[write_offset + vec_count * SIMD_COMPUTE + i] =
                buffer[vec_count * SIMD_COMPUTE + i];
        }
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void write_output(int *even_ranks,
                  int idx,
                  void *out_buffer,
                  uint32_t size,
                  int threads_already_processed,
                  void *temp_buffer[],
                  uint32_t temp_rank,
                  int size_per_buffer_kernel,
                  int buffer_index_kernel2) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> buffer;
    data_type *read_ptr = (data_type *)temp_buffer[temp_rank];
    read_ptr +=
        idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 + size_per_buffer_kernel * buffer_index_kernel2;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        //read the values
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>(read_ptr + i * SIMD_COMPUTE,
                                                properties{ alignment<align> });
    }

    int is_odd = (even_ranks[0] == 1);
    //write out the results
    data_type *write_ptr = (data_type *)out_buffer;
    uint32_t write_offset = (idx + threads_already_processed) * SIMD_COMPUTE * TEMP_WORLD +
                            (1 - is_odd) * SIMD_COMPUTE * TEMP_WORLD / 2;
    if (write_offset + SIMD_COMPUTE * TEMP_WORLD / 2 <= size) {
#pragma unroll
        for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                write_ptr + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
    }
    else if (write_offset < size) {
        int vec_count = (size - write_offset) / SIMD_COMPUTE;
        for (int i = 0; i < vec_count; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                write_ptr + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
        int count = size - write_offset - vec_count * SIMD_COMPUTE;
        for (int i = 0; i < count; i++) {
            write_ptr[write_offset + vec_count * SIMD_COMPUTE + i] =
                buffer[vec_count * SIMD_COMPUTE + i];
        }
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void nocopy_sum_and_distribute_to_remote_ranks(int *even_ranks,
                                               int myrank,
                                               int idx,
                                               void **in_buffers,
                                               uint32_t size,
                                               int threads_already_processed,
                                               void *temp_buffer[],
                                               uint32_t temp_rank,
                                               int size_per_buffer_kernel,
                                               int buffer_index_kernel2) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    int is_odd = (even_ranks[0] == 1);
    uint32_t read_offset = (idx + threads_already_processed) * SIMD_COMPUTE * TEMP_WORLD +
                           is_odd * SIMD_COMPUTE * TEMP_WORLD / 2;
    if (read_offset >= size) {
        return;
    }

    //read the input data
    data_type *ptr_even = (data_type *)in_buffers[temp_rank & 0xfffffffe];
    data_type *ptr_odd = (data_type *)in_buffers[temp_rank | 1];
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> sum;
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD> buffer;
    uint32_t i;
    if (read_offset + (TEMP_WORLD / 2) * SIMD_COMPUTE <= size) {
#pragma unroll
        for (i = 0; i < TEMP_WORLD / 2; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)ptr_even + read_offset + i * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
#pragma unroll
        for (i = TEMP_WORLD / 2; i < TEMP_WORLD; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)ptr_odd + read_offset + (i - TEMP_WORLD / 2) * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
    }
    else {
        int count = (size - read_offset + SIMD_COMPUTE - 1) / SIMD_COMPUTE;
        for (i = 0; i < count; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)ptr_even + read_offset + i * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
        for (i = TEMP_WORLD / 2; i < TEMP_WORLD / 2 + count; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)ptr_odd + read_offset + (i - TEMP_WORLD / 2) * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
    }

    sum = buffer.template select<SIMD_COMPUTE * TEMP_WORLD / 2, 1>(0) +
          buffer.template select<SIMD_COMPUTE * TEMP_WORLD / 2, 1>(SIMD_COMPUTE * TEMP_WORLD / 2);

    //store the result in at (SIMD_COMPUTE * TEMP_WORLD) offset in remote ranks' temp buffers.
    //distribute to other ranks. But even(odd) rank goes to other even(odd) rank.
#pragma unroll
    for (i = 0; i < TEMP_WORLD / 2; i++) {
        data_type *ptr = (data_type *)temp_buffer[even_ranks[i]];
        ptr += idx * SIMD_COMPUTE * TEMP_WORLD + size_per_buffer_kernel * buffer_index_kernel2;
        block_store<data_type, SIMD_COMPUTE>(ptr + (temp_rank / 2) * SIMD_COMPUTE,
                                             sum.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                                             properties{ alignment<align> });
    }
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void nocopy_all_sum(int idx,
                    const void *in_buffer,
                    uint32_t size,
                    int threads_already_processed,
                    void *temp_buffer[],
                    uint32_t temp_rank,
                    int size_per_buffer_kernel,
                    int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    data_type *ptr = (data_type *)temp_buffer[temp_rank];
    int read_offset = idx * SIMD_COMPUTE * TEMP_WORLD;
    ptr +=
        read_offset +
        size_per_buffer_kernel *
            buffer_index_kernel; //points to second half of the temp slot since that's where the data is from other ranks.
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> buffer;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>((data_type *)ptr + i * SIMD_COMPUTE,
                                                properties{ alignment<align> });
    }
    simd<data_type, SIMD_COMPUTE> sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        sum = sum + buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i);
    }
    //store the result
    // save the all sum in the second half of the temp slot.
    block_store<data_type, SIMD_COMPUTE>(ptr, sum, properties{ alignment<align> });
}

template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void nocopy_gather_from_remote_and_dist_to_rank_pair(int *even_ranks,
                                                     int idx,
                                                     void **out_buffers,
                                                     uint32_t size,
                                                     int threads_already_processed,
                                                     void *temp_buffer[],
                                                     uint32_t temp_rank,
                                                     int size_per_buffer_kernel,
                                                     int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    int is_odd = (even_ranks[0] == 1);
    //read the input data
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> buffer;

#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        //read the values
        data_type *read_ptr_int = (data_type *)temp_buffer[even_ranks[i]];
        read_ptr_int += size_per_buffer_kernel * buffer_index_kernel;
        read_ptr_int += idx * SIMD_COMPUTE * TEMP_WORLD;
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>(read_ptr_int, properties{ alignment<align> });
    }

    //write the data to the pair of ranks within the same gpu
    data_type *ptr_even = (data_type *)out_buffers[temp_rank ^ 1];
    data_type *ptr_odd = (data_type *)out_buffers[temp_rank];
    uint32_t write_offset = (idx + threads_already_processed) * SIMD_COMPUTE * TEMP_WORLD +
                            is_odd * SIMD_COMPUTE * TEMP_WORLD / 2;
    uint32_t i;
    if (write_offset + SIMD_COMPUTE * TEMP_WORLD / 2 <= size) {
#pragma unroll
        for (i = 0; i < TEMP_WORLD / 2; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                ptr_even + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
            block_store<data_type, SIMD_COMPUTE>(
                ptr_odd + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
    }
    else if (write_offset < size) {
        uint32_t vec_count = (size - write_offset) / SIMD_COMPUTE;
        for (i = 0; i < vec_count; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                ptr_even + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
            block_store<data_type, SIMD_COMPUTE>(
                ptr_odd + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
        uint32_t count = size - write_offset - vec_count * SIMD_COMPUTE;
        for (i = 0; i < count; i++) {
            ptr_even[write_offset + vec_count * SIMD_COMPUTE + i] =
                buffer[vec_count * SIMD_COMPUTE + i];
            ptr_odd[write_offset + vec_count * SIMD_COMPUTE + i] =
                buffer[vec_count * SIMD_COMPUTE + i];
        }
    }
}

#define RUN_FIRST_KERNEL \
    if (sw_pipeline_kernel_state[ii] & FIRST_KERNEL) { \
        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) { \
            int index = idx + inner_iter * HW_THREAD_COUNT; \
            if ((uint32_t)index >= total_threads_needed) \
                break; \
\
            auto nocopy_sum_and_distribute_to_remote_ranks_lambda = \
                [&]<int w, size_t a>() SYCL_ESIMD_KERNEL { \
                    nocopy_sum_and_distribute_to_remote_ranks<w, data_type, a>( \
                        (int *)even_ranks, \
                        myrank, \
                        index, \
                        (void **)in_buffers, \
                        size, \
                        threads_already_processed[ii], \
                        (void **)temp_buffer, \
                        temp_rank, \
                        size_per_buffer_kernel, \
                        ii); \
                }; \
            if (align4) \
                invoke_esimd_function<4>(nocopy_sum_and_distribute_to_remote_ranks_lambda, \
                                         temp_world); \
            else \
                invoke_esimd_function<2>(nocopy_sum_and_distribute_to_remote_ranks_lambda, \
                                         temp_world); \
        } \
    }

#define RUN_SECOND_KERNEL \
    if (sw_pipeline_kernel_state[ii] & SECOND_KERNEL) { \
        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) { \
            int index = idx + inner_iter * HW_THREAD_COUNT; \
            if ((uint32_t)index >= total_threads_needed) \
                break; \
\
            auto nocopy_all_sum_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL { \
                nocopy_all_sum<w, data_type, a>(index, \
                                                in_buffer, \
                                                size, \
                                                threads_already_processed[ii], \
                                                (void **)temp_buffer, \
                                                temp_rank, \
                                                size_per_buffer_kernel, \
                                                ii); \
            }; \
            if (align4) \
                invoke_esimd_function<4>(nocopy_all_sum_lambda, temp_world); \
            else \
                invoke_esimd_function<2>(nocopy_all_sum_lambda, temp_world); \
        } \
    }

#define RUN_THIRD_KERNEL \
    if (sw_pipeline_kernel_state[ii] & THIRD_KERNEL) { \
        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) { \
            int index = idx + inner_iter * HW_THREAD_COUNT; \
            if ((uint32_t)index >= total_threads_needed) \
                break; \
\
            auto nocopy_gather_from_remote_and_dist_to_rank_pair_lambda = \
                [&]<int w, size_t a>() SYCL_ESIMD_KERNEL { \
                    nocopy_gather_from_remote_and_dist_to_rank_pair<w, data_type, a>( \
                        (int *)even_ranks, \
                        index, \
                        (void **)out_buffers, \
                        size, \
                        threads_already_processed[ii], \
                        (void **)temp_buffer, \
                        temp_rank, \
                        size_per_buffer_kernel, \
                        ii); \
                }; \
            if (align4) \
                invoke_esimd_function<4>(nocopy_gather_from_remote_and_dist_to_rank_pair_lambda, \
                                         temp_world); \
            else \
                invoke_esimd_function<2>(nocopy_gather_from_remote_and_dist_to_rank_pair_lambda, \
                                         temp_world); \
        } \
    }

template <typename dtype, size_t align>
class Kernel_compute;
//template<typename dtype> class Kernel_rankSync;

template <typename dtype>
class NoCopyKernel_compute;
//template<typename dtype> class NoCopyKernel_GlobalSync;

template <typename dtype>
class AllreduceLargeKernel_GlobalSync;
template <typename dtype>
class AllreduceLargeKernel_LocalSync;

template <typename data_type, uint32_t max_rank = MAX_RANK>
class sycl_allreduce_large : public sycl_coll_base<data_type> {
public:
    sycl_allreduce_large() : sycl_coll_base<data_type>() {
        size_per_buffer = 0;
    }

    void init(sycl::queue &queue,
              ccl_comm *comm_in,
              ccl_stream *stream,
              uint32_t rank_in,
              uint32_t world_in) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.
        COPY_MAX_COUNT = COPY_MAX_SIZE / sizeof(data_type);

        int size_per_buffer_copy = COPY_MAX_SIZE + SYNC_BYTE;
        int alloc_size_copy = size_per_buffer_copy * COPY_BUFFER_COUNT;

        NOCOPY_MAX_COUNT = NOCOPY_MAX_SIZE / sizeof(data_type);

        int size_per_buffer_nocopy = NOCOPY_MAX_SIZE + SYNC_BYTE;
        int alloc_size_nocopy = size_per_buffer_nocopy * NOCOPY_BUFFER_COUNT;

        if (ccl::global_data::env().sycl_allreduce_tmp_buf) {
            data_size_per_buffer = COPY_MAX_COUNT;
            size_per_buffer = size_per_buffer_copy;
        }
        else {
            data_size_per_buffer = NOCOPY_MAX_COUNT;
            size_per_buffer = size_per_buffer_nocopy;
        }

        if (!allreduce_large_buffer) {
            int alloc_size =
                alloc_size_copy > alloc_size_nocopy ? alloc_size_copy : alloc_size_nocopy;
            allreduce_large_buffer = sycl::malloc_device(alloc_size, queue);
            auto e = queue.memset(allreduce_large_buffer, 0, alloc_size);
            e.wait();

            this->exchange_peer_ipc_mem(queue,
                                        comm_in,
                                        stream,
                                        allreduce_large_buffer,
                                        NULL,
                                        rank,
                                        world,
                                        data_size_per_buffer * sizeof(data_type),
                                        (void **)allreduce_large_buffers,
                                        (void **)allreduce_large_sync_buffer,
                                        allreduce_large_offsets,
                                        allreduce_large_ipc_handle,
                                        NULL,
                                        NULL /* mmap_buffers */,
                                        false /* to_cache */);
        }
        this->initialized = true;

        global_stream = stream;
        this->comm = comm_in;
        even_comm = comm_in->get_even_comm().get();
    }

    ccl::event allreduce(sycl::queue &queue,
                         const void *in_buffer,
                         void *out_buffer,
                         ccl::datatype dtype,
                         size_t size,
                         ccl::reduction reduction,
                         const ccl::vector_class<ccl::event> &deps,
                         bool &done) {
        sycl::event e;
        done = true;
        // check local alignment
        bool is_aligned = (size_t)in_buffer % 4 == 0 && (size_t)out_buffer % 4 == 0;

        if (ccl::global_data::env().sycl_allreduce_tmp_buf) {
            if (is_aligned)
                e = allreduce_copy<4>(queue, in_buffer, out_buffer, size);
            else
                e = allreduce_copy<2>(queue, in_buffer, out_buffer, size);
        }
        else {
            e = allreduce_nocopy(queue, in_buffer, out_buffer, size, done);
        }

        if (reduction == ccl::reduction::avg) {
            std::vector<sycl::event> evs;
            evs.push_back(e);
            e = sycl_average(queue, out_buffer, size, world, dtype, evs);
        }

        return ccl::event::create_from_native(e);
    }

private:
    template <size_t align>
    sycl::event allreduce_copy(sycl::queue &queue,
                               const void *in_buffer,
                               void *out_buffer,
                               size_t size) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        assert(this->initialized == true);
        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = allreduce_large_buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = allreduce_large_sync_buffer[i];
        }

        int even_ranks[max_rank];
        int myrank;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank) {
                myrank = i;
            }
        }

        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int buffer_index_kernel_for_sync = allreduce_large_buffer_index;
        int outer_iter;
        //todo:
        //1. shuffle the kernel# executions so that resource utilization can be smoothed out. DONE
        //2. increase the simd size there are less number of innerloop iterations. This mgiht be useful in reducing hte load stalls since the number of loads-consume pair is less. DONE
        //3. reduce gpu-cpu sync?? DONE
        //5. prefetch in persistent threads? DONE
        //uint32_t total_threads_needed_sync = 1;
        int wg_size = 1;
        int start, end;

        int outerloop_iter_count;
        int sync_reset_counter = 0;
        int max_threads_per_MAX_COUNT =
            (COPY_MAX_COUNT * 2 / 3) /
            (SIMD_COMPUTE *
             temp_world); // each thread uses (SIMD_COMPUTE * temp_world) * 3 / 2 space
        int max_elements_per_MAX_COUNT = max_threads_per_MAX_COUNT * (SIMD_COMPUTE * temp_world);

        outerloop_iter_count =
            size /
            max_elements_per_MAX_COUNT; //this is the outerloop count that requires full hw thread count. This doesnt include the outloop iteration that only needs partial thread count

        //init the sw pipeline
        int sw_pipeline_insert_index = 0;
        int sw_pipeline_insert_counter = 0;
        int sw_pipeline_kernel_state[COPY_KERNEL_NUM];
        int threads_already_processed[COPY_KERNEL_NUM];
        for (int i = 0; i < COPY_KERNEL_NUM; i++) {
            threads_already_processed[i] = 0;
            sw_pipeline_kernel_state[i] = NO_KERNEL;
        }
        //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // pipeline design
        // ---------------
        // During each outerloop iteration, one iteration (job) will be inserted to the SW pipeline.
        // Since there are 5 kernels in the workload, upto 5 jobs can be inflight as shown in the picture below. Hence only 5 states are needed in the implementation.
        // For each job in the SW pipeline, kernels from 1 to 5 will be executed in 5 iterations in the order. While it is being executed in 5 iterations, more jobs can be added to the SW pipeline.
        // Which means that at particular time, all five kernels will be executed together by a single rank. That means the HW resource utilization might be more balanced hence the improvements.
        // Additionally, by using the SW pipelininig, the required number of syncs are reduced. The syncs in the same column in the picture below can be done by one sync execution.
        //
        //                          time0   time1   time2   time3   time4   time5   time6   time7   time8   time9   time10  time11  time12  time13  time14  time15  time16  time17  time16
        //                          ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------
        // outerloop iteration0:    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration1:                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration2:                                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration3:                                                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration4:                                                                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration5:                                                                                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // ...
        //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        //The following outer-outer loop is handling the case where there are multiple outerloop iterations and the last iteration requires partial usage of the temp buffer with size (MAX_COUNT / 2).
        //As more space is used inside the temp buffer, higher innerloop count is required. Separating the processing into two iterations (one for full usage and another for partial temp buffer usage), the number of innerloop iteration within each iteration is uniform.
        for (int iter = 0; iter < 2; iter++) {
            uint32_t total_threads_needed;
            if (iter == 1) //if second iteration, then handle the partial usage of the temp buffer
            {
                //if there is little more left to compute, then finish them
                if (outerloop_iter_count * max_elements_per_MAX_COUNT < (int)size) {
                    start = outerloop_iter_count;
                    end = start + 1;
                    total_threads_needed = (size - start * max_elements_per_MAX_COUNT +
                                            SIMD_COMPUTE * temp_world - 1) /
                                           (SIMD_COMPUTE * temp_world);
                }
                else {
                    break;
                }
            }
            else {
                start = 0;
                end = outerloop_iter_count;
                total_threads_needed = max_threads_per_MAX_COUNT;

                if (end == 0) {
                    //there is nothing to do when end is 0 so check the next iter.
                    continue;
                }
            }
            int innerloop_iter_count =
                (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;
            uint32_t persist_threads_needed = total_threads_needed;
            if (persist_threads_needed > HW_THREAD_COUNT)
                persist_threads_needed = HW_THREAD_COUNT;

            //There are total of two SW pipeline sessions, for iter={0, 1}
            //SW pipeline is applied on outerloop processing.
            //Since sw pipeline is implemented, there will be tail at the end of hte execution. The size of the tail is (KERNEL_NUM - 1) and the tail is completed in the following loop.
            for (outer_iter = start; outer_iter < end + COPY_KERNEL_NUM - 1; outer_iter++) {
                //if more outer_iter remaining since there is more new processing to do, then insert them to the SW pipeline.
                //During the sw pipeline tail, there is nothing to dispatch.
                if (outer_iter < end) {
                    sw_pipeline_kernel_state[sw_pipeline_insert_index] = FIRST_KERNEL;
                    threads_already_processed[sw_pipeline_insert_index] =
                        sw_pipeline_insert_counter;
                    sw_pipeline_insert_index++;
                    if (sw_pipeline_insert_index >= COPY_KERNEL_NUM) {
                        //By the time the index wraps arounds, the kernel that was in this slot previously has already completed.
                        sw_pipeline_insert_index = 0;
                    }
                    sw_pipeline_insert_counter += total_threads_needed;
                }

                //The first kernel does the actual computation while the second kernel does the sync across ranks.
                e = queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for<class Kernel_compute<data_type, align>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        uint32_t idx = idx2.get_global_id();
                        //ESIMD kernel
                        //check if there is any kernel in the SW pipelines. If yes, execute them.
                        //to optimize, the order of loop i=0,1,2,.. can be shuffled so that different ranks can do different kernels at particular time. The purpose is to better balance the HW resource usage in the PVC node.
                        for (int ii = 0; ii < COPY_KERNEL_NUM; ii++) {
                            if (sw_pipeline_kernel_state[ii] & FIRST_KERNEL) {
                                for (int inner_iter = 0; inner_iter < innerloop_iter_count;
                                     inner_iter++) {
                                    int index = idx + inner_iter * HW_THREAD_COUNT;
                                    if ((uint32_t)index >= total_threads_needed)
                                        break;

                                    auto load_input_to_temp_buffer_lambda =
                                        [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                            load_input_to_temp_buffer<w, data_type, a>(
                                                index,
                                                in_buffer,
                                                size,
                                                threads_already_processed[ii],
                                                (void **)temp_buffer,
                                                temp_rank,
                                                size_per_buffer_kernel,
                                                ii);
                                        };
                                    invoke_esimd_function<align>(load_input_to_temp_buffer_lambda,
                                                                 temp_world);
                                }
                            }
                            if (sw_pipeline_kernel_state[ii] & SECOND_KERNEL) {
                                for (int inner_iter = 0; inner_iter < innerloop_iter_count;
                                     inner_iter++) {
                                    int index = idx + inner_iter * HW_THREAD_COUNT;
                                    if ((uint32_t)index >= total_threads_needed)
                                        break;

                                    auto local_sum_and_distribute_to_remote_ranks_lambda =
                                        [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                            local_sum_and_distribute_to_remote_ranks<w,
                                                                                     data_type,
                                                                                     a>(
                                                (int *)even_ranks,
                                                myrank,
                                                index,
                                                in_buffer,
                                                size,
                                                threads_already_processed[ii],
                                                (void **)temp_buffer,
                                                temp_rank,
                                                size_per_buffer_kernel,
                                                ii);
                                        };
                                    invoke_esimd_function<align>(
                                        local_sum_and_distribute_to_remote_ranks_lambda,
                                        temp_world);
                                }
                            }
                            if (sw_pipeline_kernel_state[ii] & THIRD_KERNEL) {
                                for (int inner_iter = 0; inner_iter < innerloop_iter_count;
                                     inner_iter++) {
                                    int index = idx + inner_iter * HW_THREAD_COUNT;
                                    if ((uint32_t)index >= total_threads_needed)
                                        break;

                                    auto all_sum_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                        all_sum<w, data_type, a>(index,
                                                                 in_buffer,
                                                                 size,
                                                                 threads_already_processed[ii],
                                                                 (void **)temp_buffer,
                                                                 temp_rank,
                                                                 size_per_buffer_kernel,
                                                                 ii);
                                    };
                                    invoke_esimd_function<align>(all_sum_lambda, temp_world);
                                }
                            }
                            if (sw_pipeline_kernel_state[ii] & FOURTH_KERNEL) {
                                for (int inner_iter = 0; inner_iter < innerloop_iter_count;
                                     inner_iter++) {
                                    int index = idx + inner_iter * HW_THREAD_COUNT;
                                    if ((uint32_t)index >= total_threads_needed)
                                        break;

                                    auto gather_from_remote_and_dist_to_rank_pair_lambda =
                                        [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                            gather_from_remote_and_dist_to_rank_pair<w,
                                                                                     data_type,
                                                                                     a>(
                                                (int *)even_ranks,
                                                index,
                                                out_buffer,
                                                size,
                                                threads_already_processed[ii],
                                                (void **)temp_buffer,
                                                temp_rank,
                                                size_per_buffer_kernel,
                                                ii);
                                        };
                                    invoke_esimd_function<align>(
                                        gather_from_remote_and_dist_to_rank_pair_lambda,
                                        temp_world);
                                }
                            }
                            if (sw_pipeline_kernel_state[ii] & FIFTH_KERNEL) {
                                for (int inner_iter = 0; inner_iter < innerloop_iter_count;
                                     inner_iter++) {
                                    int index = idx + inner_iter * HW_THREAD_COUNT;
                                    if ((uint32_t)index >= total_threads_needed)
                                        break;

                                    auto write_output_lambda =
                                        [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                            write_output<w, data_type, a>(
                                                (int *)even_ranks,
                                                index,
                                                out_buffer,
                                                size,
                                                threads_already_processed[ii],
                                                (void **)temp_buffer,
                                                temp_rank,
                                                size_per_buffer_kernel,
                                                ii);
                                        };
                                    invoke_esimd_function<align>(write_output_lambda, temp_world);
                                }
                            }
                        }

                        });//parallel_for
                }); //submit()

                sync_reset_counter++;

                //sync all the ranks within the single GPU.
                e = global_sync(queue,
                                temp_rank,
                                temp_world,
                                size_per_buffer_for_sync_kernel * buffer_index_kernel_for_sync,
                                4,
                                1);

                //update the sw pipeline process state so that next kernel will be processed in next round
                for (int i = 0; i < COPY_KERNEL_NUM; i++) {
                    if (sw_pipeline_kernel_state[i] & COPY_LAST_KERNEL)
                        sw_pipeline_kernel_state[i] =
                            0; //remove the kernel from the sw pipeline if it is fifth kernel. Everything is already executed.
                    else
                        sw_pipeline_kernel_state[i] <<= 1;
                }

                buffer_index_kernel_for_sync++;
                buffer_index_kernel_for_sync %= COPY_KERNEL_NUM;
            } //for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++)
        } //for (int iter = 0; iter < 2; iter++)

        allreduce_large_buffer_index += sync_reset_counter;
        allreduce_large_buffer_index %= COPY_KERNEL_NUM;

        return e;
    }

    // perform IPC exchange every time
    sycl::event allreduce_nocopy(sycl::queue &queue,
                                 const void *in_buffer,
                                 void *out_buffer,
                                 size_t size,
                                 bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        assert(this->initialized == true);
        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = allreduce_large_buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = allreduce_large_sync_buffer[i];
        }

        int even_ranks[max_rank];
        int myrank;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank)
                myrank = i;
        }

        void *in_buffers[max_rank];
        void *out_buffers[max_rank];
        this->exchange_peer_ipc_mem(queue,
                                    comm,
                                    global_stream,
                                    (void **)in_buffer,
                                    out_buffer,
                                    rank,
                                    world,
                                    0,
                                    (void **)in_buffers,
                                    NULL,
                                    NULL,
                                    NULL,
                                    (void **)out_buffers);
        // ignore size, size does not matter if buffers are aligned
        int align4 = all_aligned((void **)in_buffers, temp_world, 1, 4, 4) &&
                     all_aligned((void **)out_buffers, temp_world, 1, 4, 4);
        if (!align4) {
            // ESIMD performance on 2-byte alignment is not good, fallback
            done = false;
            return e;
        }
        done = true;

        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int buffer_index_kernel_for_sync = allreduce_large_buffer_index;
        int outer_iter;
        int wg_size = 1;
        int start, end;
        int outerloop_iter_count;
        int sync_reset_counter = 0;
        int max_threads_per_MAX_COUNT = (NOCOPY_MAX_COUNT) / (SIMD_COMPUTE * temp_world);
        int max_elements_per_MAX_COUNT = max_threads_per_MAX_COUNT * (SIMD_COMPUTE * temp_world);

        outerloop_iter_count =
            size /
            max_elements_per_MAX_COUNT; //this is the outerloop count that requires full hw thread count. This doesnt include the outloop iteration that only needs partial thread count

        //init the sw pipeline
        int sw_pipeline_insert_index = 0;
        int sw_pipeline_insert_counter = 0;
        int sw_pipeline_kernel_state[NOCOPY_KERNEL_NUM];
        int threads_already_processed[NOCOPY_KERNEL_NUM];
        for (int i = 0; i < NOCOPY_KERNEL_NUM; i++) {
            threads_already_processed[i] = 0;
            sw_pipeline_kernel_state[i] = NO_KERNEL;
        }
        //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        // pipeline design
        // ---------------
        // During each outerloop iteration, one iteration (job) will be inserted to the SW pipeline.
        // Since there are 5 kernels in the workload, upto 5 jobs can be inflight as shown in the picture below. Hence only 5 states are needed in the implementation.
        // For each job in the SW pipeline, kernels from 1 to 5 will be executed in 5 iterations in the order. While it is being executed in 5 iterations, more jobs can be added to the SW pipeline.
        // Which means that at particular time, all five kernels will be executed together by a single rank. That means the HW resource utilization might be more balanced hence the improvements.
        // Additionally, by using the SW pipelininig, the required number of syncs are reduced. The syncs in the same column in the picture below can be done by one sync execution.
        //
        //                          time0   time1   time2   time3   time4   time5   time6   time7   time8   time9   time10  time11  time12  time13  time14  time15  time16  time17  time16
        //                          ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------  ------
        // outerloop iteration0:    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration1:                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration2:                                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration3:                                                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration4:                                                                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // outerloop iteration5:                                                                                    Kernel1 sync    kernel2 sync    kernel3 sync    kernel4 sync    kernel5
        // ...
        //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // we need to sync between two tiles of the same GPU to make sure all data are ready
        e = local_sync(queue,
                       temp_rank,
                       temp_world,
                       size_per_buffer_for_sync_kernel * buffer_index_kernel_for_sync,
                       2,
                       0);

        //The following outer-outer loop is handling the case where there are multiple outerloop iterations and the last iteration requires partial usage of the temp buffer with size (MAX_COUNT / 2).
        //As more space is used inside the temp buffer, higher innerloop count is required. Separating the processing into two iterations (one for full usage and another for partial temp buffer usage), the number of innerloop iteration within each iteration is uniform.
        for (int iter = 0; iter < 2; iter++) {
            uint32_t total_threads_needed;
            if (iter == 1) //if second iteration, then handle the partial usage of the temp buffer
            {
                //if there is little more left to compute, then finish them
                if (outerloop_iter_count * max_elements_per_MAX_COUNT < (int)size) {
                    start = outerloop_iter_count;
                    end = start + 1;
                    total_threads_needed = (size - start * max_elements_per_MAX_COUNT +
                                            SIMD_COMPUTE * temp_world - 1) /
                                           (SIMD_COMPUTE * temp_world);
                }
                else {
                    break;
                }
            }
            else {
                start = 0;
                end = outerloop_iter_count;
                total_threads_needed = max_threads_per_MAX_COUNT;

                if (end == 0) {
                    //there is nothing to do when end is 0 so check the next iter.
                    continue;
                }
            }
            int innerloop_iter_count =
                (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;
            uint32_t persist_threads_needed = total_threads_needed;
            if (persist_threads_needed > HW_THREAD_COUNT)
                persist_threads_needed = HW_THREAD_COUNT;

            //There are total of two SW pipeline sessions, for iter={0, 1}
            //SW pipeline is applied on outerloop processing.
            //Since sw pipeline is implemented, there will be tail at the end of hte execution. The size of the tail is (KERNEL_NUM - 1) and the tail is completed in the following loop.
            for (outer_iter = start; outer_iter < end + NOCOPY_KERNEL_NUM - 1; outer_iter++) {
                //if more outer_iter remaining since there is more new processing to do, then insert them to the SW pipeline.
                //During the sw pipeline tail, there is nothing to dispatch.
                if (outer_iter < end) {
                    sw_pipeline_kernel_state[sw_pipeline_insert_index] = FIRST_KERNEL;
                    threads_already_processed[sw_pipeline_insert_index] =
                        sw_pipeline_insert_counter;
                    sw_pipeline_insert_index++;
                    if (sw_pipeline_insert_index >= NOCOPY_KERNEL_NUM) {
                        sw_pipeline_insert_index =
                            0; //By the time the index wraps arounds, the kernel that was in this slot previously has already completed.
                    }
                    sw_pipeline_insert_counter += total_threads_needed;
                }

                //The first kernel does the actual computation while the second kernel does the sync across ranks.
                e = queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for<class NoCopyKernel_compute<data_type>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        uint32_t idx = idx2.get_global_id();
                        //ESIMD kernel
                        //check if there is any kernel in the SW pipelines. If yes, execute them.
                        //to optimize, the order of loop i=0,1,2,.. can be shuffled so that different ranks can do different kernels at particular time. The purpose is to better balance the HW resource usage in the PVC node.
                        for (int ii = 0; ii < NOCOPY_KERNEL_NUM; ii++) {
                            RUN_FIRST_KERNEL
                            RUN_SECOND_KERNEL
                            RUN_THIRD_KERNEL
                        } // end of for (int ii

                        });//parallel_for
                }); //submit()

                //sync all the ranks within the single GPU.
                e = global_sync(queue,
                                temp_rank,
                                temp_world,
                                size_per_buffer_for_sync_kernel * buffer_index_kernel_for_sync,
                                5,
                                1);

                sync_reset_counter++;

                //update the sw pipeline process state so that next kernel will be processed in next round
                for (int i = 0; i < NOCOPY_KERNEL_NUM; i++) {
                    if (sw_pipeline_kernel_state[i] & NOCOPY_LAST_KERNEL)
                        sw_pipeline_kernel_state[i] =
                            0; //remove the kernel from the sw pipeline if it is last kernel. Everything is already executed.
                    else
                        sw_pipeline_kernel_state[i] <<= 1;
                }

                buffer_index_kernel_for_sync++;
                buffer_index_kernel_for_sync %= NOCOPY_BUFFER_COUNT;
            } //for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++)
        } //for (int iter = 0; iter < 2; iter++)

        allreduce_large_buffer_index += sync_reset_counter;
        allreduce_large_buffer_index %= NOCOPY_BUFFER_COUNT;

        return e;
    }

    //sync all the ranks here before consuming the results.
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
            temp_sync_buffer[i] = allreduce_large_sync_buffer[i];
        }
        sycl::event e;
        uint32_t total_threads_needed_sync = 1;
        int wg_size = 1;
        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class AllreduceLargeKernel_GlobalSync<data_type>>(
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
    sycl::event local_sync(sycl::queue &queue,
                           int temp_rank,
                           uint32_t temp_world,
                           int offset,
                           int index,
                           int reset) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = allreduce_large_sync_buffer[i];
        }
        sycl::event e;
        uint32_t total_threads_needed_sync = 1;
        int wg_size = 1;

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class AllreduceLargeKernel_LocalSync<data_type>>(
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
                while (status0[index] < RANKS_PER_GPU) {
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
                ZE_CALL(zeMemCloseIpcHandle,
                        (l0_ctx, (char *)allreduce_large_buffers[i] - allreduce_large_offsets[i]));
            }
        }

        sycl::free(allreduce_large_buffers[rank], queue);
        this->initialized = false;
    }

private:
    int rank{ ccl::utils::invalid_rank }, world{ ccl::utils::invalid_err_code };
    int COPY_MAX_COUNT{ ccl::utils::initial_count_value };
    int NOCOPY_MAX_COUNT{ ccl::utils::initial_count_value };
    int size_per_buffer{ ccl::utils::invalid_bytes_value };
    int data_size_per_buffer{ ccl::utils::invalid_bytes_value };
    ccl_stream *global_stream{};
    ccl_comm *comm{};
    ccl_comm *even_comm{};
};

#define ALLREDUCE_LARGE_API(TYPE) \
    void init_allreduce_large_##TYPE(ccl::datatype dtype, \
                                     sycl::queue &queue, \
                                     ccl_comm *comm, \
                                     ccl_stream *stream, \
                                     uint32_t rank_in, \
                                     uint32_t world_in) { \
        if (!ar_large_##TYPE.inited()) { \
            LOG_INFO("invoking large allreduce first time for datatype: ", dtype); \
            ar_large_##TYPE.init(queue, comm, stream, rank_in, world_in); \
        } \
    } \
\
    ccl::event run_allreduce_large_##TYPE(ccl::datatype dtype, \
                                          sycl::queue &queue, \
                                          const void *in_buf, \
                                          void *out_buf, \
                                          size_t count, \
                                          ccl::reduction reduction, \
                                          const ccl::vector_class<ccl::event> &deps, \
                                          bool &done) { \
        return ar_large_##TYPE.allreduce( \
            queue, in_buf, out_buf, dtype, count, reduction, deps, done); \
    }

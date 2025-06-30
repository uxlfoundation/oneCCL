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
#define BUFFER_COUNT     2
#define SYNC_BYTE        (SIMD_SYNC * sizeof(int) * 2)
#define ALIGNMENT_BYTE   256
#define MAX_SIZE         (128 * 1024 * 1024)
//#define EU_COUNT_PER_RANK 448
#define EU_COUNT_PER_RANK   512
#define THREAD_COUNT_PER_EU 8
#define HW_THREAD_COUNT     (EU_COUNT_PER_RANK * THREAD_COUNT_PER_EU)
#define KERNEL_NUM          11
#define RANKS_PER_GPU       2

extern void *allreduce_medium_buffer;
extern void *allreduce_medium_buffers[MAX_RANK];
extern void *allreduce_medium_sync_buffer[MAX_RANK];
extern size_t allreduce_medium_offsets[MAX_RANK];
extern ze_ipc_mem_handle_t allreduce_medium_ipc_handle[MAX_RANK];
extern int allreduce_medium_buffer_index;

// kernels for use_tmp_buf == 1
template <uint32_t TEMP_WORLD, typename data_type, size_t align>
void load_input_to_temp_buffer(int idx,
                               const void *in_buffer,
                               uint32_t size,
                               int threads_already_processed,
                               void *temp_buffer[],
                               uint32_t temp_rank,
                               int outer_iter,
                               int size_per_buffer_kernel,
                               int buffer_index_kernel) {
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
        for (uint32_t i = 0; i < TEMP_WORLD; i++) {
            buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
                block_load<data_type, SIMD_COMPUTE>(
                    (data_type *)in_buffer + read_offset + i * SIMD_COMPUTE,
                    properties{ alignment<align> });
        }
    }

    data_type *ptr = (data_type *)temp_buffer[temp_rank];
    ptr += size_per_buffer_kernel * buffer_index_kernel;
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
                                              int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    int is_odd = (even_ranks[0] == 1);
    //read the input data
    data_type *ptr_even =
        (data_type *)temp_buffer[temp_rank & 0xfffffffe] + is_odd * SIMD_COMPUTE * TEMP_WORLD / 2;
    data_type *ptr_odd =
        (data_type *)temp_buffer[temp_rank | 1] + is_odd * SIMD_COMPUTE * TEMP_WORLD / 2;
    ptr_even +=
        idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 + size_per_buffer_kernel * buffer_index_kernel;
    ptr_odd +=
        idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 + size_per_buffer_kernel * buffer_index_kernel;
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
               size_per_buffer_kernel * buffer_index_kernel + TEMP_WORLD * SIMD_COMPUTE;
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
             int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    data_type *ptr = (data_type *)temp_buffer[temp_rank];
    int read_offset =
        idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 +
        SIMD_COMPUTE *
            TEMP_WORLD; //points to second half of the temp slot since that's where the data is from other ranks.
    ptr += read_offset + size_per_buffer_kernel * buffer_index_kernel;
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
    //store the result in the second half of the temp slot.
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
                                              int outer_iter,
                                              int size_per_buffer_kernel,
                                              int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> buffer;

#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        //read the values
        data_type *read_ptr = (data_type *)temp_buffer[even_ranks[i]];
        read_ptr += size_per_buffer_kernel * buffer_index_kernel;
        read_ptr += idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 +
                    SIMD_COMPUTE * TEMP_WORLD; //get the sum from the second half of temp slot
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>(read_ptr, properties{ alignment<align> });
    }

    //write the data to the pair of ranks within the same gpu
    //gather in the first half of the slot
    data_type *mdfi_ptr = (data_type *)temp_buffer[temp_rank ^ 1];
    mdfi_ptr += size_per_buffer_kernel * buffer_index_kernel;
    mdfi_ptr += idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD / 2; i++) {
        block_store<data_type, SIMD_COMPUTE>(
            mdfi_ptr + i * SIMD_COMPUTE,
            buffer.template select<SIMD_COMPUTE, 1>(
                SIMD_COMPUTE * i), //save the results in the first half of temp slot
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
                  int outer_iter,
                  int size_per_buffer_kernel,
                  int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    //read the input data
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> buffer;
    data_type *read_ptr = (data_type *)temp_buffer[temp_rank];
    read_ptr +=
        idx * SIMD_COMPUTE * TEMP_WORLD * 3 / 2 + size_per_buffer_kernel * buffer_index_kernel;
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

// kernels for use_tmp_buf == 0
// tmp buffer is used for size: SIMD_COMPUTE * TEMP_WORLD / 2
template <uint32_t TEMP_WORLD, typename data_type, int align>
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
    //even rank and odd rank each read half
    int is_odd = (even_ranks[0] == 1);
    uint32_t read_offset = (idx + threads_already_processed) * SIMD_COMPUTE * TEMP_WORLD +
                           is_odd * SIMD_COMPUTE * TEMP_WORLD / 2;
    if (read_offset >= size) {
        return;
    }

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
    //store the result in the second half of the temp slot
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
            // save the results in the first half of temp slot
            block_store<data_type, SIMD_COMPUTE>(
                ptr_even + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
            // save the results in the first half of temp slot
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

template <typename data_type, size_t align>
void nocopy_2rank(int idx,
                  void **in_buffers,
                  uint32_t size,
                  void **out_buffers,
                  uint32_t temp_rank) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    const uint32_t TEMP_WORLD = 2;
    //read the input data
    //even rank and odd rank each read half
    uint32_t read_offset = idx * SIMD_COMPUTE * TEMP_WORLD;
    data_type *ptr_even = (data_type *)in_buffers[0] + temp_rank * SIMD_COMPUTE * TEMP_WORLD / 2;
    data_type *ptr_odd = (data_type *)in_buffers[1] + temp_rank * SIMD_COMPUTE * TEMP_WORLD / 2;
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD> buffer;
    uint32_t i;
#pragma unroll
    for (i = 0; i < TEMP_WORLD / 2; i++) {
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>(ptr_even + read_offset + i * SIMD_COMPUTE,
                                                properties{ alignment<align> });
    }
#pragma unroll
    for (i = TEMP_WORLD / 2; i < TEMP_WORLD; i++) {
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            block_load<data_type, SIMD_COMPUTE>(
                ptr_odd + read_offset + (i - TEMP_WORLD / 2) * SIMD_COMPUTE,
                properties{ alignment<align> });
    }
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD / 2> sum;
    sum = buffer.template select<SIMD_COMPUTE * TEMP_WORLD / 2, 1>(0) +
          buffer.template select<SIMD_COMPUTE * TEMP_WORLD / 2, 1>(SIMD_COMPUTE * TEMP_WORLD / 2);

    //write the data to the pair of ranks within the same gpu
    ptr_even = (data_type *)out_buffers[temp_rank ^ 1];
    ptr_odd = (data_type *)out_buffers[temp_rank];
    uint32_t write_offset =
        idx * SIMD_COMPUTE * TEMP_WORLD + temp_rank * SIMD_COMPUTE * TEMP_WORLD / 2;
    if (write_offset + SIMD_COMPUTE * TEMP_WORLD / 2 <= size) {
#pragma unroll
        for (i = 0; i < TEMP_WORLD / 2; i++) {
            //save the results in the first half of temp slot
            block_store<data_type, SIMD_COMPUTE>(
                ptr_even + write_offset + i * SIMD_COMPUTE,
                sum.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
            //save the results in the first half of temp slot
            block_store<data_type, SIMD_COMPUTE>(
                ptr_odd + write_offset + i * SIMD_COMPUTE,
                sum.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
    }
    else if (write_offset < size) {
        uint32_t vec_count = (size - write_offset) / SIMD_COMPUTE;
        for (i = 0; i < vec_count; i++) {
            block_store<data_type, SIMD_COMPUTE>(
                ptr_even + write_offset + i * SIMD_COMPUTE,
                sum.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
            block_store<data_type, SIMD_COMPUTE>(
                ptr_odd + write_offset + i * SIMD_COMPUTE,
                sum.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i),
                properties{ alignment<align> });
        }
        uint32_t count = size - write_offset - vec_count * SIMD_COMPUTE;
        for (i = 0; i < count; i++) {
            ptr_even[write_offset + vec_count * SIMD_COMPUTE + i] =
                sum[vec_count * SIMD_COMPUTE + i];
            ptr_odd[write_offset + vec_count * SIMD_COMPUTE + i] =
                sum[vec_count * SIMD_COMPUTE + i];
        }
    }
}

template <typename data_type>
void write_output_2rank(int idx,
                        void *out_buffer,
                        uint32_t size,
                        int threads_already_processed,
                        void *temp_buffer[],
                        uint32_t temp_rank,
                        int outer_iter,
                        int size_per_buffer_kernel,
                        int buffer_index_kernel) {
    using namespace __ESIMD_NS;
    using namespace __ESIMD_ENS;

    const uint32_t TEMP_WORLD = 2;
    //read the input data
    simd<data_type, SIMD_COMPUTE * TEMP_WORLD> buffer;
    data_type *read_ptr = (data_type *)temp_buffer[temp_rank];
    read_ptr += idx * SIMD_COMPUTE * TEMP_WORLD + size_per_buffer_kernel * buffer_index_kernel;
#pragma unroll
    for (uint32_t i = 0; i < TEMP_WORLD; i++) {
        //read the values
        buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i) =
            lsc_block_load<data_type,
                           SIMD_COMPUTE,
                           lsc_data_size::default_size,
                           cache_hint::uncached,
                           cache_hint::cached>(read_ptr + i * SIMD_COMPUTE);
    }

    //write out the results
    data_type *write_ptr = (data_type *)out_buffer;
    uint32_t write_offset = (idx + threads_already_processed) * SIMD_COMPUTE * TEMP_WORLD;
    if (write_offset + SIMD_COMPUTE * TEMP_WORLD <= size) {
#pragma unroll
        for (uint32_t i = 0; i < TEMP_WORLD; i++) {
            lsc_block_store<data_type,
                            SIMD_COMPUTE,
                            lsc_data_size::default_size,
                            cache_hint::uncached,
                            cache_hint::uncached>(
                write_ptr + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i));
        }
    }
    else if (write_offset < size) {
        int vec_count = (size - write_offset) / SIMD_COMPUTE;
        for (int i = 0; i < vec_count; i++) {
            lsc_block_store<data_type,
                            SIMD_COMPUTE,
                            lsc_data_size::default_size,
                            cache_hint::uncached,
                            cache_hint::uncached>(
                write_ptr + write_offset + i * SIMD_COMPUTE,
                buffer.template select<SIMD_COMPUTE, 1>(SIMD_COMPUTE * i));
        }
        int count = size - write_offset - vec_count * SIMD_COMPUTE;
        for (int i = 0; i < count; i++) {
            write_ptr[write_offset + vec_count * SIMD_COMPUTE + i] =
                buffer[vec_count * SIMD_COMPUTE + i];
        }
    }
}

template <typename dtype, size_t align>
class Kernel_load_input_to_temp_buffer;
template <typename dtype, size_t align>
class Kernel_local_sum_and_distribute_to_remote_ranks;
template <typename dtype, size_t align>
class Kernel_all_sum;
template <typename dtype, size_t align>
class Kernel_gather_from_remote_and_dist_to_rank_pair;
template <typename dtype, size_t align>
class Kernel_write_output;

template <typename dtype>
class AllreduceMediumKernel_2rank;
template <typename dtype>
class AllreduceMediumKernel_write_output_2rank;

template <typename dtype>
class NoCopyKernel_local_sum_and_distribute_to_remote_ranks;
template <typename dtype>
class NoCopyKernel_all_sum;
template <typename dtype>
class NoCopyKernel_gather_from_remote_and_dist_to_rank_pair;
template <typename dtype>
class NoCopyKernel_2rank;

template <typename dtype>
class AllreduceMediumKernel_GlobalSync;
template <typename dtype>
class AllreduceMediumKernel_LocalSync;

template <typename data_type, uint32_t max_rank = MAX_RANK, uint32_t max_buffer = 1024 /*KB*/>
class sycl_allreduce_medium : public sycl_coll_base<data_type> {
public:
    sycl_allreduce_medium() : sycl_coll_base<data_type>() {
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

        data_size_per_buffer = MAX_SIZE / sizeof(data_type);

        size_per_buffer = MAX_SIZE + SYNC_BYTE;

        if (allreduce_medium_buffer == NULL) {
            allreduce_medium_buffer = sycl::malloc_device(size_per_buffer * BUFFER_COUNT, queue);

            auto e = queue.memset(allreduce_medium_buffer, 0, size_per_buffer * BUFFER_COUNT);
            e.wait();

            // XXX: gain access to remote pointers
            this->exchange_peer_ipc_mem(queue,
                                        comm_in,
                                        stream,
                                        allreduce_medium_buffer,
                                        NULL,
                                        rank,
                                        world,
                                        data_size_per_buffer * sizeof(data_type),
                                        (void **)allreduce_medium_buffers,
                                        (void **)allreduce_medium_sync_buffer,
                                        allreduce_medium_offsets,
                                        allreduce_medium_ipc_handle,
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
        size_t is_aligned = (size_t)in_buffer % 4 == 0 && (size_t)out_buffer % 4 == 0;

        if (ccl::global_data::env().sycl_allreduce_tmp_buf) {
            if (is_aligned)
                e = allreduce_copy<4>(queue, in_buffer, out_buffer, size);
            else
                e = allreduce_copy<2>(queue, in_buffer, out_buffer, size);
        }
        else {
            if (world == 2) {
                e = allreduce_nocopy_2rank(queue, in_buffer, out_buffer, size);
            }
            else {
                e = allreduce_nocopy(queue, in_buffer, out_buffer, size);
            }
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
            temp_buffer[i] = allreduce_medium_buffers[i];
        }
        /*
        void* temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) 
        {
            temp_sync_buffer[i] = allreduce_medium_sync_buffer[i];
        }
        */

        int even_ranks[max_rank];
        int myrank;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank)
                myrank = i;
            //printf("even rank %d: %d neighbor: %d\n", i, even_ranks[i], even_ranks[i] ^ 1);
        }

        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int buffer_index_kernel = allreduce_medium_buffer_index;
        int outerloop_iter_count; //Since 16 elements in temp buffer is used to process 8 element output, the outer loop count must be doubled roughly.
        int outer_iter;
        //todo:
        //5. prefetch in persistent threads?
        int max_threads_per_MAX_COUNT =
            (data_size_per_buffer * 2 / 3) /
            (SIMD_COMPUTE *
             temp_world); // each thread uses (data_size_per_buffer * temp_world) * 3 / 2 space
        int max_elements_per_MAX_COUNT = max_threads_per_MAX_COUNT * (SIMD_COMPUTE * temp_world);

        int threads_already_processed = 0;
        outerloop_iter_count =
            (size + max_elements_per_MAX_COUNT - 1) /
            max_elements_per_MAX_COUNT; //this is the outerloop count that requires full hw thread count. This doesnt include the outloop iteration that only needs partial thread count
        //uint32_t total_threads_needed_sync = 1;
        for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++) {
            uint32_t total_threads_needed;
            if ((outer_iter + 1) * max_elements_per_MAX_COUNT < (int)size) {
                total_threads_needed = max_threads_per_MAX_COUNT;
            }
            else {
                total_threads_needed = (size - outer_iter * max_elements_per_MAX_COUNT +
                                        SIMD_COMPUTE * temp_world - 1) /
                                       (SIMD_COMPUTE * temp_world);
            }
            int wg_size = 1;

            int innerloop_iter_count =
                (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;

            uint32_t persist_threads_needed = total_threads_needed;
            if (persist_threads_needed > HW_THREAD_COUNT)
                persist_threads_needed = HW_THREAD_COUNT;

#define KERNEL_EXEC_MAP (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 + 256)

#if KERNEL_EXEC_MAP & 1
            //Data is sent to other tile within the same gpu via MDFI
            queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class Kernel_load_input_to_temp_buffer<data_type, align>>(
                    sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                    {
                    uint32_t idx = idx2.get_global_id();
                    //ESIMD kernel
                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                        int index = idx + inner_iter * HW_THREAD_COUNT;
                        if ((uint32_t)index >= total_threads_needed)
                            break;

                        auto load_input_to_temp_buffer_lambda =
                            [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                load_input_to_temp_buffer<w, data_type, a>(
                                    index,
                                    in_buffer,
                                    size,
                                    threads_already_processed,
                                    (void **)temp_buffer,
                                    temp_rank,
                                    outer_iter,
                                    size_per_buffer_kernel,
                                    buffer_index_kernel);
                            };
                        invoke_esimd_function<align>(load_input_to_temp_buffer_lambda, temp_world);
                    }

                    });//parallel_for
            }); //submit()
#endif
#if KERNEL_EXEC_MAP & 2
                //sync all the ranks within the single GPU.
            e = local_sync(queue,
                           temp_rank,
                           temp_world,
                           size_per_buffer_for_sync_kernel * buffer_index_kernel,
                           0,
                           0);
            //printf("kernel1\n");
#endif
#if KERNEL_EXEC_MAP & 4
            //local reduction kernel
            queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for<class Kernel_local_sum_and_distribute_to_remote_ranks<data_type, align>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        uint32_t idx = idx2.get_global_id();
                        //ESIMD kernel
                        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= total_threads_needed)
                                break;

                            auto local_sum_and_distribute_to_remote_ranks_lambda =
                                [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                    local_sum_and_distribute_to_remote_ranks<w, data_type, a>(
                                        (int *)even_ranks,
                                        myrank,
                                        index,
                                        in_buffer,
                                        size,
                                        threads_already_processed,
                                        (void **)temp_buffer,
                                        temp_rank,
                                        size_per_buffer_kernel,
                                        buffer_index_kernel);
                                };
                            invoke_esimd_function<align>(
                                local_sum_and_distribute_to_remote_ranks_lambda, temp_world);
                        }
                        });//parallel_for
            }); //submit()
#endif
#if KERNEL_EXEC_MAP & 8
            //sync all the ranks here before consuming the results.
            e = global_sync(queue,
                            temp_rank,
                            temp_world,
                            size_per_buffer_for_sync_kernel * buffer_index_kernel,
                            1,
                            0);
#endif
#if KERNEL_EXEC_MAP & 16
            //local reduction kernel
            queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for<class Kernel_all_sum<data_type, align>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        uint32_t idx = idx2.get_global_id();
                        //ESIMD kernel
                        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= total_threads_needed)
                                break;

                            auto all_sum_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                all_sum<w, data_type, a>(index,
                                                         in_buffer,
                                                         size,
                                                         threads_already_processed,
                                                         (void **)temp_buffer,
                                                         temp_rank,
                                                         size_per_buffer_kernel,
                                                         buffer_index_kernel);
                            };
                            invoke_esimd_function<align>(all_sum_lambda, temp_world);
                        }

                        });//parallel_for
            }); //submit()
#endif
#if KERNEL_EXEC_MAP & 32
            //sync all the ranks here before consuming the results.
            e = global_sync(queue,
                            temp_rank,
                            temp_world,
                            size_per_buffer_for_sync_kernel * buffer_index_kernel,
                            3,
                            0);
#endif
#if KERNEL_EXEC_MAP & 64
            //copy the results to all the ranks.
            queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for<class Kernel_gather_from_remote_and_dist_to_rank_pair<data_type, align>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        uint32_t idx = idx2.get_global_id();
                        //ESIMD kernel
                        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= total_threads_needed)
                                break;

                            auto gather_from_remote_and_dist_to_rank_pair_lambda =
                                [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                    gather_from_remote_and_dist_to_rank_pair<w, data_type, a>(
                                        (int *)even_ranks,
                                        index,
                                        out_buffer,
                                        size,
                                        threads_already_processed,
                                        (void **)temp_buffer,
                                        temp_rank,
                                        outer_iter,
                                        size_per_buffer_kernel,
                                        buffer_index_kernel);
                                };
                            invoke_esimd_function<align>(
                                gather_from_remote_and_dist_to_rank_pair_lambda, temp_world);
                        }
                        });//parallel_for
            }); //submit()
#endif
#if KERNEL_EXEC_MAP & 128
            //sync all the ranks within the same GPU.
            e = local_sync(queue,
                           temp_rank,
                           temp_world,
                           size_per_buffer_for_sync_kernel * buffer_index_kernel,
                           4,
                           1);
#endif
#if KERNEL_EXEC_MAP & 256
            //copy the results to all the ranks.
            e = queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for<class Kernel_write_output<data_type, align>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        uint32_t idx = idx2.get_global_id();
                        //ESIMD kernel
                        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= total_threads_needed)
                                break;

                            auto write_output_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                write_output<w, data_type, a>((int *)even_ranks,
                                                              index,
                                                              out_buffer,
                                                              size,
                                                              threads_already_processed,
                                                              (void **)temp_buffer,
                                                              temp_rank,
                                                              outer_iter,
                                                              size_per_buffer_kernel,
                                                              buffer_index_kernel);
                            };
                            invoke_esimd_function<align>(write_output_lambda, temp_world);
                        }
                        });//parallel_for
            }); //submit()
#endif

            buffer_index_kernel++;
            buffer_index_kernel &= 1;
            threads_already_processed += total_threads_needed;
        } //for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++)

        allreduce_medium_buffer_index += outerloop_iter_count;
        allreduce_medium_buffer_index &= 1;

        return e;
    }

    // perform IPC exchange every time
    sycl::event allreduce_nocopy(sycl::queue &queue,
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
            temp_buffer[i] = allreduce_medium_buffers[i];
        }

        int even_ranks[max_rank];
        int myrank;
        for (int i = 0; i < world / 2; i++) {
            even_ranks[i] = even_comm->get_node_rank(i);
            if (even_ranks[i] == (int)temp_rank) {
                myrank = i;
            }
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
        int align4 = all_aligned((void **)in_buffers, temp_world, size, sizeof(data_type), 4) &&
                     all_aligned((void **)out_buffers, temp_world, size, sizeof(data_type), 4);

        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int buffer_index_kernel = allreduce_medium_buffer_index;
        //Since 16 elements in temp buffer is used to process 8 element output, the outer loop count must be doubled roughly.
        int outerloop_iter_count;
        int outer_iter;
        //todo:
        //5. prefetch in persistent threads?
        int max_threads_per_MAX_COUNT = data_size_per_buffer / (SIMD_COMPUTE * temp_world);
        int max_elements_per_MAX_COUNT = max_threads_per_MAX_COUNT * (SIMD_COMPUTE * temp_world);

        int threads_already_processed = 0;
        //this is the outerloop count that requires full hw thread count. This doesnt include the outloop iteration that only needs partial thread count
        outerloop_iter_count = (size + max_elements_per_MAX_COUNT - 1) / max_elements_per_MAX_COUNT;
        //uint32_t total_threads_needed_sync = 1;

        // sync two tiles of a same GPU before entering the call
        e = local_sync(queue,
                       temp_rank,
                       temp_world,
                       size_per_buffer_for_sync_kernel * buffer_index_kernel,
                       0,
                       0);

        for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++) {
            uint32_t total_threads_needed;
            if ((outer_iter + 1) * max_elements_per_MAX_COUNT < (int)size) {
                total_threads_needed = max_threads_per_MAX_COUNT;
            }
            else {
                total_threads_needed = (size - outer_iter * max_elements_per_MAX_COUNT +
                                        SIMD_COMPUTE * temp_world - 1) /
                                       (SIMD_COMPUTE * temp_world);
            }
            int wg_size = 1;

            int innerloop_iter_count =
                (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;

            uint32_t persist_threads_needed = total_threads_needed;
            if (persist_threads_needed > HW_THREAD_COUNT)
                persist_threads_needed = HW_THREAD_COUNT;

            //local reduction kernel
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class NoCopyKernel_local_sum_and_distribute_to_remote_ranks<data_type>>(
                    sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                    {
                    uint32_t idx = idx2.get_global_id();
                    //ESIMD kernel
                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                        int index = idx + inner_iter * HW_THREAD_COUNT;
                        if ((uint32_t)index >= total_threads_needed)
                            break;

                        auto nocopy_sum_and_distribute_to_remote_ranks_lambda =
                            [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                nocopy_sum_and_distribute_to_remote_ranks<w, data_type, a>(
                                    (int *)even_ranks,
                                    myrank,
                                    index,
                                    (void **)in_buffers,
                                    size,
                                    threads_already_processed,
                                    (void **)temp_buffer,
                                    temp_rank,
                                    size_per_buffer_kernel,
                                    buffer_index_kernel);
                            };
                        if (align4)
                            invoke_esimd_function<4>(
                                nocopy_sum_and_distribute_to_remote_ranks_lambda, temp_world);
                        else
                            invoke_esimd_function<2>(
                                nocopy_sum_and_distribute_to_remote_ranks_lambda, temp_world);
                    }

                    });//parallel_for
            }); //submit()
            //e.wait();

            //sync all the ranks here before consuming the results.
            e = global_sync(queue,
                            temp_rank,
                            temp_world,
                            size_per_buffer_for_sync_kernel * buffer_index_kernel,
                            2,
                            0);

            //local reduction kernel
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class NoCopyKernel_all_sum<data_type>>(
                    sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                    {
                    uint32_t idx = idx2.get_global_id();
                    //ESIMD kernel
                    for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                        int index = idx + inner_iter * HW_THREAD_COUNT;
                        if ((uint32_t)index >= total_threads_needed)
                            break;

                        auto nocopy_all_sum_lambda = [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                            nocopy_all_sum<w, data_type, a>(index,
                                                            in_buffer,
                                                            size,
                                                            threads_already_processed,
                                                            (void **)temp_buffer,
                                                            temp_rank,
                                                            size_per_buffer_kernel,
                                                            buffer_index_kernel);
                        };
                        if (align4)
                            invoke_esimd_function<4>(nocopy_all_sum_lambda, temp_world);
                        else
                            invoke_esimd_function<2>(nocopy_all_sum_lambda, temp_world);
                    }

                    });//parallel_for
            }); //submit()
            //e.wait();

            //sync all the ranks here before consuming the results.
            int reset = (outer_iter == outerloop_iter_count - 1) ? 0 : 1;
            e = global_sync(queue,
                            temp_rank,
                            temp_world,
                            size_per_buffer_for_sync_kernel * buffer_index_kernel,
                            3,
                            reset);
            //e.wait();

            //copy the results to all the ranks.
            e = queue.submit([&](sycl::handler &cgh) {
                    cgh.parallel_for<class NoCopyKernel_gather_from_remote_and_dist_to_rank_pair<data_type>>(
                        sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                        {
                        //ESIMD kernel
                        uint32_t idx = idx2.get_global_id();
                        for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                            int index = idx + inner_iter * HW_THREAD_COUNT;
                            if ((uint32_t)index >= total_threads_needed)
                                break;

                            auto nocopy_gather_from_remote_and_dist_to_rank_pair_lambda =
                                [&]<int w, size_t a>() SYCL_ESIMD_KERNEL {
                                    nocopy_gather_from_remote_and_dist_to_rank_pair<w,
                                                                                    data_type,
                                                                                    a>(
                                        (int *)even_ranks,
                                        index,
                                        (void **)out_buffers,
                                        size,
                                        threads_already_processed,
                                        (void **)temp_buffer,
                                        temp_rank,
                                        size_per_buffer_kernel,
                                        buffer_index_kernel);
                                };
                            if (align4) {
                                invoke_esimd_function<4>(
                                    nocopy_gather_from_remote_and_dist_to_rank_pair_lambda,
                                    temp_world);
                            }
                            else {
                                invoke_esimd_function<2>(
                                    nocopy_gather_from_remote_and_dist_to_rank_pair_lambda,
                                    temp_world);
                            }
                        }
                        });//parallel_for
            }); //submit()
            //e.wait();

            if (outer_iter == outerloop_iter_count - 1) {
                // sync two tiles of a same GPU before exiting the call
                e = global_sync(queue,
                                temp_rank,
                                temp_world,
                                size_per_buffer_for_sync_kernel * buffer_index_kernel,
                                4,
                                1);
            }

            buffer_index_kernel++;
            buffer_index_kernel &= 1;
            threads_already_processed += total_threads_needed;
        } //for (outer_iter = 0; outer_iter < outerloop_iter_count; outer_iter++)

        allreduce_medium_buffer_index += outerloop_iter_count;
        allreduce_medium_buffer_index &= 1;

        return e;
    }

    // perform IPC exchange every time (2 rank version)
    sycl::event allreduce_nocopy_2rank(sycl::queue &queue,
                                       const void *in_buffer,
                                       void *out_buffer,
                                       uint32_t size) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        assert(this->initialized == true);

        //queue.wait();

        void *in_buffers[2];
        void *out_buffers[2];
        this->exchange_peer_ipc_mem(queue,
                                    comm,
                                    global_stream,
                                    (void **)in_buffer,
                                    out_buffer,
                                    rank,
                                    world,
                                    data_size_per_buffer,
                                    (void **)in_buffers,
                                    NULL,
                                    NULL,
                                    NULL,
                                    (void **)out_buffers);
        int align4 = all_aligned((void **)in_buffers, temp_world, size, sizeof(data_type), 4) &&
                     all_aligned((void **)out_buffers, temp_world, size, sizeof(data_type), 4);

        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));
        int buffer_index_kernel = allreduce_medium_buffer_index;

        // a GPU barrier to make sure all ranks are ready
        e = global_sync(queue,
                        temp_rank,
                        temp_world,
                        size_per_buffer_for_sync_kernel * buffer_index_kernel,
                        0,
                        0);

        uint32_t total_threads_needed;
        total_threads_needed = (size + SIMD_COMPUTE * temp_world - 1) / (SIMD_COMPUTE * temp_world);
        int wg_size = 1;

        int innerloop_iter_count = (total_threads_needed + HW_THREAD_COUNT - 1) / HW_THREAD_COUNT;

        uint32_t persist_threads_needed = total_threads_needed;
        if (persist_threads_needed > HW_THREAD_COUNT)
            persist_threads_needed = HW_THREAD_COUNT;

        //local reduction kernel
        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class NoCopyKernel_2rank<data_type>>(
                sycl::nd_range<1>({ persist_threads_needed }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL
                {
                uint32_t idx = idx2.get_global_id();
                for (int inner_iter = 0; inner_iter < innerloop_iter_count; inner_iter++) {
                    int index = idx + inner_iter * HW_THREAD_COUNT;
                    if ((uint32_t)index >= total_threads_needed)
                        break;

                    if (align4) {
                        nocopy_2rank<data_type, 4>(
                            index, (void **)in_buffers, size, (void **)out_buffers, temp_rank);
                    }
                    else {
                        nocopy_2rank<data_type, 2>(
                            index, (void **)in_buffers, size, (void **)out_buffers, temp_rank);
                    }
                }

                });//parallel_for
        }); //submit()
        //e.wait();

        // sync two tiles of a same GPU before exiting the call
        e = local_sync(queue,
                       temp_rank,
                       temp_world,
                       size_per_buffer_for_sync_kernel * buffer_index_kernel,
                       4,
                       1);

        allreduce_medium_buffer_index++; // acount for the local sync
        allreduce_medium_buffer_index &= 1;

        return e;
    }

    void release(sycl::queue &queue) {
        // Clean up, close/put ipc handles, free memory, etc.
        auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
        for (int i = 0; i < world; i++) {
            if (i != rank) {
                ZE_CALL(
                    zeMemCloseIpcHandle,
                    (l0_ctx, (char *)allreduce_medium_buffers[i] - allreduce_medium_offsets[i]));
            }
        }

        sycl::free(allreduce_medium_buffers[rank], queue);
        this->initialized = false;
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
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = allreduce_medium_sync_buffer[i];
        }
        sycl::event e;
        uint32_t total_threads_needed_sync = 1;
        int wg_size = 1;
        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class AllreduceMediumKernel_GlobalSync<data_type>>(
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
            temp_sync_buffer[i] = allreduce_medium_sync_buffer[i];
        }
        sycl::event e;
        uint32_t total_threads_needed_sync = 1;
        int wg_size = 1;

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<class AllreduceMediumKernel_LocalSync<data_type>>(
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

private:
    int rank{ ccl::utils::invalid_rank }, world{ ccl::utils::invalid_err_code };
    int size_per_buffer{ ccl::utils::invalid_bytes_value };
    int data_size_per_buffer{ ccl::utils::invalid_bytes_value };
    ccl_stream *global_stream{};
    ccl_comm *comm{};
    ccl_comm *even_comm{};
};

#define ALLREDUCE_MEDIUM_API(TYPE) \
    void init_allreduce_medium_##TYPE(ccl::datatype dtype, \
                                      sycl::queue &queue, \
                                      ccl_comm *comm, \
                                      ccl_stream *stream, \
                                      uint32_t rank_in, \
                                      uint32_t world_in) { \
        if (!ar_medium_##TYPE.inited()) { \
            LOG_INFO("invoking medium allreduce first time for datatype: ", dtype); \
            ar_medium_##TYPE.init(queue, comm, stream, rank_in, world_in); \
        } \
    } \
\
    ccl::event run_allreduce_medium_##TYPE(ccl::datatype dtype, \
                                           sycl::queue &queue, \
                                           const void *in_buf, \
                                           void *out_buf, \
                                           size_t count, \
                                           ccl::reduction reduction, \
                                           const ccl::vector_class<ccl::event> &deps, \
                                           bool &done) { \
        return ar_medium_##TYPE.allreduce( \
            queue, in_buf, out_buf, dtype, count, reduction, deps, done); \
    }

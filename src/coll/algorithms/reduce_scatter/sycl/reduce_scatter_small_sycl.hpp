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

#define SIMD_MAX              256
#define SIMD                  (SIMD_MAX / sizeof(data_type))
#define SIMD_ATOMIC           16
#define MAX_RANK              16
#define UNROLL_SIZE           1
#define TRIPLE_BUFFER         3
#define SYNC_BYTE             (SIMD_ATOMIC * sizeof(int) * 2)
#define ALIGNMENT_BYTE        256
#define EU_COUNT              512
#define THREADS_PER_EU        8
#define MAX_THREAD            (EU_COUNT * THREADS_PER_EU)
#define MAX_KERNEL_LOOP_COUNT 4
#define MAX_COUNT             (SIMD * UNROLL_SIZE * MAX_KERNEL_LOOP_COUNT * MAX_THREAD)

template <typename data_type, uint32_t N, int kernel_inner_loop_scalar>
ESIMD_INLINE void reduce_kernel(void **temp_buffer, int buf_offset, int offset, data_type result[]) {
    data_type peer[N][kernel_inner_loop_scalar];
//    gpu_kernel_copy((char*)result, (const char *)((data_type *)(temp_buffer[0]) + buf_offset + offset), kernel_inner_loop_scalar * sizeof(data_type));
#pragma unroll
    for (uint32_t r = 0; r < N; r++) {
        data_type *peer_ptr = (data_type *)(temp_buffer[r]) + buf_offset + offset;
        if ((size_t)peer_ptr % 4 == 0 && kernel_inner_loop_scalar > 1) {
            gpu_kernel_copy((char *)peer[r], (const char *)peer_ptr, kernel_inner_loop_scalar * sizeof(data_type));
        }
        else {
            for (int i = 0; i < kernel_inner_loop_scalar; i++) {
                peer[r][i] = peer_ptr[i];
            }
        }
    }
    if ((size_t)result % 4 == 0 && kernel_inner_loop_scalar > 1) {
        gpu_kernel_copy((char *)result, (const char *)peer[0], kernel_inner_loop_scalar * sizeof(data_type));
    }
    else {
        for (int i = 0; i < kernel_inner_loop_scalar; i++) {
            result[i] = peer[0][i];
        }
    }
#pragma unroll
    for (uint32_t r = 1; r < N; r++) {
        for (int j = 0; j < kernel_inner_loop_scalar; j++)
            result[j] += peer[r][j];
    }
}

template <typename dtype, int kernel_inner_loop, size_t alignment>
class Reduce_scatter_small_kernel_esimd;
template <typename dtype, int kernel_inner_loop_scalar, int wg_size>
class Reduce_scatter_small_kernel_scalar;

template <typename data_type, uint32_t max_rank = MAX_RANK, uint32_t max_buffer = 1024 /*KB*/>
class sycl_reduce_scatter_small : public sycl_coll_base<data_type> {
public:
    sycl_reduce_scatter_small() : sycl_coll_base<data_type>() {
        buffer_index = 0;
        size_per_buffer = 0;
    }

    void init(sycl::queue &queue, ccl_comm *comm, ccl_stream *stream, uint32_t rank_in, uint32_t world_in) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.
        data_size_per_buffer = ((MAX_COUNT + SIMD * UNROLL_SIZE * MAX_KERNEL_LOOP_COUNT - 1) /
                                (SIMD * UNROLL_SIZE * MAX_KERNEL_LOOP_COUNT)) *
                               SIMD * UNROLL_SIZE * MAX_KERNEL_LOOP_COUNT;
        data_size_per_buffer = ((data_size_per_buffer * sizeof(data_type) + ALIGNMENT_BYTE - 1) / ALIGNMENT_BYTE) *
                               ALIGNMENT_BYTE / sizeof(data_type); //aligned size
        size_per_buffer = data_size_per_buffer * sizeof(data_type) + SYNC_BYTE;
        void *local_triple_buffer = sycl::malloc_device(size_per_buffer * TRIPLE_BUFFER, queue);

        auto e = queue.memset(local_triple_buffer, 0, size_per_buffer * TRIPLE_BUFFER);
        e.wait();
        this->exchange_peer_ipc_mem(queue,
                                    comm,
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
    }

    ccl::event reduce_scatter(sycl::queue &queue,
                              const void *send_buf,
                              void *out_buffer,
                              ccl::datatype dtype,
                              size_t recv_size,
                              ccl::reduction reduction,
                              const ccl::vector_class<ccl::event> &deps,
                              bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        assert(this->initialized == true);
        done = false;

        size_t total_count = recv_size * world;
        if (total_count > MAX_COUNT) {
            done = false;
            return ccl::event::create_from_native(e);
        }

        // check local alignment
        int align4 = is_aligned(send_buf, out_buffer, recv_size, sizeof(data_type), 4);

        auto esimd_lambda = [&]<int kernel_inner_loop>() {
            if (align4) {
                return reduce_scatter_esimd<kernel_inner_loop, 4>(
                    queue, send_buf, out_buffer, dtype, recv_size, deps, done);
            }
            else {
                return reduce_scatter_esimd<kernel_inner_loop, 2>(
                    queue, send_buf, out_buffer, dtype, recv_size, deps, done);
            }
        };

        if (total_count * sizeof(data_type) <= 8192) {
            e = reduce_scatter_scalar<1, 16>(queue, send_buf, out_buffer, dtype, recv_size, deps, done);
        }
        else if (total_count * sizeof(data_type) <= 524288) {
            e = esimd_lambda.template operator()<1>();
            if (!done) {
                e = esimd_lambda.template operator()<2>();
            }
            if (!done) {
                e = esimd_lambda.template operator()<4>();
            }
        }
        else {
            e = esimd_lambda.template operator()<2>();
            if (!done) {
                e = esimd_lambda.template operator()<3>();
            }
            if (!done) {
                e = esimd_lambda.template operator()<4>();
            }
        }

        if (reduction == ccl::reduction::avg) {
            std::vector<sycl::event> evs;
            evs.push_back(e);
            e = sycl_average(queue, out_buffer, recv_size, world, dtype, evs);
        }

        return ccl::event::create_from_native(e);
    }

    template <int kernel_inner_loop, size_t align>
    sycl::event reduce_scatter_esimd(sycl::queue &queue,
                                     const void *send_buf,
                                     void *out_buffer,
                                     ccl::datatype dtype,
                                     size_t recv_size,
                                     const ccl::vector_class<ccl::event> &deps,
                                     bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;

        size_t total_count = recv_size * world;

        done = true;

        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }

        int size_per_buffer_kernel __attribute__((unused)) = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel __attribute__((unused)) =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));

        uint32_t total_threads_needed = (total_count + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) /
                                        (SIMD * UNROLL_SIZE * kernel_inner_loop); //ceiling
        int wg_size = 16;
        uint32_t total_threads_dispatched = (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = total_threads_dispatched / wg_size;

        uint32_t total_threads_needed_for_reduce = (recv_size + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) /
                                                   (SIMD * UNROLL_SIZE * kernel_inner_loop); //ceiling

        // checking oversubscription of hardware threads
        ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
        ssize_t dev_id{ ccl::utils::invalid_device_id };
        if (!ccl::ze::get_device_global_id(ze_dev, &dev_id)) {
            CCL_THROW("unable to get global id for device\n");
        }
        if (total_threads_dispatched > ccl::global_data::get().ze_data->devices[dev_id].total_threads) {
            done = false;
            return e;
        }

        int buffer_index_kernel = buffer_index;
        buffer_index++;
        buffer_index %= TRIPLE_BUFFER;

        std::vector<sycl::event> dep_events = get_sycl_events(deps);

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep_events);
            cgh.parallel_for<Reduce_scatter_small_kernel_esimd<data_type, kernel_inner_loop, align>>(
                sycl::nd_range<1>({ total_threads_dispatched }, wg_size),
                [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL {
                    //slm_init(1024);
                    uint32_t idx = idx2.get_global_id();

                    //ESIMD kernel
                    uint32_t offset = idx * SIMD * UNROLL_SIZE * kernel_inner_loop;
                    simd<data_type, max_rank * SIMD * UNROLL_SIZE> buffer; //64 registers
                    simd<data_type, SIMD * UNROLL_SIZE> buffer_small;
                    simd<ushort, SIMD_ATOMIC> ramp;
                    simd_mask<SIMD_ATOMIC> pred;
                    simd<int, SIMD_ATOMIC> status0;
                    int *local_sync_ptr;

#pragma unroll
                    for (uint32_t i = 0; i < SIMD_ATOMIC; i++) {
                        ramp[i] = i * sizeof(int);
                    }

                    //process the input only if the thread is useful
                    if (idx < total_threads_needed) {
                        //do copy from input buffer to temp buffer.
                        for (int i = 0; i < kernel_inner_loop; i++) {
                            if (offset + i * SIMD * UNROLL_SIZE > total_count)
                                break;
#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                buffer_small.template select<SIMD, 1>(unroll_i * SIMD) =
                                    block_load<data_type, SIMD>(
                                        (data_type *)send_buf + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                            }

                            //use the temp buffer for the current rank to copy the data to.
                            data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                            //point to the correct buffer inside the triple buffer
                            local_temp_ptr += buffer_index_kernel * size_per_buffer_kernel;

#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                block_store<data_type, SIMD>(
                                    (data_type *)local_temp_ptr + offset + unroll_i * SIMD +
                                        i * SIMD * UNROLL_SIZE,
                                    buffer_small.template select<SIMD, 1>(unroll_i * SIMD),
                                    properties{ alignment<align> });
                            }
                        }
                        //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                        //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank
                    }

                    //sync locally within local GPU first.
                    //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr = (int *)temp_sync_buffer[temp_rank];
                    local_sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;

                    //if there are more than 1 threads required per rank, then do the local sync within the rank first.
                    if (total_threads_needed > 1) {
                        //do local sync in two steps. First using TG barrier. Then global L3 atomics.
                        uint32_t local_tid = idx2.get_local_linear_id();

                        pred = false;
                        pred[0] = true;
                        if (local_tid == 0) {
                            status0 = lsc_atomic_update<atomic_op::inc,
                                                        int,
                                                        SIMD_ATOMIC,
                                                        lsc_data_size::default_size,
                                                        cache_hint::none,
                                                        cache_hint::none>(local_sync_ptr, ramp, pred);
                            //wait for all the local TG to sync. Then sync the other remote GPUs
                            while (status0[0] != total_wg_count) {
                                status0 = lsc_atomic_update<atomic_op::load,
                                                            int,
                                                            SIMD_ATOMIC,
                                                            lsc_data_size::default_size,
                                                            cache_hint::none,
                                                            cache_hint::none>(local_sync_ptr, ramp, pred);
                            }
                        }
                        barrier();
                    }

                    //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                    pred = false;
                    pred[1] = true; //use different lane for the remote gpu sync
                    if (total_threads_dispatched >= temp_world) {
                        if (idx < temp_world) {
                            status0 = total_threads_needed;
                            int *sync_ptr = (int *)temp_sync_buffer[idx];
                            sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;
                            lsc_atomic_update<atomic_op::add,
                                              int,
                                              SIMD_ATOMIC,
                                              lsc_data_size::default_size,
                                              cache_hint::none,
                                              cache_hint::none>(sync_ptr, ramp, status0, pred);
                        }
                    }
                    else if (idx == 0) { //one thread in the local gpu notifies the remote gpu of its status.
                        status0 = total_threads_needed;
                        for (uint32_t i = 0; i < temp_world; i++) {
                            int *sync_ptr;
                            sync_ptr = (int *)temp_sync_buffer
                                [i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;
                            lsc_atomic_update<atomic_op::add,
                                              int,
                                              SIMD_ATOMIC,
                                              lsc_data_size::default_size,
                                              cache_hint::none,
                                              cache_hint::none>(sync_ptr, ramp, status0, pred);
                        }
                    }

                    //once the local sync is done, retire useless threads
                    if (idx >= total_threads_needed)
                        return;

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    status0 = lsc_atomic_update<atomic_op::load,
                                                int,
                                                SIMD_ATOMIC,
                                                lsc_data_size::default_size,
                                                cache_hint::none,
                                                cache_hint::none>(local_sync_ptr, ramp, pred);
                    while (status0[1] != total_threads_needed * temp_world) {
                        status0 = lsc_atomic_update<atomic_op::load,
                                                    int,
                                                    SIMD_ATOMIC,
                                                    lsc_data_size::default_size,
                                                    cache_hint::none,
                                                    cache_hint::none>(local_sync_ptr, ramp, pred);
                    }

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx == 0) { //one thread in the local gpu notifies the remote gpu of its status.
                        int buffer_index_to_reset = (buffer_index_kernel + 2) % 3;
                        status0 = 0;
                        pred = true;
                        //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr = (int *)temp_sync_buffer[temp_rank];
                        local_sync_ptr += buffer_index_to_reset * size_per_buffer_for_sync_kernel;
                        lsc_atomic_update<atomic_op::store,
                                          int,
                                          SIMD_ATOMIC,
                                          lsc_data_size::default_size,
                                          cache_hint::none,
                                          cache_hint::none>(
                            local_sync_ptr, ramp, status0, pred); //reset the first half of sync buffer
                    }

                    if (idx > total_threads_needed_for_reduce)
                        return;

                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    //do partial reduce
                    uint32_t send_offset = recv_size * temp_rank + idx * SIMD * UNROLL_SIZE * kernel_inner_loop;

                    simd<data_type, SIMD * UNROLL_SIZE> result;
                    for (int i = 0; i < kernel_inner_loop; i++) {
                        if (temp_world == 4) {
                            data_type *peer_ptr0 =
                                ((data_type *)temp_buffer[0]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr1 =
                                ((data_type *)temp_buffer[1]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr2 =
                                ((data_type *)temp_buffer[2]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr3 =
                                ((data_type *)temp_buffer[3]) + buffer_index_kernel * size_per_buffer_kernel;

#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 0 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr0 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 1 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr1 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 2 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr2 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 3 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr3 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                            }
                            //do the actual reduction
                            result = 0;
#pragma unroll
                            for (int r = 0; r < 4; r++) {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result =
                                    result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }
                        }
                        else if (temp_world == 8) {
                            data_type *peer_ptr0 =
                                ((data_type *)temp_buffer[0]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr1 =
                                ((data_type *)temp_buffer[1]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr2 =
                                ((data_type *)temp_buffer[2]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr3 =
                                ((data_type *)temp_buffer[3]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr4 =
                                ((data_type *)temp_buffer[4]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr5 =
                                ((data_type *)temp_buffer[5]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr6 =
                                ((data_type *)temp_buffer[6]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr7 =
                                ((data_type *)temp_buffer[7]) + buffer_index_kernel * size_per_buffer_kernel;

#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 0 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr0 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 1 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr1 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 2 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr2 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 3 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr3 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 4 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr4 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 5 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr5 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 6 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr6 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 7 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr7 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                            }
                            //do the actual reduction
                            result = 0;
#pragma unroll
                            for (int r = 0; r < 8; r++) {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result =
                                    result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }
                        }
                        else if (temp_world == 16) {
                            //first 8 ranks processing
                            data_type *peer_ptr0 =
                                ((data_type *)temp_buffer[0]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr1 =
                                ((data_type *)temp_buffer[1]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr2 =
                                ((data_type *)temp_buffer[2]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr3 =
                                ((data_type *)temp_buffer[3]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr4 =
                                ((data_type *)temp_buffer[4]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr5 =
                                ((data_type *)temp_buffer[5]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr6 =
                                ((data_type *)temp_buffer[6]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr7 =
                                ((data_type *)temp_buffer[7]) + buffer_index_kernel * size_per_buffer_kernel;
                            //second 8 ranks processing
                            data_type *peer_ptr8 =
                                ((data_type *)temp_buffer[8]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr9 =
                                ((data_type *)temp_buffer[9]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr10 =
                                ((data_type *)temp_buffer[10]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr11 =
                                ((data_type *)temp_buffer[11]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr12 =
                                ((data_type *)temp_buffer[12]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr13 =
                                ((data_type *)temp_buffer[13]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr14 =
                                ((data_type *)temp_buffer[14]) + buffer_index_kernel * size_per_buffer_kernel;
                            data_type *peer_ptr15 =
                                ((data_type *)temp_buffer[15]) + buffer_index_kernel * size_per_buffer_kernel;

#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 0 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr0 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 1 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr1 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 2 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr2 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 3 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr3 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 4 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr4 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 5 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr5 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 6 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr6 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 7 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr7 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 8 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr8 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 9 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr9 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 10 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr10 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 11 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr11 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 12 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr12 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 13 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr13 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 14 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr14 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 15 * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>(
                                        peer_ptr15 + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                            }
                            //do the actual reduction
                            result = 0;
#pragma unroll
                            for (int r = 0; r < 16; r++) {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result =
                                    result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }
                        }
                        else { //this is for 2,4,6 ranks. So there is no problem of overflowing the buffer.
                            for (uint32_t r = 0; r < temp_world; r++) {
                                data_type *peer_ptr =
                                    ((data_type *)temp_buffer[r]) + buffer_index_kernel * size_per_buffer_kernel;
#pragma unroll
                                for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                    buffer.template select<SIMD, 1>(unroll_i * SIMD + r * SIMD * UNROLL_SIZE) =
                                        block_load<data_type, SIMD>(
                                            peer_ptr + send_offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                            properties{ alignment<align> });
                                }
                            }
                            //do the actual reduction
                            result = 0;
                            for (uint32_t r = 0; r < temp_world; r++) {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result =
                                    result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }
                        }

                        //write out the results
                        if (offset + i * SIMD * UNROLL_SIZE + UNROLL_SIZE * SIMD <= recv_size) {
#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                block_store<data_type, SIMD>(
                                    (data_type *)out_buffer + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                    result.template select<SIMD, 1>(unroll_i * SIMD),
                                    properties{ alignment<align> });
                            }
                        }
                        else if (offset + i * SIMD * UNROLL_SIZE < recv_size) {
                            int count = recv_size - (offset + i * SIMD * UNROLL_SIZE);
                            for (int c = 0; c < count; c++) {
                                ((data_type *)out_buffer)[offset + i * SIMD * UNROLL_SIZE + c] = result[c];
                            }
                        }
                        else
                            break;
                    }
                });
        });
        //e.wait();
        return e;
    }

    template <int kernel_inner_loop_scalar, int wg_size>
    sycl::event reduce_scatter_scalar(sycl::queue &queue,
                                      const void *send_buf,
                                      void *out_buffer,
                                      ccl::datatype dtype,
                                      size_t recv_size,
                                      const ccl::vector_class<ccl::event> &deps,
                                      bool &done) {
        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;

        assert(this->initialized == true);

        size_t total_count = recv_size * world;

        done = true;

        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }

        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel = size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));

        uint32_t total_threads_needed_for_reduce =
            (recv_size + kernel_inner_loop_scalar - 1) / kernel_inner_loop_scalar;
        uint32_t total_threads_needed;
        uint32_t copy_count;
        if (total_threads_needed_for_reduce > MAX_THREAD && kernel_inner_loop_scalar == 1) {
            total_threads_needed = total_threads_needed_for_reduce;
            copy_count = temp_world;
        }
        else {
            total_threads_needed = (total_count + kernel_inner_loop_scalar - 1) / kernel_inner_loop_scalar;
            copy_count = 1;
        }
        uint32_t total_threads_dispatched = (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = total_threads_dispatched / wg_size;

        // checking oversubscription of hardware threads
        ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
        ssize_t dev_id{ ccl::utils::invalid_device_id };
        if (!ccl::ze::get_device_global_id(ze_dev, &dev_id)) {
            CCL_THROW("unable to get global id for device\n");
        }
        if (total_threads_dispatched > ccl::global_data::get().ze_data->devices[dev_id].total_threads * 8) {
            done = false;
            return e;
        }

        int buffer_index_kernel = buffer_index;
        buffer_index++;
        buffer_index %= TRIPLE_BUFFER;

        std::vector<sycl::event> dep_events = get_sycl_events(deps);

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dep_events);
            cgh.parallel_for<Reduce_scatter_small_kernel_scalar<data_type, kernel_inner_loop_scalar, wg_size>>(
                sycl::nd_range<1>({ total_threads_dispatched }, wg_size),
                [=](sycl::nd_item<1> idx2) [[sycl::reqd_sub_group_size(wg_size)]] {
                    uint32_t idx = idx2.get_global_id();

                    int *local_sync_ptr;

                    //use the temp buffer for the current rank to copy the data to.
                    data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                    local_temp_ptr +=
                        (buffer_index_kernel *
                         size_per_buffer_kernel); //point to the correct buffer inside the triple buffer

                    //process the input only if the thread is useful
                    if (idx < total_threads_needed) {
                        uint32_t offset = idx * kernel_inner_loop_scalar * copy_count;
                        if ((size_t)((data_type *)send_buf + offset) % 4 == 0 &&
                            kernel_inner_loop_scalar * copy_count > 1) {
                            gpu_kernel_copy((char *)(local_temp_ptr + offset),
                                            (const char *)((data_type *)send_buf + offset),
                                            kernel_inner_loop_scalar * copy_count * sizeof(data_type));
                        }
                        else {
                            for (int i = 0; i < kernel_inner_loop_scalar * copy_count; i++) {
                                (local_temp_ptr + offset)[i] = ((data_type *)send_buf + offset)[i];
                            }
                        }
                    }

                    //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank

                    //sync locally within local GPU first.
                    local_sync_ptr = (int *)temp_sync_buffer
                        [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_for_sync_kernel);

                    //if there are more than 1 threads required per rank, then do the local sync within the rank first.
                    if (total_threads_needed > 1) {
                        //do local sync in two steps. First using TG barrier. Then global L3 atomics.
                        uint32_t local_tid = idx2.get_local_linear_id();
                        if (local_tid == 0) {
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(local_sync_ptr[0]);
                            atomic_p += 1;

                            //wait for all the local TG to sync. Then sync the other remote GPUs
                            uint32_t val = atomic_p.load();
                            while (val != total_wg_count) {
                                val = atomic_p.load();
                            }
                        }
                        //idx2.barrier();
                    }

                    //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                    if (total_threads_dispatched >= temp_world) {
                        if (idx < temp_world) {
                            int *sync_ptr = (int *)temp_sync_buffer[idx];
                            sync_ptr += (buffer_index_kernel * size_per_buffer_for_sync_kernel);
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(sync_ptr[1]);
                            atomic_p++;
                        }
                    }
                    else if (idx == 0) { //one thread in the local gpu notifies the remote gpu of its status.
                        for (uint32_t i = 0; i < temp_world; i++) {
                            int *sync_ptr;
                            sync_ptr = (int *)temp_sync_buffer
                                [i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr += (buffer_index_kernel * size_per_buffer_for_sync_kernel);
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(sync_ptr[1]);
                            atomic_p++;
                        }
                    }

                    //sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
                    //once the local sync is done, retire useless threads
                    if (idx >= total_threads_needed)
                        return;

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    sycl::atomic_ref<int,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_p(local_sync_ptr[1]);
                    uint32_t val = atomic_p.load();
                    while (val < temp_world) {
                        val = atomic_p.load();
                    }

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx == 0) { //one thread in the local gpu notifies the remote gpu of its status.
                        int buffer_index_to_reset = (buffer_index_kernel + 2) % 3;
                        local_sync_ptr = (int *)temp_sync_buffer
                            [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr += (buffer_index_to_reset * size_per_buffer_for_sync_kernel);
                        local_sync_ptr[0] = local_sync_ptr[1] = 0;
                    }

                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    if (idx >= total_threads_needed_for_reduce)
                        return;

                    //data_type result[kernel_inner_loop_scalar];
                    uint32_t send_offset = temp_rank * recv_size + idx * kernel_inner_loop_scalar;

                    data_type *out_ptr = (data_type *)out_buffer + idx * kernel_inner_loop_scalar;
                    switch (temp_world) {
                        case 2:
                            reduce_kernel<data_type, 2, kernel_inner_loop_scalar>(
                                (void **)temp_buffer,
                                buffer_index_kernel * size_per_buffer_kernel,
                                send_offset,
                                out_ptr);
                            break;
                        case 4:
                            reduce_kernel<data_type, 4, kernel_inner_loop_scalar>(
                                (void **)temp_buffer,
                                buffer_index_kernel * size_per_buffer_kernel,
                                send_offset,
                                out_ptr);
                            break;
                        case 6:
                            reduce_kernel<data_type, 6, kernel_inner_loop_scalar>(
                                (void **)temp_buffer,
                                buffer_index_kernel * size_per_buffer_kernel,
                                send_offset,
                                out_ptr);
                            break;
                        case 8:
                            reduce_kernel<data_type, 8, kernel_inner_loop_scalar>(
                                (void **)temp_buffer,
                                buffer_index_kernel * size_per_buffer_kernel,
                                send_offset,
                                out_ptr);
                            break;
                        case 10:
                            reduce_kernel<data_type, 10, kernel_inner_loop_scalar>(
                                (void **)temp_buffer,
                                buffer_index_kernel * size_per_buffer_kernel,
                                send_offset,
                                out_ptr);
                            break;
                        case 12:
                            reduce_kernel<data_type, 12, kernel_inner_loop_scalar>(
                                (void **)temp_buffer,
                                buffer_index_kernel * size_per_buffer_kernel,
                                send_offset,
                                out_ptr);
                            break;
                        case 14:
                            reduce_kernel<data_type, 14, kernel_inner_loop_scalar>(
                                (void **)temp_buffer,
                                buffer_index_kernel * size_per_buffer_kernel,
                                send_offset,
                                out_ptr);
                            break;
                        case 16:
                            reduce_kernel<data_type, 16, kernel_inner_loop_scalar>(
                                (void **)temp_buffer,
                                buffer_index_kernel * size_per_buffer_kernel,
                                send_offset,
                                out_ptr);
                            break;
                        default: assert(0);
                    }
                });
        });
        return e;
    }

private:
    void *buffers[max_rank]{};
    void *sync_buffer[max_rank]{};
    size_t offsets[max_rank]{};
    ze_ipc_mem_handle_t ipc_handle[max_rank]{};
    int rank{ ccl::utils::invalid_rank }, world{ ccl::utils::invalid_err_code };
    int buffer_index{ ccl::utils::invalid_err_code };
    int size_per_buffer{ ccl::utils::invalid_bytes_value };
    int data_size_per_buffer{ ccl::utils::invalid_bytes_value };
};

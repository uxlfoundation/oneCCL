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

#define MAX_REPETITION                 16
#define SIMD_MAX                       256
#define SIMD                           (SIMD_MAX / sizeof(data_type))
#define SIMD_ATOMIC                    16
#define MAX_RANK                       16
#define UNROLL_SIZE                    1
#define TRIPLE_BUFFER                  3
#define SYNC_BYTE                      (SIMD_ATOMIC * sizeof(int) * 2)
#define ALIGNMENT_BYTE                 256
#define EU_COUNT                       512
#define THREADS_PER_EU                 8
#define MAX_THREAD                     (EU_COUNT * THREADS_PER_EU)
#define MAX_COUNT                      (SIMD * UNROLL_SIZE * kernel_inner_loop * MAX_THREAD)
#define LOOP_COUNT_LIMIT               (1000000)
#define DEBUG_DATA_SIZE                16
#define DEBUG_THREAD_COUNT             2
#define DEBUG_DUMP_TO_DEDICATED_OFFSET 1
#define DEBUG                          0
#define TEST_REP                       50
#define INIT_SIZE                      64
#define INIT_COUNT                     1
#define SIMD_INIT                      (INIT_SIZE * INIT_COUNT)
#define SMALLEST_NORM_FP16             0.00006103515625

extern void *allreduce_small_buffer;
extern void *allreduce_small_buffers[MAX_RANK];
extern void *allreduce_small_sync_buffer[MAX_RANK];
extern size_t allreduce_small_offsets[MAX_RANK];
extern ze_ipc_mem_handle_t allreduce_small_ipc_handle[MAX_RANK];
extern int allreduce_small_buffer_index;

const int kernel_inner_loop = 1;

template <typename data_type, uint32_t N, int kernel_inner_loop_scalar>
ESIMD_INLINE void reduce_kernel(void **temp_buffer,
                                int buf_offset,
                                int offset,
                                data_type result[]) {
    data_type peer[N][kernel_inner_loop_scalar];
#pragma unroll
    for (uint32_t r = 0; r < N; r++) {
        data_type *peer_ptr = (data_type *)(temp_buffer[r]) + buf_offset + offset;
        gpu_kernel_copy(
            (char *)peer[r], (const char *)peer_ptr, kernel_inner_loop_scalar * sizeof(data_type));
    }
    gpu_kernel_copy(
        (char *)result, (const char *)peer[0], kernel_inner_loop_scalar * sizeof(data_type));
#pragma unroll
    for (uint32_t r = 1; r < N; r++) {
        for (int j = 0; j < kernel_inner_loop_scalar; j++)
            result[j] += peer[r][j];
    }
}

template <typename dtype, size_t align>
class Allreduce_small_kernel;
template <typename dtype, int kernel_inner_loop_scalar>
class Allreduce_small_kernel_scalar;

template <typename data_type, uint32_t max_rank = MAX_RANK, uint32_t max_buffer = 1024 /*KB*/>
class sycl_allreducer_small : public sycl_coll_base<data_type> {
public:
    sycl_allreducer_small() : sycl_coll_base<data_type>() {
        size_per_buffer = 0;
        data_size_per_buffer = 0;
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
        data_size_per_buffer = ((MAX_COUNT + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) /
                                (SIMD * UNROLL_SIZE * kernel_inner_loop)) *
                               SIMD * UNROLL_SIZE * kernel_inner_loop;
        data_size_per_buffer =
            ((data_size_per_buffer * sizeof(data_type) + ALIGNMENT_BYTE - 1) / ALIGNMENT_BYTE) *
            ALIGNMENT_BYTE / sizeof(data_type); //aligned size
        data_size_per_buffer *= 2;
        size_per_buffer = data_size_per_buffer * sizeof(data_type) + SYNC_BYTE;

        if (allreduce_small_buffer == NULL) {
            allreduce_small_buffer_index = 0;
            allreduce_small_buffer = sycl::malloc_device(size_per_buffer * TRIPLE_BUFFER, queue);

            auto e = queue.memset(allreduce_small_buffer, 0, size_per_buffer * TRIPLE_BUFFER);
            e.wait();
            this->exchange_peer_ipc_mem(queue,
                                        comm_in,
                                        stream,
                                        allreduce_small_buffer,
                                        NULL,
                                        rank,
                                        world,
                                        data_size_per_buffer * sizeof(data_type),
                                        (void **)allreduce_small_buffers,
                                        (void **)allreduce_small_sync_buffer,
                                        allreduce_small_offsets,
                                        allreduce_small_ipc_handle,
                                        NULL,
                                        NULL /* mmap_buffers */,
                                        false /* to_cache */);

            int wg_size = 1;
            //dummy kernel to avoid hang. The hang happens when there is no dummy kernel and allreduce() is called right after init().
            e = queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::nd_range<1>({ 1 }, wg_size),
                                 [=](sycl::nd_item<1> idx) SYCL_ESIMD_KERNEL {

                                 });
            });
            e.wait();
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
        assert(this->initialized == true);

        done = true;
        // check local alignment
        bool is_aligned =
            sizeof(data_type) >= 4 || ((size_t)in_buffer % 4 == 0 && (size_t)out_buffer % 4 == 0);

        //if (size * sizeof(data_type) <= 4096) {
        if (size * sizeof(data_type) <= 65536 && is_aligned) {
            if (size == 1) {
                e = allreduce_scalar<1>(queue, in_buffer, out_buffer, size, deps, done);
            }
            else if (size == 2) {
                e = allreduce_scalar<2>(queue, in_buffer, out_buffer, size, deps, done);
            }
            else {
                e = allreduce_scalar<4>(queue, in_buffer, out_buffer, size, deps, done);
            }
        }
        else {
            if (is_aligned) {
                e = allreduce_esimd<4>(queue, in_buffer, out_buffer, size, deps, done);
            }
            else {
                e = allreduce_esimd<2>(queue, in_buffer, out_buffer, size, deps, done);
            }
        }

        if (reduction == ccl::reduction::avg) {
            std::vector<sycl::event> evs;
            evs.push_back(e);
            e = sycl_average(queue, out_buffer, size, world, dtype, evs);
        }

        return ccl::event::create_from_native(e);
    }

    template <size_t align>
    sycl::event allreduce_esimd(sycl::queue &queue,
                                const void *in_buffer,
                                void *out_buffer,
                                size_t size,
                                const ccl::vector_class<ccl::event> &deps,
                                bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;

        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = allreduce_small_buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = allreduce_small_sync_buffer[i];
        }
        uint32_t total_threads_needed = (size + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) /
                                        (SIMD * UNROLL_SIZE * kernel_inner_loop); //ceiling
        const uint32_t wg_size = 16;
        int size_per_buffer_kernel = size_per_buffer;
        uint32_t total_threads_dispatched =
            (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t __attribute__((unused)) total_wg_count = total_threads_dispatched / wg_size;

        done = true;
        // checking oversubscription of hardware threads
        ze_device_handle_t ze_dev =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
        ssize_t dev_id{ ccl::utils::invalid_device_id };
        if (!ccl::ze::get_device_global_id(ze_dev, &dev_id)) {
            CCL_THROW("unable to get global id for device\n");
        }
        if (total_threads_dispatched >
            ccl::global_data::get().ze_data->devices[dev_id].total_threads) {
            done = false;
            return e;
        }

        int buffer_index_kernel = allreduce_small_buffer_index;
        allreduce_small_buffer_index++;
        allreduce_small_buffer_index %= TRIPLE_BUFFER;

        auto sycl_events = get_sycl_events(deps);

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(sycl_events);
            cgh.parallel_for<class Allreduce_small_kernel<data_type, align>>(
                sycl::nd_range<1>({ total_threads_dispatched }, wg_size), [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL{
                uint32_t idx = idx2.get_global_id();

                //ESIMD kernel
                uint offset = idx * SIMD * UNROLL_SIZE * kernel_inner_loop;
                simd<data_type, max_rank * SIMD * UNROLL_SIZE> buffer; //64 registers
                simd<data_type, SIMD * UNROLL_SIZE> buffer_small;
                simd<ushort, SIMD_ATOMIC> ramp;
                simd_mask<SIMD_ATOMIC> pred;
                simd<int, SIMD_ATOMIC> status0;
                int *local_sync_ptr;

                //to do:
                //O3 compiler optimization: not much difference after the change.
                //tune the fence: good perf improvements
                //tune the cacheability for each IO message: no noticeable improvement
                //tune the thread size: not much improvements
                //tune the polling freq

#pragma unroll
                for (uint32_t i = 0; i < SIMD_ATOMIC; i++) {
                    ramp[i] = i * sizeof(int);
                }

                //process the input only if the thread is useful
                if (idx < total_threads_needed) {
                    //do copy from input buffer to temp buffer.
                    for (int i = 0; i < kernel_inner_loop; i++) {
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer_small.template select<SIMD, 1>(unroll_i * SIMD) =
                                block_load<data_type, SIMD>((data_type *)in_buffer + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                        }

                        //use the temp buffer for the current rank to copy the data to.
                        data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                        local_temp_ptr +=
                            (buffer_index_kernel * size_per_buffer_kernel /
                             sizeof(
                                 data_type)); //point to the correct buffer inside the triple buffer

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
                }

                //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank

                //sync locally within local GPU first.
                local_sync_ptr = (int *)temp_sync_buffer
                    [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                local_sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

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
                            status0 =
                                lsc_atomic_update<atomic_op::load,
                                                  int,
                                                  SIMD_ATOMIC,
                                                  lsc_data_size::default_size,
                                                  cache_hint::none,
                                                  cache_hint::none>(local_sync_ptr, ramp, pred);
                        }
                    }
                    idx2.barrier();
                }

                //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                pred = false;
                pred[1] = true; //use different lane for the remote gpu sync
                if (total_threads_dispatched >= temp_world) {
                    if (idx < temp_world) {
                        int *sync_ptr;

                        // DEBUG: rank3 seems to have some problem where its cpu time is always ~100us
                        // to debug this, change the way inter-gpu are synced.

                        status0 = total_threads_needed;
                        sync_ptr = (int *)temp_sync_buffer
                            [idx]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        lsc_atomic_update<atomic_op::add,
                                          int,
                                          SIMD_ATOMIC,
                                          lsc_data_size::default_size,
                                          cache_hint::none,
                                          cache_hint::none>(sync_ptr, ramp, status0, pred);
                    }
                }
                else if (idx ==
                         0) //one thread in the local gpu notifies the remote gpu of its status.
                {
                    status0 = total_threads_needed;
                    for (uint32_t i = 0; i < temp_world; i++) {
                        int *sync_ptr = (int *)temp_sync_buffer
                            [i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        lsc_atomic_update<atomic_op::add,
                                          int,
                                          SIMD_ATOMIC,
                                          lsc_data_size::default_size,
                                          cache_hint::none,
                                          cache_hint::none>(sync_ptr, ramp, status0, pred);
                    }
                }

                //once the local sync is done, retire useless threads
                if (idx >= total_threads_needed) {
                    return;
                }

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
                if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                {
                    int buffer_index_to_reset =
                        (buffer_index_kernel + TRIPLE_BUFFER - 1) % TRIPLE_BUFFER;
                    status0 = 0;
                    pred = true;
                    local_sync_ptr = (int *)temp_sync_buffer
                        [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr +=
                        (buffer_index_to_reset * size_per_buffer_kernel / sizeof(int));
                    lsc_atomic_update<atomic_op::store,
                                      int,
                                      SIMD_ATOMIC,
                                      lsc_data_size::default_size,
                                      cache_hint::none,
                                      cache_hint::none>(
                        local_sync_ptr, ramp, status0, pred); //reset the first half of sync buffer
                }

                //at this point, all the threads are done copying data from input buffer to temp buffer.
                //do All reduce
                simd<data_type, SIMD * UNROLL_SIZE> result;
                for (int i = 0; i < kernel_inner_loop; i++) {
                    if (temp_world == 2) {
                        simd<data_type, 2 * SIMD * UNROLL_SIZE> buffer2; //64 registers
                        int *peer_ptr0 =
                            ((int *)temp_buffer[0]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr1 =
                            ((int *)temp_buffer[1]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer2.template select<SIMD, 1>(unroll_i * SIMD +
                                                             0 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr0 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer2.template select<SIMD, 1>(unroll_i * SIMD +
                                                             1 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr1 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                        }
                        //do the actual reduction
                        result = buffer2.template select<SIMD * UNROLL_SIZE, 1>(0);
                        result = result +
                                 buffer2.template select<SIMD * UNROLL_SIZE, 1>(SIMD * UNROLL_SIZE);
                    }
                    else if (temp_world == 4) {
                        int *peer_ptr0 =
                            ((int *)temp_buffer[0]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr1 =
                            ((int *)temp_buffer[1]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr2 =
                            ((int *)temp_buffer[2]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr3 =
                            ((int *)temp_buffer[3]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            0 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr0 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            1 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr1 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            2 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr2 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            3 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr3 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                        }
                        //do the actual reduction
                        result = 0;
#pragma unroll
                        for (int r = 0; r < 4; r++) {
                            //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(
                                                  r * SIMD * UNROLL_SIZE);
                        }
                    }
                    else if (temp_world == 8) {
                        int *peer_ptr0 =
                            ((int *)temp_buffer[0]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr1 =
                            ((int *)temp_buffer[1]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr2 =
                            ((int *)temp_buffer[2]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr3 =
                            ((int *)temp_buffer[3]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr4 =
                            ((int *)temp_buffer[4]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr5 =
                            ((int *)temp_buffer[5]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr6 =
                            ((int *)temp_buffer[6]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr7 =
                            ((int *)temp_buffer[7]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            0 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr0 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            1 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr1 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            2 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr2 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            3 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr3 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            4 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr4 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            5 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr5 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            6 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr6 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            7 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr7 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                        }
                        //do the actual reduction
                        result = 0;
#pragma unroll
                        for (int r = 0; r < 8; r++) {
                            //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(
                                                  r * SIMD * UNROLL_SIZE);
                        }
                    }
                    else if (temp_world == 12) {
                        int *peer_ptr0 =
                            ((int *)temp_buffer[0]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr1 =
                            ((int *)temp_buffer[1]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr2 =
                            ((int *)temp_buffer[2]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr3 =
                            ((int *)temp_buffer[3]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr4 =
                            ((int *)temp_buffer[4]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr5 =
                            ((int *)temp_buffer[5]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr6 =
                            ((int *)temp_buffer[6]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr7 =
                            ((int *)temp_buffer[7]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr8 =
                            ((int *)temp_buffer[8]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr9 =
                            ((int *)temp_buffer[9]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr10 =
                            ((int *)temp_buffer[10]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr11 =
                            ((int *)temp_buffer[11]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            0 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr0 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            1 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr1 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            2 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr2 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            3 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr3 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            4 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr4 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            5 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr5 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            6 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr6 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            7 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr7 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            8 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr8 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            9 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr9 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            10 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr10 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            11 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr11 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                        }
                        //do the actual reduction
                        result = 0;
#pragma unroll
                        for (int r = 0; r < 12; r++) {
                            //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(
                                                  r * SIMD * UNROLL_SIZE);
                        }
                    }
                    else if (temp_world == 16) {
                        //first 8 ranks processing
                        int *peer_ptr0 =
                            ((int *)temp_buffer[0]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr1 =
                            ((int *)temp_buffer[1]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr2 =
                            ((int *)temp_buffer[2]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr3 =
                            ((int *)temp_buffer[3]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr4 =
                            ((int *)temp_buffer[4]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr5 =
                            ((int *)temp_buffer[5]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr6 =
                            ((int *)temp_buffer[6]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr7 =
                            ((int *)temp_buffer[7]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        //second 8 ranks processing
                        int *peer_ptr8 =
                            ((int *)temp_buffer[8]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr9 =
                            ((int *)temp_buffer[9]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr10 =
                            ((int *)temp_buffer[10]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr11 =
                            ((int *)temp_buffer[11]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr12 =
                            ((int *)temp_buffer[12]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr13 =
                            ((int *)temp_buffer[13]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr14 =
                            ((int *)temp_buffer[14]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                        int *peer_ptr15 =
                            ((int *)temp_buffer[15]) +
                            (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            0 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr0 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            1 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr1 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            2 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr2 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            3 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr3 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            4 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr4 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            5 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr5 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            6 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr6 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            7 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr7 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            8 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr8 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            9 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr9 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            10 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr10 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            11 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr11 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            12 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr12 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            13 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr13 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            14 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr14 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                            buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                            15 * SIMD * UNROLL_SIZE) =
                                block_load<data_type, SIMD>((data_type *)peer_ptr15 + offset +
                                                                unroll_i * SIMD +
                                                                i * SIMD * UNROLL_SIZE,
                                                            properties{ alignment<align> });
                        }
                        //do the actual reduction
                        result = 0;
#pragma unroll
                        for (int r = 0; r < 16; r++) {
                            //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(
                                                  r * SIMD * UNROLL_SIZE);
                        }
                    }
                    else //this is for 6,10,14 ranks. So there is no problem of overflowing the buffer.
                    {
                        for (uint32_t r = 0; r < temp_world; r++) {
                            int *peer_ptr =
                                ((int *)temp_buffer[r]) +
                                (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                buffer.template select<SIMD, 1>(unroll_i * SIMD +
                                                                r * SIMD * UNROLL_SIZE) =
                                    block_load<data_type, SIMD>((data_type *)peer_ptr + offset +
                                                                    unroll_i * SIMD +
                                                                    i * SIMD * UNROLL_SIZE,
                                                                properties{ alignment<align> });
                            }
                        }
                        //do the actual reduction
                        result = 0;
                        for (uint32_t r = 0; r < temp_world; r++) {
                            //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(
                                                  r * SIMD * UNROLL_SIZE);
                        }
                    }

                    //write out the results
#pragma unroll
                    for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                        block_store<data_type, SIMD>(
                            (data_type *)out_buffer + offset + unroll_i * SIMD +
                                i * SIMD * UNROLL_SIZE,
                            result.template select<SIMD, 1>(unroll_i * SIMD),
                            properties{ alignment<align> });
                    }
                }

                });
        });

        return e;
    }

private:
    template <int kernel_inner_loop_scalar>
    sycl::event allreduce_scalar(sycl::queue &queue,
                                 const void *in_buffer,
                                 void *out_buffer,
                                 size_t size,
                                 const ccl::vector_class<ccl::event> &deps,
                                 bool &done) {
        sycl::event e;

        uint32_t temp_rank = rank;
        uint32_t temp_world = world;

        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = allreduce_small_buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = allreduce_small_sync_buffer[i];
        }
        int size_per_buffer_kernel = size_per_buffer / sizeof(data_type);
        int size_per_buffer_for_sync_kernel =
            size_per_buffer_kernel / (sizeof(int) / sizeof(data_type));

        uint32_t max_wg_size __attribute__((unused)) =
            queue.get_device().get_info<sycl::info::device::max_work_group_size>(); // 1024
        const uint32_t wg_size = 16;
        assert(wg_size <= max_wg_size);

        uint32_t total_threads_needed =
            (size + kernel_inner_loop_scalar - 1) / kernel_inner_loop_scalar;
        uint32_t total_threads_dispatched =
            (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = total_threads_dispatched / wg_size;

        done = true;
        // checking oversubscription of hardware threads
        ze_device_handle_t ze_dev =
            sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
        ssize_t dev_id{ ccl::utils::invalid_device_id };
        if (!ccl::ze::get_device_global_id(ze_dev, &dev_id)) {
            CCL_THROW("unable to get global id for device\n");
        }
        if (total_threads_dispatched >
            ccl::global_data::get().ze_data->devices[dev_id].total_threads * 8) {
            done = false;
            return e;
        }

        auto sycl_events = get_sycl_events(deps);

#ifdef CCL_DEBUG_NAN
        // check for NAN in inputabuffer
        sycl_check_nan<data_type>(
            queue, temp_rank, (data_type *)in_buffer, size, "allreduce_small", sycl_events);
#endif

        int buffer_index_kernel = allreduce_small_buffer_index;
        allreduce_small_buffer_index++;
        allreduce_small_buffer_index %= TRIPLE_BUFFER;

        // pure scalar kernel
        e = queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(sycl_events);
                cgh.parallel_for<class Allreduce_small_kernel_scalar<data_type, kernel_inner_loop_scalar>>(
                    sycl::nd_range<1>( total_threads_dispatched, wg_size), [=](sycl::nd_item<1> idx2) [[sycl::reqd_sub_group_size(wg_size)]] {
                    uint32_t idx = idx2.get_global_id();
                    uint32_t offset __attribute__((unused)) = idx * kernel_inner_loop_scalar;

                    if (idx < total_threads_needed) {
                        //do copy from input buffer to temp buffer.
                        data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                        local_temp_ptr += (buffer_index_kernel * size_per_buffer_kernel);
                        data_type *inbuf = (data_type *)in_buffer + offset;
                        gpu_kernel_copy((char *)(local_temp_ptr + offset),
                                        (const char *)inbuf,
                                        kernel_inner_loop_scalar * sizeof(data_type));
                        //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank
                    }

                    int *local_sync_ptr = (int *)temp_sync_buffer[temp_rank] +
                                          (buffer_index_kernel * size_per_buffer_for_sync_kernel);
                    //if there are more than 1 threads required per rank, then do the local sync within the rank first.
                    uint32_t local_tid = idx2.get_local_linear_id();
                    if (total_threads_needed > 1) {
                        //sync locally within local GPU first.
                        if (local_tid == 0) {
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(local_sync_ptr[0]);
                            atomic_p += 1;

                            //wait for all the local TG to sync. Then sync the other remote GPUs
                            uint32_t val = atomic_p.load();
                            //sycl::_V1::ext::oneapi::experimental::printf("HERE in: rank%d sync: %p idx:%d val: %d %d\n", temp_rank, local_sync_ptr, idx, val, total_wg_count);
                            while (val < total_wg_count) {
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
                    else if (idx == 0) {
                        //one thread in the local gpu notifies the remote gpu of its status.
                        for (uint32_t i = 0; i < temp_world; i++) {
                            int *sync_ptr = (int *)temp_sync_buffer[i];
                            sync_ptr += (buffer_index_kernel * size_per_buffer_for_sync_kernel);
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(sync_ptr[1]);
                            atomic_p++;
                        }
                    }

                    //once the local sync is done, retire useless threads
                    if (idx >= total_threads_needed)
                        return;

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    if (local_tid == 0) {
                        sycl::atomic_ref<int,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_p(local_sync_ptr[1]);
                        uint32_t val = atomic_p.load();
                        while (val < temp_world) {
                            val = atomic_p.load();
                        }
                    }

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx == 0) {
                        int buffer_index_to_reset =
                            (buffer_index_kernel + TRIPLE_BUFFER - 1) % TRIPLE_BUFFER;
                        local_sync_ptr = (int *)temp_sync_buffer[temp_rank];
                        local_sync_ptr += (buffer_index_to_reset * size_per_buffer_for_sync_kernel);
                        local_sync_ptr[0] = local_sync_ptr[1] = 0;
                    }

                    //at this point, all the threads are done copying data from input buffer to temp buffer, do All reduce
                    auto reduce_kernel_lambda = [&]<int N>() {
                        reduce_kernel<data_type, N, kernel_inner_loop_scalar>(
                            (void **)temp_buffer,
                            buffer_index_kernel * size_per_buffer_kernel,
                            offset,
                            (data_type *)out_buffer + offset);
                    };
                    switch (temp_world) {
                        case 2: reduce_kernel_lambda.template operator()<2>(); break;
                        case 4: reduce_kernel_lambda.template operator()<4>(); break;
                        case 6: reduce_kernel_lambda.template operator()<6>(); break;
                        case 8: reduce_kernel_lambda.template operator()<8>(); break;
                        case 10: reduce_kernel_lambda.template operator()<10>(); break;
                        case 12: reduce_kernel_lambda.template operator()<12>(); break;
                        case 14: reduce_kernel_lambda.template operator()<14>(); break;
                        case 16: reduce_kernel_lambda.template operator()<16>(); break;
                        default: assert(0);
                    }
                    });
        });
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

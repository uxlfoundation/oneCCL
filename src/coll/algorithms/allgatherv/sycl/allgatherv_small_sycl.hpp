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

#define SIMD_MAX       256
#define SIMD           (SIMD_MAX / sizeof(data_type))
#define SIMD_ATOMIC    16
#define MAX_RANK       16
#define UNROLL_SIZE    1
#define TRIPLE_BUFFER  3
#define SYNC_BYTE      (SIMD_ATOMIC * sizeof(int) * 2)
#define ALIGNMENT_BYTE 256
#define EU_COUNT       512
#define THREADS_PER_EU 8
#define MAX_THREAD     (EU_COUNT * THREADS_PER_EU)
#define MAX_COUNT      (SIMD * UNROLL_SIZE * MAX_THREAD)
#define INIT_SIZE      64
#define INIT_COUNT     1
#define SIMD_INIT      (INIT_SIZE * INIT_COUNT)

template <typename dtype, int kernel_inner_loop, size_t align>
class oneccl_allgatherv_small_kernel_esimd;
template <typename dtype, int kernel_inner_loop_scalar, int wg_size>
class oneccl_allgatherv_small_kernel_scalar;

template <typename data_type, uint32_t max_rank = MAX_RANK, uint32_t max_buffer = 1024 /*KB*/>
class sycl_allgatherv_small : public sycl_coll_base<data_type> {
public:
    sycl_allgatherv_small() : sycl_coll_base<data_type>() {
        buffer_index = 0;
        size_per_buffer = 0;
    }

    void init(sycl::queue &queue, ccl_comm *comm, ccl_stream *stream, uint32_t rank_in, uint32_t world_in) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.
        data_size_per_buffer = ((MAX_COUNT + SIMD * UNROLL_SIZE - 1) / (SIMD * UNROLL_SIZE)) * SIMD * UNROLL_SIZE;
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

    ccl::event allgatherv(sycl::queue &queue,
                          const void *send_buf,
                          size_t send_count,
                          void *recv_buf,
                          const ccl::vector_class<size_t> &recv_counts,
                          bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        done = false;

        sycl::event e;
        assert(this->initialized == true);

        // check local alignment
        size_t is_aligned =
            (size_t)send_buf % 4 == 0 && (size_t)recv_buf % 4 == 0 && (send_count * sizeof(data_type)) % 4 == 0;

        auto esimd_lambda = [&]<int kernel_inner_loop>() {
            if (is_aligned) {
                return allgatherv_esimd<kernel_inner_loop, 4>(
                    queue, send_buf, send_count, recv_buf, recv_counts, done);
            }
            else {
                return allgatherv_esimd<kernel_inner_loop, 2>(
                    queue, send_buf, send_count, recv_buf, recv_counts, done);
            }
        };

        if (send_count * world <= MAX_THREAD) {
            if (send_count * world <= 1024) {
                e = allgatherv_scalar<1, 16>(queue, send_buf, send_count, recv_buf, recv_counts, done);
            }
            if (!done) {
                e = allgatherv_scalar<2, 16>(queue, send_buf, send_count, recv_buf, recv_counts, done);
            }
            if (!done) {
                e = allgatherv_scalar<4, 16>(queue, send_buf, send_count, recv_buf, recv_counts, done);
            }
        }
        else {
            if (send_count <= 16384) {
                e = esimd_lambda.template operator()<1>();
            }
            if (!done) {
                e = esimd_lambda.template operator()<2>();
            }
            if (!done) {
                e = esimd_lambda.template operator()<4>();
            }
        }

        return ccl::event::create_from_native(e);
    }

    template <int kernel_inner_loop, size_t align>
    sycl::event allgatherv_esimd(sycl::queue &queue,
                                 const void *send_buf,
                                 size_t send_count,
                                 void *recv_buf,
                                 const ccl::vector_class<size_t> &recv_counts,
                                 bool &done) {
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        done = true;
        sycl::event e;

        uint32_t temp_rank = rank;
        uint32_t temp_world = world;

        size_t total_count = send_count * world;

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

        int buffer_index_kernel = buffer_index;
        buffer_index++;
        buffer_index %= TRIPLE_BUFFER;

        uint32_t threads_per_rank = (send_count + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) /
                                    (SIMD * UNROLL_SIZE * kernel_inner_loop); //ceiling
        uint32_t total_threads_needed = threads_per_rank * temp_world;

        int wg_size = 16;
        uint32_t total_threads_dispatched = (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = total_threads_dispatched / wg_size;
        uint32_t total_threads_needed_for_copy = threads_per_rank;

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

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<oneccl_allgatherv_small_kernel_esimd<data_type, kernel_inner_loop, align>>(
                sycl::nd_range<1>({ total_threads_dispatched }, wg_size),
                [=](sycl::nd_item<1> idx2) SYCL_ESIMD_KERNEL {
                    uint32_t idx = idx2.get_global_id();
                    uint32_t offset = idx * SIMD * UNROLL_SIZE * kernel_inner_loop;
                    simd<data_type, SIMD * UNROLL_SIZE> buffer_small;
                    simd<ushort, SIMD_ATOMIC> ramp;
                    simd_mask<SIMD_ATOMIC> pred;
                    simd<int, SIMD_ATOMIC> status0;
                    int *local_sync_ptr;
#pragma unroll
                    for (uint32_t i = 0; i < SIMD_ATOMIC; i++) {
                        ramp[i] = i * sizeof(int);
                    }

                    if (idx < total_threads_needed_for_copy) {
                        //do copy from input buffer to temp buffer.
                        //use the temp buffer for the current rank to copy the data to.
                        data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                        //point to the correct buffer inside the triple buffer
                        local_temp_ptr += buffer_index_kernel * size_per_buffer_kernel;
                        for (int i = 0; i < kernel_inner_loop; i++) {
#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                buffer_small.template select<SIMD, 1>(unroll_i * SIMD) =
                                    block_load<data_type, SIMD>(
                                        (data_type *)send_buf + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE,
                                        properties{ alignment<align> });
                            }
#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                block_store<data_type, SIMD>(
                                    (data_type *)local_temp_ptr + offset + unroll_i * SIMD +
                                        i * SIMD * UNROLL_SIZE,
                                    buffer_small.template select<SIMD, 1>(unroll_i * SIMD),
                                    properties{ alignment<align> });
                            }
                            if (offset + (i + 1) * SIMD * UNROLL_SIZE > send_count) {
                                break;
                            }
                        }
                    } // end of copy to temp

                    //sync locally within local GPU first.
                    local_sync_ptr = (int *)temp_sync_buffer[temp_rank];
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_for_sync_kernel);

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
                    else if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        status0 = total_threads_needed;
                        for (uint32_t i = 0; i < temp_world; i++) {
                            int *sync_ptr;
                            sync_ptr = (int *)temp_sync_buffer[i];
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
                    //fence<memory_kind::global, fence_flush_op::invalidate, fence_scope::gpus>();

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
                        int buffer_index_to_reset = (buffer_index_kernel + TRIPLE_BUFFER - 1) % TRIPLE_BUFFER;
                        status0 = 0;
                        pred = true;
                        local_sync_ptr = (int *)temp_sync_buffer[temp_rank];
                        local_sync_ptr += (buffer_index_to_reset * size_per_buffer_for_sync_kernel);
                        lsc_atomic_update<atomic_op::store,
                                          int,
                                          SIMD_ATOMIC,
                                          lsc_data_size::default_size,
                                          cache_hint::none,
                                          cache_hint::none>(
                            local_sync_ptr, ramp, status0, pred); //reset the first half of sync buffer
                    }

                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    if (idx > total_threads_needed) {
                        return;
                    }

                    int r = idx / threads_per_rank;
                    int idx_rank = idx % threads_per_rank;

                    int in_place = ((char *)recv_buf + temp_rank * send_count * sizeof(data_type)) == send_buf;
                    if (r == temp_rank && in_place) {
                        return;
                    }

                    data_type *peer_ptr =
                        (data_type *)temp_buffer[r] + (buffer_index_kernel * size_per_buffer_kernel);
                    uint32_t r_offset = idx_rank * SIMD * UNROLL_SIZE * kernel_inner_loop;
                    uint32_t w_offset = send_count * r + r_offset;
                    for (int l = 0; l < kernel_inner_loop; l++) {
                        data_type *in_ptr = peer_ptr + r_offset + l * SIMD * UNROLL_SIZE;
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer_small.template select<SIMD, 1>(unroll_i * SIMD) = block_load<data_type, SIMD>(
                                in_ptr + unroll_i * SIMD, properties{ alignment<align> });
                        }
                        data_type *out_ptr = (data_type *)recv_buf + w_offset + l * SIMD * UNROLL_SIZE;
                        if (r_offset + (l + 1) * SIMD * UNROLL_SIZE <= send_count) {
#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                                block_store<data_type, SIMD>(
                                    out_ptr + unroll_i * SIMD,
                                    buffer_small.template select<SIMD, 1>(unroll_i * SIMD),
                                    properties{ alignment<align> });
                            }
                        }
                        else {
                            int count = send_count - r_offset - l * SIMD * UNROLL_SIZE;
                            if (count > SIMD * UNROLL_SIZE) {
                                count = SIMD * UNROLL_SIZE;
                            }
                            for (int i = 0; i < count; i++) {
                                out_ptr[i] = buffer_small[i];
                            }
                            break;
                        }
                    }
                });
        });
        return e;
    }

    template <int kernel_inner_loop_scalar, int wg_size>
    sycl::event allgatherv_scalar(sycl::queue &queue,
                                  const void *send_buf,
                                  size_t send_count,
                                  void *recv_buf,
                                  const ccl::vector_class<size_t> &recv_counts,
                                  bool &done) {
        sycl::event e;
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        size_t total_count = send_count * world;

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

        assert(wg_size >= 8);
        uint32_t threads_per_rank = (send_count + kernel_inner_loop_scalar - 1) / kernel_inner_loop_scalar;
        uint32_t total_threads_needed = threads_per_rank * temp_world;
        uint32_t total_threads_dispatched = (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = total_threads_dispatched / wg_size;
        uint32_t total_threads_needed_copy = threads_per_rank;

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

        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for<oneccl_allgatherv_small_kernel_scalar<data_type, kernel_inner_loop_scalar, wg_size>>(
                sycl::nd_range<1>({ total_threads_dispatched }, wg_size),
                [=](sycl::nd_item<1> idx2) [[sycl::reqd_sub_group_size(wg_size)]] {
                    uint32_t idx = idx2.get_global_id();

                    int *local_sync_ptr;

                    //use the temp buffer for the current rank to copy the data to.
                    data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                    local_temp_ptr +=
                        buffer_index_kernel *
                        size_per_buffer_kernel; //point to the correct buffer inside the triple buffer

                    //process the input only if the thread is useful
                    if (idx < total_threads_needed_copy) {
                        uint32_t offset = idx * kernel_inner_loop_scalar;
                        if ((size_t)((data_type *)send_buf + offset) % 4 == 0 && kernel_inner_loop_scalar > 1) {
                            gpu_kernel_copy((char *)(local_temp_ptr + offset),
                                            (const char *)((data_type *)send_buf + offset),
                                            kernel_inner_loop_scalar * sizeof(data_type));
                        }
                        else {
                            for (int i = 0; i < kernel_inner_loop_scalar; i++) {
                                local_temp_ptr[offset + i] = *((data_type *)send_buf + offset + i);
                            }
                        }
                    }

                    //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank

                    //sync locally within local GPU first.
                    local_sync_ptr = (int *)temp_sync_buffer
                        [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;

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
                    const uint32_t status0 = 1;
                    if (total_threads_dispatched >= temp_world) {
                        if (idx < temp_world) {
                            int *sync_ptr = (int *)temp_sync_buffer
                                [idx]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(sync_ptr[1]);
                            atomic_p += status0;
                        }
                    }
                    else if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        //uint32_t status0 = total_threads_needed;
                        for (uint32_t i = 0; i < temp_world; i++) {
                            int *sync_ptr;
                            sync_ptr = (int *)temp_sync_buffer
                                [i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr += buffer_index_kernel * size_per_buffer_for_sync_kernel;
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(sync_ptr[1]);
                            atomic_p += status0;
                        }
                    }

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
                    while (val != temp_world) {
                        val = atomic_p.load();
                    }

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        int buffer_index_to_reset = (buffer_index_kernel + TRIPLE_BUFFER - 1) % TRIPLE_BUFFER;
                        local_sync_ptr = (int *)temp_sync_buffer
                            [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr += buffer_index_to_reset * size_per_buffer_for_sync_kernel;
                        local_sync_ptr[0] = local_sync_ptr[1] = 0;
                    }

                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    uint32_t r = idx / threads_per_rank;
                    int idx_rank = idx % threads_per_rank;

                    int in_place = ((char *)recv_buf + temp_rank * send_count * sizeof(data_type)) == send_buf;
                    if (r == temp_rank && in_place)
                        return;

                    uint32_t r_offset = idx_rank * kernel_inner_loop_scalar;
                    data_type *in_ptr =
                        (data_type *)temp_buffer[r] + buffer_index_kernel * size_per_buffer_kernel + r_offset;
                    uint32_t w_offset = send_count * r + r_offset;
                    data_type *out_ptr = (data_type *)recv_buf + w_offset;
                    uint32_t count = r_offset + kernel_inner_loop_scalar <= send_count ? kernel_inner_loop_scalar
                                                                                       : send_count - r_offset;
                    if ((size_t)out_ptr % 4 == 0 && count > 1) {
                        gpu_kernel_copy((char *)out_ptr, (const char *)in_ptr, count * sizeof(data_type));
                    }
                    else {
                        for (int i = 0; i < count; i++) {
                            out_ptr[i] = in_ptr[i];
                        }
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

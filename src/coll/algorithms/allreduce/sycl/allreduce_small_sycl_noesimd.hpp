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
#define SIMD                           128
#define SIMD_ATOMIC                    16
#define MAX_RANK                       16
#define UNROLL_SIZE                    1
#define TRIPLE_BUFFER                  3
#define SYNC_BYTE                      (SIMD_ATOMIC * sizeof(int) * 2)
#define ALIGNMENT_BYTE                 256
#define EU_COUNT                       448
#define THREADS_PER_EU                 8
#define MAX_THREAD                     (EU_COUNT * THREADS_PER_EU)
#define MAX_COUNT                      (SIMD * UNROLL_SIZE * kernel_inner_loop * MAX_THREAD)
#define LOOP_COUNT_LIMIT               (1000000)
#define DEBUG_DATA_SIZE                16
#define DEBUG_THREAD_COUNT             2
#define DEBUG_DUMP_TO_DEDICATED_OFFSET 1
#define DEBUG                          0

const int kernel_inner_loop = 1;
const uint32_t vec_size = 16;

template <typename dtype>
class oneccl_allreduce_small_kernel_scalar;
template <typename dtype>
class oneccl_allreduce_small_kernel_block;

template <typename data_type, uint32_t max_rank = MAX_RANK, uint32_t max_buffer = 1024 /*KB*/>
class sycl_allreducer_small : public sycl_coll_base<data_type> {
public:
    sycl_allreducer_small() : sycl_coll_base<data_type>() {
        buffer_index = 0;
        size_per_buffer = 0;
    }

    void init(sycl::queue &queue,
              ccl_comm *comm,
              ccl_stream *stream,
              uint32_t rank_in,
              uint32_t world_in) {
        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.
        data_size_per_buffer = MAX_COUNT;
        data_size_per_buffer =
            ((data_size_per_buffer * sizeof(data_type) + ALIGNMENT_BYTE - 1) / ALIGNMENT_BYTE) *
            ALIGNMENT_BYTE / sizeof(data_type); //aligned size
        size_per_buffer = data_size_per_buffer * sizeof(data_type) + SYNC_BYTE;
        void *local_triple_buffer = sycl::malloc_device(size_per_buffer * TRIPLE_BUFFER, queue);
        int wg_size = 1;

        auto e = queue.memset(local_triple_buffer, 0, size_per_buffer * TRIPLE_BUFFER);
        e.wait();
        this->exchange_peer_ipc_mem(queue,
                                    comm,
                                    local_triple_buffer,
                                    rank,
                                    world,
                                    data_size_per_buffer,
                                    (void **)buffers,
                                    (void **)sync_buffer,
                                    offsets,
                                    ipc_handle);
        this->initialized = true;

        //dummy kernel to avoid hang. The hang happens when there is no dummy kernel and allreduce() is called right after init().
        e = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<1>({ 1 }, wg_size), [=](sycl::item<1> idx) {

            });
        });
        e.wait();
    }
    ccl::event allreduce(sycl::queue &queue,
                         const void *in_buffer,
                         void *out_buffer,
                         uint32_t size,
                         int repetition,
                         bool print_en) {
        sycl::event e;

        if (repetition > MAX_REPETITION) {
            printf("error: repetition cannot be larger than %d\n", MAX_REPETITION);
            exit(-1);
        }
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        int r;
        assert(this->initialized == true);
        void *temp_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_buffer[i] = buffers[i];
        }
        void *temp_sync_buffer[max_rank];
        for (int i = 0; i < world; i++) {
            temp_sync_buffer[i] = sync_buffer[i];
        }
        int size_per_buffer_kernel = size_per_buffer;

        int max_wg_size __attribute__((unused)) =
            queue.get_device().get_info<sycl::info::device::max_work_group_size>(); // 1024
        const int wg_size = 32;
        // assert(wg_size <= max_wg_size); TODO

        const int subgroup_size __attribute__((unused)) = 32;

        int num_vecs = 1;
        if (size >= 262144)
            num_vecs = 4;
        else if (size > 32768)
            num_vecs = 2;

        uint32_t v = vec_size * num_vecs;
        if (size <= 2048)
            v = 1;

        assert(wg_size >= subgroup_size);

        uint32_t total_threads_needed = size > v ? (size + v - 1) / v : 1;
        uint32_t total_threads_dispatched =
            (total_threads_needed + wg_size - 1) / wg_size * wg_size;
        uint32_t total_wg_count = total_threads_dispatched / wg_size;

        for (r = 0; r < repetition; r++) {
            int buffer_index_kernel = buffer_index;
            buffer_index++;
            buffer_index %= TRIPLE_BUFFER;

            if (v == 1) {
                e = queue.submit([&](sycl::handler &cgh) {
                cgh.parallel_for<class oneccl_allreduce_small_kernel_scalar<data_type>>(
                    sycl::nd_range<1>( total_threads_dispatched, wg_size), [=](sycl::nd_item<1> idx2) {
                    uint32_t idx = idx2.get_global_id();
                    //uint32_t idx = idx2.get_linear_id();

                    uint32_t offset __attribute__((unused)) = idx;

                    //to do:
                    //O3 compiler optimization: not much difference after the change.
                    //tune the fence: good perf improvements
                    //tune the cacheability for each IO message: no noticeable improvement
                    //tune the thread size: not much improvements
                    //tune the polling freq

                    if (idx < total_threads_needed) {
                        //do copy from input buffer to temp buffer.
                        data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                        local_temp_ptr +=
                            (buffer_index_kernel * size_per_buffer_kernel /
                             sizeof(
                                 data_type)); //point to the correct buffer inside the triple buffer
                        data_type *dest_ptr = local_temp_ptr + offset;
                        data_type *src_ptr = (data_type *)in_buffer + offset;
                        *(dest_ptr) = *(src_ptr);
                        //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank
                    }

                    int *local_sync_ptr;
                    local_sync_ptr = (int *)temp_sync_buffer
                        [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                    //if there are more than 1 threads required per rank, then do the local sync within the rank first.
                    if (total_threads_needed > 1) {
                        uint32_t local_tid = idx2.get_local_linear_id();
                        if (local_tid == 0) {
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(local_sync_ptr[0]);
                            atomic_p += 1;

                            uint32_t val = atomic_p.load();
                            while (val != total_wg_count) {
                                val = atomic_p.load();
                            }
                        }
                        idx2.barrier();
                    }

                    //once the local sync is done, retire useless threads
                    if (idx >= total_threads_needed)
                        return;

                    //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                    if (idx ==
                        0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        uint32_t status0 = total_threads_needed;
                        for (uint32_t i = 0; i < temp_world; i++) {
                            int *sync_ptr;

                            sync_ptr = (int *)temp_sync_buffer
                                [i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr +=
                                (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(sync_ptr[1]);
                            atomic_p += status0;
                        }
                    }

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    sycl::atomic_ref<int,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_p(local_sync_ptr[1]);
                    uint32_t val = atomic_p.load();
                    while (val != total_threads_needed * temp_world) {
                        val = atomic_p.load();
                    }

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx ==
                        0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        int buffer_index_to_reset =
                            (buffer_index_kernel + TRIPLE_BUFFER - 1) % TRIPLE_BUFFER;
                        local_sync_ptr = (int *)temp_sync_buffer
                            [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr +=
                            (buffer_index_to_reset * size_per_buffer_kernel / sizeof(int));
                        /*
                            sycl::atomic_ref<int, sycl::memory_order::seq_cst,
                                sycl::memory_scope::system,
                                sycl::access::address_space::global_space>
                                atomic_p(local_sync_ptr[0]);
			    */
                        local_sync_ptr[0] = local_sync_ptr[1] = 0;
                    }

                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    //do All reduce
                    //data_type result;
                    data_type result;
                    for (int i = 0; i < kernel_inner_loop; i++) {
                        if (temp_world == 4) {
                            data_type *peer_ptr0 =
                                (data_type *)(((int *)temp_buffer[0]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr1 =
                                (data_type *)(((int *)temp_buffer[1]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr2 =
                                (data_type *)(((int *)temp_buffer[2]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr3 =
                                (data_type *)(((int *)temp_buffer[3]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            //do the actual reduction
                            result = *peer_ptr0;
                            result += *peer_ptr1;
                            result += *peer_ptr2;
                            result += *peer_ptr3;
                        }
                        else if (temp_world == 8) {
                            data_type *peer_ptr0 =
                                (data_type *)(((int *)temp_buffer[0]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr1 =
                                (data_type *)(((int *)temp_buffer[1]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr2 =
                                (data_type *)(((int *)temp_buffer[2]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr3 =
                                (data_type *)(((int *)temp_buffer[3]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr4 =
                                (data_type *)(((int *)temp_buffer[4]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr5 =
                                (data_type *)(((int *)temp_buffer[5]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr6 =
                                (data_type *)(((int *)temp_buffer[6]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            data_type *peer_ptr7 =
                                (data_type *)(((int *)temp_buffer[7]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int))) +
                                offset;
                            //do the actual reduction
                            result = *peer_ptr0;
                            result += *peer_ptr1;
                            result += *peer_ptr2;
                            result += *peer_ptr3;
                            result += *peer_ptr4;
                            result += *peer_ptr5;
                            result += *peer_ptr6;
                            result += *peer_ptr7;
                        }
                        else //this is for 2,4,6 ranks. So there is no problem of overflowing the buffer.
                        {
                            data_type *peer_ptr[MAX_RANK];
                            //do the actual reduction
                            result = 0;
#pragma unroll
                            for (uint32_t r = 0; r < temp_world; r++) {
                                peer_ptr[r] = (data_type *)(((int *)temp_buffer[r]) +
                                                            (buffer_index_kernel *
                                                             size_per_buffer_kernel / sizeof(int)));
                                peer_ptr[r] += offset;
                                result += *(peer_ptr[r]);
                            }
                        }

                        //write out the results
                        *((data_type *)out_buffer + offset) = result;
                    } // end of for loop

                    });
                });
                //e.wait();
            }
            else {
                // block read kernel
                e = queue.submit([&](sycl::handler &cgh) {
                    //                cgh.depends_on(memcpy_event);
                cgh.parallel_for<class oneccl_allreduce_small_kernel_block<data_type>>(
                    sycl::nd_range<1>( total_threads_dispatched, wg_size), [=](sycl::nd_item<1> idx2) [[sycl::reqd_sub_group_size(subgroup_size)]] {
                    //                    sycl::nd_range<1>( total_threads_dispatched, wg_size), [=](sycl::nd_item<1> idx2) {

                    uint32_t idx = idx2.get_global_id();
                    //uint32_t idx = idx2.get_linear_id();
                    sycl::sub_group sg = idx2.get_sub_group();
                    const size_t sgSize __attribute__((unused)) = sg.get_local_range()[0];

                    //using global_ptr = sycl::multi_ptr<data_type, sycl::access::address_space::global_space, sycl::access::decorated::yes>;
                    using global_ptr =
                        sycl::multi_ptr<data_type, sycl::access::address_space::global_space>;
                    uint32_t offset __attribute__((unused)) = idx;
                    int base = (offset / sgSize) * sgSize * vec_size * num_vecs;
                    //int base = (idx2.get_group(0) * wg_size + sg.get_group_id()[0] * sgSize * subgroup_size;
                    int use_block_rw = 1;
                    uint32_t nelem = sgSize * vec_size * num_vecs;
                    if (size - sg.get_group_id()[0] * sgSize * v < sgSize * v) {
                        use_block_rw = 0;
                        //nelem = size - sg.get_group_id()[0] * sgSize * vec_size;
                        nelem = size - idx * v;
                        if (nelem > v)
                            nelem = v;
                        else if (nelem < 0)
                            nelem = 0;
                    }

                    //to do:
                    //O3 compiler optimization: not much difference after the change.
                    //tune the fence: good perf improvements
                    //tune the cacheability for each IO message: no noticeable improvement
                    //tune the thread size: not much improvements
                    //tune the polling freq

                    if (idx < total_threads_needed) {
                        //do copy from input buffer to temp buffer.
                        data_type *local_temp_ptr = (data_type *)temp_buffer[temp_rank];
                        local_temp_ptr +=
                            (buffer_index_kernel * size_per_buffer_kernel /
                             sizeof(
                                 data_type)); //point to the correct buffer inside the triple buffer
                        if (use_block_rw) {
                            int b = base;
                            sycl::vec<data_type, vec_size> val;
                            for (int m = 0; m < num_vecs; m++) {
                                val = sg.load<vec_size>(global_ptr(&(((data_type *)in_buffer)[b])));
                                sg.store<vec_size>(global_ptr(local_temp_ptr + b), val);
                                b += sgSize * vec_size;
                            }
                        }
                        else {
                            data_type *dest_ptr = local_temp_ptr + offset * vec_size;
                            data_type *src_ptr = (data_type *)in_buffer + offset * vec_size;
#pragma unroll
                            for (uint32_t n = 0; n < nelem; n++) {
                                *(dest_ptr + n) = *(src_ptr + n);
                            }
                        }
                        //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank
                    }

                    int *local_sync_ptr;
                    local_sync_ptr = (int *)temp_sync_buffer
                        [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                    //if there are more than 1 threads required per rank, then do the local sync within the rank first.
                    if (total_threads_needed > 1) {
                        uint32_t local_tid = idx2.get_local_linear_id();
                        //sync locally within local GPU first.
                        //sycl::_V1::ext::oneapi::experimental::printf("HERE in: rank%d %d\n", temp_rank, local_tid);
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
                        idx2.barrier();
                    }

                    //once the local sync is done, retire useless threads
                    if (idx >= total_threads_needed)
                        return;

                    //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                    if (idx ==
                        0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        uint32_t status0 = total_threads_needed;
                        for (uint32_t i = 0; i < temp_world; i++) {
                            int *sync_ptr;

                            sync_ptr = (int *)temp_sync_buffer
                                [i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr +=
                                (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            sycl::atomic_ref<int,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_p(sync_ptr[1]);
                            atomic_p += status0;
                        }
                    }

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    sycl::atomic_ref<int,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_p(local_sync_ptr[1]);
                    uint32_t val = atomic_p.load();
                    while (val != total_threads_needed * temp_world) {
                        val = atomic_p.load();
                    }

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx ==
                        0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        int buffer_index_to_reset =
                            (buffer_index_kernel + TRIPLE_BUFFER - 1) % TRIPLE_BUFFER;
                        local_sync_ptr = (int *)temp_sync_buffer
                            [temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr +=
                            (buffer_index_to_reset * size_per_buffer_kernel / sizeof(int));

                        local_sync_ptr[0] = local_sync_ptr[1] = 0;
                    }

                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    //do All reduce
                    for (int i = 0; i < kernel_inner_loop; i++) {
                        if (temp_world == 4) {
                            data_type *peer_ptr0 =
                                (data_type *)(((int *)temp_buffer[0]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr1 =
                                (data_type *)(((int *)temp_buffer[1]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr2 =
                                (data_type *)(((int *)temp_buffer[2]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr3 =
                                (data_type *)(((int *)temp_buffer[3]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            if (use_block_rw) {
                                int b = base;
                                sycl::vec<data_type, vec_size> result;
                                for (int m = 0; m < num_vecs; m++) {
                                    result = sg.load<vec_size>(global_ptr(peer_ptr0 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr1 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr2 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr3 + b));
                                    sg.store<vec_size>(global_ptr((data_type *)out_buffer + b),
                                                       result);
                                    b += sgSize * vec_size;
                                }
                            }
                            else {
                                peer_ptr0 += offset * vec_size * num_vecs;
                                peer_ptr1 += offset * vec_size * num_vecs;
                                peer_ptr2 += offset * vec_size * num_vecs;
                                peer_ptr3 += offset * vec_size * num_vecs;
                                data_type *dest =
                                    (data_type *)out_buffer + offset * vec_size * num_vecs;
                                data_type res;
#pragma unroll
                                for (uint32_t n = 0; n < nelem; n++) {
                                    res = *(peer_ptr0++);
                                    res += *(peer_ptr1++);
                                    res += *(peer_ptr2++);
                                    res += *(peer_ptr3++);
                                    *dest = res;
                                    dest++;
                                }
                            }
                        }
                        else if (temp_world == 8) {
                            data_type *peer_ptr0 =
                                (data_type *)(((int *)temp_buffer[0]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr1 =
                                (data_type *)(((int *)temp_buffer[1]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr2 =
                                (data_type *)(((int *)temp_buffer[2]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr3 =
                                (data_type *)(((int *)temp_buffer[3]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr4 =
                                (data_type *)(((int *)temp_buffer[4]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr5 =
                                (data_type *)(((int *)temp_buffer[5]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr6 =
                                (data_type *)(((int *)temp_buffer[6]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            data_type *peer_ptr7 =
                                (data_type *)(((int *)temp_buffer[7]) +
                                              (buffer_index_kernel * size_per_buffer_kernel /
                                               sizeof(int)));
                            if (use_block_rw) {
                                int b = base;
                                sycl::vec<data_type, vec_size> result;
                                for (int m = 0; m < num_vecs; m++) {
                                    result = sg.load<vec_size>(global_ptr(peer_ptr0 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr1 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr2 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr3 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr4 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr5 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr6 + b));
                                    result += sg.load<vec_size>(global_ptr(peer_ptr7 + b));
                                    sg.store<vec_size>(global_ptr((data_type *)out_buffer + b),
                                                       result);
                                    b += sgSize * vec_size;
                                }
                            }
                            else {
                                peer_ptr0 += offset * vec_size * num_vecs;
                                peer_ptr1 += offset * vec_size * num_vecs;
                                peer_ptr2 += offset * vec_size * num_vecs;
                                peer_ptr3 += offset * vec_size * num_vecs;
                                peer_ptr4 += offset * vec_size * num_vecs;
                                peer_ptr5 += offset * vec_size * num_vecs;
                                peer_ptr6 += offset * vec_size * num_vecs;
                                peer_ptr7 += offset * vec_size * num_vecs;
                                data_type *dest =
                                    (data_type *)out_buffer + offset * vec_size * num_vecs;
                                data_type res;
#pragma unroll
                                for (uint32_t n = 0; n < nelem; n++) {
                                    res = *(peer_ptr0++);
                                    res += *(peer_ptr1++);
                                    res += *(peer_ptr2++);
                                    res += *(peer_ptr3++);
                                    res += *(peer_ptr4++);
                                    res += *(peer_ptr5++);
                                    res += *(peer_ptr6++);
                                    res += *(peer_ptr7++);
                                    *dest = res;
                                    dest++;
                                }
                            }
                        }
                        else //this is for 2,4,6 ranks. So there is no problem of overflowing the buffer.
                        {
                            data_type *peer_ptr[MAX_RANK];
#pragma unroll
                            for (uint32_t r = 0; r < temp_world; r++) {
                                peer_ptr[r] = (data_type *)(((int *)temp_buffer[r]) +
                                                            (buffer_index_kernel *
                                                             size_per_buffer_kernel / sizeof(int)));
                            }
                            if (use_block_rw) {
                                int b = base;
                                for (int m = 0; m < num_vecs; m++) {
                                    sycl::vec<data_type, vec_size> result(0);
                                        //sycl::vec<data_type, vec_size> result = sg.load<vec_size>(global_ptr(peer_ptr[0] + b));
#pragma unroll
                                    for (uint32_t r = 1; r < temp_world; r++) {
                                        result += sg.load<vec_size>(global_ptr(peer_ptr[r] + b));
                                    }
                                    sg.store<vec_size>(global_ptr((data_type *)out_buffer + b),
                                                       result);
                                    b += sgSize * vec_size;
                                }
                            }
                            else {
#pragma unroll
                                for (uint32_t r = 0; r < temp_world; r++) {
                                    peer_ptr[r] += offset * vec_size * num_vecs;
                                }
                                data_type *dest =
                                    (data_type *)out_buffer + offset * vec_size * num_vecs;
                                for (uint32_t n = 0; n < nelem; n++) {
                                    data_type res = 0;
#pragma unroll
                                    for (uint32_t r = 0; r < temp_world; r++) {
                                        res += *(peer_ptr[r]++);
                                    }
                                    *dest = res;
                                    dest++;
                                }
                            }
                        }

                    } // end of for loop

                    });
                });
            } // kernel

        } // for (r = 0; r < repetition; r++)

        return ccl::event::create_from_native(e);
    }
    void release(sycl::queue &queue) {
        // Clean up, close/put ipc handles, free memory, etc.
        auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_context());
        for (uint32_t i = 0; i < world; i++) {
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
    int rank, world;
    int buffer_index;
    int size_per_buffer;
    int data_size_per_buffer;
};

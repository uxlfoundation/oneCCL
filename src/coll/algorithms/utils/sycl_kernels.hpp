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

namespace ccl::v1 {
struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t &operator()(const Object &obj) const {
        return obj.get_impl();
    }
};
}; // namespace ccl::v1

struct sycl_ptrs_type {
    void *mdfi_ptr_rd{ nullptr }, *mdfi_ptr_wr{ nullptr };
    std::array<void *, MAX_GPUS> xelink_ptrs_rd, xelink_ptrs_wr;
    std::array<void *, MAX_NODE_RANKS> node_ptrs_rd, node_ptrs_wr;
};

template <typename T,
          sycl::access::address_space Space = sycl::access::address_space::global_space,
          sycl::access::decorated IsDecorated = sycl::access::decorated::yes>
sycl::multi_ptr<T, sycl::access::address_space::global_space, sycl::access::decorated::yes>
get_multi_ptr(T *ptr) {
    return sycl::address_space_cast<Space, IsDecorated>(ptr);
}

template <typename T, int N, int vec_size>
void inline copy_data(std::array<void *, MAX_GPUS> dst,
                      std::array<void *, MAX_GPUS> src,
                      const size_t count,
                      const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    const size_t packed_count = count / vec_size;

    sycl::sub_group sg = it.get_sub_group();
    const size_t sgSize = sg.get_local_range()[0];

    int base = (idx / sgSize) * sgSize * vec_size;
    const long rem_elem_count = count - base;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
#pragma unroll
        for (int i = 0; i < N; i++) {
            ((AT *)dst[i])[idx] = ((AT *)src[i])[idx];
        }
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
#pragma unroll
            for (int i = 0; i < N; i++) {
                ((T *)dst[i])[new_idx] = ((T *)src[i])[new_idx];
            }
        }
    }
}

// local barrier within gpu similar to q.ext_oneapi_submit_barrier()
inline void kernel_barrier(size_t *sync_ptr, const sycl::nd_item<1> it) {
    sycl::sub_group sg = it.get_sub_group();
    const size_t sidx = sg.get_local_id();
    if (sidx == 0) {
        // number of subgroups = global_size / sg_size
        const size_t num_sg = it.get_global_range()[0] / sg.get_local_range()[0];
        sycl::atomic_ref<size_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_p(*sync_ptr);
        atomic_p += 1;

        size_t val = atomic_p.load();
        while (val < num_sg) {
            val = atomic_p.load();
        }
    }
}

// communication barrier across ranks (gpus)
inline void comm_barrier(ccl_comm_barrier_data barrier_data,
                         const sycl::nd_item<1> it,
                         const bool use_gpu = true,
                         const bool gpu_counter_increase = false) {
    if (!use_gpu)
        return;

    const size_t idx = it.get_global_linear_id();
    sycl::sub_group sg = it.get_sub_group();
    const size_t sidx = sg.get_local_id();

    const int comm_rank = barrier_data.rank();
    const int comm_size = barrier_data.size();

    if (gpu_counter_increase) {
        if (idx == 0) {
            barrier_data.inc_gpu(1);
        }
        // this works because there is only a single workgroup
        // the barrier does not work between workgroups
        // and multiple workgroups yield incorrect results
        it.barrier();
    }

    const size_t barrier_count =
        gpu_counter_increase ? barrier_data.count_gpu() : barrier_data.count();
    const int buffer_idx = gpu_counter_increase ? barrier_data.slot_gpu() : barrier_data.slot();
    std::array<size_t *, MAX_NODE_RANKS> sync_remote_ptrs =
        gpu_counter_increase ? barrier_data.remote_ptrs_gpu() : barrier_data.remote_ptrs();

    // increment count in all remote ranks
    if (idx < (size_t)comm_size) {
        const size_t i = idx;
        sycl::atomic_ref<size_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_p(sync_remote_ptrs[i][buffer_idx]);
        atomic_p += 1;
    }

    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

    // wait for all remote ranks to update the local count
    if (sidx == 0) {
        sycl::atomic_ref<size_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_p(sync_remote_ptrs[comm_rank][buffer_idx]);

        size_t val = atomic_p.load();
        size_t counter_full = barrier_count * comm_size;
        while (val < counter_full) {
            val = atomic_p.load();
        }
    }
}

template <typename T>
inline void reduce_average_kernel(void *buf, const size_t n, size_t idx) {
    ((T *)buf)[idx] /= n;
}

template <typename T, int vec_size>
inline void reduce_average(void *reduce_buf,
                           const size_t count,
                           const size_t average_divisor,
                           const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    const size_t packed_count = count / vec_size;
    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        reduce_average_kernel<AT>(reduce_buf, average_divisor, idx);
    }
    else {
        const size_t new_idx = vec_size * packed_count + idx - packed_count;
        if (new_idx < count) {
            reduce_average_kernel<T>(reduce_buf, average_divisor, new_idx);
        }
    }
}

template <typename T, int vec_size, int SGS>
inline sycl::event reduce_average_invoke(sycl::queue &q,
                                         void *reduce_buf,
                                         const size_t reduce_count,
                                         const size_t average_divisor,
                                         std::vector<sycl::event> &dep_events) {
    constexpr int wg_size = SGS;
    constexpr int sg_size = SGS;
    int kernel_threads = reduce_count / vec_size + reduce_count % vec_size;
    int kernel_size = (kernel_threads + wg_size - 1) / wg_size * wg_size;
    sycl::event e = q.submit([=](sycl::handler &h) {
        h.depends_on(dep_events);
        h.parallel_for(sycl::nd_range<1>(kernel_size, wg_size), [=](sycl::nd_item<1> it) {
            reduce_average<T, vec_size>(reduce_buf, reduce_count, average_divisor, it);
        });
    });
    return e;
}

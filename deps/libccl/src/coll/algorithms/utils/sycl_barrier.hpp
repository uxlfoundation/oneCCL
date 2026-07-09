/*
 Copyright 2016-2026 Intel Corporation

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
// Kernel-side barrier and flag primitives used across SYCL collectives.

#include <array>
#include <sycl/sycl.hpp>

#include "coll/algorithms/utils/comm_barrier_data.hpp"
#include "coll/algorithms/utils/consts.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"

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

inline void p2p_barrier(ccl_comm_flag_data flag_data,
                        const sycl::nd_item<1> it,
                        const bool use_subgroups = false,
                        const bool use_root_sync = false) {
    const size_t idx = it.get_global_linear_id();
    sycl::sub_group sg = it.get_sub_group();
    const size_t sidx = sg.get_local_id();

    const int comm_rank = flag_data.rank();
    const int comm_size = flag_data.size();
    const int dest_rank = (comm_rank + 1) % comm_size;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    __LscFlushCache();
#endif

    if (idx == 0) {
        flag_data.inc(1);
    }

    if (use_root_sync) {
        sycl::group_barrier(it.ext_oneapi_get_root_group());
    }
    else {
        sycl::group_barrier(it.get_group());
    }

    size_t flag_count = flag_data.count();
    const int buffer_idx = flag_data.slot();
    std::array<size_t *, MAX_NODE_RANKS> sync_remote_ptrs = flag_data.remote_ptrs();

    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

    // write flag to remote gpu memory
    if (idx == 0) {
        // TODO: should every thread writing do release fence
        //sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

#if defined(CCL_SYCL_ENABLE_ARCB) && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        size_t *dst = (size_t *)&(sync_remote_ptrs[dest_rank][buffer_idx]);
        //__LscStoreUnCached(dst, flag_count);
        lscStore<16, CacheCtrl::L1UC_L3UC>(dst, flag_count);

        // assemby seems to be slower
        /*
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        size_t* addr = (size_t*)&(sync_remote_ptrs[comm_rank][buffer_idx]);
        size_t val = 0;
        while(val < flag_count) {
            __LscLoadUnCached(val, addr);
        }
        */
#else
        sync_remote_ptrs[dest_rank][buffer_idx] = flag_count;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        __LscFlushCache();
#endif
#endif
    }

    // read flag from local gpu memory
    const size_t use_idx = use_subgroups == 1 ? sidx : idx;
    if (use_idx == 0) {
#if 1
        size_t *addr = (size_t *)&(sync_remote_ptrs[comm_rank][buffer_idx]);
        size_t val = 0;
        while (val < flag_count) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscLoad<16, CacheCtrl::L1UC_L3UC>(val, addr);
#endif
        }
#else
        sycl::atomic_ref<size_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_p(sync_remote_ptrs[comm_rank][buffer_idx]);

        size_t val = atomic_p.load();
        while (val < flag_count) {
            val = atomic_p.load();
        }
#endif
    }

    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);

    if (!use_subgroups) {
        if (use_root_sync) {
            sycl::group_barrier(it.ext_oneapi_get_root_group());
        }
        else {
            sycl::group_barrier(it.get_group());
        }
    }
}

// communication barrier for PCIe-based system when remote atomics
// does not work. This is used for BMG with vISA instruction set
template <bool use_root_sync = false>
inline void comm_barrier_visa(ccl_comm_barrier_data barrier_data,
                              const sycl::nd_item<1> it,
                              const bool gpu_counter_increase = false) {
    const size_t idx = it.get_global_linear_id();
    sycl::sub_group sg = it.get_sub_group();
    const size_t sidx = sg.get_local_id();

    const int comm_rank = barrier_data.rank();
    const int comm_size = barrier_data.size();

    if (gpu_counter_increase) {
        // flush cache before the sycl barrier
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        __LscFlushCache();
#endif
        if (idx == 0) {
            barrier_data.inc_gpu(1);
        }
        if constexpr (use_root_sync) {
            // similar to oneCCL kernel_barrier
            sycl::group_barrier(it.ext_oneapi_get_root_group());
        }
        // this works because there is only a single workgroup
        // the barrier does not work between workgroups
        // and multiple workgroups yield incorrect results
        else {
            sycl::group_barrier(it.get_group());
        }
    }

    const size_t barrier_count =
        gpu_counter_increase ? barrier_data.count_gpu() : barrier_data.count();
    const int buffer_idx = gpu_counter_increase ? barrier_data.slot_gpu() : barrier_data.slot();
    std::array<size_t *, MAX_NODE_RANKS> sync_remote_ptrs =
        gpu_counter_increase ? barrier_data.remote_ptrs_gpu() : barrier_data.remote_ptrs();

    // all threads do release fence
    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
    // flush data to make sure all data goes out
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    if (!gpu_counter_increase) {
        __LscFlushCache();
    }
#endif

    // increment count in all remote ranks
    // write flag without caching
    if (idx < (size_t)comm_size) {
        const size_t i = idx;

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
        size_t *dst = (size_t *)&(sync_remote_ptrs[i][buffer_idx * comm_size + comm_rank]);
        lscStore<16, CacheCtrl::L1UC_L3UC>(dst, barrier_count);
#else
        sync_remote_ptrs[i][buffer_idx * comm_size + comm_rank] = barrier_count;
#endif
    }

#if 0 && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    __LscFlushCache();
#endif

    if (sidx < (size_t)comm_size) {
#if 1
        size_t *addr = (size_t *)&(sync_remote_ptrs[comm_rank][buffer_idx * comm_size + sidx]);
        size_t val = 0;
        while (val < barrier_count) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            lscLoad<16, CacheCtrl::L1UC_L3UC>(val, addr);
#endif
        }
#else
        // local atmoic read
        sycl::atomic_ref<size_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_p(sync_remote_ptrs[comm_rank][buffer_idx * comm_size + sidx]);

        size_t val = atomic_p.load();
        while (val < barrier_count) {
            val = atomic_p.load();
        }
#endif
    }

    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
}

// communication barrier across ranks (gpus)
template <bool use_root_sync = false>
inline void comm_barrier(ccl_comm_barrier_data barrier_data,
                         const sycl::nd_item<1> it,
                         const bool use_gpu = true,
                         const bool gpu_counter_increase = false) {
    if (!use_gpu)
        return;

    if (!barrier_data.use_remote_atomics()) {
        comm_barrier_visa<use_root_sync>(barrier_data, it, gpu_counter_increase);
        return;
    }

    const size_t idx = it.get_global_linear_id();
    sycl::sub_group sg = it.get_sub_group();
    const size_t sidx = sg.get_local_id();

    const int comm_rank = barrier_data.rank();
    const int comm_size = barrier_data.size();

    if (gpu_counter_increase) {
        if (idx == 0) {
            barrier_data.inc_gpu(1);
        }
        if constexpr (use_root_sync) {
            sycl::group_barrier(it.ext_oneapi_get_root_group());
        }
        // this works because there is only a single workgroup
        // the barrier does not work between workgroups
        // and multiple workgroups yield incorrect results
        else {
            sycl::group_barrier(it.get_group());
        }
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

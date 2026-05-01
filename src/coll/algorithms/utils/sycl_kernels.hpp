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

#include "common/global/global.hpp"
// for ccl_kernel_barrier_data and ccl_comm_barrier_data definitions
#include "coll/algorithms/utils/consts.hpp"
// reduction types
#include "coll/reduction/reduction.hpp"
#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"

#define __LscFlushCache() \
    { __asm__ __volatile__("lsc_fence.ugm.evict.gpu"); }

struct sycl_ptrs_type {
    void *mdfi_ptr_rd{ nullptr }, *mdfi_ptr_wr{ nullptr };
    std::array<void *, MAX_GPUS> xelink_ptrs_rd, xelink_ptrs_wr;
    std::array<void *, MAX_NODE_RANKS> node_ptrs_rd, node_ptrs_wr;
};

inline bool use_recording_path(const sycl::queue &q) {
    return ccl::global_data::env().sycl_force_recording_path ||
           q.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording;
}

inline bool use_recording_path(const ccl_stream *stream) {
    if (stream) {
        return use_recording_path(stream->get_native_stream());
    }
    if (ccl::global_data::env().sycl_force_recording_path) {
        // LOG_WARN("trying to force recording on null stream; falling back to non-recording");
        CCL_THROW("recording null stream"); // TODO WARN or THROW ?
    }
    return false;
}

class ccl_kernel_barrier_data {
public:
    static constexpr int slots = 3;

    ccl_kernel_barrier_data() = default;
    ccl_kernel_barrier_data(const ccl_kernel_barrier_data &) = default;
    ccl_kernel_barrier_data &operator=(const ccl_kernel_barrier_data &) = default;

    void set_sync_ptrs(size_t *ptrs) {
        m_sync_ptrs = ptrs;
    }

    // advance to the next slot
    ccl_kernel_barrier_data inc_slot(int count = 1) {
        m_count += count;
        return *this;
    }

    // get current slot pointer
    size_t *get_sync_ptr() const {
        size_t curr_slot = m_count % slots;
        return m_sync_ptrs + curr_slot;
    }

    // reset data in the farthest slot
    void reset_sync_data() {
        size_t reset_slot = (m_count + (slots - 1)) % slots;
        m_sync_ptrs[reset_slot] = 0;
    }

private:
    size_t *m_sync_ptrs;
    size_t m_count = 0;
};

class alignas(CACHELINE_SIZE) ccl_comm_barrier_data {
private:
    int m_rank;
    int m_size;
    size_t m_count = slots - 1;
    size_t *d_count = nullptr;
    bool m_is_set = false;
    bool m_use_remote_atomics;
    std::array<size_t *, MAX_NODE_RANKS> m_remote_ptrs{};
    std::array<size_t *, MAX_NODE_RANKS> d_remote_ptrs{};

public:
    static constexpr int slots = 3;

    // setting use_remote_atomics is postponed in coll_init
    // until topo manager is created.
    ccl_comm_barrier_data(int rank, int size)
            : m_rank(rank),
              m_size(size),
              m_use_remote_atomics(false) {}

    int rank() const {
        return m_rank;
    }

    int size() const {
        return m_size;
    }

    bool is_set() const {
        return m_is_set;
    }

    bool use_remote_atomics() const {
        return m_use_remote_atomics;
    }

    void set_remote_atomics(bool use_remote_atomics) {
        m_use_remote_atomics = use_remote_atomics;
    }

    size_t inc(size_t n) {
        m_count += n;
        return m_count;
    }

    size_t count() const {
        return m_count / slots;
    }

    int slot() const {
        return m_count % slots;
    }

    std::array<size_t *, MAX_NODE_RANKS> remote_ptrs() const {
        return m_remote_ptrs;
    }

    size_t inc_gpu(size_t n) {
        *d_count = *d_count + n;
        return *d_count;
    }

    size_t count_gpu() const {
        return *d_count / slots;
    }

    int slot_gpu() const {
        return *d_count % slots;
    }

    std::array<size_t *, MAX_NODE_RANKS> remote_ptrs_gpu() const {
        return d_remote_ptrs;
    }

    void set_count_gpu(size_t *count) {
        d_count = count;
    }

    void set_remote_ptrs(std::array<size_t *, MAX_NODE_RANKS> ptrs,
                         std::array<size_t *, MAX_NODE_RANKS> gpu_barrier_ptrs) {
        assert(!is_set());
        m_is_set = true;
        m_remote_ptrs = ptrs;
        d_remote_ptrs = gpu_barrier_ptrs;
    }
};

class alignas(CACHELINE_SIZE) ccl_comm_flag_data {
private:
    int m_rank;
    int m_size;
    size_t *d_count = nullptr;
    std::array<size_t *, MAX_NODE_RANKS> d_remote_ptrs{};

public:
    static constexpr int slots = 4;

    ccl_comm_flag_data(int rank, int size) : m_rank(rank), m_size(size) {}

    int rank() const {
        return m_rank;
    }

    int size() const {
        return m_size;
    }

    size_t inc(size_t n) {
        *d_count = *d_count + n;
        return *d_count;
    }

    size_t count() const {
        return *d_count / slots;
    }

    int slot() const {
        return *d_count % slots;
    }

    std::array<size_t *, MAX_NODE_RANKS> remote_ptrs() const {
        return d_remote_ptrs;
    }

    void set_count(size_t *count) {
        d_count = count;
    }

    void set_remote_ptrs(std::array<size_t *, MAX_NODE_RANKS> ptrs) {
        d_remote_ptrs = ptrs;
    }
};

/* COPY KERNELS */

template <typename T, int N, int vec_size>
inline void copy_data(std::array<void *, MAX_GPUS> dst,
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

template <typename T, int vec_size>
inline void copy_and_modify_data(std::array<void *, MAX_GPUS> dst,
                                 std::array<void *, MAX_GPUS> src,
                                 const size_t comm_size,
                                 const size_t count,
                                 const ccl_reduction_data reduction,
                                 const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        for (int i = 0; i < comm_size; i++) {
            ((AT *)dst[i])[idx] = apply_pre_operation<AT>(reduction, ((AT *)src[i])[idx]);
        }
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            for (int i = 0; i < comm_size; i++) {
                ((T *)dst[i])[new_idx] = apply_pre_operation<T>(reduction, ((T *)src[i])[new_idx]);
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
        char *dst = (char *)&(sync_remote_ptrs[dest_rank][buffer_idx]);
        //__LscStoreUnCached(dst, flag_count);
        lscStore<16, CacheCtrl::L1UC_L3UC>(dst, flag_count);

        // assemby seems to be slower
        /*
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        char* addr = (char*)&(sync_remote_ptrs[comm_rank][buffer_idx]);
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
        sycl::atomic_ref<size_t,
                         sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            atomic_p(sync_remote_ptrs[comm_rank][buffer_idx]);

        size_t val = atomic_p.load();
        while (val < flag_count) {
            val = atomic_p.load();
        }
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
        char *dst = (char *)&(sync_remote_ptrs[i][buffer_idx * comm_size + comm_rank]);
        lscStore<16, CacheCtrl::L1UC_L3UC>(dst, barrier_count);
#else
        sync_remote_ptrs[i][buffer_idx * comm_size + comm_rank] = barrier_count;
#endif
    }

#if 0 && defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
    __LscFlushCache();
#endif

    if (sidx < (size_t)comm_size) {
#if 0
        // slower with assembly
        char* addr = (char*)&(sync_remote_ptrs[comm_rank][buffer_idx * comm_size + sidx]);
        size_t val = 0;
        while (val < barrier_count) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
            __LscLoadUnCached(val, addr);
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

// Kernel name templates for kernel_memcpy
template <typename Type>
class oneccl_kernel_memcpy_typed {};

class oneccl_kernel_memcpy_bytes {};

static inline sycl::event kernel_memcpy(sycl::queue &q,
                                        const void *send_buf,
                                        void *recv_buf,
                                        int *send_buf_idx_ptr,
                                        int *recv_buf_idx_ptr,
                                        size_t count,
                                        size_t dsize,
                                        size_t tmp_buf_size,
                                        const std::vector<sycl::event> &dep_events) {
    constexpr size_t wg_size = 32;
    constexpr size_t sg_size = 32;
    const size_t bytes = dsize * count;
    bool upsize = ccl::global_data::env().sycl_kernel_memcpy_upsize;
    if (upsize) {
        auto ptr_to_datasize = [](const void *ptr) {
            const size_t mod8 = reinterpret_cast<uintptr_t>(ptr) % 8;
            switch (mod8) {
                case 0:
                    // uint64_t
                    return 8;
                case 4:
                    // uint32_t
                    return 4;
                case 2:
                case 6:
                    // uint16_t
                    return 2;
                default:
                    // uint8_t
                    return 1;
            }
        };
        auto kernel_memcpy_captured_typed = [&q,
                                             &dep_events,
                                             recv_buf,
                                             send_buf_idx_ptr,
                                             recv_buf_idx_ptr,
                                             tmp_buf_size]<typename Type>(const Type *send_buf,
                                                                          size_t bytes) {
            const size_t items = bytes / sizeof(Type);
            const size_t nof_workgroups = items / wg_size + 1;
            const size_t loop_size = nof_workgroups * wg_size;
            return q.submit([=](sycl::handler &h) {
                h.depends_on(dep_events);
                h.parallel_for<oneccl_kernel_memcpy_typed<Type>>(
                    sycl::nd_range<1>(loop_size, wg_size),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                        size_t idx = it.get_global_linear_id();
                        if (idx < items) {
                            size_t offset_recv_items = (recv_buf_idx_ptr ? *recv_buf_idx_ptr : 0) *
                                                       tmp_buf_size / sizeof(Type);
                            size_t offset_send_items = (send_buf_idx_ptr ? *send_buf_idx_ptr : 0) *
                                                       tmp_buf_size / sizeof(Type);
                            Type *recv_buf_offset =
                                static_cast<Type *>(recv_buf) + offset_recv_items;
                            const Type *send_buf_offset =
                                static_cast<const Type *>(send_buf) + offset_send_items;
                            *(recv_buf_offset + idx) = *(send_buf_offset + idx);
                        }
                    });
            });
        };
        auto kernel_memcpy_captured =
            [ptr_to_datasize, &kernel_memcpy_captured_typed, tmp_buf_size](
                const void *send_buf, void *recv_buf, size_t bytes) {
                const void *send_ptr_end = (static_cast<const uint8_t *>(send_buf) + bytes);
                size_t start_send_size = ptr_to_datasize(send_buf);
                size_t end_send_size = ptr_to_datasize(send_ptr_end);
                const void *recv_ptr_end = (static_cast<const uint8_t *>(recv_buf) + bytes);
                size_t start_recv_size = ptr_to_datasize(recv_buf);
                size_t end_recv_size = ptr_to_datasize(recv_ptr_end);
                size_t min_size = start_send_size;
                if (end_send_size < min_size) {
                    min_size = end_send_size;
                }
                if (start_recv_size < min_size) {
                    min_size = start_recv_size;
                }
                if (end_recv_size < min_size) {
                    min_size = end_recv_size;
                }
                switch (min_size) {
                    case 8:
                        return kernel_memcpy_captured_typed(static_cast<const uint64_t *>(send_buf),
                                                            bytes);
                    case 4:
                        return kernel_memcpy_captured_typed(static_cast<const uint32_t *>(send_buf),
                                                            bytes);
                    case 2:
                        return kernel_memcpy_captured_typed(static_cast<const uint16_t *>(send_buf),
                                                            bytes);
                    default:
                        return kernel_memcpy_captured_typed(static_cast<const uint8_t *>(send_buf),
                                                            bytes);
                }
            };
        return kernel_memcpy_captured(send_buf, recv_buf, bytes);
    }
    else {
        const size_t nof_workgroups = bytes / wg_size + 1;
        const size_t loop_size = nof_workgroups * wg_size;
        return q.submit([=](sycl::handler &h) {
            h.depends_on(dep_events);
            h.parallel_for<oneccl_kernel_memcpy_bytes>(
                sycl::nd_range<1>(loop_size, wg_size),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(sg_size)]] {
                    size_t idx = it.get_global_linear_id();
                    if (idx < bytes) {
                        size_t offset_recv_bytes =
                            (recv_buf_idx_ptr ? *recv_buf_idx_ptr : 0) * tmp_buf_size;
                        size_t offset_send_bytes =
                            (send_buf_idx_ptr ? *send_buf_idx_ptr : 0) * tmp_buf_size;
                        void *recv_buf_offset = static_cast<void *>(
                            static_cast<uint8_t *>(recv_buf) + offset_recv_bytes);
                        const void *send_buf_offset = static_cast<const void *>(
                            static_cast<const uint8_t *>(send_buf) + offset_send_bytes);
                        *(static_cast<uint8_t *>(recv_buf_offset) + idx) =
                            *(static_cast<const uint8_t *>(send_buf_offset) + idx);
                    }
                });
        });
    }
}

/* reduction operations supporting different data types */

// Trait for bfloat16
template <typename T>
struct is_sycl_bfloat16 : std::false_type {};
template <>
struct is_sycl_bfloat16<sycl::ext::oneapi::bfloat16> : std::true_type {};
template <typename T>
constexpr bool is_sycl_bfloat16_v = is_sycl_bfloat16<T>::value;

// Trait for floating-point types (float, double, half, bfloat16)
template <typename T>
struct is_sycl_floating_point
        : std::integral_constant<bool,
                                 std::is_floating_point_v<T> || std::is_same_v<T, sycl::half> ||
                                     std::is_same_v<T, sycl::ext::oneapi::bfloat16>> {};
template <typename T>
constexpr bool is_sycl_floating_point_v = is_sycl_floating_point<T>::value;

// Trait for integer types, excluding bfloat16
template <typename T>
constexpr bool is_sycl_integer_v = std::is_integral_v<T> && !is_sycl_bfloat16_v<T>;

/* internal operation types */

// generic max/min operations that works for all types
// it looks complex just because we use types like this sycl::marray<sycl::vec<T, vec_size>, 8>
struct sycl_max_op {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (is_sycl_bfloat16_v<T>) {
            // Use sycl::ext::oneapi::experimental::fmax for bfloat16
            return sycl::ext::oneapi::experimental::fmax(a, b);
        }
        else if constexpr (is_sycl_floating_point_v<T>) {
            // Use sycl::fmax for all other floating-point types
            return sycl::fmax(a, b);
        }
        else if constexpr (is_sycl_integer_v<T>) {
            // Use sycl::max for integer types
            return sycl::max(a, b);
        }
        else if constexpr (sycl::detail::is_vec_v<T> || sycl::detail::is_marray_v<T>) {
            // Recursively apply sycl_max_op elementwise for vectors and marrays
            T result;
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = (*this)(a[i], b[i]);
            return result;
        }
        else {
            // Fallback
            return a < b ? b : a;
        }
    }
};

struct sycl_min_op {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (is_sycl_bfloat16_v<T>) {
            // Use sycl::ext::oneapi::experimental::fmin for bfloat16
            return sycl::ext::oneapi::experimental::fmin(a, b);
        }
        else if constexpr (is_sycl_floating_point_v<T>) {
            // Use sycl::fmin for all other floating-point types
            return sycl::fmin(a, b);
        }
        else if constexpr (is_sycl_integer_v<T>) {
            // Use sycl::min for integer types
            return sycl::min(a, b);
        }
        else if constexpr (sycl::detail::is_vec_v<T> || sycl::detail::is_marray_v<T>) {
            // Recursively apply sycl_min_op elementwise for vectors and marrays
            T result;
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = (*this)(a[i], b[i]);
            return result;
        }
        else {
            // Fallback
            return a < b ? a : b;
        }
    }
};

struct sycl_prod_op {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a * b;
    }
};

// sycl_avg_op is the same as sycl_sum_op, but we will handle averaging separately
// in the reduce_average function, so we can keep it simple here
struct sycl_avg_op {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct sycl_pre_mul_sum_op {
    template <typename T, typename ScalarType>
    T pre_operation(const T &a, const ScalarType &scalar) const {
        if constexpr (sycl::detail::is_marray_v<T>) {
            T result;
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] * scalar;
            }
            return result;
        }
        else {
            return a * scalar;
        }
    }

    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct sycl_sum_op {
    static constexpr bool has_pre_operation = false;

    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

// generic operation
struct sycl_any_op {
    static constexpr bool has_pre_operation = true;

    template <typename T>
    T operator()(const ccl_reduction_data &reduction, const T &a, const T &b) const {
        if (reduction.op_type == ccl_reduction_type_internal::ccl_min) {
            return sycl_min_op()(a, b);
        }
        else if (reduction.op_type == ccl_reduction_type_internal::ccl_max) {
            return sycl_max_op()(a, b);
        }
        else if (reduction.op_type == ccl_reduction_type_internal::ccl_prod) {
            return sycl_prod_op()(a, b);
        }
        else if (reduction.op_type == ccl_reduction_type_internal::ccl_avg) {
            return sycl_avg_op()(a, b);
        }
        else if (reduction.op_type == ccl_reduction_type_internal::ccl_pre_mul_sum) {
            return sycl_pre_mul_sum_op()(a, b);
        }
        else {
            /* unknown/error */
            return a;
        }
    }
};

// helper trait to get the scalar type for vectors/marrays, otherwise T
template <typename T>
struct get_sycl_scalar_type {
    using type = T;
};
template <typename T, int N>
struct get_sycl_scalar_type<sycl::vec<T, N>> {
    using type = T;
};
template <typename T, int N>
struct get_sycl_scalar_type<sycl::marray<T, N>> {
    using type = T;
};
// specialization for sycl::marray<sycl::vec<T, N>, M>
template <typename T, int N, int M>
struct get_sycl_scalar_type<sycl::marray<sycl::vec<T, N>, M>> {
    using type = T;
};
// use to obtain T from sycl array types
template <typename T>
using get_sycl_scalar_type_t = typename get_sycl_scalar_type<T>::type;

// generic SYCL bit-cast helper for device code
template <typename T>
inline T sycl_bit_cast_device(uint64_t value) {
    using IntType = std::conditional_t<
        (sizeof(T) == 8),
        uint64_t,
        std::conditional_t<(sizeof(T) == 4),
                           uint32_t,
                           std::conditional_t<(sizeof(T) == 2), uint16_t, uint8_t>>>;
    IntType bits = static_cast<IntType>(value);
    return sycl::bit_cast<T>(bits);
}

template <typename T>
inline T apply_pre_operation(const ccl_reduction_data &reduction, const T &a) {
    using ScalarTypeDeduced = get_sycl_scalar_type_t<T>;
    if (reduction.op_type == ccl_reduction_type_internal::ccl_pre_mul_sum) {
        sycl_pre_mul_sum_op op;
        if (reduction.scalar_arg_is_ptr) {
            // scalar_arg is a pointer to the value
            ScalarTypeDeduced *scalar_ptr =
                reinterpret_cast<ScalarTypeDeduced *>(reduction.scalar_arg);
            return op.pre_operation(a, *scalar_ptr);
        }
        else {
            // scalar_arg is a value, but original type can be anything
            // use special bit-cast to handle even uint64_t -> float conversion
            ScalarTypeDeduced scalar_value =
                sycl_bit_cast_device<ScalarTypeDeduced>(reduction.scalar_arg);
            return op.pre_operation(a, scalar_value);
        }
    }
    else {
        return a;
    }
}

template <typename T, typename ReduceOp>
inline T apply_reduction(const ccl_reduction_data &reduction, const T &a, const T &b) {
    if constexpr (std::is_same_v<ReduceOp, sycl_sum_op>) {
        return sycl_sum_op()(a, b);
    }
    else {
        return ReduceOp()(reduction, a, b);
    }
}

inline ccl_reduction_data make_reduction_operation(ccl::reduction op_type) {
    if (op_type == ccl::reduction::none || op_type == ccl::reduction::custom) {
        CCL_THROW("Unsupported reduction operation type: none or custom");
    }
    ccl_reduction_data reduction_data;
    if (!ccl_reduction_type_storage::is_custom(op_type)) {
        // predefined reductions
        reduction_data.op_type = ccl_reduction_type_storage::convert_to_internal(op_type);
    }
    else {
        // user-defined reductions
        reduction_data = ccl::global_data::get().redtype_storage->get(op_type);
    }
    return reduction_data;
}

/* different reusable kernels implementation used in SYCL collectives */

/* USER-DEFINED PRE-OPERATION STANDALONE KERNEL */

template <typename T>
inline void pre_operation_kernel(void *buf, const ccl_reduction_data reduction, size_t idx) {
    T *buf_typed = (T *)buf;
    buf_typed[idx] = apply_pre_operation<T>(reduction, buf_typed[idx]);
}

template <typename T, int VS>
inline void pre_operation(void *buf,
                          const size_t count,
                          const ccl_reduction_data reduction,
                          const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, VS>;
    const size_t packed_count = count / VS;
    if (idx < packed_count) {
        pre_operation_kernel<AT>(buf, reduction, idx);
    }
    else {
        const size_t new_idx = VS * packed_count + idx - packed_count;
        if (new_idx < count) {
            pre_operation_kernel<T>(buf, reduction, new_idx);
        }
    }
}

// Kernel name template for pre_operation
template <typename T, int VS, int SGS>
class oneccl_pre_operation {};

template <typename T, int VS, int SGS>
inline sycl::event pre_operation_invoke(sycl::queue &q,
                                        void *buf,
                                        size_t count,
                                        const bool is_recording,
                                        int *tmp_buf_idx,
                                        const ccl_reduction_data reduction,
                                        size_t tmp_buf_size,
                                        const std::vector<sycl::event> &dep_events) {
    constexpr int wg_size = SGS, sg_size = SGS;
    const size_t kernel_threads = count / VS + count % VS;
    const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
    return q.submit([=](sycl::handler &h) {
        h.depends_on(dep_events);
        h.parallel_for<oneccl_pre_operation<T, VS, SGS>>(
            sycl::nd_range<1>(kernel_size, wg_size), [=](sycl::nd_item<1> it) {
                void *buf_local = buf;
                if (is_recording) {
                    size_t offset_bytes = *tmp_buf_idx * tmp_buf_size;
                    buf_local = (void *)((uintptr_t)buf + offset_bytes);
                }
                pre_operation<T, VS>(buf_local, count, reduction, it);
            });
    });
}

/* AVERAGE KERNEL */

// Kernel name template for reduce_average
template <typename T, int VS, int SGS>
class oneccl_reduce_average {};

template <typename T>
inline void reduce_average_kernel(void *buf, const size_t n, size_t idx) {
    ((T *)buf)[idx] /= n;
}

template <typename T, int VS>
inline void reduce_average(void *reduce_buf,
                           const size_t count,
                           const size_t average_divisor,
                           const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    const size_t packed_count = count / VS;
    if (idx < packed_count) {
        using AT = sycl::vec<T, VS>;
        reduce_average_kernel<AT>(reduce_buf, average_divisor, idx);
    }
    else {
        const size_t new_idx = VS * packed_count + idx - packed_count;
        if (new_idx < count) {
            reduce_average_kernel<T>(reduce_buf, average_divisor, new_idx);
        }
    }
}

template <typename T, int VS, int SGS>
inline sycl::event reduce_average_invoke(sycl::queue &q,
                                         void *reduce_buf,
                                         const size_t reduce_count,
                                         const size_t average_divisor,
                                         const std::vector<sycl::event> &dep_events) {
    constexpr int wg_size = SGS;
    constexpr int sg_size = SGS;
    int kernel_threads = reduce_count / VS + reduce_count % VS;
    int kernel_size = (kernel_threads + wg_size - 1) / wg_size * wg_size;
    sycl::event e = q.submit([=](sycl::handler &h) {
        h.depends_on(dep_events);
        h.parallel_for<oneccl_reduce_average<T, VS, SGS>>(
            sycl::nd_range<1>(kernel_size, wg_size), [=](sycl::nd_item<1> it) {
                reduce_average<T, VS>(reduce_buf, reduce_count, average_divisor, it);
            });
    });
    return e;
}

/* REDUCE PAIR (2 BUFFERS) KERNEL*/

// Kernel name template for reduce_pair
template <typename T, int VS, int SGS>
class oneccl_reduce_pair {};

template <typename T, typename ReduceOp>
inline void reduce_pair_kernel(const void *in1_,
                               const void *in2_,
                               void *out_,
                               const ccl_reduction_data reduction,
                               size_t idx) {
    T *i1 = (T *)in1_;
    T *i2 = (T *)in2_;
    T *out = (T *)out_;
    out[idx] = apply_reduction<T, ReduceOp>(reduction, i1[idx], i2[idx]);
}

template <typename T>
inline void reduce_pair_dispatch(const void *in1,
                                 const void *in2,
                                 void *out,
                                 const ccl_reduction_data reduction,
                                 size_t idx) {
    if (reduction.op_type == ccl_reduction_type_internal::ccl_sum) {
        reduce_pair_kernel<T, sycl_sum_op>(in1, in2, out, reduction, idx);
    }
    else {
        reduce_pair_kernel<T, sycl_any_op>(in1, in2, out, reduction, idx);
    }
}

// generic reduction kernel for two input buffers, used in scale-out path
template <typename T, int VS>
inline void reduce_pair(const void *in1,
                        const void *in2,
                        void *out,
                        const size_t count,
                        const ccl_reduction_data reduction,
                        const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, VS>;
    const size_t packed_count = count / VS;
    if (idx < packed_count) {
        reduce_pair_dispatch<AT>(in1, in2, out, reduction, idx);
    }
    else {
        const size_t new_idx = VS * packed_count + idx - packed_count;
        if (new_idx < count) {
            reduce_pair_dispatch<T>(in1, in2, out, reduction, new_idx);
        }
    }
}

template <typename T, int VS, int SGS>
inline sycl::event reduce_pair_invoke(sycl::queue &q,
                                      void *in1,
                                      void *in2,
                                      void *out,
                                      size_t reduce_count,
                                      const ccl_reduction_data reduction,
                                      const std::vector<sycl::event> &dep_events) {
    constexpr int wg_size = SGS, sg_size = SGS;
    const size_t kernel_threads = reduce_count / VS + reduce_count % VS;
    const size_t kernel_size = ((kernel_threads + wg_size - 1) / wg_size) * wg_size;
    return q.submit([=](sycl::handler &h) {
        h.depends_on(dep_events);
        h.parallel_for<oneccl_reduce_pair<T, VS, SGS>>(
            sycl::nd_range<1>(kernel_size, wg_size), [=](sycl::nd_item<1> it) {
                reduce_pair<T, VS>(in1, in2, out, reduce_count, reduction, it);
            });
    });
}

/* REDUCE KERNEL */

template <typename T, typename ReduceOp, int N, int read_all>
inline void reduce_base_kernel(void *recv,
                               std::array<void *, MAX_NODE_RANKS> in,
                               std::array<void *, MAX_NODE_RANKS> out,
                               const ccl_reduction_data reduction,
                               size_t idx) {
    T tmp_arr[N];
    // copy from remote to local array
    T reduce_val = ((T *)in[0])[idx];
#pragma unroll
    for (int i = 1; i < N; i++) {
        tmp_arr[i] = ((T *)in[i])[idx];
    }

    // reduce from local array
    for (int i = 1; i < N; i++) {
        reduce_val = apply_reduction<T, ReduceOp>(reduction, reduce_val, tmp_arr[i]);
    }

    // write to local recv buffer
    if constexpr (read_all) {
        ((T *)recv)[idx] = reduce_val;
    }
    // write back to remote tmp buffers
    else {
#pragma unroll
        for (int i = 0; i < N; i++) {
            ((T *)out[i])[idx] = reduce_val;
        }
    }
}

template <typename T, int N, int read_all>
inline void reduce_base_dispatch(void *recv,
                                 std::array<void *, MAX_NODE_RANKS> in,
                                 std::array<void *, MAX_NODE_RANKS> out,
                                 const ccl_reduction_data reduction,
                                 size_t idx) {
    if (reduction.op_type == ccl_reduction_type_internal::ccl_sum) {
        reduce_base_kernel<T, sycl_sum_op, N, read_all>(recv, in, out, reduction, idx);
    }
    else {
        reduce_base_kernel<T, sycl_any_op, N, read_all>(recv, in, out, reduction, idx);
    }
}

template <typename T,
          int N,
          int vec_size,
          int use_block,
          int use_local_barrier,
          int use_global_barrier,
          int read_all = 1,
          int M = 1,
          typename AT = sycl::vec<T, vec_size>>
inline void reduce_base(const void *send,
                        void *recv,
                        void *tmp,
                        std::array<void *, MAX_NODE_RANKS> in,
                        std::array<void *, MAX_NODE_RANKS> out,
                        ccl_kernel_barrier_data kernel_barrier_data,
                        const ccl_comm_barrier_data comm_barrier_data,
                        const ccl_reduction_data reduction,
                        const size_t count,
                        const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();

    const size_t packed_count = count / vec_size;

    if (use_local_barrier) {
        // copy data from send buffer to tmp buffer
        if (use_block && idx < packed_count) {
            using MAT = sycl::marray<AT, M>;
            ((MAT *)tmp)[idx] = ((MAT *)send)[idx];
        }
        else {
            const size_t new_idx = idx + (vec_size - 1) * packed_count;
            if (new_idx < count) {
                using MT = sycl::marray<T, M>;
                ((MT *)tmp)[new_idx] = ((MT *)send)[new_idx];
            }
        }

        // local barrier within gpu
        kernel_barrier(kernel_barrier_data.get_sync_ptr(), it);
    }

    if (use_global_barrier) {
        // global communication barrier across ranks
        comm_barrier(comm_barrier_data, it);
    }

    // reset local barrier counter
    if (use_local_barrier && idx == 0) {
        kernel_barrier_data.reset_sync_data();
    }

    if (use_block && idx < packed_count) {
        reduce_base_dispatch<AT, N, read_all>(recv, in, out, reduction, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            reduce_base_dispatch<T, N, read_all>(recv, in, out, reduction, new_idx);
        }
    }
}

/* REDUCE GENERAL KERNEL */

template <typename T, int vec_size, int M>
inline void copy_data_internal(void *dst,
                               const void *src,
                               const size_t count,
                               const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    constexpr int vec_size_cp = vec_size * M;
    const size_t packed_count = count / vec_size_cp;

    if (idx < packed_count) {
        using MAT = sycl::marray<AT, M>;
        ((MAT *)dst)[idx] = ((MAT *)src)[idx];
    }
    else {
        const size_t new_idx = idx + (vec_size_cp - 1) * packed_count;
        if (new_idx < count) {
            ((T *)dst)[new_idx] = ((T *)src)[new_idx];
        }
    }
}

template <typename T,
          int N,
          int vec_size,
          int use_block,
          int use_local_barrier,
          int use_global_barrier,
          int read_all,
          int M>
inline void reduce_base_general(const void *send,
                                void *recv,
                                void *tmp,
                                std::array<void *, MAX_NODE_RANKS> in,
                                std::array<void *, MAX_NODE_RANKS> out,
                                ccl_kernel_barrier_data kernel_barrier_data,
                                const ccl_comm_barrier_data comm_barrier_data,
                                const ccl_reduction_data reduction,
                                const size_t count_cp,
                                const size_t count_red,
                                const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();
    using AT = sycl::vec<T, vec_size>;

    if (use_local_barrier) {
        // copy data from send buffer to local temp buffer
        copy_data_internal<T, vec_size, M>(tmp, send, count_cp, it);

        // local barrier within gpu
        kernel_barrier(kernel_barrier_data.get_sync_ptr(), it);
    }

    if (use_global_barrier) {
        // global communication barrier across ranks
        comm_barrier(comm_barrier_data, it);
    }

    // reset local barrier counter
    if (use_local_barrier && idx == 0) {
        kernel_barrier_data.reset_sync_data();
    }

    const size_t packed_count = count_red / vec_size;

    // reduce data from all ranks
    if (idx < packed_count) {
        reduce_base_dispatch<AT, N, read_all>(recv, in, out, reduction, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count_red) {
            reduce_base_dispatch<T, N, read_all>(recv, in, out, reduction, new_idx);
        }
    }
}

/* READ-REDUCE-WRITE KERNEL*/

template <typename T, typename ReduceOp, int N>
inline void read_reduce_write_kernel(std::array<void *, MAX_GPUS> pair_ptrs,
                                     std::array<void *, MAX_GPUS> local_ptrs,
                                     std::array<void *, MAX_GPUS> even_ptrs,
                                     const ccl_reduction_data reduction,
                                     const bool is_multi_tile,
                                     size_t idx) {
    if (is_multi_tile) {
#pragma unroll
        for (int i = 0; i < N; i++) {
            const T pair_val = ((T *)pair_ptrs[i])[idx];
            T local_val = ((T *)local_ptrs[i])[idx];
            if constexpr (ReduceOp::has_pre_operation) {
                local_val = apply_pre_operation<T>(reduction, local_val);
            }
            const T red_val = apply_reduction<T, ReduceOp>(reduction, pair_val, local_val);
            ((T *)even_ptrs[i])[idx] = red_val;
        }
    }
    else {
        if constexpr (ReduceOp::has_pre_operation) {
#pragma unroll
            for (int i = 0; i < N; i++) {
                const T local_val = ((T *)local_ptrs[i])[idx];
                ((T *)even_ptrs[i])[idx] = apply_pre_operation<T>(reduction, local_val);
            }
        }
        else {
#pragma unroll
            for (int i = 0; i < N; i++) {
                ((T *)even_ptrs[i])[idx] = ((T *)local_ptrs[i])[idx];
            }
        }
    }
}

template <typename T, int N>
inline void read_reduce_write_dispatch(std::array<void *, MAX_GPUS> pair_ptrs,
                                       std::array<void *, MAX_GPUS> local_ptrs,
                                       std::array<void *, MAX_GPUS> even_ptrs,
                                       const ccl_reduction_data reduction,
                                       const bool is_multi_tile,
                                       size_t idx) {
    if (reduction.op_type == ccl_reduction_type_internal::ccl_sum) {
        read_reduce_write_kernel<T, sycl_sum_op, N>(
            pair_ptrs, local_ptrs, even_ptrs, reduction, is_multi_tile, idx);
    }
    else {
        read_reduce_write_kernel<T, sycl_any_op, N>(
            pair_ptrs, local_ptrs, even_ptrs, reduction, is_multi_tile, idx);
    }
}

template <typename T, int N, int vec_size>
inline void read_reduce_write(std::array<void *, MAX_GPUS> pair_ptrs,
                              std::array<void *, MAX_GPUS> local_ptrs,
                              std::array<void *, MAX_GPUS> even_ptrs,
                              const ccl_reduction_data reduction,
                              const bool is_multi_tile,
                              const size_t count,
                              const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id();

    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        read_reduce_write_dispatch<AT, N>(
            pair_ptrs, local_ptrs, even_ptrs, reduction, is_multi_tile, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            read_reduce_write_dispatch<T, N>(
                pair_ptrs, local_ptrs, even_ptrs, reduction, is_multi_tile, new_idx);
        }
    }
}

/* READ-WRITE KERNEL*/

template <typename T, int N>
inline void read_write_kernel(std::array<void *, MAX_GPUS> even_ptrs,
                              std::array<void *, MAX_GPUS> local_ptrs,
                              std::array<void *, MAX_GPUS> pair_ptrs,
                              const bool is_multi_tile,
                              const size_t idx) {
#pragma unroll
    for (int i = 0; i < N; i++) {
        const T val = ((T *)even_ptrs[i])[idx];
        if (is_multi_tile) {
            ((T *)pair_ptrs[i])[idx] = val;
        }
        ((T *)local_ptrs[i])[idx] = val;
    }
}

template <typename T, int N, int vec_size>
inline void read_write(std::array<void *, MAX_GPUS> even_ptrs,
                       std::array<void *, MAX_GPUS> local_ptrs,
                       std::array<void *, MAX_GPUS> pair_ptrs,
                       const bool is_multi_tile,
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
        read_write_kernel<AT, N>(even_ptrs, local_ptrs, pair_ptrs, is_multi_tile, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            read_write_kernel<T, N>(even_ptrs, local_ptrs, pair_ptrs, is_multi_tile, new_idx);
        }
    }
}

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
// SYCL copy kernels and kernel-name tag classes.

#include <array>
#include <sycl/sycl.hpp>
#include <vector>

#include "common/global/global.hpp"
#include "coll/algorithms/utils/consts.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/utils/tvisa/include/gen_visa_templates.hpp"

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

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

using sycl_copy_message_t = sycl::vec<uint32_t, 4>;

/**
 * Copies one scalar or vector element to two destinations.
 *
 * @tparam T Scalar or vector element type.
 * @param dst_ptr1 First destination buffer.
 * @param dst_ptr2 Second destination buffer.
 * @param src_ptr Source buffer.
 * @param idx Element index to copy.
 */
template <typename T>
inline void copy2_kernel(void *dst_ptr1, void *dst_ptr2, const void *src_ptr, size_t idx) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__) && defined(CCL_SYCL_ENABLE_ARCB)
    sycl_copy_message_t value;
    if (sizeof(T) == sizeof(sycl_copy_message_t)) {
        lscLoad<16, CacheCtrl::L1UC_L3C>(
            value,
            const_cast<sycl_copy_message_t *>(static_cast<const sycl_copy_message_t *>(src_ptr)) +
                idx);
        lscStore<16, CacheCtrl::L1WB_L3WB>(static_cast<sycl_copy_message_t *>(dst_ptr1) + idx,
                                           value);
        lscStore<16, CacheCtrl::L1WB_L3WB>(static_cast<sycl_copy_message_t *>(dst_ptr2) + idx,
                                           value);
    }
    else {
        static_cast<T *>(dst_ptr1)[idx] = static_cast<const T *>(src_ptr)[idx];
        static_cast<T *>(dst_ptr2)[idx] = static_cast<const T *>(src_ptr)[idx];
    }
#else
    static_cast<T *>(dst_ptr1)[idx] = static_cast<const T *>(src_ptr)[idx];
    static_cast<T *>(dst_ptr2)[idx] = static_cast<const T *>(src_ptr)[idx];
#endif
}

/**
 * Copies the packed element and scalar tail assigned to a work-item.
 *
 * @tparam T Scalar element type.
 * @tparam vec_size Number of scalar elements in each packed copy.
 * @param dst_ptr1 First destination buffer.
 * @param dst_ptr2 Second destination buffer.
 * @param src_ptr Source buffer.
 * @param count Number of scalar elements to copy.
 * @param offset Work-item offset used for grid-stride iteration.
 * @param it Current SYCL work-item.
 */
template <typename T, int vec_size>
inline void copy2(void *dst_ptr1,
                  void *dst_ptr2,
                  const void *src_ptr,
                  const size_t count,
                  const size_t offset,
                  const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id() + offset;
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        copy2_kernel<AT>(dst_ptr1, dst_ptr2, src_ptr, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            copy2_kernel<T>(dst_ptr1, dst_ptr2, src_ptr, new_idx);
        }
    }
}

/**
 * Copies one source buffer to two destinations using one load per element.
 *
 * @tparam T Scalar element type.
 * @tparam full_vec_size Vector width used when all buffers are aligned.
 * @tparam partial_vec_size Vector width used for the unaligned path.
 * @param dst_ptr1 First destination buffer.
 * @param dst_ptr2 Second destination buffer.
 * @param src_ptr Source buffer.
 * @param count Number of scalar elements to copy.
 * @param use_full_vector Whether to use the full-width vector path.
 */
template <typename T, int full_vec_size, int partial_vec_size>
inline void copy_2dst_wrap(void *dst_ptr1,
                           void *dst_ptr2,
                           const void *src_ptr,
                           const size_t count,
                           int use_full_vector) {
    const sycl::nd_item<1> it = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;
    const size_t req_threads = count / vec_size + count % vec_size;

    if (use_full_vector) {
#pragma unroll
        for (size_t offset = 0; offset < req_threads; offset += num_threads) {
            copy2<T, full_vec_size>(dst_ptr1, dst_ptr2, src_ptr, count, offset, it);
        }
    }
    else {
#pragma unroll
        for (size_t offset = 0; offset < req_threads; offset += num_threads) {
            copy2<T, partial_vec_size>(dst_ptr1, dst_ptr2, src_ptr, count, offset, it);
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

/* COPY KERNELS */

#ifndef RS_USE_PARALLEL_FOR
#define RS_USE_PARALLEL_FOR 1
#endif

using message_t = sycl::vec<uint32_t, 4>;

template <typename T>
void inline copy_kernel(void *dst_ptr, const void *src_ptr, size_t idx) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__) && defined(CCL_SYCL_ENABLE_ARCB)
    message_t v;
    if (sizeof(T) == 16) {
        lscLoad<16, CacheCtrl::L1UC_L3C>(
            v, const_cast<message_t *>(static_cast<const message_t *>(src_ptr)) + idx);
        lscStore<16, CacheCtrl::L1WB_L3WB>(static_cast<message_t *>(dst_ptr) + idx, v);
    }
    else {
        static_cast<T *>(dst_ptr)[idx] = static_cast<const T *>(src_ptr)[idx];
    }
#else
    static_cast<T *>(dst_ptr)[idx] = static_cast<const T *>(src_ptr)[idx];
#endif
}

template <typename T, int vec_size>
void inline copy(void *dst_ptr,
                 const void *src_ptr,
                 const size_t count,
                 const size_t offset,
                 const sycl::nd_item<1> it) {
    const size_t idx = it.get_global_linear_id() + offset;
    const size_t packed_count = count / vec_size;

    if (idx < packed_count) {
        using AT = sycl::vec<T, vec_size>;
        copy_kernel<AT>(dst_ptr, src_ptr, idx);
    }
    else {
        const size_t new_idx = idx + (vec_size - 1) * packed_count;
        if (new_idx < count) {
            copy_kernel<T>(dst_ptr, src_ptr, new_idx);
        }
    }
}

// Single vec_size variant (used by send_simple/recv_simple)
template <typename T, int vec_size>
#if !RS_USE_PARALLEL_FOR
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
#endif
void inline copy_kernel_wrap(void *dst_ptr, const void *src_ptr, const size_t count) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const size_t req_threads = count / vec_size + count % vec_size;

#pragma unroll
    for (size_t offset = 0; offset < req_threads; offset += num_threads) {
        copy<T, vec_size>(dst_ptr, src_ptr, count, offset, it);
    }
}

// Dual vec_size variant with use_full_vector (used by alltoall, inherited from p2p_barriers)
template <typename T, int full_vec_size, int partial_vec_size>
#if !RS_USE_PARALLEL_FOR
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
#endif
void inline copy_kernel_wrap(void *dst_ptr, const void *src_ptr, const size_t count, int use_full_vector) {
    const sycl::nd_item<1> it = syclext::this_work_item::get_nd_item<1>();
    const size_t num_threads = it.get_global_range(0);
    const int vec_size = use_full_vector ? full_vec_size : partial_vec_size;
    const size_t req_threads = count / vec_size + count % vec_size;

    if (use_full_vector) {
#pragma unroll
        for (size_t offset = 0; offset < req_threads; offset += num_threads) {
            copy<T, full_vec_size>(dst_ptr, src_ptr, count, offset, it);
        }
    }
    else {
#pragma unroll
        for (size_t offset = 0; offset < req_threads; offset += num_threads) {
            copy<T, partial_vec_size>(dst_ptr, src_ptr, count, offset, it);
        }
    }
}

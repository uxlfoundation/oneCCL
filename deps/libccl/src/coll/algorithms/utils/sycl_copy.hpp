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

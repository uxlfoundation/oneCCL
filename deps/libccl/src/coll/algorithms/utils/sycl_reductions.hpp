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
// Reduction kernels (pre_operation, reduce_average/pair/base, read/write)
// and the reduction-aware copy_and_modify_data helper.

#include <array>
#include <sycl/sycl.hpp>
#include <vector>

#include "common/global/global.hpp"
#include "coll/algorithms/utils/comm_barrier_data.hpp"
#include "coll/algorithms/utils/consts.hpp"
#include "coll/algorithms/utils/sycl_barrier.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "coll/algorithms/utils/sycl_traits.hpp"
#include "coll/reduction/reduction.hpp"

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

// Reduction-aware copy used by scale-out; depends on apply_pre_operation so
// it lives here (not in sycl_copy.hpp) to avoid a cycle. See plan §2.2.
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

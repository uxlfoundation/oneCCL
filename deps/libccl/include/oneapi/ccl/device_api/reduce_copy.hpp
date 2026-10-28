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

/**
 * oneCCL device-side reduce-copy engine — cooperative vectorized sum from N
 * sources to M destinations, in three phases: scalar prefix to reach alignment,
 * 16-byte vectorized bulk (sycl::vec<T, 16/sizeof(T)>), scalar remainder.
 *
 * Pointers come from lambdas: src_lambda(i) / dst_lambda(i) -> T*.
 */
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

#ifdef CCL_ENABLE_SYCL

namespace ccl {
namespace v1 {
namespace detail {

// bf16/half accumulate in float — repeated 16-bit additions lose precision.
template <typename T> struct accum_type { using type = T; };
template <> struct accum_type<sycl::half> { using type = float; };
template <> struct accum_type<sycl::ext::oneapi::bfloat16> { using type = float; };

template <typename T>
using accum_t = typename accum_type<T>::type;

// Bytes needed to align `ptr` to `align`; 0 if already aligned.
inline uintptr_t get_misalignment(const void* ptr, unsigned align) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t rem = addr % align;
    return rem == 0 ? 0 : align - rem;
}

// Elements to skip so every source and destination reaches pack alignment,
// or -1 if they don't share the same sub-pack alignment.
template <typename T, typename SrcLambda, typename DstLambda>
inline int common_align_prefix(
    SrcLambda src_lambda, int n_src,
    DstLambda dst_lambda, int n_dst,
    unsigned pack_bytes) {

    if (n_src <= 0) return 0;

    uintptr_t ref = reinterpret_cast<uintptr_t>(src_lambda(0));

    // Every source and destination must share ref's addr % pack_bytes
    for (int s = 1; s < n_src; s++) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(src_lambda(s));
        if ((addr - ref) % pack_bytes != 0) {
            return -1;
        }
    }

    for (int d = 0; d < n_dst; d++) {
        uintptr_t addr = reinterpret_cast<uintptr_t>(dst_lambda(d));
        if ((addr - ref) % pack_bytes != 0) {
            return -1;
        }
    }

    uintptr_t misalign = get_misalignment(src_lambda(0), pack_bytes);
    if (misalign % sizeof(T) != 0) {
        return -1;  // not element-aligned
    }
    return static_cast<int>(misalign / sizeof(T));
}

// Element-by-element, cooperative-strided.
template <typename T, typename Coop, typename SrcLambda, typename DstLambda>
inline void reduce_copy_scalar(
    Coop coop,
    SrcLambda src_lambda, int n_src,
    DstLambda dst_lambda, int n_dst,
    int offset, int count) {

    const int tid = coop.thread_rank();
    const int stride = coop.size();

    using Acc = accum_t<T>;

    for (int i = tid; i < count; i += stride) {
        int idx = offset + i;
        Acc acc = static_cast<Acc>(src_lambda(0)[idx]);
        for (int s = 1; s < n_src; s++) {
            acc += static_cast<Acc>(src_lambda(s)[idx]);
        }
        for (int d = 0; d < n_dst; d++) {
            dst_lambda(d)[idx] = static_cast<T>(acc);
        }
    }
}

// Vectorized bulk loop over sycl::vec<T,N>: UNROLL packs per thread per
// iteration, cooperative-strided. Caller must pass a multiple of pack_count.
template <typename T, int PackCount, int UNROLL, typename Coop,
          typename SrcLambda, typename DstLambda>
inline void reduce_copy_vec(
    Coop coop,
    SrcLambda src_lambda, int n_src,
    DstLambda dst_lambda, int n_dst,
    int offset,       // element offset into src/dst pointers
    int vec_count) {  // number of elements to process (multiple of PackCount)

    using Pack = sycl::vec<T, PackCount>;
    using Acc = accum_t<T>;
    using AccPack = sycl::vec<Acc, PackCount>;
    const int total_packs = vec_count / PackCount;
    const int tid = coop.thread_rank();
    const int stride = coop.size();

    for (int base = tid; base < total_packs; base += stride * UNROLL) {
        AccPack acc[UNROLL];

        // Seed from source 0 (load as T, widen to Acc)
        {
            const Pack* sp = reinterpret_cast<const Pack*>(src_lambda(0) + offset);
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                int idx = base + u * stride;
                if (idx < total_packs) {
                    acc[u] = sp[idx].template convert<Acc>();
                }
            }
        }

        for (int s = 1; s < n_src; s++) {
            const Pack* sp = reinterpret_cast<const Pack*>(src_lambda(s) + offset);
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                int idx = base + u * stride;
                if (idx < total_packs) {
                    acc[u] += sp[idx].template convert<Acc>();
                }
            }
        }

        // Narrow back to T on store
        for (int d = 0; d < n_dst; d++) {
            Pack* dp = reinterpret_cast<Pack*>(dst_lambda(d) + offset);
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
                int idx = base + u * stride;
                if (idx < total_packs) {
                    dp[idx] = acc[u].template convert<T>();
                }
            }
        }
    }
}

// Top-level entry point: scalar prefix, vectorized bulk, scalar remainder.
template <typename T, int UNROLL = 4, typename Coop,
          typename SrcLambda, typename DstLambda>
inline void reduce_copy(
    Coop coop,
    SrcLambda src_lambda, int n_src,
    DstLambda dst_lambda, int n_dst,
    int count) {

    if (count <= 0 || n_src <= 0) return;

    constexpr int pack_bytes = 16;
    constexpr int pack_count = pack_bytes / sizeof(T);
    int processed = 0;

    int prefix = common_align_prefix<T>(src_lambda, n_src,
                                        dst_lambda, n_dst, pack_bytes);
    // Mutually unalignable, or the prefix covers everything: all-scalar
    if (prefix < 0 || prefix >= count) {
        reduce_copy_scalar<T>(coop, src_lambda, n_src,
                              dst_lambda, n_dst, 0, count);
        return;
    }

    if (prefix > 0) {
        reduce_copy_scalar<T>(coop, src_lambda, n_src,
                              dst_lambda, n_dst, 0, prefix);
        processed = prefix;
    }

    int remaining = count - processed;
    int vec_elts = (remaining / pack_count) * pack_count;

    if (vec_elts > 0) {
        reduce_copy_vec<T, pack_count, UNROLL>(
            coop, src_lambda, n_src,
            dst_lambda, n_dst, processed, vec_elts);
        processed += vec_elts;
    }

    int tail = count - processed;
    if (tail > 0) {
        reduce_copy_scalar<T>(coop, src_lambda, n_src,
                              dst_lambda, n_dst, processed, tail);
    }
}

} // namespace detail
} // namespace v1
} // namespace ccl

#endif // CCL_ENABLE_SYCL

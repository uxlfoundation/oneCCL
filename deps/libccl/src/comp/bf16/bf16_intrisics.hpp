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

#ifdef CCL_BF16_TARGET_ATTRIBUTES

#ifdef CCL_BF16_AVX512BF_COMPILER
#define BF16_ALL_ATTRS "avx512bw,avx512vl,avx512f,avx512bf16"
#else
#define BF16_ALL_ATTRS "avx512bw,avx512vl,avx512f"
#endif

#define BF16_TARGET_ATTRIBUTE_BWF __attribute__((target("avx512bw,avx512f")))
#define BF16_TARGET_ATTRIBUTE_ALL __attribute__((target(BF16_ALL_ATTRS)))
#define BF16_INLINE_TARGET_ATTRIBUTE_BW \
    __attribute__((__always_inline__, target("avx512bw"))) inline
#define BF16_INLINE_TARGET_ATTRIBUTE __attribute__((__always_inline__, target("avx512bf16"))) inline
#define BF16_INLINE_TARGET_ATTRIBUTE_ALL \
    __attribute__((__always_inline__, target(BF16_ALL_ATTRS))) inline

#else // CCL_BF16_TARGET_ATTRIBUTES

#define BF16_TARGET_ATTRIBUTE_BWF
#define BF16_TARGET_ATTRIBUTE_ALL
#define BF16_INLINE_TARGET_ATTRIBUTE_BW  __attribute__((__always_inline__)) inline
#define BF16_INLINE_TARGET_ATTRIBUTE     __attribute__((__always_inline__)) inline
#define BF16_INLINE_TARGET_ATTRIBUTE_ALL __attribute__((__always_inline__)) inline

#endif // CCL_BF16_TARGET_ATTRIBUTES

#ifdef CCL_BF16_COMPILER

#include <immintrin.h>
#include <inttypes.h>

#include "common/global/global.hpp"
#include "comp/bf16/bf16_utils.hpp"
#include "oneapi/ccl/types.hpp"

#define CCL_BF16_IN_M256 16

typedef __m512 (*ccl_bf16_reduction_func_ptr)(__m512 a, __m512 b);
BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_sum_wrap(__m512 a, __m512 b);
BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_prod_wrap(__m512 a, __m512 b);
BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_min_wrap(__m512 a, __m512 b);
BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_max_wrap(__m512 a, __m512 b);
BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_reduce(__m512 a, __m512 b, ccl_bf16_reduction_func_ptr op);

BF16_INLINE_TARGET_ATTRIBUTE_BW void ccl_bf16_load_as_fp32(const void* src, void* dst) {
    __m512i y = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src));
    _mm512_storeu_si512(dst, _mm512_bslli_epi128(y, 2));
}

BF16_INLINE_TARGET_ATTRIBUTE_BW void ccl_fp32_store_as_bf16_avx512f(const void* src, void* dst) {
    _mm256_storeu_si256((__m256i*)(dst),
                        _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_loadu_si512(src), 2)));
}

#ifdef CCL_BF16_AVX512BF_COMPILER
BF16_INLINE_TARGET_ATTRIBUTE void ccl_fp32_store_as_bf16_avx512bf(const void* src, void* dst) {
    _mm256_storeu_si256((__m256i*)(dst), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src)));
}
#endif // CCL_BF16_AVX512BF_COMPILER

#define CCL_BF16_DEFINE_REDUCE_FUNC(impl_type) \
\
    BF16_INLINE_TARGET_ATTRIBUTE_ALL void ccl_bf16_reduce_inputs_##impl_type( \
        const void* a, const void* b, void* res, ccl_bf16_reduction_func_ptr op) { \
        __m512 vfp32_in, vfp32_inout; \
        ccl_bf16_load_as_fp32(a, (void*)&vfp32_in); \
        ccl_bf16_load_as_fp32(b, (void*)&vfp32_inout); \
        __m512 vfp32_out = bf16_reduce(vfp32_in, vfp32_inout, op); \
        ccl_fp32_store_as_bf16_##impl_type((const void*)&vfp32_out, res); \
    } \
\
    BF16_INLINE_TARGET_ATTRIBUTE_ALL void ccl_bf16_reduce_main_##impl_type( \
        const void* in, const void* inout, ccl_bf16_reduction_func_ptr op) { \
        ccl_bf16_reduce_inputs_##impl_type(in, inout, (void*)(inout), op); \
    } \
\
    BF16_INLINE_TARGET_ATTRIBUTE_ALL void ccl_bf16_reduce_tile_##impl_type( \
        const void* in, void* inout, uint8_t len, ccl_bf16_reduction_func_ptr op) { \
        if (len == 0) \
            return; \
        uint16_t mask = ((uint16_t)0xFFFF) >> (CCL_BF16_IN_M256 - len); \
        __m256i a = _mm256_maskz_loadu_epi16(mask, in); \
        __m256i b = _mm256_maskz_loadu_epi16(mask, inout); \
        __m256i res; \
        ccl_bf16_reduce_inputs_##impl_type(&a, &b, &res, op); \
        _mm256_mask_storeu_epi16(inout, (__mmask16)mask, res); \
    } \
\
    BF16_INLINE_TARGET_ATTRIBUTE_ALL void ccl_bf16_reduce_impl_##impl_type( \
        const void* in_buf, void* inout_buf, size_t in_cnt, ccl_bf16_reduction_func_ptr op) { \
        int i = 0; \
        for (i = 0; i <= (int)in_cnt - CCL_BF16_IN_M256; i += CCL_BF16_IN_M256) { \
            ccl_bf16_reduce_main_##impl_type((uint16_t*)in_buf + i, (uint16_t*)inout_buf + i, op); \
        } \
        ccl_bf16_reduce_tile_##impl_type( \
            (uint16_t*)in_buf + i, (uint16_t*)inout_buf + i, (uint8_t)(in_cnt - i), op); \
    }

CCL_BF16_DEFINE_REDUCE_FUNC(avx512f);
#ifdef CCL_BF16_AVX512BF_COMPILER
CCL_BF16_DEFINE_REDUCE_FUNC(avx512bf);
#endif // CCL_BF16_AVX512BF_COMPILER

BF16_INLINE_TARGET_ATTRIBUTE_ALL void ccl_bf16_reduce_impl(const void* in_buf,
                                                           void* inout_buf,
                                                           size_t in_cnt,
                                                           ccl::reduction op) {
    ccl_bf16_reduction_func_ptr func = nullptr;
    switch (op) {
        case ccl::reduction::sum: func = &bf16_sum_wrap; break;
        case ccl::reduction::prod: func = &bf16_prod_wrap; break;
        case ccl::reduction::min: func = &bf16_min_wrap; break;
        case ccl::reduction::max: func = &bf16_max_wrap; break;
        default: CCL_FATAL("unexpected value ", ccl::utils::enum_to_underlying(op));
    }

    auto impl_type = ccl::global_data::env().bf16_impl_type;

    if (impl_type == ccl_bf16_avx512f) {
        ccl_bf16_reduce_impl_avx512f(in_buf, inout_buf, in_cnt, func);
    }
#ifdef CCL_BF16_AVX512BF_COMPILER
    else if (impl_type == ccl_bf16_avx512bf) {
        ccl_bf16_reduce_impl_avx512bf(in_buf, inout_buf, in_cnt, func);
    }
#endif // CCL_BF16_AVX512BF_COMPILER
    else {
        CCL_THROW("unexpected bf16_impl_type: ", impl_type);
    }
}

#endif // CCL_BF16_COMPILER

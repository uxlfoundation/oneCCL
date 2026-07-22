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

#ifdef CCL_FP16_COMPILER

#include <immintrin.h>
#include <inttypes.h>
#include <string.h>

#include "common/global/global.hpp"
#include "comp/fp16/fp16_utils.hpp"
#include "oneapi/ccl/types.hpp"

#define CCL_FP16_STEP_512FP16 32
#define CCL_FP16_STEP_512     16
#define CCL_FP16_STEP_256     8

#ifdef CCL_FP16_TARGET_ATTRIBUTES
#define FP16_TARGET_ATTRIBUTE_F16C        __attribute__((target("f16c")))
#define FP16_TARGET_ATTRIBUTE_AVX512      __attribute__((target("avx512f")))
#define FP16_TARGET_ATTRIBUTE_AVX512FP16  __attribute__((target("avx512fp16")))
#define FP16_INLINE_TARGET_ATTRIBUTE_F16C __attribute__((target("f16c"))) inline
#define FP16_INLINE_TARGET_ATTRIBUTE_AVX512F \
    __attribute__((target("avx512f,avx512bw,avx512vl"))) inline
#define FP16_INLINE_TARGET_ATTRIBUTE_AVX512FP16 \
    __attribute__((target("avx512fp16,avx512bw"))) inline
#else // CCL_FP16_TARGET_ATTRIBUTES
#define FP16_TARGET_ATTRIBUTE_F16C
#define FP16_TARGET_ATTRIBUTE_AVX512
#define FP16_TARGET_ATTRIBUTE_AVX512FP16
#define FP16_INLINE_TARGET_ATTRIBUTE_F16C       inline
#define FP16_INLINE_TARGET_ATTRIBUTE_AVX512F    inline
#define FP16_INLINE_TARGET_ATTRIBUTE_AVX512FP16 inline
#endif // CCL_FP16_TARGET_ATTRIBUTES

#define FP16_TARGET_ATTRIBUTE_256     FP16_TARGET_ATTRIBUTE_F16C
#define FP16_TARGET_ATTRIBUTE_512     FP16_TARGET_ATTRIBUTE_AVX512
#define FP16_TARGET_ATTRIBUTE_512FP16 FP16_TARGET_ATTRIBUTE_AVX512FP16

#define CCL_FP16_DECLARE_ELEM_FUNCS(VLEN) \
    typedef __m##VLEN (*ccl_fp16_reduction_func_ptr_##VLEN)(__m##VLEN a, __m##VLEN b); \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_sum_wrap_##VLEN(__m##VLEN a, __m##VLEN b); \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_prod_wrap_##VLEN(__m##VLEN a, __m##VLEN b); \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_min_wrap_##VLEN(__m##VLEN a, __m##VLEN b); \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_max_wrap_##VLEN(__m##VLEN a, __m##VLEN b); \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_reduce_##VLEN( \
        __m##VLEN a, __m##VLEN b, ccl_fp16_reduction_func_ptr_##VLEN op);

#define CCL_FP16_DEFINE_ELEM_FUNCS(VLEN) \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_sum_wrap_##VLEN(__m##VLEN a, __m##VLEN b) { \
        return _mm##VLEN##_add_ps(a, b); \
    } \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_prod_wrap_##VLEN(__m##VLEN a, __m##VLEN b) { \
        return _mm##VLEN##_mul_ps(a, b); \
    } \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_min_wrap_##VLEN(__m##VLEN a, __m##VLEN b) { \
        return _mm##VLEN##_min_ps(a, b); \
    } \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_max_wrap_##VLEN(__m##VLEN a, __m##VLEN b) { \
        return _mm##VLEN##_max_ps(a, b); \
    } \
    FP16_TARGET_ATTRIBUTE_##VLEN __m##VLEN fp16_reduce_##VLEN( \
        __m##VLEN a, __m##VLEN b, ccl_fp16_reduction_func_ptr_##VLEN op) { \
        return (*op)(a, b); \
    }

#ifdef CCL_FP16_AVX512FP16_COMPILER
typedef __m512h (*ccl_fp16_reduction_func_ptr_512FP16)(__m512h a, __m512h b);
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_sum_wrap_512FP16(__m512h a, __m512h b);
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_prod_wrap_512FP16(__m512h a, __m512h b);
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_min_wrap_512FP16(__m512h a, __m512h b);
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_max_wrap_512FP16(__m512h a, __m512h b);
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_reduce_512FP16(__m512h a,
                                                          __m512h b,
                                                          ccl_fp16_reduction_func_ptr_512FP16 op);
#endif // CCL_FP16_AVX512FP16_COMPILER

CCL_FP16_DECLARE_ELEM_FUNCS(256);
CCL_FP16_DECLARE_ELEM_FUNCS(512);

FP16_INLINE_TARGET_ATTRIBUTE_F16C void ccl_fp16_reduce_inputs_256(
    const void* a,
    const void* b,
    void* res,
    ccl_fp16_reduction_func_ptr_256 op) {
    __m256 vfp32_in, vfp32_inout;
    vfp32_in = (__m256)(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)a)));
    vfp32_inout = (__m256)(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)b)));
    __m256 vfp32_out = fp16_reduce_256(vfp32_in, vfp32_inout, op);
    _mm_storeu_si128((__m128i*)(res), _mm256_cvtps_ph(vfp32_out, 0));
}

FP16_INLINE_TARGET_ATTRIBUTE_F16C void ccl_fp16_reduce_tile_256(
    const void* in,
    void* inout,
    uint8_t len,
    ccl_fp16_reduction_func_ptr_256 op) {
    if (len == 0)
        return;
    uint16_t a[CCL_FP16_STEP_256];
    uint16_t b[CCL_FP16_STEP_256];
    uint16_t res[CCL_FP16_STEP_256];
    memcpy(a, in, len * sizeof(uint16_t));
    memcpy(b, inout, len * sizeof(uint16_t));
    ccl_fp16_reduce_inputs_256(a, b, res, op);
    memcpy(inout, res, len * sizeof(uint16_t));
}

FP16_INLINE_TARGET_ATTRIBUTE_AVX512F void ccl_fp16_reduce_inputs_512(
    const void* a,
    const void* b,
    void* res,
    ccl_fp16_reduction_func_ptr_512 op) {
    __m512 vfp32_in, vfp32_inout;
    vfp32_in = (__m512)(_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)a)));
    vfp32_inout = (__m512)(_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)b)));
    __m512 vfp32_out = fp16_reduce_512(vfp32_in, vfp32_inout, op);
    _mm256_storeu_si256((__m256i*)(res), _mm512_cvtps_ph(vfp32_out, 0));
}

FP16_INLINE_TARGET_ATTRIBUTE_AVX512F void ccl_fp16_reduce_tile_512(
    const void* in,
    void* inout,
    uint8_t len,
    ccl_fp16_reduction_func_ptr_512 op) {
    if (len == 0)
        return;
    uint16_t mask = ((uint16_t)0xFFFF) >> (CCL_FP16_STEP_512 - len);
    __m256i a = _mm256_maskz_loadu_epi16(mask, in);
    __m256i b = _mm256_maskz_loadu_epi16(mask, inout);
    __m256i res;
    ccl_fp16_reduce_inputs_512(&a, &b, &res, op);
    _mm256_mask_storeu_epi16(inout, (__mmask16)mask, res);
}

#ifdef CCL_FP16_AVX512FP16_COMPILER
FP16_INLINE_TARGET_ATTRIBUTE_AVX512FP16 void ccl_fp16_reduce_inputs_512FP16(
    const void* a,
    const void* b,
    void* res,
    ccl_fp16_reduction_func_ptr_512FP16 op) {
    __m512h vfp16_in, vfp16_inout;
    vfp16_in = _mm512_loadu_ph(a);
    vfp16_inout = _mm512_loadu_ph(b);
    __m512h vfp16_out = fp16_reduce_512FP16(vfp16_in, vfp16_inout, op);
    _mm512_storeu_ph(res, vfp16_out);
}

FP16_INLINE_TARGET_ATTRIBUTE_AVX512FP16 void ccl_fp16_reduce_tile_512FP16(
    const void* in,
    void* inout,
    uint8_t len,
    ccl_fp16_reduction_func_ptr_512FP16 op) {
    if (len == 0)
        return;
    uint32_t mask = ((uint32_t)0xFFFFFFFF) >> (CCL_FP16_STEP_512FP16 - len);
    __m512i a = _mm512_maskz_loadu_epi16(mask, in);
    __m512i b = _mm512_maskz_loadu_epi16(mask, inout);
    __m512i res;
    ccl_fp16_reduce_inputs_512FP16(&a, &b, &res, op);
    _mm512_mask_storeu_epi16(inout, (__mmask32)mask, res);
}
#endif // CCL_FP16_AVX512FP16_COMPILER

#define CCL_FP16_DEFINE_REDUCE_FUNC(VLEN) \
\
    void inline ccl_fp16_reduce_main_##VLEN( \
        const void* in, const void* inout, ccl_fp16_reduction_func_ptr_##VLEN op) { \
        ccl_fp16_reduce_inputs_##VLEN(in, inout, (void*)inout, op); \
    } \
\
    void inline ccl_fp16_reduce_impl_##VLEN(const void* in_buf, \
                                            void* inout_buf, \
                                            size_t in_cnt, \
                                            ccl_fp16_reduction_func_ptr_##VLEN op) { \
        int i = 0; \
        for (i = 0; i <= (int)in_cnt - CCL_FP16_STEP_##VLEN; i += CCL_FP16_STEP_##VLEN) { \
            ccl_fp16_reduce_main_##VLEN((uint16_t*)in_buf + i, (uint16_t*)inout_buf + i, op); \
        } \
        ccl_fp16_reduce_tile_##VLEN( \
            (uint16_t*)in_buf + i, (uint16_t*)inout_buf + i, (uint8_t)(in_cnt - i), op); \
    }

#ifdef CCL_FP16_AVX512FP16_COMPILER
CCL_FP16_DEFINE_REDUCE_FUNC(512FP16);
#endif // CCL_FP16_AVX512FP16_COMPILER
CCL_FP16_DEFINE_REDUCE_FUNC(512);
CCL_FP16_DEFINE_REDUCE_FUNC(256);

void inline ccl_fp16_reduce_impl(const void* in_buf,
                                 void* inout_buf,
                                 size_t in_cnt,
                                 ccl::reduction op) {
    ccl_fp16_reduction_func_ptr_256 func_256 = nullptr;
    ccl_fp16_reduction_func_ptr_512 func_512 = nullptr;
#ifdef CCL_FP16_AVX512FP16_COMPILER
    ccl_fp16_reduction_func_ptr_512FP16 func_512FP16 = nullptr;
#endif // CCL_FP16_AVX512FP16_COMPILER

    auto impl_type = ccl::global_data::env().fp16_impl_type;

    if (impl_type == ccl_fp16_f16c) {
        switch (op) {
            case ccl::reduction::sum: func_256 = &fp16_sum_wrap_256; break;
            case ccl::reduction::prod: func_256 = &fp16_prod_wrap_256; break;
            case ccl::reduction::min: func_256 = &fp16_min_wrap_256; break;
            case ccl::reduction::max: func_256 = &fp16_max_wrap_256; break;
            default: CCL_FATAL("unexpected value ", ccl::utils::enum_to_underlying(op));
        }
        ccl_fp16_reduce_impl_256(in_buf, inout_buf, in_cnt, func_256);
    }
    else if (impl_type == ccl_fp16_avx512f) {
        switch (op) {
            case ccl::reduction::sum: func_512 = &fp16_sum_wrap_512; break;
            case ccl::reduction::prod: func_512 = &fp16_prod_wrap_512; break;
            case ccl::reduction::min: func_512 = &fp16_min_wrap_512; break;
            case ccl::reduction::max: func_512 = &fp16_max_wrap_512; break;
            default: CCL_FATAL("unexpected value ", ccl::utils::enum_to_underlying(op));
        }
        ccl_fp16_reduce_impl_512(in_buf, inout_buf, in_cnt, func_512);
    }
#ifdef CCL_FP16_AVX512FP16_COMPILER
    else if (impl_type == ccl_fp16_avx512fp16) {
        switch (op) {
            case ccl::reduction::sum: func_512FP16 = &fp16_sum_wrap_512FP16; break;
            case ccl::reduction::prod: func_512FP16 = &fp16_prod_wrap_512FP16; break;
            case ccl::reduction::min: func_512FP16 = &fp16_min_wrap_512FP16; break;
            case ccl::reduction::max: func_512FP16 = &fp16_max_wrap_512FP16; break;
            default: CCL_FATAL("unexpected value ", ccl::utils::enum_to_underlying(op));
        }
        ccl_fp16_reduce_impl_512FP16(in_buf, inout_buf, in_cnt, func_512FP16);
    }
#endif // CCL_FP16_AVX512FP16_COMPILER
}

#endif // CCL_FP16_COMPILER

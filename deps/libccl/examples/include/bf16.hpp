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

#include <cassert>
#include <immintrin.h>

#define FLOATS_IN_M512 16
#define BF16_SHIFT     16

/*

 https://www.johndcook.com/blog/2018/11/15/bfloat16/

 In this example we use the accuracy 0.00781250
 of calculations performed in the bfloat16, but don't take
 into account the error that may occur during conversion
 from float32 datatype to bfloat16.

 */

#define BF16_PRECISION 0.00781250 /* 2^-7 */
#define FP16_PRECISION (9.77e-4) // 2^-10
#define FP32_PRECISION (1.19e-7) // 2^-23
#define FP64_PRECISION (2.22e-16) // 2^-52

void convert_fp32_to_bf16_arrays(void*, void*, int);
void convert_bf16_to_fp32_arrays(void*, float*, int);

int is_bf16_enabled() {
#ifdef CCL_BF16_COMPILER
    int is_avx512f_enabled = 0;
    uint32_t reg[4];

    __asm__ __volatile__("cpuid"
                         : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                         : "a"(7), "c"(0));
    is_avx512f_enabled = ((reg[1] & (1u << 16)) >> 16) & ((reg[1] & (1u << 30)) >> 30) &
                         ((reg[1] & (1u << 31)) >> 31);

    return (is_avx512f_enabled) ? 1 : 0;
#else
    return 0;
#endif
}

int is_avx512bf_enabled() {
#ifdef CCL_BF16_AVX512BF_COMPILER
    static int is_enabled = -1;

    if (is_enabled == -1) {
        uint32_t reg[4];

        __asm__ __volatile__("cpuid"
                             : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                             : "a"(7), "c"(1));
        is_enabled = (reg[0] & (1 << 5)) >> 5;
    }

    return is_enabled;
#else
    return 0;
#endif
}

#ifdef CCL_BF16_COMPILER

/* float32 -> bfloat16 */
#ifdef CCL_BF16_TARGET_ATTRIBUTES
#ifdef CCL_BF16_AVX512BF_COMPILER
void convert_fp32_to_bf16(const void* src, void* dst)
    __attribute__((target("avx512bw,avx512bf16")));
#else
void convert_fp32_to_bf16(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
#endif
void convert_fp32_to_bf16(const void* src, void* dst) {
#ifdef CCL_BF16_AVX512BF_COMPILER
    if (is_avx512bf_enabled()) {
        _mm256_storeu_si256((__m256i*)(dst), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src)));
    }
    else
#endif
    {
        _mm256_storeu_si256((__m256i*)(dst),
                            _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_loadu_si512(src), 2)));
    }
}

/* bfloat16 -> float32 */
#ifdef CCL_BF16_TARGET_ATTRIBUTES
#ifdef CCL_BF16_AVX512BF_COMPILER
void convert_bf16_to_fp32(const void* src, void* dst)
    __attribute__((target("avx512bw,avx512bf16")));
#else
void convert_bf16_to_fp32(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
#endif
void convert_bf16_to_fp32(const void* src, void* dst) {
    _mm512_storeu_si512(
        dst,
        _mm512_bslli_epi128(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src)), 2));
}

void convert_fp32_to_bf16_arrays(void* send_buf, void* send_buf_bf16, int count) {
    int int_val = 0, int_val_shifted = 0;
    float* send_buf_float = (float*)send_buf;
    int limit = (count / FLOATS_IN_M512) * FLOATS_IN_M512;

    for (int i = 0; i < limit; i += FLOATS_IN_M512) {
        convert_fp32_to_bf16(send_buf_float + i, ((unsigned char*)send_buf_bf16) + (2 * i));
    }

    /* proceed remaining float's in buffer */
    for (int i = limit; i < count; i++) {
        /* iterate over send_buf_bf16 */
        int* send_bfp_tail = (int*)(((char*)send_buf_bf16) + (2 * i));
        /* copy float (4 bytes) data as is to int variable, */
        memcpy(&int_val, &send_buf_float[i], 4);
        /* then perform shift and */
        int_val_shifted = int_val >> BF16_SHIFT;
        /* save pointer to result */
        *send_bfp_tail = int_val_shifted;
    }
}

void convert_bf16_to_fp32_arrays(void* recv_buf_bf16, float* recv_buf, int count) {
    int int_val = 0, int_val_shifted = 0;
    int limit = (count / FLOATS_IN_M512) * FLOATS_IN_M512;

    for (int i = 0; i < limit; i += FLOATS_IN_M512) {
        convert_bf16_to_fp32((char*)recv_buf_bf16 + (2 * i), recv_buf + i);
    }

    /* proceed remaining bf16's in buffer */
    for (int i = limit; i < count; i++) {
        /* iterate over recv_buf_bf16 */
        int* recv_bfp_tail = (int*)((char*)recv_buf_bf16 + (2 * i));
        /* copy bf16 data as is to int variable, */
        memcpy(&int_val, recv_bfp_tail, 4);
        /* then perform shift and */
        int_val_shifted = int_val << BF16_SHIFT;
        /* copy result to output */
        memcpy((recv_buf + i), &int_val_shifted, 4);
    }
}
#else // CCL_BF16_COMPILER

void convert_fp32_to_bf16_arrays(void* send_buf, void* send_buf_bf16, int count) {
    printf("unsupported\n");
    assert(0);
}

void convert_bf16_to_fp32_arrays(void* recv_buf_bf16, float* recv_buf, int count) {
    printf("unsupported\n");
    assert(0);
}

#endif // CCL_BF16_COMPILER

// Routines to convert between fp32 and bf16 without relying on AVX instructions.
// These are useful when bf16 is only natively supported on a device.
void convert_fp32_to_bf16_arrays_generic(float* send_buf_float, void* send_buf_bf16, int count) {
    int int_val = 0, int_val_shifted = 0;

    for (int i = 0; i < count; ++i) {
        /* iterate over send_buf_bf16 */
        int* send_bfp_tail = (int*)(((char*)send_buf_bf16) + (2 * i));
        /* copy float (4 bytes) data as is to int variable, */
        memcpy(&int_val, &send_buf_float[i], 4);
        /* then perform shift and */
        int_val_shifted = int_val >> BF16_SHIFT;
        /* save pointer to result */
        *send_bfp_tail = int_val_shifted;
    }
}

void convert_bf16_to_fp32_arrays_generic(void* recv_buf_bf16, float* recv_buf_float, int count) {
    int int_val = 0, int_val_shifted = 0;

    /* proceed remaining bf16's in buffer */
    for (int i = 0; i < count; i++) {
        /* iterate over recv_buf_bf16 */
        int* recv_bfp_tail = (int*)((char*)recv_buf_bf16 + (2 * i));
        /* copy bf16 data as is to int variable, */
        memcpy(&int_val, recv_bfp_tail, 4);
        /* then perform shift and */
        int_val_shifted = int_val << BF16_SHIFT;
        /* copy result to output */
        memcpy((recv_buf_float + i), &int_val_shifted, 4);
    }
}

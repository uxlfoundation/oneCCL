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

#include "lp.hpp"

bool is_lp_datatype(ccl_data_type dtype) {
    return (dtype == DATATYPE_FLOAT16 || dtype == DATATYPE_BFLOAT16) ? true : false;
}

int is_fp16_enabled() {
#ifdef CCL_FP16_COMPILER
    static int is_fp16_enabled = -1;
    if (is_fp16_enabled == -1) {
        uint32_t reg[4];

        __asm__ __volatile__("cpuid"
                             : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                             : "a"(1));
        is_fp16_enabled = (reg[2] & (1 << 29)) >> 29;
    }
    return is_fp16_enabled;
#else
    printf("FUNC_TESTS: no FP16 compiler\n");
    return 0;
#endif
}

int is_bf16_enabled() {
#ifdef CCL_BF16_COMPILER
    static int is_bf16_enabled = -1;
    if (is_bf16_enabled == -1) {
        uint32_t reg[4];

        __asm__ __volatile__("cpuid"
                             : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                             : "a"(7), "c"(0));
        is_bf16_enabled = ((reg[1] & (1u << 16)) >> 16) & ((reg[1] & (1u << 30)) >> 30) &
                          ((reg[1] & (1u << 31)) >> 31);
    }
    return is_bf16_enabled;
#else
    printf("FUNC_TESTS: no BF16 compiler\n");
    return 0;
#endif
}

int is_avx512bf_enabled() {
#ifdef CCL_BF16_AVX512BF_COMPILER
    static int is_avx512bf_enabled = -1;
    if (is_avx512bf_enabled == -1) {
        uint32_t reg[4];
        __asm__ __volatile__("cpuid"
                             : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                             : "a"(7), "c"(1));
        is_avx512bf_enabled = (reg[0] & (1u << 5)) >> 5;
    }
    return is_avx512bf_enabled;
#else
    return 0;
#endif
}

#ifdef CCL_FP16_COMPILER
void convert_fp32_to_fp16(const void* src, void* dst) {
    // _mm256_storeu_si256((__m256i*)dst, _mm512_cvtps_ph(_mm512_loadu_ps((float*)src), 0));
    _mm_storeu_si128((__m128i*)dst,
                     _mm256_cvtps_ph((__m256)(_mm256_loadu_si256((__m256i*)src)), 0));
}
void convert_fp16_to_fp32(const void* src, void* dst) {
    // _mm512_storeu_si512(dst, (__m512i)(_mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)src))));
    _mm256_storeu_si256((__m256i*)dst, (__m256i)(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src))));
}
#else // CCL_FP16_COMPILER
void convert_fp32_to_fp16(const void* src, void* dst) {
    ASSERT(0, "FP16 is unsupported");
}
void convert_fp16_to_fp32(const void* src, void* dst) {
    ASSERT(0, "FP16 is unsupported");
}
#endif // CCL_FP16_COMPILER

#ifdef CCL_BF16_COMPILER
void convert_fp32_to_bf16(const void* src, void* dst) {
#ifdef CCL_BF16_AVX512BF_COMPILER
    if (is_avx512bf_enabled())
        _mm256_storeu_si256((__m256i*)(dst), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(src)));
    else
#endif
        _mm256_storeu_si256((__m256i*)(dst),
                            _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_loadu_si512(src), 2)));
}
void convert_bf16_to_fp32(const void* src, void* dst) {
    __m512i y = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i const*)src));
    _mm512_storeu_si512(dst, _mm512_bslli_epi128(y, 2));
}
#else // CCL_BF16_COMPILER
void convert_fp32_to_bf16(const void* src, void* dst) {
    ASSERT(0, "BF16 is unsupported");
}
void convert_bf16_to_fp32(const void* src, void* dst) {
    ASSERT(0, "BF16 is unsupported");
}
#endif // CCL_BF16_COMPILER

void convert_lp_to_fp32(const void* src, void* dst, ccl_data_type dtype) {
    if (dtype == DATATYPE_FLOAT16) {
        convert_fp16_to_fp32(src, dst);
    }
    else if (dtype == DATATYPE_BFLOAT16) {
        convert_bf16_to_fp32(src, dst);
    }
    else {
        ASSERT(0, "unexpected data_type %d", dtype);
    }
}

void convert_fp32_to_lp(const void* src, void* dst, ccl_data_type dtype) {
    if (dtype == DATATYPE_FLOAT16) {
        convert_fp32_to_fp16(src, dst);
    }
    else if (dtype == DATATYPE_BFLOAT16) {
        convert_fp32_to_bf16(src, dst);
    }
    else {
        ASSERT(0, "unexpected data_type %d", dtype);
    }
}

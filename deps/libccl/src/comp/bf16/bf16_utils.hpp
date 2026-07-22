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

#include <map>
#include <set>
#include <stdint.h>

#ifdef CCL_BF16_COMPILER
#include <immintrin.h>
#endif // CCL_BF16_COMPILER

typedef enum { ccl_bf16_scalar = 0, ccl_bf16_avx512f, ccl_bf16_avx512bf } ccl_bf16_impl_type;

extern std::map<ccl_bf16_impl_type, std::string> bf16_impl_names;

__attribute__((__always_inline__)) inline std::set<ccl_bf16_impl_type> ccl_bf16_get_impl_types() {
    std::set<ccl_bf16_impl_type> result;

    result.insert(ccl_bf16_scalar);

#ifdef CCL_BF16_COMPILER
    int is_avx512f_enabled = 0;
    int is_avx512bf_enabled = 0;

    uint32_t reg[4];

    /* baseline AVX512 capabilities for BF16 implementation */
    /* CPUID.(EAX=07H, ECX=0):EBX.AVX512F  [bit 16] */
    /* CPUID.(EAX=07H, ECX=0):EBX.AVX512BW [bit 30] */
    /* CPUID.(EAX=07H, ECX=0):EBX.AVX512VL [bit 31] */
    __asm__ __volatile__("cpuid"
                         : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                         : "a"(7), "c"(0));
    is_avx512f_enabled = ((reg[1] & (1u << 16)) >> 16) & ((reg[1] & (1u << 30)) >> 30) &
                         ((reg[1] & (1u << 31)) >> 31);

#ifdef CCL_BF16_AVX512BF_COMPILER
    /* capabilities for optimized BF16/FP32 conversions */
    /* CPUID.(EAX=07H, ECX=1):EAX[bit 05] */
    __asm__ __volatile__("cpuid"
                         : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                         : "a"(7), "c"(1));
    is_avx512bf_enabled = (reg[0] & (1 << 5)) >> 5;
#endif // CCL_BF16_AVX512BF_COMPILER

    if (is_avx512f_enabled)
        result.insert(ccl_bf16_avx512f);

    if (is_avx512bf_enabled)
        result.insert(ccl_bf16_avx512bf);
#endif // CCL_BF16_COMPILER

    return result;
}

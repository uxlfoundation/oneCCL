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

#ifdef CCL_FP16_COMPILER
#include <immintrin.h>
#endif

typedef enum {
    ccl_fp16_no_compiler_support = 0,
    ccl_fp16_no_hardware_support,
    ccl_fp16_f16c,
    ccl_fp16_avx512f,
    ccl_fp16_avx512fp16
} ccl_fp16_impl_type;

extern std::map<ccl_fp16_impl_type, std::string> fp16_impl_names;
extern std::map<ccl_fp16_impl_type, std::string> fp16_env_impl_names;

__attribute__((__always_inline__)) inline std::set<ccl_fp16_impl_type> ccl_fp16_get_impl_types() {
    std::set<ccl_fp16_impl_type> result;

#ifdef CCL_FP16_COMPILER
    int is_f16c_enabled = 0;
    int is_avx512f_enabled = 0;
    int is_avx512fp16_enabled = 0;

    uint32_t reg[4];

    /* F16C capabilities for FP16 implementation */
    /* CPUID.(EAX=01H):ECX.AVX [bit 29] */
    __asm__ __volatile__("cpuid" : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3]) : "a"(1));
    is_f16c_enabled = (reg[2] & (1 << 29)) >> 29;

    /* AVX512 capabilities for FP16 implementation */
    /* CPUID.(EAX=07H, ECX=0):EBX.AVX512F  [bit 16] */
    /* CPUID.(EAX=07H, ECX=0):EBX.AVX512BW [bit 30] */
    /* CPUID.(EAX=07H, ECX=0):EBX.AVX512VL [bit 31] */
    __asm__ __volatile__("cpuid"
                         : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                         : "a"(7), "c"(0));
    is_avx512f_enabled = ((reg[1] & (1u << 16)) >> 16) & ((reg[1] & (1u << 30)) >> 30) &
                         ((reg[1] & (1u << 31)) >> 31);

#ifdef CCL_FP16_AVX512FP16_COMPILER
    /* AVX512 capabilities for FP16 arithmetic operations support */
    /* CPUID.(EAX=07H, ECX=0):EBX.AVX512BW [bit 30] */
    /* CPUID.(EAX=07H, ECX=0):EDX.AVX512_FP16 [bit 23] */
    __asm__ __volatile__("cpuid"
                         : "=a"(reg[0]), "=b"(reg[1]), "=c"(reg[2]), "=d"(reg[3])
                         : "a"(7), "c"(0));
    is_avx512fp16_enabled = ((reg[1] & (1u << 30)) >> 30) & ((reg[3] & (1u << 23)) >> 23);
#endif // CCL_FP16_AVX512FP16_COMPILER

    if (is_avx512fp16_enabled)
        result.insert(ccl_fp16_avx512fp16);

    if (is_avx512f_enabled)
        result.insert(ccl_fp16_avx512f);

    if (is_f16c_enabled)
        result.insert(ccl_fp16_f16c);

    if (!is_avx512fp16_enabled && !is_avx512f_enabled && !is_f16c_enabled)
        result.insert(ccl_fp16_no_hardware_support);
#else
    result.insert(ccl_fp16_no_compiler_support);
#endif

    return result;
}

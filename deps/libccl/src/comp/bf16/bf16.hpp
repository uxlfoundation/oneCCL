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

#include "oneapi/ccl/types.hpp"

#ifdef CCL_BF16_TARGET_ATTRIBUTES
#ifdef CCL_BF16_AVX512BF_COMPILER
__attribute__((target("avx512bw,avx512vl,avx512bf16")))
#else
__attribute__((target("avx512bw,avx512vl")))
#endif
void ccl_bf16_reduce(const void* in_buf, size_t in_cnt,
                     void* inout_buf, size_t* out_cnt,
                     ccl::reduction reduction_op);
#else // CCL_BF16_TARGET_ATTRIBUTES
void ccl_bf16_reduce(const void* in_buf,
                     size_t in_cnt,
                     void* inout_buf,
                     size_t* out_cnt,
                     ccl::reduction reduction_op);
#endif // CCL_BF16_TARGET_ATTRIBUTES

void ccl_convert_fp32_to_bf16_arrays(void*, void*, size_t);
void ccl_convert_bf16_to_fp32_arrays(void*, float*, size_t);

#ifdef CCL_BF16_COMPILER

#ifdef CCL_BF16_TARGET_ATTRIBUTES
#ifdef CCL_BF16_AVX512BF_COMPILER
void ccl_convert_fp32_to_bf16(const void* src, void* dst)
    __attribute__((target("avx512bw,avx512bf16")));
#else
void ccl_convert_fp32_to_bf16(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
#endif // CCL_BF16_TARGET_ATTRIBUTES

#ifdef CCL_BF16_TARGET_ATTRIBUTES
#ifdef CCL_BF16_AVX512BF_COMPILER
void ccl_convert_bf16_to_fp32(const void* src, void* dst)
    __attribute__((target("avx512bw,avx512bf16")));
#else
void ccl_convert_bf16_to_fp32(const void* src, void* dst) __attribute__((target("avx512bw")));
#endif
#endif // CCL_BF16_TARGET_ATTRIBUTES

#endif // CCL_BF16_COMPILER
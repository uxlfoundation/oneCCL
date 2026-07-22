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

#include "oneapi/ccl/types.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "comp/fp16/fp16.hpp"
#include "comp/fp16/fp16_intrisics.hpp"
#include "common/utils/enums.hpp"

#define CCL_FLOATS_IN_M512 16

std::map<ccl_fp16_impl_type, std::string> fp16_impl_names = {
    std::make_pair(ccl_fp16_no_compiler_support, "no_compiler_support"),
    std::make_pair(ccl_fp16_no_hardware_support, "no_hardware_support"),
    std::make_pair(ccl_fp16_f16c, "f16c"),
    std::make_pair(ccl_fp16_avx512f, "avx512f"),
    std::make_pair(ccl_fp16_avx512fp16, "avx512fp16")
};

std::map<ccl_fp16_impl_type, std::string> fp16_env_impl_names = {
    std::make_pair(ccl_fp16_f16c, "f16c"),
    std::make_pair(ccl_fp16_avx512f, "avx512f"),
    std::make_pair(ccl_fp16_avx512fp16, "avx512fp16")
};

#ifdef CCL_FP16_COMPILER

void ccl_fp16_reduce(const void* in_buf,
                     size_t in_cnt,
                     void* inout_buf,
                     size_t* out_cnt,
                     ccl::reduction op) {
    LOG_DEBUG("FP16 reduction for ", in_cnt, " elements");

    if (out_cnt != nullptr) {
        *out_cnt = in_cnt;
    }

    ccl_fp16_reduce_impl(in_buf, inout_buf, in_cnt, op);
}

void ccl_convert_fp32_to_fp16(const void* src, void* dst) {
    _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph((__m256)_mm256_loadu_si256((__m256i*)src), 0));
}

void ccl_convert_fp16_to_fp32(const void* src, void* dst) {
    _mm256_storeu_si256((__m256i*)dst, (__m256i)_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src)));
}

#else // CCL_FP16_COMPILER

void ccl_fp16_reduce(const void* in_buf,
                     size_t in_cnt,
                     void* inout_buf,
                     size_t* out_cnt,
                     ccl::reduction op) {
    CCL_FATAL("FP16 reduction was requested but CCL was compiled w/o FP16 support");
}

void ccl_convert_fp32_to_fp16(const void* src, void* dst) {
    CCL_FATAL("FP32->FP16 conversion was requested but CCL was compiled w/o FP16 support");
}

void ccl_convert_fp16_to_fp32(const void* src, void* dst) {
    CCL_FATAL("FP16->FP32 conversion was requested but CCL was compiled w/o FP16 support");
}

#endif // CCL_FP16_COMPILER

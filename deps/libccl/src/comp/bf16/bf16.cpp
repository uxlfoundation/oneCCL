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
#include "comp/bf16/bf16.hpp"
#include "comp/bf16/bf16_intrisics.hpp"
#include "common/utils/enums.hpp"

#define CCL_FLOATS_IN_M512 16
#define CCL_BF16_SHIFT     16

std::map<ccl_bf16_impl_type, std::string> bf16_impl_names = {
    std::make_pair(ccl_bf16_scalar, "scalar"),
    std::make_pair(ccl_bf16_avx512f, "avx512f"),
    std::make_pair(ccl_bf16_avx512bf, "avx512bf")
};

typedef float (*ccl_bf16_reduction_scalar_func_ptr)(float a, float b);

inline float bf16_sum_scalar(float a, float b) {
    return a + b;
}

inline float bf16_prod_scalar(float a, float b) {
    return a * b;
}

inline float bf16_min_scalar(float a, float b) {
    return std::min(a, b);
}

inline float bf16_max_scalar(float a, float b) {
    return std::max(a, b);
}

inline uint16_t ccl_convert_fp32_to_bf16_scalar(float val) {
    uint16_t int_val = 0;
    memcpy(&int_val, reinterpret_cast<uint8_t*>(&val) + 2, sizeof(int_val));
    return int_val;
}

inline float ccl_convert_bf16_to_fp32_scalar(uint16_t val) {
    float ret = 0;
    uint32_t temp = static_cast<uint32_t>(val) << CCL_BF16_SHIFT;
    memcpy(&ret, &temp, sizeof(temp));
    return ret;
}

void ccl_bf16_reduce_scalar_impl(const void* in_buf,
                                 void* inout_buf,
                                 size_t in_count,
                                 ccl::reduction op) {
    ccl_bf16_reduction_scalar_func_ptr func = nullptr;
    switch (op) {
        case ccl::reduction::sum: func = &bf16_sum_scalar; break;
        case ccl::reduction::prod: func = &bf16_prod_scalar; break;
        case ccl::reduction::min: func = &bf16_min_scalar; break;
        case ccl::reduction::max: func = &bf16_max_scalar; break;
        default: CCL_FATAL("unexpected value ", ccl::utils::enum_to_underlying(op));
    }

    uint16_t* in_buf_int = (uint16_t*)in_buf;
    uint16_t* inout_buf_int = (uint16_t*)inout_buf;

    for (size_t i = 0; i < in_count; i++) {
        float in_value_1 = ccl_convert_bf16_to_fp32_scalar(in_buf_int[i]);
        float in_value_2 = ccl_convert_bf16_to_fp32_scalar(inout_buf_int[i]);
        float out_value = func(in_value_1, in_value_2);
        inout_buf_int[i] = ccl_convert_fp32_to_bf16_scalar(out_value);
    }
}

void ccl_bf16_reduce(const void* in_buf,
                     size_t in_count,
                     void* inout_buf,
                     size_t* out_count,
                     ccl::reduction op) {
    LOG_DEBUG("BF16 reduction for", in_count, " elements");

    if (out_count != nullptr) {
        *out_count = in_count;
    }

    auto bf16_impl_type = ccl::global_data::env().bf16_impl_type;

    if (bf16_impl_type == ccl_bf16_scalar) {
        ccl_bf16_reduce_scalar_impl(in_buf, inout_buf, in_count, op);
    }
    else {
#ifdef CCL_BF16_COMPILER
        ccl_bf16_reduce_impl(in_buf, inout_buf, in_count, op);
#else // CCL_BF16_COMPILER
        CCL_THROW("unexpected bf16_impl_type: ", bf16_impl_type);
#endif // CCL_BF16_COMPILER
    }
}

#ifdef CCL_BF16_COMPILER
void ccl_convert_fp32_to_bf16(const void* src, void* dst) {
#ifdef CCL_BF16_AVX512BF_COMPILER
    if (ccl::global_data::env().bf16_impl_type == ccl_bf16_avx512bf) {
        ccl_fp32_store_as_bf16_avx512bf(src, dst);
    }
    else
#endif // CCL_BF16_AVX512BF_COMPILER
    {
        ccl_fp32_store_as_bf16_avx512f(src, dst);
    }
}

void ccl_convert_bf16_to_fp32(const void* src, void* dst) {
    ccl_bf16_load_as_fp32(src, dst);
}
#endif // CCL_BF16_COMPILER

void ccl_convert_fp32_to_bf16_arrays(void* fp32_buf, void* bf16_buf, size_t count) {
    float* fp32_buf_float = (float*)fp32_buf;
    uint16_t* bf16_buf_int = (uint16_t*)bf16_buf;

    size_t limit = 0;

#ifdef CCL_BF16_COMPILER
    if (ccl::global_data::env().bf16_impl_type != ccl_bf16_scalar) {
        limit = (count / CCL_FLOATS_IN_M512) * CCL_FLOATS_IN_M512;
        for (size_t i = 0; i < limit; i += CCL_FLOATS_IN_M512) {
            ccl_convert_fp32_to_bf16(fp32_buf_float + i, ((unsigned char*)bf16_buf) + (2 * i));
        }
    }
#endif // CCL_BF16_COMPILER

    /* process remaining fp32 values */
    for (size_t i = limit; i < count; i++) {
        bf16_buf_int[i] = ccl_convert_fp32_to_bf16_scalar(fp32_buf_float[i]);
    }
}

void ccl_convert_bf16_to_fp32_arrays(void* bf16_buf, float* fp32_buf, size_t count) {
    uint16_t* bf16_buf_int = (uint16_t*)bf16_buf;

    size_t limit = 0;

#ifdef CCL_BF16_COMPILER
    if (ccl::global_data::env().bf16_impl_type != ccl_bf16_scalar) {
        limit = (count / CCL_FLOATS_IN_M512) * CCL_FLOATS_IN_M512;
        for (size_t i = 0; i < limit; i += CCL_FLOATS_IN_M512) {
            ccl_convert_bf16_to_fp32((char*)bf16_buf + (2 * i), fp32_buf + i);
        }
    }
#endif // CCL_BF16_COMPILER

    /* process remaining bf16 values */
    for (size_t i = limit; i < count; i++) {
        fp32_buf[i] = ccl_convert_bf16_to_fp32_scalar(bf16_buf_int[i]);
    }
}

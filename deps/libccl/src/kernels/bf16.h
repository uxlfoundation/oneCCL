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

#ifdef CCL_BF16_GPU_TRUNCATE

float __bf16_to_fp32(ushort V) {
    uint temp = convert_uint(V) << 16;
    return as_float(temp);
}

ushort __fp32_to_bf16(float V) {
    ushort2 temp = as_ushort2(V);
    return temp.s1;
}

#else // CCL_BF16_GPU_TRUNCATE

#ifdef cl_intel_bfloat16_conversions
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable
#else // cl_intel_bfloat16_conversions
#error "cl_intel_bfloat16_conversions are not defined, compilation failed."
#endif // cl_intel_bfloat16_conversions

float __bf16_to_fp32(ushort V) {
    return intel_convert_as_bfloat16_float(V);
}

ushort __fp32_to_bf16(float V) {
    return intel_convert_bfloat16_as_ushort(V);
}

#endif // CCL_BF16_GPU_TRUNCATE

#define DEFINE_BF16SUM_OP(T) \
    T __bf16_sum_##T(T lhs, T rhs) { \
        return __fp32_to_bf16(__bf16_to_fp32(lhs) + __bf16_to_fp32(rhs)); \
    }

#define DEFINE_BF16PROD_OP(T) \
    T __bf16_prod_##T(T lhs, T rhs) { \
        return __fp32_to_bf16(__bf16_to_fp32(lhs) * __bf16_to_fp32(rhs)); \
    }

#define DEFINE_BF16MIN_OP(T) \
    T __bf16_min_##T(T lhs, T rhs) { \
        return __fp32_to_bf16(min(__bf16_to_fp32(lhs), __bf16_to_fp32(rhs))); \
    }

#define DEFINE_BF16MAX_OP(T) \
    T __bf16_max_##T(T lhs, T rhs) { \
        return __fp32_to_bf16(max(__bf16_to_fp32(lhs), __bf16_to_fp32(rhs))); \
    }

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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef CCL_FP16_GPU_TRUNCATE
/*
Truncation routines for converting fp32 <-> fp16

fp16 has 1 sign bit, 5 exponent bits and 10 significand bits with exponent
offset 15 - https://en.wikipedia.org/wiki/Half-precision_floating-point_format

For fp16 -> fp32

The sign & significand bits are unchanged, but the exponent must be properly
re-offset (i.e. convert the fp16 offset -> fp32 offset). Care must also be taken
to saturate the fp32 result if the fp16 result is saturated. Denormals must be
flushed to 0.

For fp32 -> fp16

Similar to fp16 -> fp32 except that the exponent must be checked for saturation
since the range of the exponent is signficantly smaller than that of fp32.
*/
float __fp16_to_fp32(half V) {
    uint ans_bits = 0;
    uint exp_bits = as_ushort(V) & 0x7C00;
    uint significand_bits = as_ushort(V) & 0x03FF;
    if (exp_bits == 0x7C00) {
        ans_bits = ((as_ushort(V) & 0x8000) << 16) | 0x7F800000 | (significand_bits << 13);
    }
    else if (exp_bits == 0x0000) {
        if (significand_bits != 0x00000000) {
            ans_bits = ((as_ushort(V) & 0x8000) << 16);
        }
        else {
            ans_bits = ((as_ushort(V) & 0x8000) << 16) | (significand_bits << 13);
        }
    }
    else {
        ans_bits = ((as_ushort(V) & 0x8000) << 16) | ((exp_bits + 0x1C000) << 13) |
                   (significand_bits << 13);
    }
    return as_float(ans_bits);
}

half __fp32_to_fp16(float V) {
    ushort ans;
    uint exp_bits = (as_uint(V) & 0x7F800000);
    uint significand_bits = (as_uint(V) & 0x007FFFFF);
    if (exp_bits == 0x00000000) {
        ans = (as_uint(V) & 0x80000000) >> 16;
    }
    else if (exp_bits == 0x7F800000) {
        if (significand_bits != 0) {
            ans = ((as_uint(V) & 0x80000000) >> 16) | 0x00007C01;
        }
        else {
            ans = ((as_uint(V) & 0x80000000) >> 16) | 0x00007C00;
        }
    }
    else if (exp_bits < 0x38800000) {
        ans = 0xFC00;
    }
    else if (exp_bits > 0x47000000) {
        ans = 0x7C00;
    }
    else {
        ans = ((as_uint(V) & 0x80000000) >> 16) |
              ((((as_uint(V) & 0x7F800000) >> 23) - 112) << 10) | ((as_uint(V) & 0x007FFFFF) >> 13);
    }
    return as_half(ans);
}

#define DEFINE_FP16SUM_OP(T) \
    T __sum_##T(T lhs, T rhs) { \
        return __fp32_to_fp16(__fp16_to_fp32(lhs) + __fp16_to_fp32(rhs)); \
    }

#define DEFINE_FP16PROD_OP(T) \
    T __prod_##T(T lhs, T rhs) { \
        return __fp32_to_fp16(__fp16_to_fp32(lhs) * __fp16_to_fp32(rhs)); \
    }

#define DEFINE_FP16MIN_OP(T) \
    T __min_##T(T lhs, T rhs) { \
        return __fp32_to_fp16(min(__fp16_to_fp32(lhs), __fp16_to_fp32(rhs))); \
    }

#define DEFINE_FP16MAX_OP(T) \
    T __max_##T(T lhs, T rhs) { \
        return __fp32_to_fp16(max(__fp16_to_fp32(lhs), __fp16_to_fp32(rhs))); \
    }

#else // CCL_FP16_GPU_TRUNCATE

#define DEFINE_FP16SUM_OP(T) \
    T __sum_##T(T lhs, T rhs) { \
        return lhs + rhs; \
    }

#define DEFINE_FP16PROD_OP(T) \
    T __prod_##T(T lhs, T rhs) { \
        return lhs * rhs; \
    }

#define DEFINE_FP16MIN_OP(T) \
    T __min_##T(T lhs, T rhs) { \
        return min(lhs, rhs); \
    }

#define DEFINE_FP16MAX_OP(T) \
    T __max_##T(T lhs, T rhs) { \
        return max(lhs, rhs); \
    }

#endif // CCL_FP16_GPU_TRUNCATE

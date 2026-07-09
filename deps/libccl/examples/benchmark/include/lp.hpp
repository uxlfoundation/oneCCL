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

#include "oneapi/ccl/lp_types.hpp"

ccl::float16 fp32_to_fp16(float val) {
    uint32_t ans;
    uint32_t* val_ptr = (reinterpret_cast<uint32_t*>(&val));
    uint32_t exp_bits = (*val_ptr & 0x7F800000);
    uint32_t significand_bits = (*val_ptr & 0x007FFFFF);
    if (exp_bits == 0x00000000) {
        ans = (*val_ptr & 0x80000000) >> 16;
    }
    else if (exp_bits == 0x7F800000) {
        if (significand_bits != 0) {
            ans = ((*val_ptr & 0x80000000) >> 16) | 0x00007C01;
        }
        else {
            ans = ((*val_ptr & 0x80000000) >> 16) | 0x00007C00;
        }
    }
    else if (exp_bits < 0x38800000) {
        ans = 0xFC00;
    }
    else if (exp_bits > 0x47000000) {
        ans = 0x7C00;
    }
    else {
        ans = ((*val_ptr & 0x80000000) >> 16) | ((((*val_ptr & 0x7F800000) >> 23) - 112) << 10) |
              ((*val_ptr & 0x007FFFFF) >> 13);
    }
    return ccl::float16(ans);
}

float fp16_to_fp32(ccl::float16 val) {
    uint16_t val_data = val.get_data();
    float ans = 0.0f;
    uint32_t ans_bits = 0;
    uint32_t exp_bits = val_data & 0x7C00;
    uint32_t significand_bits = val_data & 0x03FF;
    if (exp_bits == 0x7C00) {
        ans_bits = ((val_data & 0x8000) << 16) | 0x7F800000 | (significand_bits << 13);
    }
    else if (exp_bits == 0x0000) {
        if (significand_bits != 0x00000000) {
            ans_bits = ((val_data & 0x8000) << 16);
        }
        else {
            ans_bits = ((val_data & 0x8000) << 16) | (significand_bits << 13);
        }
    }
    else {
        ans_bits =
            ((val_data & 0x8000) << 16) | ((exp_bits + 0x1C000) << 13) | (significand_bits << 13);
    }
    std::memcpy(reinterpret_cast<void*>(&ans), reinterpret_cast<void*>(&ans_bits), 4);
    return ans;
}

ccl::bfloat16 fp32_to_bf16(float val) {
    // Truncate
    uint16_t int_val = 0;
    memcpy(&int_val, reinterpret_cast<uint8_t*>(&val) + 2, 2);
    return ccl::bfloat16(int_val);
}

float bf16_to_fp32(ccl::bfloat16 val) {
    float ret = 0;
    uint32_t temp = static_cast<uint32_t>(val.get_data()) << 16;
    memcpy(&ret, &temp, sizeof(temp));
    return ret;
}

std::ostream& operator<<(std::ostream& out, const ccl::float16& v) {
    out << fp16_to_fp32(v) << "|" << v.get_data();
    return out;
}

std::ostream& operator<<(std::ostream& out, const ccl::bfloat16& v) {
    out << bf16_to_fp32(v) << "|" << v.get_data();
    return out;
}

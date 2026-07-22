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

#ifdef CCL_FP16_TARGET_ATTRIBUTES
void ccl_fp16_reduce(const void* in_buf,
                     size_t in_cnt,
                     void* inout_buf,
                     size_t* out_cnt,
                     ccl::reduction reduction_op);
__attribute__((target("f16c"))) void ccl_convert_fp32_to_fp16(const void* src, void* dst);
__attribute__((target("f16c"))) void ccl_convert_fp16_to_fp32(const void* src, void* dst);
#else // CCL_FP16_TARGET_ATTRIBUTES
void ccl_fp16_reduce(const void* in_buf,
                     size_t in_cnt,
                     void* inout_buf,
                     size_t* out_cnt,
                     ccl::reduction reduction_op);
void ccl_convert_fp32_to_fp16(const void* src, void* dst);
void ccl_convert_fp16_to_fp32(const void* src, void* dst);
#endif // CCL_FP16_TARGET_ATTRIBUTES

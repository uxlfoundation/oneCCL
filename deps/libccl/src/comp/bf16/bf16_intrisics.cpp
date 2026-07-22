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

#include "comp/bf16/bf16_intrisics.hpp"

#ifdef CCL_BF16_COMPILER

BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_sum_wrap(__m512 a, __m512 b) {
    return _mm512_add_ps(a, b);
}

BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_prod_wrap(__m512 a, __m512 b) {
    return _mm512_mul_ps(a, b);
}

BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_min_wrap(__m512 a, __m512 b) {
    return _mm512_min_ps(a, b);
}

BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_max_wrap(__m512 a, __m512 b) {
    return _mm512_max_ps(a, b);
}

BF16_TARGET_ATTRIBUTE_BWF __m512 bf16_reduce(__m512 a, __m512 b, ccl_bf16_reduction_func_ptr op) {
    return (*op)(a, b);
}

#endif // CCL_BF16_COMPILER

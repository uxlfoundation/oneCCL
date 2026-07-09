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

#include "comp/fp16/fp16_intrisics.hpp"

#ifdef CCL_FP16_COMPILER

CCL_FP16_DEFINE_ELEM_FUNCS(256);
CCL_FP16_DEFINE_ELEM_FUNCS(512);

#ifdef CCL_FP16_AVX512FP16_COMPILER
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_sum_wrap_512FP16(__m512h a, __m512h b) {
    return _mm512_add_ph(a, b);
}
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_prod_wrap_512FP16(__m512h a, __m512h b) {
    return _mm512_mul_ph(a, b);
}
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_min_wrap_512FP16(__m512h a, __m512h b) {
    return _mm512_min_ph(a, b);
}
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_max_wrap_512FP16(__m512h a, __m512h b) {
    return _mm512_max_ph(a, b);
}
FP16_TARGET_ATTRIBUTE_512FP16 __m512h fp16_reduce_512FP16(__m512h a,
                                                          __m512h b,
                                                          ccl_fp16_reduction_func_ptr_512FP16 op) {
    return (*op)(a, b);
}
#endif // CCL_FP16_AVX512FP16_COMPILER

#endif // CCL_FP16_COMPILER

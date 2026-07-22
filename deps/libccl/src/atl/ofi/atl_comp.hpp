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

#include "comp/fp16/fp16.hpp"
#include "comp/bf16/bf16.hpp"

#define ATL_REDUCE(type) \
    do { \
        type* in_buf_##type = (type*)in_buf; \
        type* inout_buf_##type = (type*)inout_buf; \
        switch (reduction) { \
            case ATL_REDUCTION_SUM: \
                for (i = 0; i < in_count; i++) { \
                    inout_buf_##type[i] += in_buf_##type[i]; \
                } \
                break; \
            case ATL_REDUCTION_PROD: \
                for (i = 0; i < in_count; i++) { \
                    inout_buf_##type[i] *= in_buf_##type[i]; \
                } \
                break; \
            case ATL_REDUCTION_MIN: \
                for (i = 0; i < in_count; i++) { \
                    inout_buf_##type[i] = std::min(in_buf_##type[i], inout_buf_##type[i]); \
                } \
                break; \
            case ATL_REDUCTION_MAX: \
                for (i = 0; i < in_count; i++) { \
                    inout_buf_##type[i] = std::max(in_buf_##type[i], inout_buf_##type[i]); \
                } \
                break; \
            default: CCL_FATAL("unexpected reduction op value ", reduction); \
        } \
    } while (0)

atl_status_t atl_comp_reduce_regular(const void* in_buf,
                                     size_t in_count,
                                     void* inout_buf,
                                     size_t* out_count,
                                     const atl_datatype_t dtype,
                                     atl_reduction_t reduction) {
    size_t i;
    switch (dtype) {
        case ATL_DTYPE_INT8: ATL_REDUCE(int8_t); break;
        case ATL_DTYPE_UINT8: ATL_REDUCE(uint8_t); break;
        case ATL_DTYPE_INT16: ATL_REDUCE(int16_t); break;
        case ATL_DTYPE_UINT16: ATL_REDUCE(uint16_t); break;
        case ATL_DTYPE_INT32: ATL_REDUCE(int32_t); break;
        case ATL_DTYPE_UINT32: ATL_REDUCE(uint32_t); break;
        case ATL_DTYPE_INT64: ATL_REDUCE(int64_t); break;
        case ATL_DTYPE_UINT64: ATL_REDUCE(uint64_t); break;
        case ATL_DTYPE_FLOAT16: {
            ccl::reduction reduction_op = static_cast<ccl::reduction>(reduction);
            ccl_fp16_reduce(in_buf, in_count, inout_buf, out_count, reduction_op);
            break;
        }
        case ATL_DTYPE_FLOAT32: ATL_REDUCE(float); break;
        case ATL_DTYPE_FLOAT64: ATL_REDUCE(double); break;
        case ATL_DTYPE_BFLOAT16: {
            ccl::reduction reduction_op = static_cast<ccl::reduction>(reduction);
            ccl_bf16_reduce(in_buf, in_count, inout_buf, out_count, reduction_op);
            break;
        }
        default: CCL_FATAL("unexpected value ", dtype); break;
    }

    return ATL_STATUS_SUCCESS;
}

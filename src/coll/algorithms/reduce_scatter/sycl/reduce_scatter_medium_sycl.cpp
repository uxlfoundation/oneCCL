/*
 Copyright 2016-2020 Intel Corporation
 
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
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_medium_sycl.hpp"

sycl_reduce_scatter_medium<sycl::half> rs_medium_fp16;
sycl_reduce_scatter_medium<sycl::_V1::ext::oneapi::bfloat16> rs_medium_bf16;
sycl_reduce_scatter_medium<int32_t> rs_medium_int32;
sycl_reduce_scatter_medium<float> rs_medium_fp32;

#define SWITCH_INIT_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        if (!rs_medium_##TYPE.inited()) { \
            LOG_INFO("invoking medium reduce_scatter first time for datatype: ", ccl_type); \
            rs_medium_##TYPE.init(queue, comm, stream, rank_in, world_in); \
        } \
        break;

void init_reduce_scatter_medium(ccl::datatype dtype,
                                sycl::queue &queue,
                                ccl_comm *comm,
                                ccl_stream *stream,
                                uint32_t rank_in,
                                uint32_t world_in) {
    switch (dtype) {
        SWITCH_INIT_TYPE(fp16, ccl::datatype::float16)
        SWITCH_INIT_TYPE(bf16, ccl::datatype::bfloat16)
        SWITCH_INIT_TYPE(fp32, ccl::datatype::float32)
        SWITCH_INIT_TYPE(int32, ccl::datatype::int32)
        default: assert(0);
    }
}

#define SWITCH_RUN_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        e = rs_medium_##TYPE.reduce_scatter(queue, send_buf, recv_buf, dtype, recv_count, reduction, deps, done); \
        break;

ccl::event run_reduce_scatter_medium(ccl::datatype dtype,
                                     sycl::queue &queue,
                                     const void *send_buf,
                                     void *recv_buf,
                                     size_t recv_count,
                                     ccl::reduction reduction,
                                     const ccl::vector_class<ccl::event> &deps,
                                     bool &done) {
    ccl::event e;
    switch (dtype) {
        SWITCH_RUN_TYPE(fp16, ccl::datatype::float16)
        SWITCH_RUN_TYPE(bf16, ccl::datatype::bfloat16)
        SWITCH_RUN_TYPE(fp32, ccl::datatype::float32)
        SWITCH_RUN_TYPE(int32, ccl::datatype::int32)
        default: assert(0);
    }
    return e;
}

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
#include "coll/algorithms/allreduce/sycl/allreduce_medium_sycl.hpp"

#define MAX_RANK 16

void *allreduce_medium_buffer = NULL;
void *allreduce_medium_buffers[MAX_RANK];
void *allreduce_medium_sync_buffer[MAX_RANK];
size_t allreduce_medium_offsets[MAX_RANK];
ze_ipc_mem_handle_t allreduce_medium_ipc_handle[MAX_RANK];
int allreduce_medium_buffer_index = 0;

#define ALLREDUCE_MEDIUM_API_DECL(TYPE) \
    void init_allreduce_medium_##TYPE(ccl::datatype dtype, \
                                      sycl::queue &queue, \
                                      ccl_comm *comm, \
                                      ccl_stream *stream, \
                                      uint32_t rank_in, \
                                      uint32_t world_in); \
    ccl::event run_allreduce_medium_##TYPE(ccl::datatype dtype, \
                                           sycl::queue &queue, \
                                           const void *in_buf, \
                                           void *out_buf, \
                                           size_t count, \
                                           ccl::reduction reduction, \
                                           const ccl::vector_class<ccl::event> &deps, \
                                           bool &done)

ALLREDUCE_MEDIUM_API_DECL(fp16);
ALLREDUCE_MEDIUM_API_DECL(bf16);
ALLREDUCE_MEDIUM_API_DECL(fp32);
ALLREDUCE_MEDIUM_API_DECL(int32);

#define SWITCH_INIT_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        init_allreduce_medium_##TYPE(dtype, queue, comm, stream, rank_in, world_in); \
        break;

void init_allreduce_medium(ccl::datatype dtype,
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
        default: CCL_THROW("unsupported datatype for allreduce"); assert(0);
    }
}

#define SWITCH_RUN_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        e = run_allreduce_medium_##TYPE( \
            dtype, queue, in_buf, out_buf, count, reduction, deps, done); \
        break;

ccl::event run_allreduce_medium(ccl::datatype dtype,
                                sycl::queue &queue,
                                const void *in_buf,
                                void *out_buf,
                                size_t count,
                                ccl::reduction reduction,
                                const ccl::vector_class<ccl::event> &deps,
                                bool &done) {
    ccl::event e;
    switch (dtype) {
        SWITCH_RUN_TYPE(fp16, ccl::datatype::float16)
        SWITCH_RUN_TYPE(bf16, ccl::datatype::bfloat16)
        SWITCH_RUN_TYPE(fp32, ccl::datatype::float32)
        SWITCH_RUN_TYPE(int32, ccl::datatype::int32)
        default: CCL_THROW("unsupported datatype for allreduce"); assert(0);
    }
    return e;
}

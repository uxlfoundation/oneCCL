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
#include "coll/algorithms/allreduce/sycl/allreduce_small_sycl.hpp"

#define MAX_RANK 16

void *allreduce_small_buffer = NULL;
void *allreduce_small_buffers[MAX_RANK];
void *allreduce_small_sync_buffer[MAX_RANK];
size_t allreduce_small_offsets[MAX_RANK];
ze_ipc_mem_handle_t allreduce_small_ipc_handle[MAX_RANK];
int allreduce_small_buffer_index = 0;

#define SWITCH_INIT_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        if (!ar_small_##TYPE.inited()) { \
            LOG_INFO("invoking small allreduce first time for datatype: ", ccl_type); \
            ar_small_##TYPE.init(queue, comm, stream, rank_in, world_in); \
        } \
        break;

#define SWITCH_RUN_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        e = ar_small_##TYPE.allreduce( \
            queue, in_buf, out_buf, dtype, count, reduction, deps, done); \
        break;

#define SWITCH_TYPE_UNSUPPORTED(TYPE, ccl_type) \
    case ccl_type: \
        fprintf(stderr, "allreduce with bf16 not supported!\n"); \
        CCL_THROW("allreduce with bf16 not supported"); \
        break;

#include "coll/algorithms/allreduce/sycl/allreduce_small_sycl.hpp"

sycl_allreducer_small<sycl::half> ar_small_fp16;
sycl_allreducer_small<sycl::_V1::ext::oneapi::bfloat16> ar_small_bf16;
sycl_allreducer_small<float> ar_small_fp32;
sycl_allreducer_small<int> ar_small_int32;

void init_allreduce_small(ccl::datatype dtype,
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

ccl::event run_allreduce_small(ccl::datatype dtype,
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

#include "coll/algorithms/allreduce/sycl/allreduce_small_sycl_impl.hpp"
ccl::event allreduce_small(const void *send_buf,
                           void *recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           ccl::reduction reduction,
                           ccl_comm *comm,
                           ccl_stream *global_stream,
                           const ccl::vector_class<ccl::event> &deps) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("|MT|: invoking allreduce_small");
        coll_initExt(comm, ccl::global_data::get().shared_data->hash_table, global_stream);
    }
    else {
        LOG_DEBUG("invoking allreduce_small");
        coll_init(comm, global_stream);
    }

    auto lambda = [&]<typename T, int NE, int NP>() {
        return allreduce_small_impl<T, NE, NP>(
            send_buf, recv_buf, count, dtype, reduction, comm, global_stream, deps);
    };

    return invoke_collective(lambda, comm, dtype);
}

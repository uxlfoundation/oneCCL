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
#include "coll/algorithms/allgatherv/sycl/allgatherv_small_sycl.hpp"

sycl_allgatherv_small<sycl::half> agv_small_fp16;
sycl_allgatherv_small<sycl::_V1::ext::oneapi::bfloat16> agv_small_bf16;
sycl_allgatherv_small<float> agv_small_fp32;
sycl_allgatherv_small<int> agv_small_int32;

#define SWITCH_INIT_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        if (!agv_small_##TYPE.inited()) { \
            LOG_INFO("invoking allgatherv small kernel first time for datatype: ", ccl_type); \
            agv_small_##TYPE.init(queue, comm, stream, rank_in, world_in); \
        } \
        break;

void init_allgatherv_small(ccl::datatype dtype,
                           sycl::queue& queue,
                           ccl_comm* comm,
                           ccl_stream* stream,
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
        e = agv_small_##TYPE.allgatherv(queue, send_buf, send_count, recv_buf, recv_counts, done); \
        break;

ccl::event run_allgatherv_small(ccl::datatype dtype,
                                sycl::queue& queue,
                                const void* send_buf,
                                size_t send_count,
                                void* recv_buf,
                                const ccl::vector_class<size_t>& recv_counts,
                                bool& done) {
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

#include "coll/algorithms/allgatherv/sycl/allgatherv_small_sycl_impl.hpp"

ccl::event allgatherv_small(const void* send_buf,
                            size_t send_count,
                            void* recv_buf,
                            const ccl::vector_class<size_t>& recv_counts,
                            const ccl::vector_class<size_t>& offsets,
                            ccl::datatype dtype,
                            ccl_comm* comm,
                            ccl_stream* global_stream,
                            const ccl::vector_class<ccl::event>& deps) {
    LOG_DEBUG("invoking allgatherv_small");

    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("|MT|: invoking allgatherv_small");
        coll_initExt(comm, ccl::global_data::get().shared_data->hash_table, global_stream);
    }
    else {
        LOG_DEBUG("invoking allgatherv_small");
        coll_init(comm, global_stream);
    }
    auto lambda = [&]<typename T, int NE, int NP>() {
        return allgatherv_small_impl<T, NE, NP>(
            send_buf, send_count, recv_buf, recv_counts, offsets, dtype, comm, global_stream, deps);
    };
    return invoke_collective(lambda, comm, dtype);
}

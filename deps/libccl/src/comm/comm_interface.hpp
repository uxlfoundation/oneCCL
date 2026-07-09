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
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/event.hpp"

#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"

#include "oneapi/ccl/stream_attr_ids.hpp"
#include "oneapi/ccl/stream_attr_ids_traits.hpp"
#include "oneapi/ccl/stream.hpp"

#include "comm/comm_selector.hpp"
#include "internal_types.hpp"

namespace native {
struct ccl_device;
}

namespace ccl {
namespace v1 {
class allgather_attr;
class allgatherv_attr;
class allreduce_attr;
class alltoall_attr;
class alltoallv_attr;
class barrier_attr;
class broadcast_attr;
class pt2pt_attr;
class reduce_attr;
class reduce_scatter_attr;
class scatter_attr;
} // namespace v1
} // namespace ccl

#include "types_generator_defines.hpp"

#define COMM_INTERFACE_COLL_METHODS(TYPE) \
\
    COMM_INTERFACE_COLL_##TYPE##__VOID; \
    COMM_INTERFACE_COLL_##TYPE(int8_t); \
    COMM_INTERFACE_COLL_##TYPE(uint8_t); \
    COMM_INTERFACE_COLL_##TYPE(int16_t); \
    COMM_INTERFACE_COLL_##TYPE(uint16_t); \
    COMM_INTERFACE_COLL_##TYPE(int32_t); \
    COMM_INTERFACE_COLL_##TYPE(uint32_t); \
    COMM_INTERFACE_COLL_##TYPE(int64_t); \
    COMM_INTERFACE_COLL_##TYPE(uint64_t); \
    COMM_INTERFACE_COLL_##TYPE(float); \
    COMM_INTERFACE_COLL_##TYPE(double);

#define SYCL_COMM_INTERFACE_COLL_METHODS(TYPE) \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<int8_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<uint8_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<int16_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<uint16_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<int32_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<uint32_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<int64_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<uint64_t COMMA 1>); \
    /*COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<ccl::float16 COMMA 1>);*/ \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<float COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<double COMMA 1>); \
    /*COMM_INTERFACE_COLL_CLASS_##TYPE(sycl::buffer<ccl::bfloat16 COMMA 1>);*/

#define COMM_INTERFACE_COLL_INSTANTIATION(COMM) \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, int8_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, uint8_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, int16_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, uint16_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, int32_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, uint32_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, int64_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, uint64_t); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, float); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, double); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, ccl::bfloat16); \
    COMM_INTERFACE_COLL_INSTANTIATIONS(COMM, ccl::float16);

#define SYCL_COMM_INTERFACE_COLL_INSTANTIATION(COMM) \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, sycl::buffer<int8_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, sycl::buffer<int32_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, sycl::buffer<int64_t COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, sycl::buffer<uint64_t COMMA 1>); \
    /*COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, sycl::buffer<ccl::float16 COMMA 1>);*/ \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, sycl::buffer<float COMMA 1>); \
    COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, sycl::buffer<double COMMA 1>); \
    /*COMM_INTERFACE_COLL_CLASS_INSTANTIATIONS(COMM, sycl::buffer<ccl::bfloat16 COMMA 1>);*/

namespace ccl {
struct comm_interface : public comm_selector {
    virtual ~comm_interface() = default;

    virtual int rank() const = 0;
    virtual int size() const = 0;

    virtual ccl::comm_interface_ptr split(int color, int key, bool split_external_use = false) = 0;

    virtual void finalize() = 0;

    // collectives operation declarations
    virtual ccl::event barrier(const stream::impl_value_t& op_stream,
                               const barrier_attr& attr,
                               const vector_class<event>& deps = {}) = 0;

    COMM_INTERFACE_COLL_METHODS(DECLARATION);
#ifdef CCL_ENABLE_SYCL
    SYCL_COMM_INTERFACE_COLL_METHODS(DECLARATION);
#endif // CCL_ENABLE_SYCL
};
} // namespace ccl

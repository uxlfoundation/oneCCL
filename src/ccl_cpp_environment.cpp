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
#include "environment_impl.hpp"
#include "common/global/global.hpp"
#include "exec/exec.hpp"
#include "common/utils/version.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "comm/comm_interface.hpp"
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include <memory>

namespace ccl {

namespace detail {

environment::environment() {
    static auto result = global_data::get().init();
    CCL_CHECK_AND_THROW(result, "failed to initialize CCL");
}

environment::~environment() {}

CCL_API environment& environment::instance() {
    static environment env;
    return env;
}

ccl::library_version CCL_API environment::get_library_version() {
    return utils::get_library_version();
}

/******************** KVS ********************/

shared_ptr_class<kvs> environment::create_main_kvs(const kvs_attr& attr) const {
    return std::shared_ptr<kvs>(new kvs(attr));
}

shared_ptr_class<kvs> environment::create_kvs(const kvs::address_type& addr,
                                              const kvs_attr& attr) const {
    return std::shared_ptr<kvs>(new kvs(addr, attr));
}

/******************** DEVICE ********************/

device environment::create_device(empty_t empty) const {
    static typename ccl::unified_device_type::ccl_native_t default_native_device;
    return device::create_device(default_native_device);
}

/******************** CONTEXT ********************/

context environment::create_context(empty_t empty) const {
    static typename ccl::unified_context_type::ccl_native_t default_native_context;
    return context::create_context(default_native_context);
}

/******************** DATATYPE ********************/

ccl::datatype environment::register_datatype(const datatype_attr& attr) {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    LOG_DEBUG("register datatype");

    return ccl::global_data::get().dtypes->create(attr);
}

void environment::deregister_datatype(ccl::datatype dtype) {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    LOG_DEBUG("deregister datatype");

    ccl::global_data::get().dtypes->free(dtype);
}

size_t environment::get_datatype_size(ccl::datatype dtype) const {
    while (unlikely(ccl::global_data::get().executor->is_locked)) {
        std::this_thread::yield();
    }

    return ccl::global_data::get().dtypes->get(dtype).size();
}

/*************** CUSTOM REDUCTIONS ****************/

void environment::reduction_create_pre_mul_sum(reduction* rtype,
                                               void* scalar,
                                               datatype dtype,
                                               scalar_residence_type residence,
                                               const communicator&) {
    reduction_create_pre_mul_sum_impl(rtype, scalar, dtype, residence);
}

void environment::reduction_destroy(reduction rtype, const communicator&) {
    reduction_destroy_impl(rtype);
}

/******************** COMMUNICATOR ********************/

communicator environment::create_communicator(const comm_attr& attr) const {
    return communicator::create_communicator(attr);
}

communicator environment::create_communicator(const size_t size,
                                              ccl::shared_ptr_class<kvs_interface> kvs,
                                              const comm_attr& attr) const {
    return communicator::create_communicator(size, kvs, attr);
}

communicator environment::create_communicator(const size_t size,
                                              const int rank,
                                              ccl::shared_ptr_class<kvs_interface> kvs,
                                              const comm_attr& attr) const {
    return communicator::create_communicator(size, rank, kvs, attr);
}

communicator environment::split_communicator(const communicator& comm, int color, int key) const {
    return communicator::split_communicator(comm, color, key);
}

} // namespace detail

} // namespace ccl

/******************** TypeGenerations ********************/

CREATE_COMM_INSTANTIATION(ccl::device, ccl::context)

CREATE_STREAM_INSTANTIATION(typename ccl::unified_stream_type::ccl_native_t)
CREATE_CONTEXT_INSTANTIATION(typename ccl::unified_context_type::ccl_native_t)
CREATE_DEVICE_INSTANTIATION(typename ccl::unified_device_type::ccl_native_t)

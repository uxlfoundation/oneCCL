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
#include "comm/comm_selector.hpp"
#include "comm/comm_interface.hpp"

#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"

#include "comm/comm_common_attr.hpp"
#include "comm/comm_impl.hpp"
#include "comm/comm_split_common_attr.hpp"
#include "common/global/global.hpp"

#include "comm_attr_impl.hpp"
#include "comm_split_attr_impl.hpp"

#ifdef CCL_ENABLE_STUB_BACKEND
#include "comm/stub_comm.hpp"
#endif

#include "kvs_impl.hpp"

namespace ccl {

comm_interface_ptr comm_selector::create_comm_impl() {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "host communicator is only supported for native backend");

    return comm_interface_ptr(new ccl_comm());
}

comm_interface_ptr comm_selector::create_comm_impl(const comm_attr& attr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "host communicator is only supported for native backend");

    ccl_comm_attr_impl internal_attr(attr);
    return comm_interface_ptr(new ccl_comm(internal_attr));
}

comm_interface_ptr comm_selector::create_comm_impl(const size_t size,
                                                   shared_ptr_class<kvs_interface> kvs,
                                                   const comm_attr& attr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "host communicator is only supported for native backend");

    ccl_comm_attr_impl internal_attr(attr);
    return comm_interface_ptr(ccl_comm::create(size, std::move(kvs), internal_attr));
}

comm_interface_ptr comm_selector::create_comm_impl(const size_t size,
                                                   const int rank,
                                                   shared_ptr_class<kvs_interface> kvs,
                                                   const comm_attr& attr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "host communicator is only supported for native backend");

    ccl_comm_attr_impl internal_attr(attr);
    return comm_interface_ptr(ccl_comm::create(size, rank, std::move(kvs), internal_attr));
}

comm_interface_ptr comm_selector::create_comm_impl(const size_t size,
                                                   const int rank,
                                                   device_t device,
                                                   context_t context,
                                                   shared_ptr_class<kvs_interface> kvs,
                                                   const comm_attr& attr) {
    ccl_comm_attr_impl internal_attr(attr);
#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    if (ccl::global_data::env().backend == backend_mode::native) {
        if (device.get_native().is_gpu()) {
            CCL_THROW_IF_NOT(ccl::global_data::get().ze_data, "ze_data was not initialized");
        }
    }
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_STUB_BACKEND
    if (ccl::global_data::env().backend == backend_mode::stub) {
        // does not pass comm_attr to stub_comm for now
        return comm_interface_ptr(
            ccl::stub_comm::create(device, context, size, rank, std::move(kvs)));
    }
#endif // CCL_ENABLE_STUB_BACKEND

    return comm_interface_ptr(
        ccl_comm::create(device, context, size, rank, std::move(kvs), internal_attr));
}

comm_interface_ptr comm_selector::create_comm_implExt() {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "host communicator is only supported for native backend");

    return comm_interface_ptr(new ccl_comm());
}

comm_interface_ptr comm_selector::create_comm_implExt(const size_t size,
                                                      shared_ptr_class<kvs_interface> kvs,
                                                      const comm_attr& attr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "host communicator is only supported for native backend");
    ccl_comm_attr_impl internal_attr(attr);
    return comm_interface_ptr(ccl_comm::create(size, kvs, internal_attr));
}

comm_interface_ptr comm_selector::create_comm_implExt(const size_t size,
                                                      const int rank,
                                                      shared_ptr_class<kvs_interface> kvs,
                                                      const comm_attr& attr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::native,
                     "host communicator is only supported for native backend");
    ccl_comm_attr_impl internal_attr(attr);
    return comm_interface_ptr(ccl_comm::create(size, rank, std::move(kvs), internal_attr));
}

comm_interface_ptr comm_selector::create_comm_implExt(const size_t size,
                                                      const int rank,
                                                      device_t device,
                                                      context_t context,
                                                      shared_ptr_class<kvs_interface> kvs,
                                                      const comm_attr& attr) {
    ccl_comm_attr_impl internal_attr(attr);
#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    if (ccl::global_data::env().backend == backend_mode::native) {
        if (device.get_native().is_gpu()) {
            CCL_THROW_IF_NOT(ccl::global_data::get().ze_data, "ze_data was not initialized");
        }
    }
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_STUB_BACKEND
    if (ccl::global_data::env().backend == backend_mode::stub) {
        // does not pass comm_attr to stub_comm for now
        return comm_interface_ptr(ccl::stub_comm::create(device, context, size, rank, kvs));
    }
#endif // CCL_ENABLE_STUB_BACKEND

    return comm_interface_ptr(ccl_comm::createExt(device, context, size, rank, kvs, internal_attr));
}
} // namespace ccl

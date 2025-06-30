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
#pragma once

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/environment.hpp"

#include "coll/attr/ccl_attrs.hpp"

#include "comm/comm_common_attr.hpp"
#include "comm_attr_impl.hpp"

#include "comm/comm_split_common_attr.hpp"
#include "comm_split_attr_impl.hpp"

#include "stream_impl.hpp"

#include "common/global/global.hpp"
#include "comm/comm.hpp"

#include "oneapi/ccl/communicator.hpp"

#include "oneapi/ccl/native_device_api/export_api.hpp"
#include "common/utils/version.hpp"

#include "internal_types.hpp"

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif

#define CCL_CHECK_AND_THROW(result, diagnostic) \
    do { \
        if (result != ccl::status::success) { \
            throw ccl::exception(diagnostic); \
        } \
    } while (0);

namespace ccl {

namespace detail {

/******************** DEVICE ********************/

template <class native_device_type, typename T>
device CCL_API environment::create_device(native_device_type&& native_device) const {
    return device::create_device(std::forward<native_device_type>(native_device));
}

/******************** CONTEXT ********************/

template <class native_device_contex_type, typename T>
context CCL_API environment::create_context(native_device_contex_type&& native_context) const {
    return context::create_context(std::forward<native_device_contex_type>(native_context));
}

/******************** STREAM ********************/

template <class native_stream_type, typename T>
stream CCL_API environment::create_stream(native_stream_type& native_stream) {
    return stream::create_stream(native_stream);
}

/******************** COMMUNICATOR ********************/

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API
environment::create_communicators(const int comm_size,
                                  const vector_class<DeviceType>& local_devices,
                                  const ContextType& context,
                                  shared_ptr_class<kvs_interface> kvs,
                                  const comm_attr& attr) const {
    return communicator::create_communicators(comm_size, local_devices, context, kvs, attr);
}

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API
environment::create_communicatorsExt(const int comm_size,
                                     const vector_class<DeviceType>& local_devices,
                                     const ContextType& context,
                                     shared_ptr_class<kvs_interface> kvs,
                                     const comm_attr& attr) const {
    return communicator::create_communicatorsExt(comm_size, local_devices, context, kvs, attr);
}

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API environment::create_communicators(
    const int comm_size,
    const vector_class<pair_class<int, DeviceType>>& local_rank_device_map,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs,
    const comm_attr& attr) const {
    return communicator::create_communicators(comm_size, local_rank_device_map, context, kvs, attr);
}

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API environment::create_communicatorsExt(
    const int comm_size,
    const vector_class<pair_class<int, DeviceType>>& local_rank_device_map,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs,
    const comm_attr& attr) const {
    return communicator::create_communicatorsExt(
        comm_size, local_rank_device_map, context, kvs, attr);
}

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API
environment::create_communicators(const int comm_size,
                                  const map_class<int, DeviceType>& local_rank_device_map,
                                  const ContextType& context,
                                  shared_ptr_class<kvs_interface> kvs,
                                  const comm_attr& attr) const {
    return communicator::create_communicators(comm_size, local_rank_device_map, context, kvs, attr);
}

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API
environment::create_communicatorsExt(const int comm_size,
                                     const map_class<int, DeviceType>& local_rank_device_map,
                                     const ContextType& context,
                                     shared_ptr_class<kvs_interface> kvs,
                                     const comm_attr& attr) const {
    return communicator::create_communicatorsExt(
        comm_size, local_rank_device_map, context, kvs, attr);
}

} // namespace detail

} // namespace ccl

/******************** TypeGenerations ********************/

#define CREATE_COMM_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::detail::environment::create_communicators<DeviceType, ContextType>( \
        const int comm_size, \
        const ccl::vector_class<DeviceType>& local_devices, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr) const; \
\
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::detail::environment::create_communicatorsExt<DeviceType, ContextType>( \
        const int comm_size, \
        const ccl::vector_class<DeviceType>& local_devices, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr) const; \
\
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::detail::environment::create_communicators<DeviceType, ContextType>( \
        const int comm_size, \
        const ccl::vector_class<ccl::pair_class<int, DeviceType>>& local_devices, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr) const; \
\
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::detail::environment::create_communicatorsExt<DeviceType, ContextType>( \
        const int comm_size, \
        const ccl::vector_class<ccl::pair_class<int, DeviceType>>& local_devices, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr) const; \
\
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::detail::environment::create_communicators<DeviceType, ContextType>( \
        const int comm_size, \
        const ccl::map_class<int, DeviceType>& local_devices, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr) const; \
\
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::detail::environment::create_communicatorsExt<DeviceType, ContextType>( \
        const int comm_size, \
        const ccl::map_class<int, DeviceType>& local_devices, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr) const;

#define CREATE_STREAM_INSTANTIATION(native_stream_type) \
    template ccl::stream CCL_API ccl::detail::environment::create_stream( \
        native_stream_type& native_stream);

#define CREATE_STREAM_EXT_INSTANTIATION(device_type, native_context_type) \
    template ccl::stream CCL_API ccl::detail::environment::create_stream( \
        device_type& device, native_context_type& native_ctx);

#define CREATE_CONTEXT_INSTANTIATION(native_context_type) \
    template ccl::context CCL_API ccl::detail::environment::create_context( \
        native_context_type&& native_ctx) const; \
    template ccl::context CCL_API ccl::detail::environment::create_context( \
        native_context_type& native_ctx) const; \
    template ccl::context CCL_API ccl::detail::environment::create_context( \
        const native_context_type& native_ctx) const;

#define CREATE_DEVICE_INSTANTIATION(native_device_type) \
    template ccl::device CCL_API ccl::detail::environment::create_device( \
        native_device_type&& native_device) const; \
    template ccl::device CCL_API ccl::detail::environment::create_device( \
        native_device_type& native_device) const; \
    template ccl::device CCL_API ccl::detail::environment::create_device( \
        const native_device_type& native_device) const;

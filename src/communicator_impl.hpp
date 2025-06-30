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

#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"
#include "oneapi/ccl/communicator.hpp"

#include "kvs_impl.hpp"

namespace ccl {

namespace v1 {

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicators(
    const int size,
    const vector_class<DeviceType>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs,
    const comm_attr& attr) {
    vector_class<communicator> ret;
    throw std::runtime_error(std::string(__FUNCTION__) + " - not implemented");
    return ret;
}

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicatorsExt(
    const int size,
    const vector_class<DeviceType>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs,
    const comm_attr& attr) {
    vector_class<communicator> ret;
    throw std::runtime_error(std::string(__FUNCTION__) + " - not implemented");
    return ret;
}

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicators(
    const int size,
    const vector_class<pair_class<int, DeviceType>>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs,
    const comm_attr& attr) {
    LOG_DEBUG("size: ", size, ", local ranks: ", devices.size());

    CCL_THROW_IF_NOT(devices.size() == 1, "multiple devices per process are not supported");

    ccl::comm_interface_ptr impl = ccl::comm_interface::create_comm_impl(
        size, devices.begin()->first, devices.begin()->second, context, kvs, attr);

    ccl::vector_class<ccl::communicator> ret;
    ret.push_back(ccl::communicator(std::move(impl)));

    return ret;
}

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicatorsExt(
    const int size,
    const vector_class<pair_class<int, DeviceType>>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs,
    const comm_attr& attr) {
    LOG_DEBUG("size: ", size, ", local ranks: ", devices.size());

    CCL_THROW_IF_NOT(devices.size() == 1, "multiple devices per thread are not supported");

    ccl::comm_interface_ptr impl = ccl::comm_interface::create_comm_implExt(
        size, devices.begin()->first, devices.begin()->second, context, kvs, attr);

    ccl::vector_class<ccl::communicator> ret;
    ret.push_back(ccl::communicator(std::move(impl)));

    return ret;
}

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicators(
    const int size,
    const map_class<int, DeviceType>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs,
    const comm_attr& attr) {
    std::vector<pair_class<int, DeviceType>> vec_devices;
    for (const auto& d : devices) {
        vec_devices.push_back(std::make_pair(d.first, d.second));
    }
    return create_communicators(size, vec_devices, context, kvs, attr);
}

template <class DeviceType, class ContextType>
CCL_API vector_class<communicator> communicator::create_communicatorsExt(
    const int size,
    const map_class<int, DeviceType>& devices,
    const ContextType& context,
    shared_ptr_class<kvs_interface> kvs,
    const comm_attr& attr) {
    std::vector<pair_class<int, DeviceType>> vec_devices;
    for (const auto& d : devices) {
        vec_devices.push_back(std::make_pair(d.first, d.second));
    }
    return create_communicators(size, vec_devices, context, kvs, attr);
}

/**
 * Creates a new host communicator with externally provided size, rank and kvs.
 * Implementation is platform specific and non portable.
 * @return host communicator
 */
communicator communicator::create_communicator(const comm_attr& attr) {
    LOG_DEBUG("create communicator");

    comm_interface_ptr impl = comm_interface::create_comm_impl(attr);

    return communicator(std::move(impl));
}

/**
 * Creates a new host communicator with user supplied size and kvs.
 * Rank will be assigned automatically.
 * @param size user-supplied total number of ranks
 * @param kvs key-value store for ranks wire-up
 * @return host communicator
 */
communicator communicator::create_communicator(const int size,
                                               shared_ptr_class<kvs_interface> kvs,
                                               const comm_attr& attr) {
    throw ccl::exception(std::string(__PRETTY_FUNCTION__) + " - is not implemented");

    LOG_DEBUG("size: ", size);

    comm_interface_ptr impl = comm_interface::create_comm_impl(size, kvs, attr);

    return communicator(std::move(impl));
}

/**
 * Creates a new host communicator with user supplied size, rank and kvs.
 * @param size user-supplied total number of ranks
 * @param rank user-supplied rank
 * @param kvs key-value store for ranks wire-up
 * @return host communicator
 */
communicator communicator::create_communicator(const int size,
                                               const int rank,
                                               shared_ptr_class<kvs_interface> kvs,
                                               const comm_attr& attr) {
    LOG_DEBUG("size ", size, ", rank ", rank);

    comm_interface_ptr impl = comm_interface::create_comm_impl(size, rank, kvs, attr);

    return communicator(std::move(impl));
}

communicator communicator::split_communicator(const communicator& comm, int color, int key) {
    LOG_DEBUG("create splitted communicator");
    return const_cast<communicator&>(comm).split(color, key, true);
}

} // namespace v1

} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define API_COMM_CREATE_WO_RANK_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::communicator> CCL_API ccl::communicator::create_communicators( \
        const int comm_size, \
        const ccl::vector_class<DeviceType>& local_devices, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr); \
\
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::communicator::create_communicatorsExt(const int comm_size, \
                                               const ccl::vector_class<DeviceType>& local_devices, \
                                               const ContextType& context, \
                                               ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
                                               const comm_attr& attr);

#define API_COMM_CREATE_WITH_RANK_IN_VECTOR_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::communicator> CCL_API ccl::communicator::create_communicators( \
        const int comm_size, \
        const ccl::vector_class<ccl::pair_class<int, DeviceType>>& local_rank_device_map, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr); \
\
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::communicator::create_communicatorsExt( \
        const int comm_size, \
        const ccl::vector_class<ccl::pair_class<int, DeviceType>>& local_rank_device_map, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr);

#define API_COMM_CREATE_WITH_RANK_IN_MAP_EXPLICIT_INSTANTIATION(DeviceType, ContextType) \
    template ccl::vector_class<ccl::communicator> CCL_API ccl::communicator::create_communicators( \
        const int comm_size, \
        const ccl::map_class<int, DeviceType>& local_rank_device_map, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr); \
\
    template ccl::vector_class<ccl::communicator> CCL_API \
    ccl::communicator::create_communicatorsExt( \
        const int comm_size, \
        const ccl::map_class<int, DeviceType>& local_rank_device_map, \
        const ContextType& context, \
        ccl::shared_ptr_class<ccl::kvs_interface> kvs, \
        const comm_attr& attr);

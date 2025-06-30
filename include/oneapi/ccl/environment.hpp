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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"
#include "oneapi/ccl/coll_attr.hpp"

#include "oneapi/ccl/comm_attr_ids.hpp"
#include "oneapi/ccl/comm_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_attr.hpp"

#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"

#include "oneapi/ccl/context_attr_ids.hpp"
#include "oneapi/ccl/context_attr_ids_traits.hpp"
#include "oneapi/ccl/context.hpp"

#include "oneapi/ccl/datatype_attr_ids.hpp"
#include "oneapi/ccl/datatype_attr_ids_traits.hpp"
#include "oneapi/ccl/datatype_attr.hpp"

#include "oneapi/ccl/device_attr_ids.hpp"
#include "oneapi/ccl/device_attr_ids_traits.hpp"
#include "oneapi/ccl/device.hpp"

#include "oneapi/ccl/init_attr_ids.hpp"
#include "oneapi/ccl/init_attr_ids_traits.hpp"
#include "oneapi/ccl/init_attr.hpp"

#include "oneapi/ccl/kvs_attr_ids.hpp"
#include "oneapi/ccl/kvs_attr_ids_traits.hpp"
#include "oneapi/ccl/kvs_attr.hpp"

#include "oneapi/ccl/kvs.hpp"

#include "oneapi/ccl/event.hpp"

#include "oneapi/ccl/stream_attr_ids.hpp"
#include "oneapi/ccl/stream_attr_ids_traits.hpp"
#include "oneapi/ccl/stream.hpp"

#include "oneapi/ccl/communicator.hpp"

#include "oneapi/ccl/exception.hpp"

namespace ccl {

namespace detail {

/**
 * CCL environment singleton
 */
class environment {
public:
    ~environment();

    /**
     * Retrieves the unique environment object
     * and makes the first-time initialization of CCL library
     */
    static environment& instance();

    static ccl::library_version get_library_version();

    template <class... attr_val_type>
    static init_attr create_init_attr(attr_val_type&&... avs) {
        auto init_create_attr = create_postponed_api_type<init_attr>();
        int expander[]{ (init_create_attr.template set<attr_val_type::idx()>(avs.val()), 0)... };
        (void)expander;
        return init_create_attr;
    }

    template <class coll_attribute_type, class... attr_val_type>
    static coll_attribute_type create_operation_attr(attr_val_type&&... avs) {
        auto op_attr = create_postponed_api_type<coll_attribute_type>();
        int expander[]{ (op_attr.template set<attr_val_type::idx()>(avs.val()), 0)... };
        (void)expander;
        return op_attr;
    }

    /******************** DATATYPE ********************/

    template <class... attr_val_type>
    static datatype_attr create_datatype_attr(attr_val_type&&... avs) {
        static_assert(sizeof...(avs) > 0, "At least one argument must be specified");
        auto attr = create_postponed_api_type<datatype_attr>();
        int expander[]{ (attr.template set<attr_val_type::idx()>(avs.val()), 0)... };
        (void)expander;
        return attr;
    }

    ccl::datatype register_datatype(const datatype_attr& attr);
    void deregister_datatype(ccl::datatype dtype);
    size_t get_datatype_size(ccl::datatype dtype) const;

    /******************** KVS ********************/

    template <class... attr_val_type>
    static kvs_attr create_kvs_attr(attr_val_type&&... avs) {
        auto kvs_create_attr = create_postponed_api_type<kvs_attr>();
        int expander[]{ (kvs_create_attr.template set<attr_val_type::idx()>(avs.val()), 0)... };
        (void)expander;
        return kvs_create_attr;
    }

    shared_ptr_class<kvs> create_main_kvs(const kvs_attr& attr) const;
    shared_ptr_class<kvs> create_kvs(const kvs::address_type& addr, const kvs_attr& attr) const;

    /******************** DEVICE ********************/

    device create_device(empty_t empty) const;

    template <class native_device_type,
              class = typename std::enable_if<is_device_supported<native_device_type>()>::type>
    device create_device(native_device_type&& native_device) const;

    template <class... attr_val_type>
    device create_device_from_attr(typename unified_device_type::ccl_native_t dev,
                                   attr_val_type&&... avs) const {
        device str = create_postponed_api_type<device>(dev);
        int expander[]{ (str.template set<attr_val_type::idx()>(avs.val()), 0)... };
        (void)expander;
        str.build_from_params();
        return str;
    }

    /******************** CONTEXT ********************/

    context create_context(empty_t empty) const;

    template <
        class native_device_contex_type,
        class = typename std::enable_if<is_device_supported<native_device_contex_type>()>::type>
    context create_context(native_device_contex_type&& native_context) const;

    template <class... attr_val_type>
    context create_context_from_attr(typename unified_context_type::ccl_native_t ctx,
                                     attr_val_type&&... avs) const {
        context str = create_postponed_api_type<context>(ctx);
        int expander[]{ (str.template set<attr_val_type::idx()>(avs.val()), 0)... };
        (void)expander;
        str.build_from_params();
        return str;
    }

    /******************** EVENT ********************/

    template <class event_type,
              class = typename std::enable_if<is_event_supported<event_type>()>::type>
    event create_event(event_type& native_event) {
        return event::create_from_native(native_event);
    }

    /******************** STREAM ********************/

    template <class native_stream_type,
              class = typename std::enable_if<is_stream_supported<native_stream_type>()>::type>
    stream create_stream(native_stream_type& native_stream);

    /******************** COMMUNICATOR ********************/
    template <class... attr_val_type>
    static comm_split_attr create_comm_split_attr(attr_val_type&&... avs) {
        auto split_attr = create_postponed_api_type<comm_split_attr>();
        int expander[]{ (split_attr.template set<attr_val_type::idx()>(avs.val()), 0)... };
        (void)expander;
        return split_attr;
    }

    template <class... attr_val_type>
    static comm_attr create_comm_attr(attr_val_type&&... avs) {
        auto comm_create_attr = create_postponed_api_type<comm_attr>();
        int expander[]{ (comm_create_attr.template set<attr_val_type::idx()>(avs.val()), 0)... };
        (void)expander;
        return comm_create_attr;
    }

    communicator create_communicator(const comm_attr& attr) const;
    communicator create_communicator(size_t size,
                                     shared_ptr_class<kvs_interface> kvs,
                                     const comm_attr& attr) const;
    communicator create_communicator(size_t size,
                                     int rank,
                                     shared_ptr_class<kvs_interface> kvs,
                                     const comm_attr& attr) const;

    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicators(int comm_size,
                                                    const vector_class<DeviceType>& local_devices,
                                                    const ContextType& context,
                                                    shared_ptr_class<kvs_interface> kvs,
                                                    const comm_attr& attr) const;
    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicatorsExt(
        int comm_size,
        const vector_class<DeviceType>& local_devices,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr) const;

    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicators(
        int comm_size,
        const vector_class<pair_class<int, DeviceType>>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr) const;

    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicatorsExt(
        int comm_size,
        const vector_class<pair_class<int, DeviceType>>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr) const;

    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicators(
        int comm_size,
        const map_class<int, DeviceType>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr) const;

    template <class DeviceType, class ContextType>
    vector_class<communicator> create_communicatorsExt(
        int comm_size,
        const map_class<int, DeviceType>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr) const;

    communicator split_communicator(const communicator& comm, int color, int key) const;

private:
    environment();

    template <class ccl_api_type, class... args_type>
    static ccl_api_type create_postponed_api_type(args_type... args) {
        auto version = get_library_version();
        return ccl_api_type(std::forward<args_type>(args)..., version);
    }
};

} // namespace detail

} // namespace ccl

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

#include "oneapi/ccl/device_attr_ids.hpp"
#include "oneapi/ccl/device_attr_ids_traits.hpp"
#include "oneapi/ccl/device.hpp"

#include "common/device/device.hpp"
#include "common/utils/version.hpp"

namespace ccl {

namespace v1 {

template <class device_type, class... attr_val_type>
CCL_API device device::create_device_from_attr(device_type& native_device_handle,
                                               attr_val_type&&... avs) {
    auto version = utils::get_library_version();

    device str{ device::impl_value_t(new device::impl_t(native_device_handle, version)) };
    int expander[]{ (str.template set<attr_val_type::idx()>(avs.val()), 0)... };
    (void)expander;
    str.build_from_params();

    return str;
}

template <class device_type, typename T>
CCL_API device device::create_device(device_type&& native_device) {
    auto version = utils::get_library_version();

    return { device::impl_value_t(
        new device::impl_t(std::forward<device_type>(native_device), version)) };
}

template <device_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<device_attr_id, attrId>::return_type&
device::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<device_attr_id, attrId>{});
}

template<device_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<device_attr_id, attrId>::return_type device::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<device_attr_id, attrId>{});
}

} // namespace v1

} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define API_DEVICE_CREATION_FORCE_INSTANTIATION(native_device_type) \
    template CCL_API ccl::device ccl::device::create_device(native_device_type&& dev); \
    template CCL_API ccl::device ccl::device::create_device(native_device_type& dev); \
    template CCL_API ccl::device ccl::device::create_device(const native_device_type& dev);

#define API_DEVICE_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    template CCL_API typename ccl::detail::ccl_api_type_attr_traits<ccl::device_attr_id, \
                                                                    IN_attrId>::return_type \
    ccl::device::set<IN_attrId, IN_Value>(const IN_Value& v);

#define API_DEVICE_FORCE_INSTANTIATION_GET(IN_attrId) \
    template CCL_API const typename ccl::detail::ccl_api_type_attr_traits<ccl::device_attr_id, \
                                                                          IN_attrId>::return_type& \
    ccl::device::get<IN_attrId>() const;

#define API_DEVICE_FORCE_INSTANTIATION(IN_attrId, IN_Value) \
    API_DEVICE_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    API_DEVICE_FORCE_INSTANTIATION_GET(IN_attrId)

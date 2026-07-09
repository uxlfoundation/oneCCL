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

#include "oneapi/ccl/context_attr_ids.hpp"
#include "oneapi/ccl/context_attr_ids_traits.hpp"
#include "oneapi/ccl/context.hpp"

#include "common/context/context.hpp"
#include "common/utils/version.hpp"

namespace ccl {

namespace v1 {

template <class context_type, class... attr_val_type>
CCL_API context context::create_context_from_attr(context_type& native_context_handle,
                                                  attr_val_type&&... avs) {
    auto version = utils::get_library_version();

    context str{ context::impl_value_t(new context::impl_t(native_context_handle, version)) };
    int expander[]{ (str.template set<attr_val_type::idx()>(avs.val()), 0)... };
    (void)expander;
    str.build_from_params();

    return str;
}

template <class context_type, typename T>
CCL_API context context::create_context(context_type&& native_context) {
    auto version = utils::get_library_version();

    return { context::impl_value_t(
        new context::impl_t(std::forward<context_type>(native_context), version)) };
}

template <context_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<context_attr_id, attrId>::return_type&
context::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<context_attr_id, attrId>{});
}

template<context_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<context_attr_id, attrId>::return_type context::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<context_attr_id, attrId>{});
}

} // namespace v1

} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define API_CONTEXT_CREATION_FORCE_INSTANTIATION(native_context_type) \
    template CCL_API ccl::context ccl::context::create_context(native_context_type&& ctx); \
    template CCL_API ccl::context ccl::context::create_context(native_context_type& ctx); \
    template CCL_API ccl::context ccl::context::create_context(const native_context_type& ctx);

#define API_CONTEXT_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    template CCL_API typename ccl::detail::ccl_api_type_attr_traits<ccl::context_attr_id, \
                                                                    IN_attrId>::return_type \
    ccl::context::set<IN_attrId, IN_Value>(const IN_Value& v);

#define API_CONTEXT_FORCE_INSTANTIATION_GET(IN_attrId) \
    template CCL_API const typename ccl::detail::ccl_api_type_attr_traits<ccl::context_attr_id, \
                                                                          IN_attrId>::return_type& \
    ccl::context::get<IN_attrId>() const;

#define API_CONTEXT_FORCE_INSTANTIATION(IN_attrId, IN_Value) \
    API_CONTEXT_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    API_CONTEXT_FORCE_INSTANTIATION_GET(IN_attrId)

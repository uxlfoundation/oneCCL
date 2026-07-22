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
#include "oneapi/ccl/aliases.hpp"

#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/stream_attr_ids.hpp"
#include "oneapi/ccl/stream_attr_ids_traits.hpp"
#include "oneapi/ccl/stream.hpp"
#include "common/stream/stream.hpp"
#include "common/utils/version.hpp"

namespace ccl {

namespace v1 {

template <class native_stream_type, typename T>
stream stream::create_stream(native_stream_type& native_stream) {
    auto version = utils::get_library_version();
    return { stream_selector::create(native_stream, version) };
}

template <stream_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<stream_attr_id, attrId>::return_type&
stream::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<stream_attr_id, attrId>{});
}

template<stream_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<stream_attr_id, attrId>::return_type stream::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<stream_attr_id, attrId>{});
}

} // namespace v1

} // namespace ccl

/***************************TypeGenerations*********************************************************/
#define API_STREAM_CREATION_FORCE_INSTANTIATION(native_stream_type) \
    template CCL_API ccl::stream ccl::stream::create_stream(native_stream_type& native_stream);

#define API_STREAM_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    template CCL_API typename ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, \
                                                                    IN_attrId>::return_type \
    ccl::stream::set<IN_attrId, IN_Value>(const IN_Value& v);

#define API_STREAM_FORCE_INSTANTIATION_GET(IN_attrId) \
    template CCL_API const typename ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, \
                                                                          IN_attrId>::return_type& \
    ccl::stream::get<IN_attrId>() const;

#define API_STREAM_FORCE_INSTANTIATION(IN_attrId, IN_Value) \
    API_STREAM_FORCE_INSTANTIATION_SET(IN_attrId, IN_Value) \
    API_STREAM_FORCE_INSTANTIATION_GET(IN_attrId)

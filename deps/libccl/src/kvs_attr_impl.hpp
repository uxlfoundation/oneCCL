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
#include "oneapi/ccl/kvs_attr.hpp"

namespace ccl {

namespace v1 {

/**
 * kvs_attr attributes definition
 */
template <kvs_attr_id attrId, class Value>
CCL_API Value kvs_attr::set(const Value& v) {
    return get_impl()->set_attribute_value(v,
                                           detail::ccl_api_type_attr_traits<kvs_attr_id, attrId>{});
}

template <kvs_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<kvs_attr_id, attrId>::type& kvs_attr::get()
    const {
    return get_impl()->get_attribute_value(detail::ccl_api_type_attr_traits<kvs_attr_id, attrId>{});
}

template <kvs_attr_id attrId>
CCL_API bool kvs_attr::is_valid() const noexcept {
    return get_impl()->is_valid<attrId>();
}

} // namespace v1

} // namespace ccl

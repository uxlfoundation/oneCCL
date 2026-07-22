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
#include "oneapi/ccl/init_attr_ids.hpp"
#include "oneapi/ccl/init_attr_ids_traits.hpp"
#include "oneapi/ccl/init_attr.hpp"

namespace ccl {

class init_attr_impl {
public:
    /**
     * `version` operations
     */
    using version_traits_t = detail::ccl_api_type_attr_traits<init_attr_id, init_attr_id::version>;

    const typename version_traits_t::return_type& get_attribute_value(
        const version_traits_t& id) const {
        return version;
    }

    typename version_traits_t::return_type set_attribute_value(typename version_traits_t::type val,
                                                               const version_traits_t& t) {
        (void)t;
        throw ccl::exception("Set value for 'ccl::init_attr_id::version' is not allowed");
        return version;
    }

    init_attr_impl(const typename version_traits_t::return_type& version) : version(version) {}

protected:
    typename version_traits_t::return_type version;
};

namespace v1 {

/**
 * init_attr attributes definition
 */
template <init_attr_id attrId, class Value>
Value init_attr::set(const Value& v) {
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<init_attr_id, attrId>{});
}

template <init_attr_id attrId>
const typename detail::ccl_api_type_attr_traits<init_attr_id, attrId>::return_type& init_attr::get()
    const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<init_attr_id, attrId>{});
}

} // namespace v1

} // namespace ccl

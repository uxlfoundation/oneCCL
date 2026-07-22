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

#include "coll/attr/ccl_pt2pt_op_attr.hpp"

namespace ccl {

ccl_pt2pt_attr_impl_t::ccl_pt2pt_attr_impl_t(
    const typename ccl_operation_attr_impl_t::version_traits_t::type& version)
        : base_t(version) {}

typename ccl_pt2pt_attr_impl_t::group_id_traits_t::return_type
ccl_pt2pt_attr_impl_t::set_attribute_value(typename group_id_traits_t::type val,
                                           const group_id_traits_t& t) {
    auto old = group_id;
    std::swap(group_id, val);
    return old;
}

const typename ccl_pt2pt_attr_impl_t::group_id_traits_t::return_type&
ccl_pt2pt_attr_impl_t::get_attribute_value(const group_id_traits_t& id) const {
    return group_id;
}
} // namespace ccl

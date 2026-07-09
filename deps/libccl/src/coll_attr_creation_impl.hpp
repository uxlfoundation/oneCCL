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
#include "oneapi/ccl/coll_attr.hpp"
#include "coll/attr/ccl_attrs.hpp"
#include "common/utils/version.hpp"

namespace ccl {

namespace v1 {

/* TODO temporary function for UT compilation: would be part of ccl::detail::environment in final*/
template <class coll_attribute_type, class... attr_val_type>
coll_attribute_type create_coll_attr(attr_val_type&&... avs) {
    auto version = utils::get_library_version();
    auto coll_attr = coll_attribute_type(version);

    int expander[]{ (coll_attr.template set<attr_val_type::idx()>(avs.val()), 0)... };
    (void)expander;
    return coll_attr;
}

} // namespace v1

} // namespace ccl

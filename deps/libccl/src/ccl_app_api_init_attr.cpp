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

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/aliases.hpp"
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/init_attr_ids.hpp"
#include "oneapi/ccl/init_attr_ids_traits.hpp"
#include "oneapi/ccl/init_attr.hpp"

// Core file with PIMPL implementation
#include "init_attr_impl.hpp"

namespace ccl {

namespace v1 {

#define API_FORCE_SETTER_INSTANTIATION(class_name, IN_attrId, IN_Value, OUT_Traits_Value) \
    template CCL_API IN_Value class_name::set<IN_attrId, IN_Value>(const IN_Value& v);

#define API_FORCE_GETTER_INSTANTIATION(class_name, IN_attrId, OUT_Traits_Value) \
    template CCL_API const typename OUT_Traits_Value<init_attr_id, IN_attrId>::return_type& \
    class_name::get<IN_attrId>() const;

/**
 * init_attr attributes definition
 */
CCL_API init_attr::init_attr(init_attr&& src) : base_t(std::move(src)) {}

CCL_API init_attr::init_attr(const init_attr& src) : base_t(src) {}

CCL_API init_attr::init_attr(
    const typename detail::ccl_api_type_attr_traits<init_attr_id,
                                                    init_attr_id::version>::return_type& version)
        : base_t(impl_value_t(new impl_t(version))) {}

CCL_API init_attr::~init_attr() noexcept {}

CCL_API init_attr& init_attr::operator=(const init_attr& src) {
    this->acc_policy_t::create(this, src);
    return *this;
}

CCL_API init_attr& init_attr::operator=(init_attr&& src) {
    this->acc_policy_t::create(this, std::move(src));
    return *this;
}

API_FORCE_GETTER_INSTANTIATION(init_attr, init_attr_id::version, detail::ccl_api_type_attr_traits);

#undef API_FORCE_SETTER_INSTANTIATION
#undef API_FORCE_GETTER_INSTANTIATION

} // namespace v1

} // namespace ccl

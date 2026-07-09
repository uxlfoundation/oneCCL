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
#include "stream_impl.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

namespace ccl {

namespace v1 {

CCL_API stream::stream(
    const typename detail::ccl_api_type_attr_traits<stream_attr_id, stream_attr_id::version>::type&
        version)
        : base_t(impl_value_t()) {}

CCL_API stream::stream(stream&& src) : base_t(std::move(src)) {}

CCL_API stream::stream(impl_value_t&& impl) : base_t(std::move(impl)) {}

CCL_API stream::~stream() {}

CCL_API stream::stream(const stream& src) : base_t(src) {}

CCL_API stream& stream::operator=(stream&& src) noexcept {
    this->acc_policy_t::create(this, std::move(src));
    return *this;
}

CCL_API stream& stream::operator=(const stream& src) {
    this->acc_policy_t::create(this, src);
    return *this;
}

CCL_API stream::native_t& stream::get_native() {
    return const_cast<stream::native_t&>(static_cast<const stream*>(this)->get_native());
}

CCL_API const stream::native_t& stream::get_native() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<stream_attr_id, stream_attr_id::native_handle>{});
}

} // namespace v1

} // namespace ccl

API_STREAM_CREATION_FORCE_INSTANTIATION(typename ccl::unified_stream_type::ccl_native_t)
API_STREAM_FORCE_INSTANTIATION(ccl::stream_attr_id::version, ccl::library_version);
API_STREAM_FORCE_INSTANTIATION_GET(ccl::stream_attr_id::native_handle);

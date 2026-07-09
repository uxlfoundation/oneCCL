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
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"
#include "oneapi/ccl/context_attr_ids.hpp"
#include "oneapi/ccl/context_attr_ids_traits.hpp"
#include "common/utils/utils.hpp"

class ccl_context_impl {
public:
    using context_native_handle_t = typename ccl::unified_context_type::handle_t;
    using context_native_t = typename ccl::unified_context_type::ccl_native_t;

    ccl_context_impl() = delete;
    ccl_context_impl(const ccl_context_impl& other) = delete;
    ccl_context_impl& operator=(const ccl_context_impl& other) = delete;

    ccl_context_impl(context_native_t& ctx, const ccl::library_version& version);
    ccl_context_impl(const context_native_t& ctx, const ccl::library_version& version);
    ccl_context_impl(context_native_t&& ctx, const ccl::library_version& version);
    ccl_context_impl(context_native_handle_t ctx_handle, const ccl::library_version& version);
    ~ccl_context_impl() = default;

    //Export Attributes
    using version_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::context_attr_id, ccl::context_attr_id::version>;
    typename version_traits_t::type set_attribute_value(typename version_traits_t::type val,
                                                        const version_traits_t& t);
    const typename version_traits_t::return_type& get_attribute_value(
        const version_traits_t& id) const;

    using cl_backend_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::context_attr_id,
                                              ccl::context_attr_id::cl_backend>;
    const typename cl_backend_traits_t::return_type& get_attribute_value(
        const cl_backend_traits_t& id) const;

    using native_handle_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::context_attr_id,
                                              ccl::context_attr_id::native_handle>;
    typename native_handle_traits_t::return_type& get_attribute_value(
        const native_handle_traits_t& id);

    void build_from_params();

private:
    const ccl::library_version version;
    context_native_t native_context;
    bool creation_is_postponed{ false };
};

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

#include "common/log/log.hpp"
#include "common/context/context.hpp"
#include "oneapi/ccl/native_device_api/export_api.hpp"

ccl_context_impl::ccl_context_impl(context_native_t& ctx, const ccl::library_version& version)
        : version(version),
          native_context(ctx) {}

ccl_context_impl::ccl_context_impl(const context_native_t& ctx, const ccl::library_version& version)
        : version(version),
          native_context(ctx) {}

ccl_context_impl::ccl_context_impl(context_native_t&& ctx, const ccl::library_version& version)
        : version(version),
          native_context(std::move(ctx)) {}

ccl_context_impl::ccl_context_impl(context_native_handle_t ctx_handle,
                                   const ccl::library_version& version)
        : version(version) {}

void ccl_context_impl::build_from_params() {
    if (!creation_is_postponed) {
        throw ccl::exception("error");
    }
#ifdef CCL_ENABLE_SYCL
    /* TODO unavailable??
    event_native_t event_candidate{native_context};
    std::swap(event_candidate, native_event); //TODO USE attributes fro sycl queue construction
    */

    throw ccl::exception("build_from_attr is not availbale for sycl::device");
#else

    //TODO use attributes

#endif
    creation_is_postponed = false;
}

//Export Attributes
typename ccl_context_impl::version_traits_t::type ccl_context_impl::set_attribute_value(
    typename version_traits_t::type val,
    const version_traits_t& t) {
    (void)t;
    throw ccl::exception("Set value for 'ccl::event_attr_id::library_version' is not allowed");
    return version;
}

const typename ccl_context_impl::version_traits_t::return_type&
ccl_context_impl::get_attribute_value(const version_traits_t& id) const {
    return version;
}

const typename ccl_context_impl::cl_backend_traits_t::return_type&
ccl_context_impl::get_attribute_value(const cl_backend_traits_t& id) const {
    //TODO
    throw ccl::exception(
        "TODO - Get value for 'ccl::device_attr_id::cl_backend_traits_t' is not inmlemented");
    static constexpr ccl::cl_backend_type ret{ ccl::cl_backend_type::empty_backend };
    return ret;
}

typename ccl_context_impl::native_handle_traits_t::return_type&
ccl_context_impl::get_attribute_value(const native_handle_traits_t& id) {
    return native_context;
}

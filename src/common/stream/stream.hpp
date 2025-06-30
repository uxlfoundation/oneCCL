/*
 Copyright 2016-2020 Intel Corporation
 
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

#include "coll/attr/ccl_common_op_attrs.hpp"
#include "common/stream/stream_selector.hpp"
#include "common/utils/utils.hpp"
#include "oneapi/ccl/stream_attr_ids.hpp"
#include "oneapi/ccl/stream_attr_ids_traits.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/type_traits.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

namespace ccl {

enum class device_family { unknown, family1, family2, family3, family4, family5, family6, family7 };

std::string to_string(device_family family);

} // namespace ccl

std::string to_string(const stream_type& type);

class alignas(CACHELINE_SIZE) ccl_stream : public stream_selector {
public:
    friend class stream_selector;

    using stream_native_t = stream_selector::stream_native_t;

    ccl_stream() = delete;
    ccl_stream(const ccl_stream& other) = delete;
    ccl_stream& operator=(const ccl_stream& other) = delete;

    ~ccl_stream() = default;

    std::string to_string() const;

    stream_type get_type() const;
    ccl::device_family get_device_family() const;
    bool is_sycl_device_stream() const;
    bool is_gpu() const;

#ifdef CCL_ENABLE_SYCL
    sycl::backend get_backend() const;
#ifdef CCL_ENABLE_ZE
    ze_device_handle_t get_ze_device() const;
    ze_context_handle_t get_ze_context() const;
#endif // CCL_ENABLE_ZE
#endif // CCL_ENBALE_SYCL

    static std::unique_ptr<ccl_stream> create(stream_native_t& native_stream,
                                              const ccl::library_version& version);

    // export attributes
    using version_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id, ccl::stream_attr_id::version>;
    typename version_traits_t::return_type set_attribute_value(typename version_traits_t::type val,
                                                               const version_traits_t& t);

    const typename version_traits_t::return_type& get_attribute_value(
        const version_traits_t& id) const;

    using native_handle_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::stream_attr_id,
                                              ccl::stream_attr_id::native_handle>;
    typename native_handle_traits_t::return_type& get_attribute_value(
        const native_handle_traits_t& id);

private:
    ccl_stream(stream_type type,
               stream_native_t& native_stream,
               const ccl::library_version& version);

    const ccl::library_version version;

    stream_type type;
    ccl::device_family device_family;

#ifdef CCL_ENABLE_SYCL
    sycl::backend backend;

#ifdef CCL_ENABLE_ZE
    ze_device_handle_t device{};
    ze_context_handle_t context{};
#endif // CCL_ENABLE_ZE
#endif // CCL_ENBALE_SYCL
};

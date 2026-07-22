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
#include "common/event/ccl_event_attr_ids.hpp"
#include "common/event/ccl_event_attr_ids_traits.hpp"
#include "common/utils/utils.hpp"

namespace ccl {
namespace detail {
class environment;
}
} // namespace ccl

class alignas(CACHELINE_SIZE) ccl_event {
public:
    friend class ccl::detail::environment;
    using event_native_handle_t = typename ccl::unified_event_type::handle_t;
    using event_native_t = typename ccl::unified_event_type::ccl_native_t;

    ccl_event() = delete;
    ccl_event(const ccl_event& other) = delete;
    ccl_event& operator=(const ccl_event& other) = delete;

    ccl_event(event_native_t& event, const ccl::library_version& version);
    ~ccl_event() = default;

    //Export Attributes
    using version_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::event_attr_id, ccl::event_attr_id::version>;
    typename version_traits_t::type set_attribute_value(typename version_traits_t::type val,
                                                        const version_traits_t& t);

    const typename version_traits_t::return_type& get_attribute_value(
        const version_traits_t& id) const;

    using native_handle_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::event_attr_id,
                                              ccl::event_attr_id::native_handle>;
    typename native_handle_traits_t::return_type& get_attribute_value(
        const native_handle_traits_t& id);

    using command_type_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::event_attr_id, ccl::event_attr_id::command_type>;
    typename command_type_traits_t::return_type set_attribute_value(
        typename command_type_traits_t::type val,
        const command_type_traits_t& t);

    const typename command_type_traits_t::return_type& get_attribute_value(
        const command_type_traits_t& id) const;

    using command_execution_status_traits_t =
        ccl::detail::ccl_api_type_attr_traits<ccl::event_attr_id,
                                              ccl::event_attr_id::command_execution_status>;
    typename command_execution_status_traits_t::return_type set_attribute_value(
        typename command_execution_status_traits_t::type val,
        const command_execution_status_traits_t& t);

    const typename command_execution_status_traits_t::return_type& get_attribute_value(
        const command_execution_status_traits_t& id) const;

    void build_from_params();

private:
    const ccl::library_version version;
    event_native_t native_event;
    bool creation_is_postponed{ false };

    typename command_type_traits_t::return_type command_type_val;
    typename command_execution_status_traits_t::return_type command_execution_status_val;
};

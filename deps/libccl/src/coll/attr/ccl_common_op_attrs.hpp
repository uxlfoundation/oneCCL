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
#include "oneapi/ccl/types_policy.hpp"
#include "oneapi/ccl/coll_attr_ids.hpp"
#include "oneapi/ccl/coll_attr_ids_traits.hpp"

namespace ccl {
struct ccl_operation_attr_impl_t {
    ccl_operation_attr_impl_t(const ccl::library_version& version);
    /**
     * `version` operations
     */
    using version_traits_t =
        detail::ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::version>;
    const typename version_traits_t::return_type& get_attribute_value(
        const version_traits_t& id) const;

    typename version_traits_t::return_type set_attribute_value(typename version_traits_t::type val,
                                                               const version_traits_t& t);

    /**
     * `priority` operations
     */
    using priority_traits_t =
        detail::ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::priority>;
    const typename priority_traits_t::return_type& get_attribute_value(
        const priority_traits_t& id) const;

    typename priority_traits_t::return_type set_attribute_value(
        typename priority_traits_t::type val,
        const priority_traits_t& t);

    /**
     * `synchronous` operations
     */
    using synchronous_traits_t =
        detail::ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::synchronous>;
    const typename synchronous_traits_t::return_type& get_attribute_value(
        const synchronous_traits_t& id) const;

    typename synchronous_traits_t::return_type set_attribute_value(
        typename synchronous_traits_t::type val,
        const synchronous_traits_t& t);

    /**
     * `to_cache` operations
     */
    using to_cache_traits_t =
        detail::ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::to_cache>;
    const typename to_cache_traits_t::return_type& get_attribute_value(
        const to_cache_traits_t& id) const;

    typename to_cache_traits_t::return_type set_attribute_value(
        typename to_cache_traits_t::type val,
        const to_cache_traits_t& t);

    /**
     * `match_id` operations
     */
    using match_id_traits_t =
        detail::ccl_api_type_attr_traits<operation_attr_id, operation_attr_id::match_id>;
    const typename match_id_traits_t::return_type& get_attribute_value(
        const match_id_traits_t& id) const;

    typename match_id_traits_t::return_type set_attribute_value(
        typename match_id_traits_t::type val,
        const match_id_traits_t& t);

    /* Priority for collective operation */
    size_t priority = 0;

    /* Blocking/non-blocking */
    bool synchronous = 0;

    /* Persistent/non-persistent */
    bool to_cache = 0;

    /**
     * Id of the operation. If specified, new communicator will be created and collective
     * operations with the same @b match_id will be executed in the same order.
     */
    typename match_id_traits_t::return_type match_id{};

protected:
    ccl::library_version version;
};
} // namespace ccl

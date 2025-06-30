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
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/comm_attr_ids_traits.hpp"

namespace ccl {

class ccl_comm_attr_impl {
public:
    /**
     * `version` operations
     */
    using version_traits_t = detail::ccl_api_type_attr_traits<comm_attr_id, comm_attr_id::version>;

    const typename version_traits_t::return_type& get_attribute_value(
        const version_traits_t& id) const {
        return version;
    }

    typename version_traits_t::return_type set_attribute_value(typename version_traits_t::type val,
                                                               const version_traits_t& t) {
        (void)t;
        throw ccl::exception("Set value for 'ccl::comm_attr_id::version' is not allowed");
        return version;
    }

    /**
     * `blocking` operations
     */
    using blocking_traits_t =
        detail::ccl_api_type_attr_traits<comm_attr_id, comm_attr_id::blocking>;

    const typename blocking_traits_t::return_type& get_attribute_value(
        const blocking_traits_t& id) const {
        return blocking;
    }

    typename blocking_traits_t::return_type set_attribute_value(
        typename blocking_traits_t::type val,
        const blocking_traits_t& t) {
        auto old = blocking;
        std::swap(blocking, val);
        return old;
    }

    ccl_comm_attr_impl(const typename version_traits_t::return_type& version) : version(version) {}
    ccl_comm_attr_impl(const ccl::comm_attr& attr) {
        version = attr.get<ccl::comm_attr_id::version>();
        blocking = attr.get<ccl::comm_attr_id::blocking>();
    }

    template <comm_attr_id attr_id>
    bool is_valid() const noexcept {
        return (attr_id == comm_attr_id::version);
    }

protected:
    typename version_traits_t::return_type version;

public:
    /**
     * Indicate the collectives in a communicator is blocking or non-blocking
     * Currently, only works for MPI transport.
     * Default value is -1, which means unspecified, blocking or not depends on CCL_ATL_SYNC_COLL.
     * 0 means non-blocking and 1 means blocking, has higher priority than CCL_ATL_SYNC_COLL.
     */
    int blocking = -1;
};

} // namespace ccl

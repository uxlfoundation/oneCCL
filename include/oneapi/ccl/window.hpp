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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {
namespace detail {
class environment;
}

// Forward declaration of the implementation class
class ccl_window_impl;

namespace v1 {

class communicator;
struct impl_dispatch;

/**
 * A window object is an abstraction over registered memory windows.
 * Has no defined public constructor. Use ccl::comm_window_register
 * for window objects creation.
 */
class window : public ccl_api_base_movable<window,
                                           direct_access_policy,
                                           ccl_window_impl,
                                           std::shared_ptr> {
public:
    using base_t =
        ccl_api_base_movable<window, direct_access_policy, ccl_window_impl, std::shared_ptr>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    window(window&& src);
    window(impl_value_t&& impl);
    window& operator=(window&& src);
    ~window();

    // Prevent copying - windows are unique resources
    window(const window&) = delete;
    window& operator=(const window&) = delete;

private:
    friend class ccl::detail::environment;
    friend class ccl::v1::communicator;
    friend struct ccl::v1::impl_dispatch;
};

} // namespace v1

using v1::window;

} // namespace ccl

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

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {
namespace detail {
class environment;
}

class event_impl;

namespace v1 {

/**
 * event's interface that allows users to track communication operation progress
 */
class event : public ccl_api_base_movable<event, direct_access_policy, event_impl> {
public:
    using base_t = ccl_api_base_movable<event, direct_access_policy, event_impl>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    using native_t = typename unified_event_type::ccl_native_t;
    using native_handle_t = typename unified_event_type::handle_t;
    using context_t = typename unified_context_type::ccl_native_t;

    event() noexcept;
    event(event&& src) noexcept;
    event(impl_value_t&& impl) noexcept;
    ~event() noexcept;

    event& operator=(event&& src) noexcept;

    bool operator==(const event& rhs) const noexcept;
    bool operator!=(const event& rhs) const noexcept;

    /**
     * Non-blocking check for operation completion
     * @retval true if the operation has been completed or the event was not initialized
     * @retval false if the operation has not been completed
     */
    explicit operator bool();

    /**
     * Blocking wait for operation completion
     */
    void wait();

    /**
     * Non-blocking check for operation completion
     * @retval true if the operation has been completed or the event was not initialized
     * @retval false if the operation has not been completed
     */
    bool test();

    /**
     * Cancel a pending asynchronous operation
     * @retval true if the operation has been canceled or the event was not initialized
     * @retval false if the operation has not been canceled
     */
    bool cancel();

    /**
      * Retrieve a native event object to be used for synchronization
      * with computation or other communication operations
      * @return pointer to native event object
      */
    native_t& get_native();
    const native_t& get_native() const;

    static event create_from_native(native_t& native_event);

private:
    friend class ccl::detail::environment;

    static event create_from_native(native_handle_t native_event_handle, context_t context);
};

} // namespace v1

using v1::event;

} // namespace ccl

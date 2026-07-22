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

class ccl_context_impl;
namespace ccl {
namespace detail {
class environment;
}

namespace v1 {
class communicator;

/**
 * A context object is an abstraction over CPU/GPU context
 * Has no defined public constructor. Use ccl::create_context
 * for context objects creation
 */
/**
 * Stream class
 */
class context : public ccl_api_base_copyable<context, direct_access_policy, ccl_context_impl> {
public:
    using base_t = ccl_api_base_copyable<context, direct_access_policy, ccl_context_impl>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    /**
     * Declare native context type
     */
    using native_t =
        typename detail::ccl_api_type_attr_traits<context_attr_id,
                                                  context_attr_id::native_handle>::return_type;
    context(context&& src);
    context(const context& src);
    context& operator=(const context& src);
    context& operator=(context&& src);
    ~context();

    bool operator==(const context& rhs) const noexcept;
    bool operator!=(const context& rhs) const noexcept;
    bool operator<(const context& rhs) const noexcept;

    /**
     * Get specific attribute value by @attrId
     */
    template <context_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<context_attr_id, attrId>::return_type& get()
        const;

    /**
     * Get native context object
     */
    native_t& get_native();
    const native_t& get_native() const;

private:
    friend class ccl::detail::environment;
    friend class ccl::v1::communicator;
    context(impl_value_t&& impl);

    /**
     *Parametrized context creation helper
     */
    template <context_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<context_attr_id, attrId>::return_type set(const Value& v);

    void build_from_params();
    context(
        const typename detail::ccl_api_type_attr_traits<context_attr_id,
                                                        context_attr_id::version>::type& version);

    /**
     * Factory methods
     */
    template <class context_type,
              class = typename std::enable_if<is_context_supported<context_type>()>::type>
    static context create_context(context_type&& native_context);

    template <class context_handle_type, class... attr_val_type>
    static context create_context_from_attr(context_handle_type& native_context_handle,
                                            attr_val_type&&... avs);
};

template <context_attr_id t, class value_type>
constexpr auto attr_val(value_type v) -> detail::attr_value_triple<context_attr_id, t, value_type> {
    return detail::attr_value_triple<context_attr_id, t, value_type>(v);
}

} // namespace v1

using v1::context;
using v1::attr_val;

} // namespace ccl

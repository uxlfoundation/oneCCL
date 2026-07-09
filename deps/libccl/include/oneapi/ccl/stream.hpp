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

class ccl_stream;
namespace ccl {
namespace detail {
class environment;
}

namespace v1 {
struct ccl_empty_attr;
class communicator;
struct impl_dispatch;

/**
 * A stream object is an abstraction over CPU/GPU streams
 * Has no defined public constructor. Use ccl::create_stream
 * for stream objects creation
 */
/**
 * Stream class
 */
class stream : public ccl_api_base_copyable<stream, direct_access_policy, ccl_stream> {
public:
    using base_t = ccl_api_base_copyable<stream, direct_access_policy, ccl_stream>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    /**
     * Declare native stream type
     */
    using native_t =
        typename detail::ccl_api_type_attr_traits<stream_attr_id,
                                                  stream_attr_id::native_handle>::return_type;

    ~stream();

    stream(stream&& src);
    stream(const stream& src);
    stream& operator=(stream&& src) noexcept;
    stream& operator=(const stream& src);

    /**
     * Get specific attribute value by @attrId
     */
    template <stream_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<stream_attr_id, attrId>::return_type& get()
        const;

    /**
     * Get native stream object
     */
    native_t& get_native();
    const native_t& get_native() const;

private:
    friend class ccl::detail::environment;
    friend class ccl::v1::communicator;
    friend struct ccl::ccl_empty_attr;
    friend struct ccl::v1::impl_dispatch;

    stream(impl_value_t&& impl);

    /**
     * Parameterized stream creation helper
     */
    template <stream_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<stream_attr_id, attrId>::return_type set(const Value& v);

    stream(const typename detail::ccl_api_type_attr_traits<stream_attr_id,
                                                           stream_attr_id::version>::type& version);

    /**
     *  Factory methods
     */
    template <class native_stream_type,
              class = typename std::enable_if<is_stream_supported<native_stream_type>()>::type>
    static stream create_stream(native_stream_type& native_stream);
};

/**
 * Declare extern empty attributes
 */
extern stream default_stream;

template <stream_attr_id t, class value_type>
constexpr auto attr_val(value_type v) -> detail::attr_value_triple<stream_attr_id, t, value_type> {
    return detail::attr_value_triple<stream_attr_id, t, value_type>(v);
}

} // namespace v1

using v1::stream;
using v1::default_stream;
using v1::attr_val;

} // namespace ccl

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

class ccl_allgather_attr_impl_t;
class ccl_allgatherv_attr_impl_t;
class ccl_allreduce_attr_impl_t;
class ccl_alltoall_attr_impl_t;
class ccl_alltoallv_attr_impl_t;
class ccl_barrier_attr_impl_t;
class ccl_broadcast_attr_impl_t;
class ccl_pt2pt_attr_impl_t;
class ccl_reduce_attr_impl_t;
class ccl_reduce_scatter_attr_impl_t;
class ccl_scatter_attr_impl_t;

namespace v1 {

struct ccl_empty_attr;

/**
 * Allgather coll attributes
 */

class allgather_attr : public ccl_api_base_copyable<allgather_attr,
                                                    copy_on_write_access_policy,
                                                    ccl_allgather_attr_impl_t> {
public:
    using base_t = ccl_api_base_copyable<allgather_attr,
                                         copy_on_write_access_policy,
                                         ccl_allgather_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    allgather_attr(allgather_attr&& src);
    allgather_attr(const allgather_attr& src);
    allgather_attr& operator=(allgather_attr&& src) noexcept;
    allgather_attr& operator=(const allgather_attr& src);
    ~allgather_attr();

    /**
     * Set specific value for selft attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <allgather_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<allgather_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);
    /**
     * Get specific attribute value by @attrId
     */
    template <allgather_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<allgather_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    allgather_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Allgatherv coll attributes
 */
class allgatherv_attr : public ccl_api_base_copyable<allgatherv_attr,
                                                     copy_on_write_access_policy,
                                                     ccl_allgatherv_attr_impl_t> {
public:
    using base_t = ccl_api_base_copyable<allgatherv_attr,
                                         copy_on_write_access_policy,
                                         ccl_allgatherv_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    allgatherv_attr(allgatherv_attr&& src);
    allgatherv_attr(const allgatherv_attr& src);
    allgatherv_attr& operator=(allgatherv_attr&& src) noexcept;
    allgatherv_attr& operator=(const allgatherv_attr& src);
    ~allgatherv_attr();

    /**
     * Set specific value for selft attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <allgatherv_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<allgatherv_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);
    /**
     * Get specific attribute value by @attrId
     */
    template <allgatherv_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<allgatherv_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    allgatherv_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Allreduce coll attributes
 */
class allreduce_attr : public ccl_api_base_copyable<allreduce_attr,
                                                    copy_on_write_access_policy,
                                                    ccl_allreduce_attr_impl_t> {
public:
    using base_t = ccl_api_base_copyable<allreduce_attr,
                                         copy_on_write_access_policy,
                                         ccl_allreduce_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    allreduce_attr(allreduce_attr&& src);
    allreduce_attr(const allreduce_attr& src);
    allreduce_attr& operator=(allreduce_attr&& src) noexcept;
    allreduce_attr& operator=(const allreduce_attr& src);
    ~allreduce_attr();

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <allreduce_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<allreduce_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <allreduce_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<allreduce_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    allreduce_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * alltoall coll attributes
 */
class alltoall_attr : public ccl_api_base_copyable<alltoall_attr,
                                                   copy_on_write_access_policy,
                                                   ccl_alltoall_attr_impl_t> {
public:
    using base_t =
        ccl_api_base_copyable<alltoall_attr, copy_on_write_access_policy, ccl_alltoall_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    alltoall_attr(alltoall_attr&& src);
    alltoall_attr(const alltoall_attr& src);
    alltoall_attr& operator=(alltoall_attr&& src) noexcept;
    alltoall_attr& operator=(const alltoall_attr& src);
    ~alltoall_attr();

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <alltoall_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<alltoall_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <alltoall_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<alltoall_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    alltoall_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Alltoallv coll attributes
 */
class alltoallv_attr : public ccl_api_base_copyable<alltoallv_attr,
                                                    copy_on_write_access_policy,
                                                    ccl_alltoallv_attr_impl_t> {
public:
    using base_t = ccl_api_base_copyable<alltoallv_attr,
                                         copy_on_write_access_policy,
                                         ccl_alltoallv_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    alltoallv_attr(alltoallv_attr&& src);
    alltoallv_attr(const alltoallv_attr& src);
    alltoallv_attr& operator=(alltoallv_attr&& src) noexcept;
    alltoallv_attr& operator=(const alltoallv_attr& src);
    ~alltoallv_attr();

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <alltoallv_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<alltoallv_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <alltoallv_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<alltoallv_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    alltoallv_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Barrier coll attributes
 */
class barrier_attr : public ccl_api_base_copyable<barrier_attr,
                                                  copy_on_write_access_policy,
                                                  ccl_barrier_attr_impl_t> {
public:
    using base_t =
        ccl_api_base_copyable<barrier_attr, copy_on_write_access_policy, ccl_barrier_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    barrier_attr(barrier_attr&& src);
    barrier_attr(const barrier_attr& src);
    barrier_attr& operator=(barrier_attr&& src) noexcept;
    barrier_attr& operator=(const barrier_attr& src);
    ~barrier_attr();

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <barrier_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<barrier_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <barrier_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<barrier_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    barrier_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Broadcast coll attributes
 */
class broadcast_attr : public ccl_api_base_copyable<broadcast_attr,
                                                    copy_on_write_access_policy,
                                                    ccl_broadcast_attr_impl_t> {
public:
    using base_t = ccl_api_base_copyable<broadcast_attr,
                                         copy_on_write_access_policy,
                                         ccl_broadcast_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    broadcast_attr(broadcast_attr&& src);
    broadcast_attr(const broadcast_attr& src);
    broadcast_attr& operator=(broadcast_attr&& src) noexcept;
    broadcast_attr& operator=(const broadcast_attr& src);
    ~broadcast_attr();

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <broadcast_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<broadcast_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <broadcast_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<broadcast_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    broadcast_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Reduce coll attributes
 */
class reduce_attr : public ccl_api_base_copyable<reduce_attr,
                                                 copy_on_write_access_policy,
                                                 ccl_reduce_attr_impl_t> {
public:
    using base_t =
        ccl_api_base_copyable<reduce_attr, copy_on_write_access_policy, ccl_reduce_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    reduce_attr(reduce_attr&& src);
    reduce_attr(const reduce_attr& src);
    reduce_attr& operator=(reduce_attr&& src) noexcept;
    reduce_attr& operator=(const reduce_attr& src);
    ~reduce_attr();

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <reduce_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<reduce_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <reduce_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<reduce_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    reduce_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Reduce_scatter coll attributes
 */
class reduce_scatter_attr : public ccl_api_base_copyable<reduce_scatter_attr,
                                                         copy_on_write_access_policy,
                                                         ccl_reduce_scatter_attr_impl_t> {
public:
    using base_t = ccl_api_base_copyable<reduce_scatter_attr,
                                         copy_on_write_access_policy,
                                         ccl_reduce_scatter_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    reduce_scatter_attr(reduce_scatter_attr&& src);
    reduce_scatter_attr(const reduce_scatter_attr& src);
    reduce_scatter_attr& operator=(reduce_scatter_attr&& src) noexcept;
    reduce_scatter_attr& operator=(const reduce_scatter_attr& src);
    ~reduce_scatter_attr();

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <reduce_scatter_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<reduce_scatter_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <reduce_scatter_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<reduce_scatter_attr_id, attrId>::return_type&
    get() const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    reduce_scatter_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Point to point operation attributes
 */
class pt2pt_attr : public ccl_api_base_copyable<pt2pt_attr,
                                                copy_on_write_access_policy,
                                                ccl_pt2pt_attr_impl_t> {
public:
    using base_t =
        ccl_api_base_copyable<pt2pt_attr, copy_on_write_access_policy, ccl_pt2pt_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    pt2pt_attr(pt2pt_attr&& src);
    pt2pt_attr(const pt2pt_attr& src);
    pt2pt_attr& operator=(pt2pt_attr&& src) noexcept;
    pt2pt_attr& operator=(const pt2pt_attr& src);
    ~pt2pt_attr();

    /**
     * Set specific value for attribute by @attrId.
     * Previous attibute value would be returned
     */
    template <pt2pt_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<pt2pt_attr_id, attrId>::return_type set(const Value& v);

    template <operation_attr_id attrId,
              class Value/*,
              class = typename std::enable_if<is_attribute_value_supported<attrId, Value>()>::type*/>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(const Value& v);

    /**
     * Get specific attribute value by @attrId
     */
    template <pt2pt_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<pt2pt_attr_id, attrId>::return_type& get()
        const;

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    pt2pt_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Scatter coll attributes
 */
class scatter_attr : public ccl_api_base_copyable<scatter_attr,
                                                  copy_on_write_access_policy,
                                                  ccl_scatter_attr_impl_t> {
public:
    using base_t =
        ccl_api_base_copyable<scatter_attr, copy_on_write_access_policy, ccl_scatter_attr_impl_t>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    scatter_attr(scatter_attr&& src);
    scatter_attr(const scatter_attr& src);
    scatter_attr& operator=(scatter_attr&& src) noexcept;
    scatter_attr& operator=(const scatter_attr& src);
    ~scatter_attr();

    template <operation_attr_id attrId, class Value>
    typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type set(
        const Value& v);

    template <operation_attr_id attrId>
    const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type& get()
        const;

private:
    friend class ccl::detail::environment;
    friend struct ccl::ccl_empty_attr;
    scatter_attr(
        const typename detail::ccl_api_type_attr_traits<operation_attr_id,
                                                        operation_attr_id::version>::type& version);
};

/**
 * Declare extern empty attributes
 */
extern allgather_attr default_allgather_attr;
extern allgatherv_attr default_allgatherv_attr;
extern allreduce_attr default_allreduce_attr;
extern alltoall_attr default_alltoall_attr;
extern alltoallv_attr default_alltoallv_attr;
extern barrier_attr default_barrier_attr;
extern broadcast_attr default_broadcast_attr;
extern pt2pt_attr default_pt2pt_attr;
extern reduce_attr default_reduce_attr;
extern reduce_scatter_attr default_reduce_scatter_attr;
extern scatter_attr default_scatter_attr;

/**
 * Fabric helpers
 */

template <allgather_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> detail::attr_value_triple<allgather_attr_id, t, value_type> {
    return detail::attr_value_triple<allgather_attr_id, t, value_type>(v);
}

template <allgatherv_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> detail::attr_value_triple<allgatherv_attr_id, t, value_type> {
    return detail::attr_value_triple<allgatherv_attr_id, t, value_type>(v);
}

template <allreduce_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> detail::attr_value_triple<allreduce_attr_id, t, value_type> {
    return detail::attr_value_triple<allreduce_attr_id, t, value_type>(v);
}

template <alltoall_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> detail::attr_value_triple<alltoall_attr_id, t, value_type> {
    return detail::attr_value_triple<alltoall_attr_id, t, value_type>(v);
}

template <alltoallv_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> detail::attr_value_triple<alltoallv_attr_id, t, value_type> {
    return detail::attr_value_triple<alltoallv_attr_id, t, value_type>(v);
}

template <barrier_attr_id t, class value_type>
constexpr auto attr_val(value_type v) -> detail::attr_value_triple<barrier_attr_id, t, value_type> {
    return detail::attr_value_triple<barrier_attr_id, t, value_type>(v);
}

template <broadcast_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> detail::attr_value_triple<broadcast_attr_id, t, value_type> {
    return detail::attr_value_triple<broadcast_attr_id, t, value_type>(v);
}

template <pt2pt_attr_id t, class value_type>
constexpr auto attr_val(value_type v) -> detail::attr_value_triple<pt2pt_attr_id, t, value_type> {
    return detail::attr_value_triple<pt2pt_attr_id, t, value_type>(v);
}

template <reduce_attr_id t, class value_type>
constexpr auto attr_val(value_type v) -> detail::attr_value_triple<reduce_attr_id, t, value_type> {
    return detail::attr_value_triple<reduce_attr_id, t, value_type>(v);
}

template <reduce_scatter_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> detail::attr_value_triple<reduce_scatter_attr_id, t, value_type> {
    return detail::attr_value_triple<reduce_scatter_attr_id, t, value_type>(v);
}

template <operation_attr_id t, class value_type>
constexpr auto attr_val(value_type v)
    -> detail::attr_value_triple<operation_attr_id, t, value_type> {
    return detail::attr_value_triple<operation_attr_id, t, value_type>(v);
}

/* TODO temporary function for UT compilation: would be part of detail::environment in final*/
template <class coll_attribute_type, class... attr_val_type>
coll_attribute_type create_coll_attr(attr_val_type&&... avs);

} // namespace v1

using v1::attr_val;

using v1::allgather_attr;
using v1::allgatherv_attr;
using v1::allreduce_attr;
using v1::alltoall_attr;
using v1::alltoallv_attr;
using v1::barrier_attr;
using v1::broadcast_attr;
using v1::pt2pt_attr;
using v1::reduce_attr;
using v1::reduce_scatter_attr;
using v1::scatter_attr;

using v1::default_allgather_attr;
using v1::default_allgatherv_attr;
using v1::default_allreduce_attr;
using v1::default_alltoall_attr;
using v1::default_alltoallv_attr;
using v1::default_barrier_attr;
using v1::default_broadcast_attr;
using v1::default_pt2pt_attr;
using v1::default_reduce_attr;
using v1::default_reduce_scatter_attr;
using v1::default_scatter_attr;

} // namespace ccl

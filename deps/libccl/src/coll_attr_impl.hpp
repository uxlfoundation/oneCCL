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
#include "oneapi/ccl/coll_attr.hpp"
#include "coll/attr/ccl_attrs.hpp"

namespace ccl {

namespace v1 {

/**
 * Allgather attributes definition
 */

template<allgather_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<allgather_attr_id, attrId>::return_type allgather_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<allgather_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type allgather_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <allgather_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<allgather_attr_id, attrId>::return_type&
allgather_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<allgather_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
allgather_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * Allgatherv attributes definition
 */
template<allgatherv_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<allgatherv_attr_id, attrId>::return_type allgatherv_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<allgatherv_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type allgatherv_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <allgatherv_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<allgatherv_attr_id, attrId>::return_type&
allgatherv_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<allgatherv_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
allgatherv_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * Allreduce attributes definition
 */
template<allreduce_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<allreduce_attr_id, attrId>::return_type allreduce_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<allreduce_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type allreduce_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <allreduce_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<allreduce_attr_id, attrId>::return_type&
allreduce_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<allreduce_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
allreduce_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * alltoall attributes definition
 */
template<alltoall_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<alltoall_attr_id, attrId>::return_type alltoall_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<alltoall_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type alltoall_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <alltoall_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<alltoall_attr_id, attrId>::return_type&
alltoall_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<alltoall_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
alltoall_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * alltoallv attributes definition
 */
template<alltoallv_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<alltoallv_attr_id, attrId>::return_type alltoallv_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<alltoallv_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type alltoallv_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <alltoallv_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<alltoallv_attr_id, attrId>::return_type&
alltoallv_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<alltoallv_attr_id, attrId>{});
}

template <operation_attr_id attrId>
const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
alltoallv_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * broadcast attributes definition
 */
template<broadcast_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<broadcast_attr_id, attrId>::return_type broadcast_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<broadcast_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type broadcast_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <broadcast_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<broadcast_attr_id, attrId>::return_type&
broadcast_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<broadcast_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
broadcast_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * reduce attributes definition
 */
template<reduce_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<reduce_attr_id, attrId>::return_type reduce_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<reduce_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type reduce_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <reduce_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<reduce_attr_id, attrId>::return_type&
reduce_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<reduce_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
reduce_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * reduce_scatter attributes definition
 */
template<reduce_scatter_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<reduce_scatter_attr_id, attrId>::return_type reduce_scatter_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<reduce_scatter_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type reduce_scatter_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <reduce_scatter_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<reduce_scatter_attr_id,
                                                        attrId>::return_type&
reduce_scatter_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<reduce_scatter_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
reduce_scatter_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * barrier attributes definition
 */
template<barrier_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<barrier_attr_id, attrId>::return_type barrier_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<barrier_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type barrier_attr::set(const Value& v)
{
    return get_impl().get()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <barrier_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<barrier_attr_id, attrId>::return_type&
barrier_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<barrier_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
barrier_attr::get() const {
    return get_impl().get()->get_attribute_value(
        detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * pt2pt attributes definition
 */
template<pt2pt_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<pt2pt_attr_id, attrId>::return_type pt2pt_attr::set(const Value& v)
{
    return get_impl()->set_attribute_value(
        v, detail::ccl_api_type_attr_traits<pt2pt_attr_id, attrId>{});
}

template<operation_attr_id attrId,
             class Value/*,
             typename T*/>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type pt2pt_attr::set(const Value& v)
{
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <pt2pt_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<pt2pt_attr_id, attrId>::return_type&
pt2pt_attr::get() const {
    return get_impl()->get_attribute_value(
        detail::ccl_api_type_attr_traits<pt2pt_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
pt2pt_attr::get() const {
    return static_cast<const ccl_operation_attr_impl_t*>(get_impl().get())
        ->get_attribute_value(detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

/**
 * scatter attributes definition
 */
template <operation_attr_id attrId, class Value>
CCL_API typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type
scatter_attr::set(const Value& v) {
    return static_cast<ccl_operation_attr_impl_t*>(get_impl().get())
        ->set_attribute_value(v, detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

template <operation_attr_id attrId>
CCL_API const typename detail::ccl_api_type_attr_traits<operation_attr_id, attrId>::return_type&
scatter_attr::get() const {
    return get_impl().get()->get_attribute_value(
        detail::ccl_api_type_attr_traits<operation_attr_id, attrId>{});
}

} // namespace v1

} // namespace ccl

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

namespace v1 {

/******************** INIT ********************/

/** @defgroup init
 * @{
 */
/** @} */ // end of init

/**
 * \ingroup init
 * \brief Creates an attribute object that may be used to control the init operation
 * @return an attribute object
 */
template <class... attr_val_type>
init_attr CCL_API create_init_attr(attr_val_type&&... avs) {
    return detail::environment::create_init_attr(std::forward<attr_val_type>(avs)...);
}

/**
 * \ingroup init
 * \brief Initializes the library. Optional for invocation.
 * @param attr optional init attributes
 */
void CCL_API init(const init_attr& attr = default_init_attr);

/**
 * \ingroup init
 * \brief Retrieves the library version
 */
library_version CCL_API get_library_version();

/******************** DATATYPE ********************/

/** @defgroup datatype
 * @{
 */
/** @} */ // end of datatype

/**
 * \ingroup datatype
 * \brief Creates an attribute object that may be used to register custom datatype
 * @return an attribute object
 */
template <class... attr_val_type>
datatype_attr CCL_API create_datatype_attr(attr_val_type&&... avs) {
    return detail::environment::create_datatype_attr(std::forward<attr_val_type>(avs)...);
}

/**
 * \ingroup datatype
 * \brief Registers custom datatype to be used in communication operations
 * @param attr datatype attributes
 * @return datatype handle
 */
datatype CCL_API register_datatype(const datatype_attr& attr);

/**
 * \ingroup datatype
 * \brief Deregisters custom datatype
 * @param dtype custom datatype handle
 */
void CCL_API deregister_datatype(datatype dtype);

/**
 * \ingroup datatype
 * \brief Retrieves a datatype size in bytes
 * @param dtype datatype handle
 * @return datatype size
 */
size_t CCL_API get_datatype_size(datatype dtype);

/******************** KVS ********************/

/** @defgroup kvs
 * @{
 */
/** @} */ // end of kvs

/**
 * \ingroup kvs
 */
template <class... attr_val_type>
kvs_attr CCL_API create_kvs_attr(attr_val_type&&... avs) {
    return detail::environment::create_kvs_attr(std::forward<attr_val_type>(avs)...);
}

/**
 * \ingroup kvs
 * \brief Creates a main key-value store.
 *        Its address should be distributed using out of band communication mechanism
 *        and be used to create key-value stores on other processes.
 * @param attr optional kvs attributes
 * @return kvs object
 */
shared_ptr_class<kvs> CCL_API create_main_kvs(const kvs_attr& attr = default_kvs_attr);

/**
 * \ingroup kvs
 * \brief Creates a new key-value store from main kvs address
 * @param addr address of main kvs
 * @param attr optional kvs attributes
 * @return kvs object
 */
shared_ptr_class<kvs> CCL_API create_kvs(const kvs::address_type& addr,
                                         const kvs_attr& attr = default_kvs_attr);

/******************** DEVICE ********************/

/** @defgroup device
 * @{
 */
/** @} */ // end of device

/**
 * \ingroup device
 * \brief Creates a new device from @native_device_type
 * @param native_device the existing handle of device
 * @return device object
 */
template <class native_device_type,
          class = typename std::enable_if<is_device_supported<native_device_type>()>::type>
device CCL_API create_device(native_device_type&& native_device) {
    return detail::environment::instance().create_device(
        std::forward<native_device_type>(native_device));
}

/**
 * \ingroup device
 */
device CCL_API create_device();

/******************** CONTEXT ********************/

/** @defgroup context
 * @{
 */
/** @} */ // end of context

/**
 * \ingroup context
 * \brief Creates a new context from @native_contex_type
 * @param native_context the existing handle of context
 * @return context object
 */
template <class native_context_type,
          class = typename std::enable_if<is_context_supported<native_context_type>()>::type>
context CCL_API create_context(native_context_type&& native_context) {
    return detail::environment::instance().create_context(
        std::forward<native_context_type>(native_context));
}

/**
 * \ingroup context
 */
context CCL_API create_context();

/******************** EVENT ********************/

/** @defgroup event
 * @{
 */
/** @} */ // end of event

/**
 * \ingroup event
 * \brief Creates a new event from @native_event_type
 * @param native_event the existing event
 * @return event object
 */
template <class event_type, class = typename std::enable_if<is_event_supported<event_type>()>::type>
event CCL_API create_event(event_type& native_event) {
    return detail::environment::instance().create_event(native_event);
}

/******************** STREAM ********************/

/** @defgroup stream
 * @{
 */
/** @} */ // end of stream

/**
 * \ingroup stream
 * \brief Creates a new stream from @native_stream_type
 * @param native_stream the existing handle of stream
 * @return stream object
 */
template <class native_stream_type,
          class = typename std::enable_if<is_stream_supported<native_stream_type>()>::type>
stream CCL_API create_stream(native_stream_type& native_stream) {
    return detail::environment::instance().create_stream(native_stream);
}

/**
 * \ingroup stream
 */
stream CCL_API create_stream();

/******************** COMMUNICATOR ********************/

/** @defgroup communicator
 * @{
 */
/** @} */ // end of communicator

/**
 * \ingroup communicator
 * \brief Creates an attribute object that may be used to control the create_communicator operation
 * @return an attribute object
 */
template <class... attr_val_type>
comm_attr CCL_API create_comm_attr(attr_val_type&&... avs) {
    return detail::environment::create_comm_attr(std::forward<attr_val_type>(avs)...);
}

} // namespace v1

namespace preview {

/**
 * \ingroup communicator
 * \brief Creates an attribute object that may be used to control the split_communicator operation
 * @return an attribute object
 */
template <class... attr_val_type>
comm_split_attr CCL_API create_comm_split_attr(attr_val_type&&... avs) {
    return detail::environment::create_comm_split_attr(std::forward<attr_val_type>(avs)...);
}

} // namespace preview

namespace v1 {

/**
 * \ingroup communicator
 * \brief Creates new communicators with user supplied size, ranks, local device-rank mapping and kvs.
 * @param size user-supplied total number of ranks
 * @param rank user-supplied rank
 * @param device local device
 * @param devices user-supplied mapping of local ranks on devices
 * @param context context containing the devices
 * @param kvs key-value store for ranks wire-up
 * @param attr optional communicator attributes
 * @return vector of communicators / communicator
 */
template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API
create_communicators(int size,
                     const vector_class<pair_class<int, DeviceType>>& devices,
                     const ContextType& context,
                     shared_ptr_class<kvs_interface> kvs,
                     const comm_attr& attr = default_comm_attr) {
    return detail::environment::instance().create_communicators(size, devices, context, kvs, attr);
}

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API
create_communicatorsExt(int size,
                        const vector_class<pair_class<int, DeviceType>>& devices,
                        const ContextType& context,
                        shared_ptr_class<kvs_interface> kvs,
                        const comm_attr& attr = default_comm_attr) {
    return detail::environment::instance().create_communicatorsExt(
        size, devices, context, kvs, attr);
}

/*!
 * \ingroup communicator
 * \overload
 */
template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API create_communicators(int size,
                                                        const map_class<int, DeviceType>& devices,
                                                        const ContextType& context,
                                                        shared_ptr_class<kvs_interface> kvs,
                                                        const comm_attr& attr = default_comm_attr) {
    return detail::environment::instance().create_communicators(size, devices, context, kvs, attr);
}

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API
create_communicatorsExt(int size,
                        const map_class<int, DeviceType>& devices,
                        const ContextType& context,
                        shared_ptr_class<kvs_interface> kvs,
                        const comm_attr& attr = default_comm_attr) {
    return detail::environment::instance().create_communicatorsExt(
        size, devices, context, kvs, attr);
}

/*!
 * \ingroup communicator
 * \overload
 */
template <class DeviceType, class ContextType>
communicator CCL_API create_communicator(int size,
                                         int rank,
                                         DeviceType& device,
                                         const ContextType& context,
                                         shared_ptr_class<kvs_interface> kvs,
                                         const comm_attr& attr = default_comm_attr) {
    auto comms = detail::environment::instance().create_communicators(
        size,
        ccl::vector_class<ccl::pair_class<int, ccl::device>>{ { rank, device } },
        context,
        kvs,
        attr);

    if (comms.size() != 1)
        throw ccl::exception("unexpected comm vector size");

    return std::move(comms[0]);
}

/*!
 * \ingroup communicator
 * \overload
 */
template <class DeviceType, class ContextType>
communicator CCL_API create_communicatorExt(int size,
                                            int rank,
                                            DeviceType& device,
                                            const ContextType& context,
                                            shared_ptr_class<kvs_interface> kvs,
                                            const comm_attr& attr = default_comm_attr) {
    auto comms = detail::environment::instance().create_communicatorsExt(
        size,
        ccl::vector_class<ccl::pair_class<int, ccl::device>>{ { rank, device } },
        context,
        kvs,
        attr);
    if (comms.size() != 1)
        throw ccl::exception("unexpected comm vector size");

    return std::move(comms[0]);
}

/*!
 * \ingroup communicator
 * \overload
 */
communicator CCL_API create_communicator(int size,
                                         int rank,
                                         shared_ptr_class<kvs_interface> kvs,
                                         const comm_attr& attr = default_comm_attr);

} // namespace v1

namespace preview {

/**
 * \ingroup communicator
 * \brief Creates a new communicators with user supplied size, local devices and kvs.
 *        Ranks will be assigned automatically.
 * @param size user-supplied total number of ranks
 * @param devices user-supplied device objects for local ranks
 * @param context context containing the devices
 * @param kvs key-value store for ranks wire-up
 * @param attr optional communicator attributes
 * @return vector of communicators / communicator
 */
template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API create_communicators(int size,
                                                        const vector_class<DeviceType>& devices,
                                                        const ContextType& context,
                                                        shared_ptr_class<kvs_interface> kvs,
                                                        const comm_attr& attr = default_comm_attr) {
    return detail::environment::instance().create_communicators(size, devices, context, kvs, attr);
}

template <class DeviceType, class ContextType>
vector_class<communicator> CCL_API
create_communicatorsExt(int size,
                        const vector_class<DeviceType>& devices,
                        const ContextType& context,
                        shared_ptr_class<kvs_interface> kvs,
                        const comm_attr& attr = default_comm_attr) {
    return detail::environment::instance().create_communicatorsExt(
        size, devices, context, kvs, attr);
}

/*!
 * \ingroup communicator
 * \overload
 */
communicator CCL_API create_communicator(int size,
                                         shared_ptr_class<kvs_interface> kvs,
                                         const comm_attr& attr = default_comm_attr);

/*!
 * \overload
 */
/**
 * \ingroup communicator
 * \brief Creates a new communicator with externally provided size, rank and kvs.
 *        Implementation is platform specific and non portable.
  * @param attr optional communicator attributes
 * @return communicator
 */
communicator CCL_API create_communicator(const comm_attr& attr = default_comm_attr);

} // namespace preview
namespace v1 {

/*************** SPLIT COMMUNICATOR *****************/
/**
 * \ingroup communicator
 * \brief Splits communicators according to attributes.
 * @param attrs split attributes for local communicators
 * @return vector of communicators
 */
communicator CCL_API split_communicator(const communicator& comm, int color, int key);

/****************** GROUP CALLS ********************/

/** @defgroup group_calls
 * @{
 */
/**
 * \ingroup group_calls
 * \brief group_start() starts a group call. You can use the group_start() function 
 * to initiate a group call operation which indicates that 
 * successive operations should not get blocked due to CPU synchronization.
 */
void CCL_API group_start();
/**
 * \ingroup group_calls
 * \brief group_end() ends a group call. The group_end() call returns when all the operations 
 * between group_start() and group_end() have been enqueued for execution, 
 * but not necessarily completed.
 */
void CCL_API group_end();
/** @} */ // end of group_calls

/******************** OPERATION ********************/

/** @defgroup operation
 * @{
 */
/** @} */ // end of operation

/**
 * \ingroup operation
 * \brief Creates an attribute object that may be used to customize communication operation
 * @return an attribute object
 */
template <class coll_attribute_type, class... attr_val_type>
coll_attribute_type CCL_API create_operation_attr(attr_val_type&&... avs) {
    return detail::environment::create_operation_attr<coll_attribute_type>(
        std::forward<attr_val_type>(avs)...);
}

/** @defgroup allgather
 * \ingroup operation
 * @{
 */

/**
 * \brief Allgather is a collective communication operation that collects
 *        send_count elements from all the ranks within the communicator and places the results
 *        into recv_buf, in such a way that data from rank i can be found at offset rank i * send_count.
 *        The resulting data in the output recv_buf buffer is the same for each rank.
 *
 * @param send_buf the buffer with @c send_count elements of @c dtype that stores local data to be gathered
 * @param recv_buf [out] the buffer to store gathered result of @c dtype, must be large enough
 *        to hold values from all ranks, i.e. size should be equal to @c dtype size in bytes * @c send_count
 * @param send_count the number of elements of type @c dtype to be sent to or to received from each rank
 * @param dtype the datatype of elements in @c send_buf and @c recv_buf
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API allgather(const void* send_buf,
                        void* recv_buf,
                        size_t send_count,
                        datatype dtype,
                        const communicator& comm,
                        const stream& stream,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API allgather(const void* send_buf,
                        void* recv_buf,
                        size_t send_count,
                        datatype dtype,
                        const communicator& comm,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * @param send_buf  the buffer of @c dtype  with @c send_count elements that stores local data to be gathered
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 *                   each buffer must be large enough to keep @c send_count elements of @c dtype size
 */
event CCL_API allgather(const void* send_buf,
                        const vector_class<void*>& recv_bufs,
                        size_t send_count,
                        datatype dtype,
                        const communicator& comm,
                        const stream& stream,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 */
event CCL_API allgather(const void* send_buf,
                        const vector_class<void*>& recv_bufs,
                        size_t send_count,
                        datatype dtype,
                        const communicator& comm,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
 * @param recv_buf [out] the buffer to store gathered result of @c BufferType.
 * It must be large enough to hold values from all ranks,
 * i.e. size should be equal to @c BufferType size in bytes * @c send_count
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allgather(const BufferType* send_buf,
                        BufferType* recv_buf,
                        size_t send_count,
                        const communicator& comm,
                        const stream& stream,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allgather(const BufferType* send_buf,
                        BufferType* recv_buf,
                        size_t send_count,
                        const communicator& comm,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf  the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 *                   each buffer must be large enough to keep @c send_count elements of @c BufferType size
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allgather(const BufferType* send_buf,
                        vector_class<BufferType*>& recv_bufs,
                        size_t send_count,
                        const communicator& comm,
                        const stream& stream,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allgather(const BufferType* send_buf,
                        vector_class<BufferType*>& recv_bufs,
                        size_t send_count,
                        const communicator& comm,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer of @c BufferObjectType with @c send_count elements that stores local data to be gathered
 * @param recv_buf [out] the buffer to store gathered result of @c BufferObjectType.
 *                 It must be large enough to hold values from all ranks,
 *                 i.e. size should be equal to @c BufferObjectType size in bytes * @c send_count
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allgather(const BufferObjectType& send_buf,
                        BufferObjectType& recv_buf,
                        size_t send_count,
                        const communicator& comm,
                        const stream& stream,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allgather(const BufferObjectType& send_buf,
                        BufferObjectType& recv_buf,
                        size_t send_count,
                        const communicator& comm,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer of @c BufferObjectType with @c send_count elements that stores local data to be gathered
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 *                   each buffer must be large enough to keep send_count elements of @c BufferObjectType size
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allgather(const BufferObjectType& send_buf,
                        vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                        size_t send_count,
                        const communicator& comm,
                        const stream& stream,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allgather(const BufferObjectType& send_buf,
                        vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                        size_t send_count,
                        const communicator& comm,
                        const allgather_attr& attr = default_allgather_attr,
                        const vector_class<event>& deps = {});
/** @} */ // end of allgather

/** @defgroup allgatherv
 * \ingroup operation
 * @{
 */

/**
 * \brief Allgatherv is a collective communication operation that collects data
 *        from all the ranks within a communicator into a single buffer.
 *        Different ranks may contribute segments of different sizes.
 *        The resulting data in the output buffer is the same for each rank.
 *.
 * @param send_buf the buffer with @c send_count elements of @c dtype that stores local data to be gathered
 * @param send_count the number of elements of type @c dtype in @c send_buf
 * @param recv_buf [out] the buffer to store gathered result of @c dtype, must be large enough
 *                      to hold values from all ranks, i.e. size should be equal
 *                      to @c dtype size in bytes * sum of all values in @c recv_counts
 * @param recv_counts array with the number of elements of type @c dtype to be received from each rank
 * @param dtype the datatype of elements in @c send_buf and @c recv_buf
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API allgatherv(const void* send_buf,
                         size_t send_count,
                         void* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         const communicator& comm,
                         const stream& stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API allgatherv(const void* send_buf,
                         size_t send_count,
                         void* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         const communicator& comm,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * This overloaded function takes separate receive buffer per rank.
 *
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 * each buffer must be large enough to keep the corresponding @c recv_counts elements of @c dtype size
 */
event CCL_API allgatherv(const void* send_buf,
                         size_t send_count,
                         const vector_class<void*>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         const communicator& comm,
                         const stream& stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * This overloaded function takes separate receive buffer per rank.
 *
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 * each buffer must be large enough to keep the corresponding @c recv_counts elements of @c dtype size
 */
event CCL_API allgatherv(const void* send_buf,
                         size_t send_count,
                         const vector_class<void*>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         datatype dtype,
                         const communicator& comm,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
 * @param recv_buf [out] the buffer to store gathered result of @c BufferType, must be large enough
 *                      to hold values from all ranks, i.e. size should be equal
 *                      to @c BufferType size in bytes * sum of all values in @c recv_counts
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allgatherv(const BufferType* send_buf,
                         size_t send_count,
                         BufferType* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const stream& stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
 * @param recv_buf [out] the buffer to store gathered result of @c BufferType, must be large enough
 *                      to hold values from all ranks, i.e. size should be equal
 *                      to @c BufferType size in bytes * sum of all values in @c recv_counts
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allgatherv(const BufferType* send_buf,
                         size_t send_count,
                         BufferType* recv_buf,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 * each buffer must be large enough to keep the corresponding @c recv_counts elements of @c BufferType size
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allgatherv(const BufferType* send_buf,
                         size_t send_count,
                         vector_class<BufferType*>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const stream& stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer with @c send_count elements of @c BufferType that stores local data to be gathered
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 * each buffer must be large enough to keep the corresponding @c recv_counts elements of @c BufferType size
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allgatherv(const BufferType* send_buf,
                         size_t send_count,
                         vector_class<BufferType*>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer of @c BufferObjectType with @c send_count elements that stores local data to be gathered
 * @param recv_buf [out] the buffer of @c BufferObjectType to store gathered result, must be large enough
 *                      to hold values from all ranks, i.e. size should be equal
 *                      to @c BufferType size in bytes * sum of all values in @c recv_counts
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allgatherv(const BufferObjectType& send_buf,
                         size_t send_count,
                         BufferObjectType& recv_buf,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const stream& stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer of @c BufferObjectType with @c send_count elements that stores local data to be gathered
 * @param recv_buf [out] the buffer of @c BufferObjectType to store gathered result, must be large enough
 *                      to hold values from all ranks, i.e. size should be equal
 *                      to @c BufferType size in bytes * sum of all values in @c recv_counts
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allgatherv(const BufferObjectType& send_buf,
                         size_t send_count,
                         BufferObjectType& recv_buf,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer of @c BufferObjectType with @c send_count elements that stores local data to be gathered
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 * each buffer must be large enough to keep the corresponding @c recv_counts elements of @c BufferObjectType size
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allgatherv(const BufferObjectType& send_buf,
                         size_t send_count,
                         vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const stream& stream,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_buf the buffer of @c BufferObjectType with @c send_count elements that stores local data to be gathered
 * @param recv_bufs [out] array of buffers to store gathered result, one buffer per rank;
 * each buffer must be large enough to keep the corresponding @c recv_counts elements of @c BufferObjectType size
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allgatherv(const BufferObjectType& send_buf,
                         size_t send_count,
                         vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                         const vector_class<size_t>& recv_counts,
                         const communicator& comm,
                         const allgatherv_attr& attr = default_allgatherv_attr,
                         const vector_class<event>& deps = {});
/** @} */ // end of allgatherv

/** @defgroup allreduce
 * \ingroup operation
 * @{
 */

/**
 * \brief Allreduce is a collective communication operation that performs the global reduction operation
 *        on values from all ranks of communicator and distributes the result back to all ranks.
 * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
 * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf
 * @param count the number of elements of type @c dtype in @c send_buf and @c recv_buf
 * @param dtype the datatype of elements in @c send_buf and @c recv_buf`
 * @param rtype the type of the reduction operation to be applied
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API allreduce(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        datatype dtype,
                        reduction rtype,
                        const communicator& comm,
                        const stream& stream,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API allreduce(const void* send_buf,
                        void* recv_buf,
                        size_t count,
                        datatype dtype,
                        reduction rtype,
                        const communicator& comm,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allreduce(const BufferType* send_buf,
                        BufferType* recv_buf,
                        size_t count,
                        reduction rtype,
                        const communicator& comm,
                        const stream& stream,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API allreduce(const BufferType* send_buf,
                        BufferType* recv_buf,
                        size_t count,
                        reduction rtype,
                        const communicator& comm,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allreduce(const BufferObjectType& send_buf,
                        BufferObjectType& recv_buf,
                        size_t count,
                        reduction rtype,
                        const communicator& comm,
                        const stream& stream,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API allreduce(const BufferObjectType& send_buf,
                        BufferObjectType& recv_buf,
                        size_t count,
                        reduction rtype,
                        const communicator& comm,
                        const allreduce_attr& attr = default_allreduce_attr,
                        const vector_class<event>& deps = {});
/** @} */ // end of allreduce

/** @defgroup alltoall
 * \ingroup operation
 * @{
 */

/**
 * \brief Alltoall is a collective communication operation in which each rank
 *        sends distinct equal-sized blocks of data to each rank.
 *        The j-th block of @c send_buf sent from the i-th rank is received by the j-th rank
 *        and is placed in the i-th block of @c recvbuf.
 *
 * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be sent
 * @param recv_buf [out] the buffer to store received result, must be large enough
 *        to hold values from all ranks, i.e. at least @c comm_size * @c count
 * @param count the number of elements of type @c dtype to be send to or to received from each rank
 * @param dtype the datatype of elements in @c send_buf and @c recv_buf
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API alltoall(const void* send_buf,
                       void* recv_buf,
                       size_t count,
                       datatype dtype,
                       const communicator& comm,
                       const stream& stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API alltoall(const void* send_buf,
                       void* recv_buf,
                       size_t count,
                       datatype dtype,
                       const communicator& comm,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * @param send_bufs array of buffers with local data to be sent, one buffer per rank
 * @param recv_bufs [out] array of buffers to store received result, one buffer per rank
 */
event CCL_API alltoall(const vector_class<void*>& send_buf,
                       const vector_class<void*>& recv_buf,
                       size_t count,
                       datatype dtype,
                       const communicator& comm,
                       const stream& stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * @param send_bufs array of buffers with local data to be sent, one buffer per rank
 * @param recv_bufs [out] array of buffers to store received result, one buffer per rank
 */
event CCL_API alltoall(const vector_class<void*>& send_buf,
                       const vector_class<void*>& recv_buf,
                       size_t count,
                       datatype dtype,
                       const communicator& comm,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API alltoall(const BufferType* send_buf,
                       BufferType* recv_buf,
                       size_t count,
                       const communicator& comm,
                       const stream& stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API alltoall(const BufferType* send_buf,
                       BufferType* recv_buf,
                       size_t count,
                       const communicator& comm,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_bufs array of buffers with local data to be sent, one buffer per rank
 * @param recv_bufs [out] array of buffers to store received result, one buffer per rank
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API alltoall(const vector_class<BufferType*>& send_buf,
                       const vector_class<BufferType*>& recv_buf,
                       size_t count,
                       const communicator& comm,
                       const stream& stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 *
 * @param send_bufs array of buffers with local data to be sent, one buffer per rank
 * @param recv_bufs [out] array of buffers to store received result, one buffer per rank
 */
template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API alltoall(const vector_class<BufferType*>& send_buf,
                       const vector_class<BufferType*>& recv_buf,
                       size_t count,
                       const communicator& comm,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API alltoall(const BufferObjectType& send_buf,
                       BufferObjectType& recv_buf,
                       size_t count,
                       const communicator& comm,
                       const stream& stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_bufs array of buffers with local data to be sent, one buffer per rank
 * @param recv_bufs [out] array of buffers to store received result, one buffer per rank
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API alltoall(const BufferObjectType& send_buf,
                       BufferObjectType& recv_buf,
                       size_t count,
                       const communicator& comm,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_bufs array of buffers with local data to be sent, one buffer per rank
 * @param recv_bufs [out] array of buffers to store received result, one buffer per rank
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
                       const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                       size_t count,
                       const communicator& comm,
                       const stream& stream,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 *
 * @param send_bufs array of buffers with local data to be sent, one buffer per rank
 * @param recv_bufs [out] array of buffers to store received result, one buffer per rank
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API alltoall(const vector_class<reference_wrapper_class<BufferObjectType>>& send_buf,
                       const vector_class<reference_wrapper_class<BufferObjectType>>& recv_buf,
                       size_t count,
                       const communicator& comm,
                       const alltoall_attr& attr = default_alltoall_attr,
                       const vector_class<event>& deps = {});
/** @} */ // end of alltoall

/** @defgroup alltoallv
 * \ingroup operation
 * @{
 */

/**
 * \brief Alltoallv is a collective communication operation in which each rank
 *        sends distinct blocks of data to each rank. Block sizes may differ.
 *        The j-th block of @c send_buf sent from the i-th rank is received by the j-th rank
 *        and is placed in the i-th block of @c recvbuf.
 * @param send_buf the buffer with elements of @c dtype that stores local blocks to be sent to each rank
 * @param send_bufs array of buffers to store send blocks, one buffer per rank
 * @param recv_buf [out] the buffer to store received result, must be large enough to hold blocks from all ranks
 * @param recv_bufs [out] array of buffers to store receive blocks, one buffer per rank
 * @param send_counts array with the number of elements of type @c dtype in send blocks for each rank
 * @param recv_counts array with the number of elements of type @c dtype in receive blocks from each rank
 * @param dtype the datatype of elements in @c send_buf and @c recv_buf
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API alltoallv(const void* send_buf,
                        const vector_class<size_t>& send_counts,
                        void* recv_buf,
                        const vector_class<size_t>& recv_counts,
                        datatype dtype,
                        const communicator& comm,
                        const stream& stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API alltoallv(const void* send_buf,
                        const vector_class<size_t>& send_counts,
                        void* recv_buf,
                        const vector_class<size_t>& recv_counts,
                        datatype dtype,
                        const communicator& comm,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

event CCL_API alltoallv(const vector_class<void*>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<void*>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        datatype dtype,
                        const communicator& comm,
                        const stream& stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

event CCL_API alltoallv(const vector_class<void*>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<void*>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        datatype dtype,
                        const communicator& comm,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API alltoallv(const BufferType* send_buf,
                        const vector_class<size_t>& send_counts,
                        BufferType* recv_buf,
                        const vector_class<size_t>& recv_counts,
                        const communicator& comm,
                        const stream& stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API alltoallv(const BufferType* send_buf,
                        const vector_class<size_t>& send_counts,
                        BufferType* recv_buf,
                        const vector_class<size_t>& recv_counts,
                        const communicator& comm,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API alltoallv(const vector_class<BufferType*>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<BufferType*>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        const communicator& comm,
                        const stream& stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API alltoallv(const vector_class<BufferType*>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<BufferType*>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        const communicator& comm,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API alltoallv(const BufferObjectType& send_buf,
                        const vector_class<size_t>& send_counts,
                        BufferObjectType& recv_buf,
                        const vector_class<size_t>& recv_counts,
                        const communicator& comm,
                        const stream& stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API alltoallv(const BufferObjectType& send_buf,
                        const vector_class<size_t>& send_counts,
                        BufferObjectType& recv_buf,
                        const vector_class<size_t>& recv_counts,
                        const communicator& comm,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        const communicator& comm,
                        const stream& stream,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API alltoallv(const vector_class<reference_wrapper_class<BufferObjectType>>& send_bufs,
                        const vector_class<size_t>& send_counts,
                        const vector_class<reference_wrapper_class<BufferObjectType>>& recv_bufs,
                        const vector_class<size_t>& recv_counts,
                        const communicator& comm,
                        const alltoallv_attr& attr = default_alltoallv_attr,
                        const vector_class<event>& deps = {});

/** @} */ // end of alltoallv

/** @defgroup barrier
 * \ingroup operation
 * @{
 */

/**
 * \brief Barrier synchronization is performed across all ranks of the communicator
 *        and it is completed only after all the ranks in the communicator have called it.
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */

event CCL_API barrier(const communicator& comm,
                      const stream& stream,
                      const barrier_attr& attr = default_barrier_attr,
                      const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API barrier(const communicator& comm,
                      const barrier_attr& attr = default_barrier_attr,
                      const vector_class<event>& deps = {});

/** @} */ // end of barrier

/** @defgroup broadcast
 * \ingroup operation
 * @{
 */

/**
 * \brief Broadcast is a collective communication operation that broadcasts data
 *        from one rank of communicator (denoted as root) to all other ranks.
 * @param buf [in/out] the buffer with @c count elements of @c dtype
 *        serves as send buffer for root and as the output buffer for all other ranks
 * @param count the number of elements of type @c dtype in @c buf
 * @param dtype the datatype of elements in @c buf
 * @param root the rank that broadcasts @c buf
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API broadcast(void* buf,
                        size_t count,
                        datatype dtype,
                        int root,
                        const communicator& comm,
                        const stream& stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */
event CCL_API broadcast(void* buf,
                        size_t count,
                        datatype dtype,
                        int root,
                        const communicator& comm,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API broadcast(BufferType* buf,
                        size_t count,
                        int root,
                        const communicator& comm,
                        const stream& stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API broadcast(BufferType* buf,
                        size_t count,
                        int root,
                        const communicator& comm,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API broadcast(BufferObjectType& buf,
                        size_t count,
                        int root,
                        const communicator& comm,
                        const stream& stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API broadcast(BufferObjectType& buf,
                        size_t count,
                        int root,
                        const communicator& comm,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/** @} */ // end of broadcast

/** @defgroup broadcast
 * \ingroup operation
 * @{
 */

/**
 * \brief Broadcast is a collective communication operation that broadcasts data
 *        from one rank of communicator (denoted as root) to all other ranks.
 * @param send_buf [in] the buffer with @c count elements of @c dtype
 *        serves as send buffer for root. The send_buf is only used by the root rank (and ignored by all the other ranks).
 * @param recv_buf [out] the buffer with @c count elements of @c dtype
 *        serves as receive buffer for all ranks, including the root.
 * @param count the number of elements of type @c dtype in @c send_buf
 * @param dtype the datatype of elements in @c send_buf
 * @param root the rank that broadcasts @c send_buf
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API broadcast(void* send_buf,
                        void* recv_buf,
                        size_t count,
                        datatype dtype,
                        int root,
                        const communicator& comm,
                        const stream& stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */
event CCL_API broadcast(void* send_buf,
                        void* recv_buf,
                        size_t count,
                        datatype dtype,
                        int root,
                        const communicator& comm,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API broadcast(BufferType* send_buf,
                        BufferType* recv_buf,
                        size_t count,
                        int root,
                        const communicator& comm,
                        const stream& stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API broadcast(BufferType* send_buf,
                        BufferType* recv_buf,
                        size_t count,
                        int root,
                        const communicator& comm,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API broadcast(BufferObjectType& send_buf,
                        BufferObjectType& recv_buf,
                        size_t count,
                        int root,
                        const communicator& comm,
                        const stream& stream,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API broadcast(BufferObjectType& send_buf,
                        BufferObjectType& recv_buf,
                        size_t count,
                        int root,
                        const communicator& comm,
                        const broadcast_attr& attr = default_broadcast_attr,
                        const vector_class<event>& deps = {});

/** @} */ // end of broadcast

/** @defgroup recv
 * \ingroup operation
 * @{
 */

/**
 * \brief Recv is a pt2pt communication operation that receives data
 *        from one rank of communicator.
 * @param buf [in,out] the buffer with @c count elements of @c dtype
 *        serves as send buffer for root and as receive buffer for other ranks
 * @param count the number of elements of type @c dtype in @c buf
 * @param dtype the datatype of elements in @c buf
 * @param peer the rank that sends @c buf
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API recv(void* buf,
                   size_t count,
                   datatype dtype,
                   int peer,
                   const communicator& comm,
                   const stream& stream,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API recv(void* buf,
                   size_t count,
                   datatype dtype,
                   int peer,
                   const communicator& comm,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API recv(BufferType* buf,
                   size_t count,
                   int peer,
                   const communicator& comm,
                   const stream& stream,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API recv(BufferType* buf,
                   size_t count,
                   int peer,
                   const communicator& comm,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API recv(BufferObjectType& buf,
                   size_t count,
                   int peer,
                   const communicator& comm,
                   const stream& stream,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */
template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API recv(BufferObjectType& buf,
                   size_t count,
                   int peer,
                   const communicator& comm,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/** @} */ // end of recv

/** @defgroup send
 * \ingroup operation
 * @{
 */

/**
 * \brief Send is a pt2pt communication operation that sends data
 *        from one rank of communicator.
 * @param buf [in,out] the buffer with @c count elements of @c dtype
 *        serves as send buffer for root and as receive buffer for other ranks
 * @param count the number of elements of type @c dtype in @c buf
 * @param dtype the datatype of elements in @c buf
 * @param peer the rank that receives @c buf
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API send(void* buf,
                   size_t count,
                   datatype dtype,
                   int peer,
                   const communicator& comm,
                   const stream& stream,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API send(void* buf,
                   size_t count,
                   datatype dtype,
                   int peer,
                   const communicator& comm,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API send(BufferType* buf,
                   size_t count,
                   int peer,
                   const communicator& comm,
                   const stream& stream,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API send(BufferType* buf,
                   size_t count,
                   int peer,
                   const communicator& comm,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API send(BufferObjectType& buf,
                   size_t count,
                   int peer,
                   const communicator& comm,
                   const stream& stream,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API send(BufferObjectType& buf,
                   size_t count,
                   int peer,
                   const communicator& comm,
                   const pt2pt_attr& attr = default_pt2pt_attr,
                   const vector_class<event>& deps = {});

/** @} */ // end of send

/** @defgroup reduce
 * \ingroup operation
 * @{
 */

/**
 * \brief Reduce is a collective communication operation that performs the global reduction operation
 *        on values from all ranks of the communicator and returns the result to the root rank.
 * @param send_buf the buffer with @c count elements of @c dtype that stores local data to be reduced
 * @param recv_buf [out] the buffer to store reduced result, must have the same dimension as @c send_buf.
 *        Used by the @c root rank only, ignored by other ranks.
 * @param count the number of elements of type @c dtype in @c send_buf and @c recv_buf
 * @param dtype the datatype of elements in @c send_buf and @c recv_buf
 * @param rtype the type of the reduction operation to be applied
 * @param root the rank that gets the result of reduction
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API reduce(const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction rtype,
                     int root,
                     const communicator& comm,
                     const stream& stream,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API reduce(const void* send_buf,
                     void* recv_buf,
                     size_t count,
                     datatype dtype,
                     reduction rtype,
                     int root,
                     const communicator& comm,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API reduce(const BufferType* send_buf,
                     BufferType* recv_buf,
                     size_t count,
                     reduction rtype,
                     int root,
                     const communicator& comm,
                     const stream& stream,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API reduce(const BufferType* send_buf,
                     BufferType* recv_buf,
                     size_t count,
                     reduction rtype,
                     int root,
                     const communicator& comm,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API reduce(const BufferObjectType& send_buf,
                     BufferObjectType& recv_buf,
                     size_t count,
                     reduction rtype,
                     int root,
                     const communicator& comm,
                     const stream& stream,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API reduce(const BufferObjectType& send_buf,
                     BufferObjectType& recv_buf,
                     size_t count,
                     reduction rtype,
                     int root,
                     const communicator& comm,
                     const reduce_attr& attr = default_reduce_attr,
                     const vector_class<event>& deps = {});

/** @} */ // end of reduce

/** @defgroup reducescatter
 * \ingroup operation
 * @{
 */

/**
 * \brief Reduce-scatter is a collective communication operation that performs the global reduction operation
 *        on values from all ranks of the communicator and scatters the result in blocks back to all ranks.
 * @param send_buf the buffer with @c comm_size * @c count elements of @c dtype that stores local data to be reduced
 * @param recv_buf [out] the buffer to store result block containing @c recv_count elements of type @c dtype
 * @param recv_count the number of elements of type @c dtype in receive block
 * @param dtype the datatype of elements in @c send_buf and @c recv_buf
 * @param rtype the type of the reduction operation to be applied
 * @param comm the communicator for which the operation will be performed
 * @param stream abstraction over a device queue constructed via ccl::create_stream
 * @param attr optional attributes to customize operation
 * @param deps an optional vector of the events that the operation should depend on
 * @return @ref ccl::event an object to track the progress of the operation
 */
event CCL_API reduce_scatter(const void* send_buf,
                             void* recv_buf,
                             size_t recv_count,
                             datatype dtype,
                             reduction rtype,
                             const communicator& comm,
                             const stream& stream,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

/*!
 * \overload
 */
event CCL_API reduce_scatter(const void* send_buf,
                             void* recv_buf,
                             size_t recv_count,
                             datatype dtype,
                             reduction rtype,
                             const communicator& comm,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API reduce_scatter(const BufferType* send_buf,
                             BufferType* recv_buf,
                             size_t recv_count,
                             reduction rtype,
                             const communicator& comm,
                             const stream& stream,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferType,
          class = typename std::enable_if<is_native_type_supported<BufferType>(), event>::type>
event CCL_API reduce_scatter(const BufferType* send_buf,
                             BufferType* recv_buf,
                             size_t recv_count,
                             reduction rtype,
                             const communicator& comm,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

/*!
 * \overload
 *
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API reduce_scatter(const BufferObjectType& send_buf,
                             BufferObjectType& recv_buf,
                             size_t recv_count,
                             reduction rtype,
                             const communicator& comm,
                             const stream& stream,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

/*!
 * \overload
 *
 * Type-safe version.
 */

template <class BufferObjectType,
          class = typename std::enable_if<is_class_supported<BufferObjectType>(), event>::type>
event CCL_API reduce_scatter(const BufferObjectType& send_buf,
                             BufferObjectType& recv_buf,
                             size_t recv_count,
                             reduction rtype,
                             const communicator& comm,
                             const reduce_scatter_attr& attr = default_reduce_scatter_attr,
                             const vector_class<event>& deps = {});

/** @} */ // end of reduce_scatter
} // namespace v1

using namespace v1;

} // namespace ccl

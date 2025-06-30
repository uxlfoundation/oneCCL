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

struct comm_interface;

template <cl_backend_type type>
struct comm_impl_dispatch_selector;

namespace v1 {
class context;
class device;
class kvs_interface;
struct impl_dispatch;

/**
 * A communicator that permits communication operations
 * Has no defined public constructor.
 * Use ccl::create_communicator for communicator objects creation.
 */
class communicator final : public ccl_api_base_movable<communicator,
                                                       direct_access_policy,
                                                       comm_interface,
                                                       std::shared_ptr> {
public:
    using base_t =
        ccl_api_base_movable<communicator, direct_access_policy, comm_interface, std::shared_ptr>;

    /**
     * Declare PIMPL type
     */
    using impl_value_t = typename base_t::impl_value_t;

    /**
     * Declare implementation type
     */
    using impl_t = typename impl_value_t::element_type;

    /**
     * Type allows to get underlying device type,
     * which was used as communicator construction argument
     */
    using device_type = typename unified_device_type::ccl_native_t;

    /**
     * Declare communicator device context native type
     */
    using context_type = typename unified_context_type::ccl_native_t;

    communicator(communicator&& src);
    communicator& operator=(communicator&& src);
    ~communicator();

    /**
     * Retrieves the rank in a communicator
     * @return rank corresponding to communicator object
     */
    int rank() const;

    /**
     * Retrieves the number of rank in a communicator
     * @return number of the ranks
     */
    int size() const;

    /**
     * Retrieves underlying device, which was used as communicator construction argument
     */
    device get_device() const;

    /**
     * Retrieves underlying context, which was used as communicator construction argument
     */
    context get_context() const;

    communicator split(int color, int key, bool split_external_use = false);

private:
    friend class ccl::detail::environment;
    friend struct ccl::v1::impl_dispatch;

    template <cl_backend_type type>
    friend struct ccl::comm_impl_dispatch_selector;

    communicator(impl_value_t&& impl);

    // factory methods
    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicators(
        int comm_size,
        const vector_class<DeviceType>& local_devices,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr);

    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicatorsExt(
        int comm_size,
        const vector_class<DeviceType>& local_devices,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr);

    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicators(
        int comm_size,
        const vector_class<pair_class<int, DeviceType>>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr);

    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicatorsExt(
        int comm_size,
        const vector_class<pair_class<int, DeviceType>>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr);

    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicators(
        int comm_size,
        const map_class<int, DeviceType>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr);

    template <class DeviceType, class ContextType>
    static vector_class<communicator> create_communicatorsExt(
        int comm_size,
        const map_class<int, DeviceType>& local_rank_device_map,
        const ContextType& context,
        shared_ptr_class<kvs_interface> kvs,
        const comm_attr& attr);

    static communicator create_communicator(const comm_attr& attr);
    static communicator create_communicator(int size,
                                            shared_ptr_class<kvs_interface> kvs,
                                            const comm_attr& attr);
    static communicator create_communicator(int size,
                                            int rank,
                                            shared_ptr_class<kvs_interface> kvs,
                                            const comm_attr& attr);
    static communicator split_communicator(const communicator& comm, int color, int key);
};

} // namespace v1

using v1::communicator;

} // namespace ccl

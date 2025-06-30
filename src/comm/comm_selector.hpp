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

#include <cstddef>
#include <memory>

#include "oneapi/ccl/types.hpp"
#include "atl/atl_base_comm.hpp"

namespace native {
struct ccl_device;
}
namespace ccl {
namespace v1 {
class comm_split_attr;
}

struct comm_interface;

using comm_interface_ptr = std::shared_ptr<comm_interface>;

struct comm_selector {
    using device_t = ccl::device;
    using context_t = ccl::context;

    using device_ptr_t = std::shared_ptr<ccl::device>;
    using context_ptr_t = std::shared_ptr<ccl::context>;

    virtual ~comm_selector() = default;

    virtual device_ptr_t get_device() const = 0;
    virtual context_ptr_t get_context() const = 0;

    static comm_interface_ptr create_comm_impl();
    static comm_interface_ptr create_comm_impl(const comm_attr& attr);
    static comm_interface_ptr create_comm_impl(const size_t size,
                                               shared_ptr_class<kvs_interface> kvs,
                                               const comm_attr& attr);
    static comm_interface_ptr create_comm_impl(const size_t size,
                                               const int rank,
                                               shared_ptr_class<kvs_interface> kvs,
                                               const comm_attr& attr);
    static comm_interface_ptr create_comm_impl(const size_t size,
                                               const int rank,
                                               device_t ccl_device,
                                               context_t ccl_context,
                                               shared_ptr_class<kvs_interface> kvs,
                                               const comm_attr& attr);
    static comm_interface_ptr create_comm_implExt();
    static comm_interface_ptr create_comm_implExt(const size_t size,
                                                  shared_ptr_class<kvs_interface> kvs,
                                                  const comm_attr& attr);
    static comm_interface_ptr create_comm_implExt(const size_t size,
                                                  const int rank,
                                                  shared_ptr_class<kvs_interface> kvs,
                                                  const comm_attr& attr);
    static comm_interface_ptr create_comm_implExt(const size_t size,
                                                  const int rank,
                                                  device_t ccl_device,
                                                  context_t ccl_context,
                                                  shared_ptr_class<kvs_interface> kvs,
                                                  const comm_attr& attr);
};

} // namespace ccl

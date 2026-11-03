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

/**
 * oneCCL cooperative work decomposition — group abstractions for SYCL kernels.
 *
 * All types conform to one interface: thread_rank(), size(),
 * num_threads() (== size()), sync().
 */
#pragma once

#include <sycl/sycl.hpp>

#ifdef CCL_ENABLE_SYCL

namespace ccl {
namespace v1 {

/** Single work-item — nothing to synchronize. */
struct CoopWorkItem {
    int thread_rank() const { return 0; }
    constexpr int size() const { return 1; }
    constexpr int num_threads() const { return 1; }
    void sync() const {}
};

/** Sub-group level. */
struct CoopSubGroup {
    CoopSubGroup(const sycl::nd_item<1>& it) : sg_(it.get_sub_group()) {}

    int thread_rank() const { return static_cast<int>(sg_.get_local_linear_id()); }
    int size()        const { return static_cast<int>(sg_.get_local_range()[0]); }
    int num_threads() const { return size(); }
    void sync()       const { sycl::group_barrier(sg_); }

    const sycl::sub_group& get() const { return sg_; }

private:
    sycl::sub_group sg_;
};

/** Work-group level. */
struct CoopWorkGroup {
    CoopWorkGroup(const sycl::nd_item<1>& it) : it_(it) {}

    int thread_rank() const { return static_cast<int>(it_.get_local_linear_id()); }
    int size()        const { return static_cast<int>(it_.get_local_range(0)); }
    int num_threads() const { return size(); }
    void sync()       const { sycl::group_barrier(it_.get_group()); }

private:
    sycl::nd_item<1> it_;
};

} // namespace v1

using v1::CoopWorkItem;
using v1::CoopSubGroup;
using v1::CoopWorkGroup;

} // namespace ccl

#endif // CCL_ENABLE_SYCL

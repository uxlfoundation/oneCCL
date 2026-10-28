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
 * Device-side utility functions for oneCCL kernel code.
 */
#pragma once

#include <sycl/sycl.hpp>

namespace ccl {
namespace v1 {

// Acquire component of a memory order: release -> relaxed, acq_rel ->
// acquire, everything else unchanged.
constexpr sycl::memory_order acquireOrderOf(sycl::memory_order ord) {
    return ord == sycl::memory_order::release ? sycl::memory_order::relaxed :
           ord == sycl::memory_order::acq_rel ? sycl::memory_order::acquire :
           ord;
}

// Release component: acquire -> relaxed, acq_rel -> release, rest unchanged.
constexpr sycl::memory_order releaseOrderOf(sycl::memory_order ord) {
    return ord == sycl::memory_order::acquire ? sycl::memory_order::relaxed :
           ord == sycl::memory_order::acq_rel ? sycl::memory_order::release :
           ord;
}

} // namespace v1

using v1::releaseOrderOf;
using v1::acquireOrderOf;

} // namespace ccl

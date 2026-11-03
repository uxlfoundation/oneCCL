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

#include <sycl/sycl.hpp>

#include "oneapi/ccl.hpp"

// send_memcpy: alternative to send_simple that uses q.memcpy (L0 driver
// copy engine) for the actual data transfer instead of a SYCL kernel.
// Same staging-slot handshake (invoke_p2p_barrier_pair); the memcpy
// replaces the in-kernel copy_kernel_wrap.
//
// Selected when message size > CCL_SYCL_PT2PT_SIMPLE_THRESHOLD and CCL_SYCL_PT2PT_COPY_ENGINE=1.
ccl::event send_memcpy(const void *send_buf,
                       size_t send_count,
                       ccl::datatype dtype,
                       int peer_rank,
                       ccl_comm *comm,
                       ccl_stream *global_stream,
                       const ccl::vector_class<ccl::event> &deps,
                       bool &done);

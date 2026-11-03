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

#include "oneapi/ccl.hpp"
#include <memory>

class ccl_comm;
class ccl_stream;

// Single-node small SYCL scatter (push-mode flat tree).
//
// Root parallel_for writes each peer's block from its own send_buf into the
// peer's tmp-buf slot via IPC remote pointers, ranks synchronize on a node
// barrier, and each rank then memcpy's its slot back into recv_buf. The slot
// is sized as CCL_SYCL_TMP_BUF_SIZE / local_proc_count and the caller must
// route any count*dsize that exceeds the slot through scatter_large; the impl
// CCL_THROW's if this contract is violated.
//
// Inputs:
//   send_buf      — root only: count*comm_size elements; ignored on non-root
//   recv_buf      — every rank: count elements written on return
//   count         — per-rank block size in elements
//   dtype         — element type used for vec-path alignment + byte fallback
//   root          — global root rank (translated to node-comm)
//   comm          — communicator (multi-thread instance handled with a
//                   shared pthread barrier wait)
//   global_stream — SYCL stream the kernels are submitted on
//   deps          — caller's input ccl::events to chain on
//
// Selected by scatter_sycl_single_node when count*dsize <=
// CCL_SYCL_SCATTER_SMALL_THRESHOLD; for count*dsize above that, the caller
// must use scatter_large instead.
ccl::event scatter_small(const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         int root,
                         ccl_comm* comm,
                         const std::shared_ptr<ccl_comm>& p2p_comm,
                         ccl_stream* global_stream,
                         const ccl::vector_class<ccl::event>& deps);

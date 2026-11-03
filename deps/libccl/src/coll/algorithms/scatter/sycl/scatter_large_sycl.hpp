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

#include "coll/algorithms/utils/sycl_selection.hpp"
#include "oneapi/ccl.hpp"
#include <memory>

class ccl_comm;
class ccl_stream;

// Single-node large SYCL scatter (single-node, flat tree).
//
// Two implementations, picked at runtime:
//
//  - IPC pull (default, no extra staging): peers do q.memcpy directly from
//    root's IPC-exchanged send_buf into their own recv_buf, with a node
//    barrier ordering the reads after root's writes.
//  - Tmp-buf push (CCL_SYCL_SCATTER_LARGE_TMP_BUF=1, explicitly requested by
//    the caller, or forced for multi-thread instances since IPC exchange is not
//    safe across threads): root chunks its send_buf into the tmp-buf staging
//    slots, ranks barrier, peers memcpy their slot to recv_buf. Chunk size is
//    the per-peer slot size (CCL_SYCL_TMP_BUF_SIZE / local_proc_count,
//    kernel_mem_align'd).
//
// Both paths handle the unaligned case by falling back to byte-granularity
// copies when the IPC base or recv_buf is not alignof(T)-aligned.
//
// Inputs:
//   send_buf      — root only: count*comm_size elements; ignored on non-root
//   recv_buf      — every rank: count elements written on return
//   count         — per-rank block size in elements
//   dtype         — element type used for the alignment + chunking decision
//   root          — global root rank (translated to node-comm)
//   comm          — communicator (multi-thread forces tmp-buf path)
//   p2p_comm      — local communicator that participates in this scatter
//   global_stream — SYCL stream the kernels are submitted on
//   deps          — caller's input ccl::events to chain on
//   coll_attr     — caller-selected scale-up behavior, including force_use_tmp
//
// Selected by scatter_sycl_single_node when count*dsize >
// CCL_SYCL_SCATTER_SMALL_THRESHOLD.
ccl::event scatter_large(const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         int root,
                         ccl_comm* comm,
                         const std::shared_ptr<ccl_comm>& p2p_comm,
                         ccl_stream* global_stream,
                         const ccl::vector_class<ccl::event>& deps,
                         sycl_scaleup_options coll_attr = {});

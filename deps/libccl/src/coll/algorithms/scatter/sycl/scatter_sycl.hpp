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

#include <memory>

namespace ccl {
namespace v1 {

// Run scatter inside an arbitrary local p2p domain instead of assuming node_comm.
event scatter_sycl_single_domain(sycl::queue& q,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 datatype dtype,
                                 int root,
                                 ccl_comm* comm,
                                 const std::shared_ptr<ccl_comm>& p2p_comm,
                                 ccl_stream* global_stream,
                                 const vector_class<event>& deps,
                                 bool& done,
                                 sycl_scaleup_options coll_attr = {});

event scatter_sycl_single_node(sycl::queue& q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               datatype dtype,
                               int root,
                               ccl_comm* comm,
                               ccl_stream* global_stream,
                               const vector_class<event>& deps,
                               bool& done);


// r2r-comm scaleout scatter used by the multi-node paths below. Root
// staging: `send_buf` on root must hold `comm->size() * send_count`
// elements in [peer_0][peer_1]... layout. Every rank returns `send_count`
// elements in recv_buf.
ccl::event scatter_scaleout_sycl(sycl::queue& q,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t send_count,
                                 ccl::datatype dtype,
                                 int root,
                                 ccl_comm* comm,
                                 const ccl::vector_class<ccl::event>& deps,
                                 bool& done,
                                 bool copy_to_host = false,
                                 bool is_cpu_buffers = false,
                                 void* aux_buf = nullptr);

// Two multi-node scatter variants, selected by
// CCL_SYCL_SCATTER_SCALEOUT_SMALL_THRESHOLD:
//
// small_msg (reverse-phase, no pack, no chunking):
//   Node-major numbering means the root's send_buf already lays out each
//   node's node_size*count payload contiguously.
//   1. r2r scaleout on r2r_comm[root_local_idx]: global root scatters
//      node_size*count-sized blocks; one rank per node receives its
//      node's block into device scratch.
//   2. Intra-node scatter on node_comm: each node's local-root_local_idx
//      rank scatters its scratch to node_comm peers (`count` each).
//   Falls back to large_msg if the per-node slab exceeds one scratch slot.
//
// large_msg (pack + chunked, optional cross-chunk overlap):
//   1. Root packs each chunk into r2r-major layout and scatters it across
//      node_comm so local_idx L on the root node holds r2r_size*chunk_count
//      elements for its r2r peer group.
//   2. Each local_idx L runs the r2r scaleout as root of its r2r_comm
//      (only local_idx-L on root_node_idx is the true sender).
event scatter_sycl_multi_node_small_msg(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        datatype dtype,
                                        int root,
                                        ccl_comm* global_comm,
                                        ccl_stream* global_stream,
                                        const vector_class<event>& deps,
                                        bool& done);

event scatter_sycl_multi_node_large_msg(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        datatype dtype,
                                        int root,
                                        ccl_comm* global_comm,
                                        ccl_stream* global_stream,
                                        const vector_class<event>& deps,
                                        bool& done);

event scatter_sycl(sycl::queue q,
                   const void* send_buf,
                   void* recv_buf,
                   size_t count,
                   datatype dtype,
                   int root,
                   ccl_comm* global_comm,
                   ccl_stream* global_stream,
                   const scatter_attr& attr,
                   const vector_class<event>& deps,
                   bool& done);

} // namespace v1
} // namespace ccl

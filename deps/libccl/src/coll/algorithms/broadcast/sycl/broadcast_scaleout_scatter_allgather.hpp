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
#include "coll/algorithms/utils/sycl_coll_base.hpp"

namespace ccl {
namespace v1 {

// Scaleout broadcast via Scatter-AllGather (SAG).
//
// Three phases, replacing the default single-rail cross-node broadcast:
//   1) local SCATTER on the root p2p domain - each of the P local ranks
//      gets a distinct 1/P-th of send_buf at its assigned recv_buf offset.
//   2) parallel RAIL - each rank position i does a broadcast among its
//      column of rank-i peers across p2p domains. All P rails run in
//      parallel over the fabric.
//   3) local ALLGATHER - every p2p domain pools its chunks into recv_buf.
//
// Returns done=false to tell the caller "not applicable, fall back to
// the default chunked broadcast_sycl_multi_domain path." Reasons this
// returns done=false include unsupported communicator shape, size below
// the configured threshold, uneven chunks, or the optional max-ppn guard.
//
// Gated on is_arc_card() at the caller - non-ARC families (PVC, Xeon
// Max) use the original path.
ccl::event broadcast_scaleout_scatter_allgather(sycl::queue& q,
                                                const void* send_buf,
                                                void* recv_buf,
                                                size_t count,
                                                ccl::datatype dtype,
                                                int root,
                                                ccl_comm* global_comm,
                                                const std::shared_ptr<ccl_comm>& p2p_comm,
                                                const std::shared_ptr<ccl_comm>& r2r_comm,
                                                ccl_stream* global_stream,
                                                const ccl::vector_class<ccl::event>& deps,
                                                bool& done);

} // namespace v1
} // namespace ccl

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
#include "coll/algorithms/utils/sycl_copy.hpp"
#include "coll/algorithms/utils/sycl_barrier.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;


// SendSimple<T, vec_size>: single-kernel scaleup send. Streams the user
// buffer through the peer's staging slot in chunks of at most chunk_count
// elements (sized so chunks fit in the staging slot).
//
// Per chunk:
//   1. copy_kernel_wrap: src + offset -> dst_staging
//   2. barrier #1: data visible to peer
//   3. barrier #2: peer drained the slot, safe to reuse
//
// Always uses root-group sync inside p2p_barrier_pair so multi-WG launches
// fence stores from every work-group (not just WG 0) before the flag write.
// On single-WG launches root-group equals the only group, so there's no
// measurable cost — the barrier collapses to the same hardware op.
template <typename T, int vec_size>
struct SendSimple {
    SendSimple(void *dst_staging,
               const void *src,
               const size_t count,
               const size_t chunk_count,
               int peer_rank,
               ccl_comm *node_comm)
            : dst_staging(dst_staging),
              src(src),
              count(count),
              chunk_count(chunk_count),
              peer_rank(peer_rank),
              pflag(node_comm->pair_flag_data()) {}

    void operator()(sycl::nd_item<1> it) const {
        const T *src_t = static_cast<const T *>(src);
        T *dst_t = static_cast<T *>(dst_staging);
        for (size_t off = 0; off < count; off += chunk_count) {
            const size_t this_chunk = (count - off) < chunk_count ? (count - off) : chunk_count;
            copy_kernel_wrap<T, vec_size>(dst_t, src_t + off, this_chunk);
            p2p_barrier_pair</*use_root_sync=*/true>(pflag, peer_rank, it);
            p2p_barrier_pair</*use_root_sync=*/true>(pflag, peer_rank, it);
        }
    }

    // sub_group_size<16> replaces the [[sycl::reqd_sub_group_size(16)]] attribute.
    // use_root_sync is the launch property required by ext_oneapi_get_root_group():
    // without it the runtime rejects the kernel with UR_RESULT_ERROR_INVALID_ARGUMENT.
    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    void *dst_staging;
    const void *src;
    size_t count;
    size_t chunk_count;
    int peer_rank;
    ccl_comm_pair_flag_data pflag;
};

ccl::event send_simple(const void *send_buf,
                       size_t send_count,
                       ccl::datatype dtype,
                       int peer_rank,
                       ccl_comm *comm,
                       ccl_stream *global_stream,
                       const ccl::vector_class<ccl::event> &deps,
                       bool &done);

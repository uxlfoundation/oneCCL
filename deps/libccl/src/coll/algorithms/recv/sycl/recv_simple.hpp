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

// RecvSimple<T, vec_size>: single-kernel scaleup recv. Drains the staging
// slot in chunks of at most chunk_count elements; matches SendSimple's
// chunking so the staging slot (default 2 MiB) is never overflowed.
//
// Per chunk:
//   1. barrier #1: wait for sender's chunk write
//   2. copy_kernel_wrap: src_staging -> dst + offset
//   3. barrier #2: release staging slot for next chunk
//
// Always uses root-group sync inside p2p_barrier_pair so the post-flag
// fence covers every work-group before any WG starts the staging->dst
// copy.
template <typename T, int vec_size>
struct RecvSimple {
    RecvSimple(void *dst,
               const void *src_staging,
               const size_t count,
               const size_t chunk_count,
               int peer_rank,
               ccl_comm *node_comm)
            : dst(dst),
              src_staging(src_staging),
              count(count),
              chunk_count(chunk_count),
              peer_rank(peer_rank),
              pflag(node_comm->pair_flag_data()) {}

    void operator()(sycl::nd_item<1> it) const {
        T *dst_t = static_cast<T *>(dst);
        const T *src_t = static_cast<const T *>(src_staging);
        for (size_t off = 0; off < count; off += chunk_count) {
            const size_t this_chunk = (count - off) < chunk_count ? (count - off) : chunk_count;
            p2p_barrier_pair</*use_root_sync=*/true>(pflag, peer_rank, it);
            copy_kernel_wrap<T, vec_size>(dst_t + off, src_t, this_chunk);
            p2p_barrier_pair</*use_root_sync=*/true>(pflag, peer_rank, it);
        }
    }

    // sub_group_size<16> replaces [[sycl::reqd_sub_group_size(16)]];
    // use_root_sync is the launch property required by ext_oneapi_get_root_group().
    auto get(syclexp::properties_tag) const {
        return syclexp::properties{ syclexp::sub_group_size<16>, syclexp::use_root_sync };
    }

    void *dst;
    const void *src_staging;
    size_t count;
    size_t chunk_count;
    int peer_rank;
    ccl_comm_pair_flag_data pflag;
};

ccl::event recv_simple(void *recv_buf,
                       size_t recv_count,
                       ccl::datatype dtype,
                       int peer_rank,
                       ccl_comm *comm,
                       ccl_stream *global_stream,
                       const ccl::vector_class<ccl::event> &deps,
                       bool &done);

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
 * oneCCL lsa_reduce_sum — device-side reduce-sum from LSA peers. Overloads:
 *   1. lsa_reduce_sum(coop, srcLambda, nSrc, dst, count)
 *   2. lsa_reduce_sum(coop, win, offset, dst, count, team)
 *   3. lsa_reduce_sum(coop, win, offset, dst, count, devcomm)
 */
#pragma once

#include "oneapi/ccl/device_api/reduce_copy.hpp"
#include "oneapi/ccl/device_api/team.hpp"
#include "oneapi/ccl/device_api/window_data.hpp"
#include "oneapi/ccl/device_api/dev_comm.hpp"

#ifdef CCL_ENABLE_SYCL

namespace ccl {
namespace v1 {

// Repeated from device_api.hpp so this header stands alone.
typedef window_data* window_t;

// [1] Caller supplies srcLambda(i) -> T*.
template <typename T, int UNROLL = 4, typename Coop, typename SrcLambda>
inline void lsa_reduce_sum(
    Coop coop,
    SrcLambda src_lambda, int n_src,
    T* dst,
    int count) {

    auto dst_lambda = [=](int /*ignored*/) -> T* { return dst; };

    detail::reduce_copy<T, UNROLL>(
        coop, src_lambda, n_src, dst_lambda, /*n_dst=*/1, count);
}

// [2] Reads every peer in the team through the window.
template <typename T, int UNROLL = 4, typename Coop>
inline void lsa_reduce_sum(
    Coop coop,
    window_t win, size_t offset,
    T* dst,
    int count,
    team tm) {

    // Map team member i into win->ptrs[] (LSA-rank space) via the team's rank
    // and stride — for the LSA team (stride 1) this is just i.
    auto src_lambda = [=](int i) -> T* {
        int base_slot = win->lsa_rank - tm.rank * tm.stride;
        int slot = base_slot + i * tm.stride;
        return reinterpret_cast<T*>(
            static_cast<char*>(win->ptrs[slot]) + offset);
    };

    lsa_reduce_sum<T, UNROLL>(coop, src_lambda, tm.nranks, dst, count);
}

// [3] Takes the LSA team from the device communicator.
template <typename T, int UNROLL = 4, typename Coop>
inline void lsa_reduce_sum(
    Coop coop,
    window_t win, size_t offset,
    T* dst,
    int count,
    const dev_comm_data& dc) {

    team tm = team_lsa(dc);
    lsa_reduce_sum<T, UNROLL>(coop, win, offset, dst, count, tm);
}

} // namespace v1

using v1::lsa_reduce_sum;

} // namespace ccl

#endif // CCL_ENABLE_SYCL

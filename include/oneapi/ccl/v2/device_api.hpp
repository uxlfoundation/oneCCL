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
 * oneCCL v2 device-side API — functions/types callable from SYCL kernels.
 *
 * Unlike the host half (a C ABI dispatched through a plugin), device symbols
 * cannot cross a runtime plugin boundary — they compile into the user's kernel.
 * So this header wraps the v1 device headers under oneccl-prefixed names,
 * passing the v2 opaque handles straight through.
 *
 * Requires the v1 device headers on the include path — present whenever the
 * target links the v1 `ccl` library.
 */
#pragma once

#include "oneapi/ccl/v2/types.h"

// Defines CCL_ENABLE_SYCL. Needed here, not from the v1 header below: the macro
// gates that include, and the v1 device headers include <sycl/sycl.hpp> outside
// their own guard, so they cannot come first on a non-SYCL TU.
#include "oneapi/ccl/config.h"

#ifdef CCL_ENABLE_SYCL

#include "oneapi/ccl/device_api.hpp"

// Cooperative-group types the kernel instantiates.
using onecclCoopWorkItem = ccl::CoopWorkItem;
using onecclCoopSubGroup = ccl::CoopSubGroup;
using onecclCoopWorkGroup = ccl::CoopWorkGroup;

// Team tags selecting the peer set a barrier / reduce operates over.
using onecclTeamTagWorld = ccl::team_tag_world;
using onecclTeamTagLsa = ccl::team_tag_lsa;

namespace oneccl_detail {
// Sole cast sites — every entry below routes through them, so the conversion
// never appears in user code. onecclWindow_t carries the device pointer
// directly; onecclDevComm_t is an opaque blob, read in place.
inline ccl::window_t as_window(onecclWindow_t handle) {
    return static_cast<ccl::window_t>(handle);
}

static_assert(sizeof(ccl::dev_comm_data) <= sizeof(onecclDevComm_t),
              "onecclDevComm_t is too small for ccl::dev_comm_data");
static_assert(alignof(ccl::dev_comm_data) <= alignof(onecclDevComm_t),
              "onecclDevComm_t is under-aligned for ccl::dev_comm_data");

inline const ccl::dev_comm_data &as_dev_comm(const onecclDevComm_t &dev_comm) {
    return *reinterpret_cast<const ccl::dev_comm_data *>(&dev_comm);
}
} // namespace oneccl_detail

/** LSA-team rank / size read from the device communicator. */
inline int onecclLsaRank(const onecclDevComm_t &dev_comm) {
    return oneccl_detail::as_dev_comm(dev_comm).lsa_rank;
}
inline int onecclLsaSize(const onecclDevComm_t &dev_comm) {
    return oneccl_detail::as_dev_comm(dev_comm).lsa_size;
}

/** Peer-pointer resolves, taking the host API's window handle unwrapped. */
inline void *onecclGetLsaPointer(onecclWindow_t win, size_t offset,
                                 int lsa_peer) {
    return ccl::get_lsa_pointer(oneccl_detail::as_window(win), offset,
                                lsa_peer);
}
inline void *onecclGetPeerPointer(onecclWindow_t win, size_t offset, int peer) {
    return ccl::get_peer_pointer(oneccl_detail::as_window(win), offset, peer);
}
inline void *onecclGetLocalPointer(onecclWindow_t win, size_t offset) {
    return ccl::get_local_pointer(oneccl_detail::as_window(win), offset);
}

/** LSA barrier session, constructed straight from the device communicator. */
template <typename Coop>
struct onecclLsaBarrierSession : ccl::LsaBarrierSession<Coop> {
    template <typename Tag>
    onecclLsaBarrierSession(Coop coop, const onecclDevComm_t &dev_comm, Tag tag,
                            uint32_t index)
        : ccl::LsaBarrierSession<Coop>(
              coop, oneccl_detail::as_dev_comm(dev_comm), tag, index) {}
};

/** Fused reduce-sum over all LSA peers. */
template <typename T, int UNROLL = 4, typename Coop>
inline void onecclLsaReduceSum(Coop coop, onecclWindow_t win, size_t offset,
                               T *dst, int count,
                               const onecclDevComm_t &dev_comm) {
    ccl::lsa_reduce_sum<T, UNROLL>(coop, oneccl_detail::as_window(win), offset,
                                   dst, count,
                                   oneccl_detail::as_dev_comm(dev_comm));
}

#endif // CCL_ENABLE_SYCL

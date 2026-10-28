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
 * oneCCL Device API — functions callable from SYCL kernels: the window_t opaque
 * pointer, the dev_comm_data value struct, and the get_*_pointer() resolves.
 */
#pragma once

#include <sycl/sycl.hpp>
#include "oneapi/ccl/config.h"
#include "oneapi/ccl/device_api/window_data.hpp"
#include "oneapi/ccl/device_api/dev_comm.hpp"
#include "oneapi/ccl/device_api/device_utility.hpp"
#include "oneapi/ccl/device_api/coop.hpp"
#include "oneapi/ccl/device_api/team.hpp"
#include "oneapi/ccl/device_api/lsa_barrier.hpp"
#include "oneapi/ccl/device_api/reduce_copy.hpp"
#include "oneapi/ccl/device_api/lsa_reduce_sum.hpp"

#ifdef CCL_ENABLE_SYCL

#ifndef CCL_PRODUCT_FULL
#error "Do not include this file directly. Please include 'ccl.hpp'"
#endif

namespace ccl {
namespace v1 {

/**
 * Pointer to a peer's buffer within a registered window, addressed by world
 * communicator rank. Header-inline, so it lands in the caller's kernel image —
 * no device symbol resolved out of libccl.
 *
 * win->ptrs[] is indexed by LSA (node) rank, so a world rank converts as
 * slot = lsa_rank + (peer - world_rank): a node's ranks are contiguous in world
 * order (enforced at registration), so the signed world distance carries over
 * to LSA space. Peers outside this rank's LSA team fall out of range.
 *
 * Example — 2 nodes x 8 GPUs, this rank world 10 (lsa_rank=2, ptrs[0..7] map to
 * world 8..15): peer 13 -> slot 5; peer 8 -> slot 0; peer 3 -> slot -5, nullptr.
 *
 * @param win    registered window handle (opaque pointer)
 * @param offset byte offset within the window, below win->size
 * @param peer   peer's rank in the world communicator
 * @return       pointer to the peer's buffer at the given offset, or nullptr on error
 */
inline void* get_peer_pointer(window_t win, size_t offset, int peer) {
    if (!win || !win->ptrs || offset >= win->size) {
        return nullptr;
    }
    int slot = win->lsa_rank + (peer - win->world_rank);
    if (slot < 0 || slot >= win->lsa_size) {
        return nullptr;
    }
    return static_cast<char*>(win->ptrs[slot]) + offset;
}

/**
 * Pointer to a peer's buffer addressed by LSA team rank, which indexes
 * win->ptrs[] directly.
 *
 * @param win      registered window handle
 * @param offset   byte offset within the peer's buffer, below win->size
 * @param lsa_peer peer's rank within the LSA team (0 .. lsa_size-1)
 * @return         pointer to the peer's buffer at the given offset, or nullptr on error
 */
inline void* get_lsa_pointer(window_t win, size_t offset, int lsa_peer) {
    if (!win || !win->ptrs || offset >= win->size || lsa_peer < 0 ||
        lsa_peer >= win->lsa_size) {
        return nullptr;
    }
    return static_cast<char*>(win->ptrs[lsa_peer]) + offset;
}

/**
 * Pointer to this rank's own buffer within a registered window.
 *
 * @param win    registered window handle
 * @param offset byte offset within the local buffer, below win->size
 * @return       pointer to the local buffer at the given offset, or nullptr on error
 */
inline void* get_local_pointer(window_t win, size_t offset) {
    if (!win || !win->ptrs || offset >= win->size || win->lsa_rank < 0 ||
        win->lsa_rank >= win->lsa_size) {
        return nullptr;
    }
    return static_cast<char*>(win->ptrs[win->lsa_rank]) + offset;
}

} // namespace v1

using v1::window_t;
using v1::get_peer_pointer;
using v1::get_lsa_pointer;
using v1::get_local_pointer;
using v1::dev_comm_t;

} // namespace ccl

#endif // CCL_ENABLE_SYCL

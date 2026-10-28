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

#include <array>
#include <optional>
#include <sycl/sycl.hpp>

#include "coll/algorithms/utils/consts.hpp"
#include "common/ze/ze_wrapper.hpp"

// Owns the resources behind the counter-based events used for copy engine (CE)
// synchronization: the doorbell allocation the events are created over, and the events
// themselves. One instance per comm_impl.
//
// Each rank holds two events per peer:
//  - remote[p]: signalled by this rank to tell peer p that the data destined for p has
//               landed. Its doorbell cell lives in p's allocation, mapped in here.
//  - local[p]:  waited on by this rank for the data peer p sends. Its doorbell cell lives
//               in this rank's own allocation.
//
// The events are created without host storage, so they can only be signalled and waited
// on from the device - zeEventHostSynchronize and friends are not usable on them. Nothing
// in the CE flow needs a host wait; both the signal and the wait are appended to the copy
// engine command list.
//
// Set-up is two-phase because the doorbell allocation has to join the ipc_ptrs exchange
// long before the events can be created: allocate_device_memory() first, then init() once
// the exchange has produced the mapped peer pointers.
class ccl_ce_cb_events {
public:
    ccl_ce_cb_events() = default;
    ~ccl_ce_cb_events();

    ccl_ce_cb_events(const ccl_ce_cb_events &) = delete;
    ccl_ce_cb_events &operator=(const ccl_ce_cb_events &) = delete;

    // Allocates the local doorbell region, one zeroed cell per rank in the node, and
    // returns it so the caller can add it to the ipc_ptrs exchange. The allocation stays
    // owned here and is released by the destructor.
    cb_event_mem_t *allocate_device_memory(sycl::queue &q, int size);

    // device_ptrs[p] is the doorbell allocation exported by rank p, already mapped into
    // this process - the result of the ipc_ptrs exchange for the slot that
    // allocate_device_memory() fed. Only the peer slots are read: those mapped pointers
    // are borrowed (ipc_handle_manager owns and closes them) and are the only way to
    // reach a peer's memory. The self slot is our own allocation and is asserted to match
    // m_device_storage rather than used.
    void init(const std::array<void *, MAX_NODE_RANKS> &device_ptrs,
              ze_device_ptr ze_device,
              int rank,
              int size);

    bool is_allocated() const {
        return m_device_storage != nullptr;
    }

    bool is_initialized() const {
        return m_is_set;
    }

    ze_cb_event_ptr &get_remote(int peer_rank) {
        return m_remote_events[peer_rank];
    }

    ze_cb_event_ptr &get_local(int peer_rank) {
        return m_local_events[peer_rank];
    }

private:
    // Declared before the events and the allocation on purpose: members are destroyed in
    // reverse declaration order, so the context outlives everything created from it.
    std::optional<sycl::context> m_context;
    cb_event_mem_t *m_device_storage = nullptr;
    std::array<ze_cb_event_ptr, MAX_NODE_RANKS> m_remote_events{};
    std::array<ze_cb_event_ptr, MAX_NODE_RANKS> m_local_events{};
    bool m_is_set = false;
};

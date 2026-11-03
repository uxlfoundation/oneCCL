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
 * oneCCL LSA Barrier Session — arrive/wait/sync scoped to the LSA team.
 *
 * 2D inbox layout, as in comm_barrier_visa:
 *   barrier_inboxes[owner][barrier_index * nranks + sender]
 * Each sender plain-stores to its own slot (no RMW); the owner polls its own
 * slots locally at memory_scope::device.
 *
 * Usage:
 *   ccl::LsaBarrierSession<ccl::CoopWorkGroup> bar(wg, dev_comm, ccl::team_tag_lsa{}, slot);
 *   bar.sync(wg, sycl::memory_order::acq_rel);
 *   // ... read/write peer memory ...
 *   bar.sync(wg, sycl::memory_order::acq_rel);
 */
#pragma once

#include <sycl/sycl.hpp>
#include "oneapi/ccl/device_api/dev_comm.hpp"
#include "oneapi/ccl/device_api/device_utility.hpp"
#include "oneapi/ccl/device_api/team.hpp"

#ifdef CCL_ENABLE_SYCL

namespace ccl {
namespace v1 {

template<typename Coop>
struct LsaBarrierSession {
    /**
     * @param coop     cooperative group executing this barrier
     * @param dev_comm device communicator data
     * @param tm       team to barrier across
     * @param index    barrier slot index (0 .. n_barriers-1)
     */
    LsaBarrierSession(Coop coop, const dev_comm_data& dev_comm, team tm, uint32_t index)
        : coop_(coop), dev_comm_(dev_comm), team_(tm), index_(index) {
        epoch_ = dev_comm_.barrier_epochs[index_];
    }

    /** LSA team barrier. */
    LsaBarrierSession(Coop coop, const dev_comm_data& dev_comm, team_tag_lsa, uint32_t index)
        : LsaBarrierSession(coop, dev_comm, team_lsa(dev_comm), index) {}

    /** World team barrier. */
    LsaBarrierSession(Coop coop, const dev_comm_data& dev_comm, team_tag_world, uint32_t index)
        : LsaBarrierSession(coop, dev_comm, team_world(dev_comm), index) {}

    ~LsaBarrierSession() = default;

    LsaBarrierSession(const LsaBarrierSession&) = delete;
    LsaBarrierSession& operator=(const LsaBarrierSession&) = delete;

    /**
     * Signal all peers that this rank has reached the barrier. Collective over
     * the coop group. Leaves epoch_ alone — wait() advances it.
     */
    void arrive(Coop coop, sycl::memory_order order = sycl::memory_order::acq_rel) {
        // Prior writes must be visible before signaling
        coop_.sync();

        if (team_.nranks > 1) {
            sycl::atomic_fence(releaseOrderOf(order), sycl::memory_scope::system);
        }

        for (int i = coop_.thread_rank(); i < team_.nranks; i += coop_.size()) {
            if (i != team_.rank) {
                dev_comm_.barrier_inboxes[i][index_ * team_.nranks + team_.rank] = epoch_ + 1;
            }
        }
    }

    /**
     * Block until all peers have arrived, then advance epoch_. Collective over
     * the coop group; polling of this rank's own inbox is split across its
     * threads.
     */
    void wait(Coop coop, sycl::memory_order order = sycl::memory_order::acq_rel) {
        for (int i = coop_.thread_rank(); i < team_.nranks; i += coop_.size()) {
            if (i != team_.rank) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space>
                    slot(dev_comm_.barrier_inboxes[team_.rank][index_ * team_.nranks + i]);

                // Modular compare, not `got < target` — the latter deadlocks
                // across uint32 wrap (got=0 vs target=0xFFFFFFFF looks like
                // "not arrived" when the peer is in fact an epoch ahead).
                // Difference in 0..0x7FFFFFFF means arrived; above means spin.
                while (slot.load() - (epoch_ + 1) > uint32_t(-1) >> 1) {
                }
            }
        }

        epoch_ += 1;
        dev_comm_.barrier_epochs[index_] = epoch_;

        if (team_.nranks > 1) {
            sycl::atomic_fence(acquireOrderOf(order), sycl::memory_scope::system);
        }

        // Broadcast completion to all coop threads
        coop_.sync();
    }

    /** Full barrier: arrive + wait. */
    void sync(Coop coop, sycl::memory_order order = sycl::memory_order::acq_rel) {
        arrive(coop, order);
        wait(coop, order);
    }

private:
    Coop coop_;
    const dev_comm_data& dev_comm_;
    team team_;
    uint32_t index_;
    uint32_t epoch_;  /**< monotonic barrier round counter */
};

} // namespace v1

using v1::LsaBarrierSession;

} // namespace ccl

#endif // CCL_ENABLE_SYCL

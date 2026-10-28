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

#include "coll/algorithms/utils/cb_events_data.hpp"

ccl_ce_cb_events::~ccl_ce_cb_events() {
    if (!m_context) {
        return;
    }

    // Teardown order for events over shared external storage, per the Level Zero
    // developers: (1) destroy the event whose storage is on the remote device, (2) destroy
    // the event whose storage is on the allocating device, (3) free the device memory.
    // The sequence below follows it. Doing it in the destructor body is deliberate: member
    // destruction runs only after the body, which would release the doorbell allocation
    // while the events still referenced it.
    //
    // Step 1 is valid because the peers' doorbell allocations are still mapped here: the
    // exchange imports them with to_cache=false, which makes open_handle() flag them
    // is_cached so ipc_handle_manager::clear() skips the close and the mapping lives for
    // the process. See ze_handle_manager.cpp's open_handle().
    //
    // One gap remains, deferred to the commit that reworks the whole ipc exchange
    // teardown: step 3 is only safe once every peer has finished its step 1, since each
    // peer's remote event points into this allocation. No barrier enforces that - see the
    // matching TODO in sched_base.cpp, which hits the same problem from the other side.
    LOG_DEBUG("releasing cb events, doorbell storage ", m_device_storage);

    for (auto &ev : m_remote_events) {
        ev.reset();
    }
    for (auto &ev : m_local_events) {
        ev.reset();
    }

    if (m_device_storage) {
        sycl::free(m_device_storage, *m_context);
        LOG_DEBUG("freed cb event doorbell storage ", m_device_storage);
        m_device_storage = nullptr;
    }
}

cb_event_mem_t *ccl_ce_cb_events::allocate_device_memory(sycl::queue &q, int size) {
    if (is_allocated()) {
        // Reuse rather than reallocate: overwriting the member would leak the first
        // allocation and leave already-created events pointing at it.
        LOG_WARN("cb event doorbell storage is already allocated, reusing ", m_device_storage);
        return m_device_storage;
    }

    // Holding the context keeps it alive for as long as this object does, which is what
    // makes both the free above and the event teardown safe.
    m_context = q.get_context();

    m_device_storage = sycl::malloc_device<cb_event_mem_t>(size, q);
    CCL_THROW_IF_NOT(m_device_storage, "failed to allocate cb event doorbell storage");

    // zero it before anyone can signal into it: signal() writes a single byte into a cell
    // and relies on the rest already being zero
    q.memset(m_device_storage, 0, size * sizeof(cb_event_mem_t)).wait();

    LOG_DEBUG("allocated cb event doorbell storage ", m_device_storage, " for ", size, " ranks");
    return m_device_storage;
}

void ccl_ce_cb_events::init(const std::array<void *, MAX_NODE_RANKS> &device_ptrs,
                            ze_device_ptr ze_device,
                            int rank,
                            int size) {
    if (m_is_set) {
        // Not fatal: the existing events are still valid, so keep them and carry on. It
        // does mean a caller-side guard let coll_init reach here twice for one comm.
        LOG_WARN("cb events are already initialized for rank ", rank, ", skipping");
        return;
    }

    CCL_THROW_IF_NOT(is_allocated(),
                     "allocate_device_memory() must run before init() for rank ",
                     rank);
    CCL_THROW_IF_NOT(ze_device, "no L0 device for cb events, rank ", rank);

    // get_ipc_ptrs() fills the self slot verbatim with the local pointer it was given, so
    // this catches a caller that exchanged a different slot than the one we allocated -
    // the handle index is positional and nothing else verifies it.
    CCL_THROW_IF_NOT(device_ptrs[rank] == m_device_storage,
                     "exchanged doorbell storage ",
                     device_ptrs[rank],
                     " does not match the allocated one ",
                     m_device_storage,
                     " on rank ",
                     rank);

    // must match the value ze_cb_event_t::signal() writes into the doorbell cell
    const uint64_t completion_val = 1;

    for (int i = 1; i < size; i++) {
        int peer_rank = (rank + i) % size;

        CCL_THROW_IF_NOT(device_ptrs[peer_rank],
                         "no mapped doorbell storage from peer ",
                         peer_rank,
                         " on rank ",
                         rank);

        // hostStorage stays null - see the class comment: these events are device-only.
        // signalled by us into peer_rank's mapped allocation, at our own slot
        cb_event_arg_t remote_arg{ completion_val,
                                   &static_cast<uint64_t *>(device_ptrs[peer_rank])[rank],
                                   nullptr };
        // waited on by us in our own allocation, at peer_rank's slot - taken from the
        // member rather than device_ptrs[rank], which is the same address
        cb_event_arg_t local_arg{ completion_val, &m_device_storage[peer_rank], nullptr };

        m_remote_events[peer_rank] = ze_device->context()->create_cb_event(*ze_device, remote_arg);
        m_local_events[peer_rank] = ze_device->context()->create_cb_event(*ze_device, local_arg);

        CCL_THROW_IF_NOT(m_remote_events[peer_rank] && m_remote_events[peer_rank]->is_valid(),
                         "failed to create remote cb event for peer ",
                         peer_rank);
        CCL_THROW_IF_NOT(m_local_events[peer_rank] && m_local_events[peer_rank]->is_valid(),
                         "failed to create local cb event for peer ",
                         peer_rank);

        LOG_DEBUG("rank(",
                  rank,
                  ") cb events for peer(",
                  peer_rank,
                  "): remote handle {",
                  m_remote_events[peer_rank]->handle(),
                  "} deviceStorage ",
                  remote_arg.deviceStorage,
                  ", local handle {",
                  m_local_events[peer_rank]->handle(),
                  "} deviceStorage ",
                  local_arg.deviceStorage);
    }

    m_is_set = true;
}

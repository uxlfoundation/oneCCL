/*
 Copyright 2016-2020 Intel Corporation
 
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
#include "comm/comm_id_allocator.hpp"

namespace ccl {

// comm_id_allocator implementation

uint64_t comm_id_allocator::compute_split_signature(int parent_comm_id,
                                                    int color,
                                                    const std::vector<int>& global_ranks) {
    uint64_t hash = 14695981039346656037ULL; // FNV-1a offset basis

    // Hash parent_comm_id
    hash = hash_combine(hash, static_cast<uint64_t>(parent_comm_id));

    // Hash color (determines sub-communicator membership)
    hash = hash_combine(hash, static_cast<uint64_t>(color));

    // Hash the size of the group
    hash = hash_combine(hash, static_cast<uint64_t>(global_ranks.size()));

    // Hash all global ranks in order
    // The ranks should be sorted to ensure determinism across all processes
    for (int rank : global_ranks) {
        hash = hash_combine(hash, static_cast<uint64_t>(rank));
    }

    return hash;
}

int comm_id_allocator::allocate_unique_comm_id(uint64_t signature,
                                               const std::set<int>& used_comm_ids,
                                               int max_comm_id) {
    // Map signature to initial comm_id candidate
    int candidate = static_cast<int>(signature % static_cast<uint64_t>(max_comm_id));

    // Linear probing for collision resolution
    // All ranks will deterministically compute the same candidate and
    // probe in the same order, so they will arrive at the same final comm_id
    int probes = 0;
    while (probes < max_comm_id) {
        if (used_comm_ids.find(candidate) == used_comm_ids.end()) {
            LOG_DEBUG("allocated unique comm_id: ",
                      candidate,
                      " (signature: ",
                      signature,
                      ", probes: ",
                      probes,
                      ")");
            return candidate;
        }
        // Linear probe with a secondary hash to reduce clustering
        candidate = (candidate + 1 + static_cast<int>((signature >> 32) % 7)) % max_comm_id;
        probes++;
    }

    CCL_THROW("Failed to allocate unique comm_id: all slots exhausted");
    return atl_comm_id_storage::invalid_comm_id;
}

std::set<int> comm_id_allocator::collect_base_used_comm_ids(int parent_comm_id,
                                                            int global_comm_id) {
    std::set<int> used;

    // Always reserve the parent comm_id
    if (parent_comm_id != atl_comm_id_storage::invalid_comm_id) {
        used.insert(parent_comm_id);
    }

    // Reserve global comm_id if provided and different
    if (global_comm_id != atl_comm_id_storage::invalid_comm_id &&
        global_comm_id != parent_comm_id) {
        used.insert(global_comm_id);
    }

    return used;
}

uint64_t comm_id_allocator::hash_combine(uint64_t hash, uint64_t value) {
    const uint64_t fnv_prime = 1099511628211ULL;
    hash ^= value;
    hash *= fnv_prime;
    return hash;
}

// split_comm_id_registry implementation

split_comm_id_registry& split_comm_id_registry::instance() {
    static split_comm_id_registry registry;
    return registry;
}

int split_comm_id_registry::get_or_allocate(uint64_t signature,
                                            const std::set<int>& base_used_comm_ids) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if already allocated
    auto it = signature_to_comm_id_.find(signature);
    if (it != signature_to_comm_id_.end()) {
        LOG_DEBUG("reusing existing comm_id: ", it->second, " for signature: ", signature);
        return it->second;
    }

    // Build full set of used comm_ids
    std::set<int> all_used = base_used_comm_ids;
    all_used.insert(allocated_comm_ids_.begin(), allocated_comm_ids_.end());

    // Allocate new comm_id
    int new_comm_id = comm_id_allocator::allocate_unique_comm_id(signature, all_used);

    // Register the allocation
    signature_to_comm_id_[signature] = new_comm_id;
    allocated_comm_ids_.insert(new_comm_id);

    LOG_DEBUG("registered new comm_id: ", new_comm_id, " for signature: ", signature);
    return new_comm_id;
}

void split_comm_id_registry::release(int comm_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    allocated_comm_ids_.erase(comm_id);

    // Remove from signature map
    for (auto it = signature_to_comm_id_.begin(); it != signature_to_comm_id_.end();) {
        if (it->second == comm_id) {
            it = signature_to_comm_id_.erase(it);
        }
        else {
            ++it;
        }
    }
}

bool split_comm_id_registry::is_allocated(int comm_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_comm_ids_.find(comm_id) != allocated_comm_ids_.end();
}

std::set<int> split_comm_id_registry::get_allocated_comm_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_comm_ids_;
}

} // namespace ccl

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
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "atl/atl_base_transport.hpp"
#include "common/log/log.hpp"

namespace ccl {

/**
 * Unique comm_id allocator for independent comm_split operations.
 *
 * This class provides deterministic, collision-free comm_id allocation
 * for sub-communicators created via comm_split. All ranks participating
 * in the same sub-communicator will compute the same comm_id.
 *
 * The algorithm:
 * 1. Compute a signature hash from parent_comm_id, color, and membership (global ranks)
 * 2. Map the signature to a valid comm_id range [0, max_comm_id)
 * 3. Use deterministic collision resolution (linear probing with salt) if needed
 */
class comm_id_allocator {
public:
    /**
     * Compute a split signature that uniquely identifies a sub-communicator.
     *
     * @param parent_comm_id The comm_id of the parent communicator
     * @param color The color used in comm_split (determines membership)
     * @param global_ranks Ordered vector of global ranks in the sub-communicator
     * @return A 64-bit hash signature
     *
     * Note: The 'key' parameter from comm_split is NOT included in the signature
     * because it only affects rank ordering within the sub-communicator, not membership.
     * All ranks with the same color end up in the same sub-communicator regardless of key.
     */
    static uint64_t compute_split_signature(int parent_comm_id,
                                            int color,
                                            const std::vector<int>& global_ranks);

    /**
     * Allocate a unique comm_id based on the signature.
     *
     * This function maps the signature to a valid comm_id and resolves
     * collisions deterministically using linear probing.
     *
     * @param signature The split signature computed by compute_split_signature
     * @param used_comm_ids Set of currently used comm_ids (for collision detection)
     * @param max_comm_id Maximum valid comm_id value (exclusive)
     * @return A unique comm_id in range [0, max_comm_id)
     */
    static int allocate_unique_comm_id(uint64_t signature,
                                       const std::set<int>& used_comm_ids,
                                       int max_comm_id = atl_comm_id_storage::max_comm_id);

    /**
     * Collect used comm_ids from the parent communicator for collision detection.
     *
     * This gathers the comm_ids that are in use across all ranks in the
     * parent communicator to ensure the new sub-communicator gets a unique id.
     *
     * @param parent_comm_id The parent communicator's comm_id
     * @param global_comm_id The global communicator's comm_id (if different)
     * @return Set of comm_ids that should be considered as used
     */
    static std::set<int> collect_base_used_comm_ids(int parent_comm_id, int global_comm_id = -1);

private:
    /**
     * FNV-1a style hash combine function.
     */
    static uint64_t hash_combine(uint64_t hash, uint64_t value);
};

/**
 * Process-global registry for tracking allocated split comm_ids.
 *
 * This singleton maintains a mapping from split signatures to allocated comm_ids
 * to ensure consistency across repeated splits with the same parameters and
 * to track which comm_ids are in use.
 */
class split_comm_id_registry {
public:
    static split_comm_id_registry& instance();

    /**
     * Get or allocate a comm_id for the given signature.
     *
     * If the signature was previously allocated, returns the same comm_id.
     * Otherwise, allocates a new unique comm_id.
     *
     * @param signature The split signature
     * @param base_used_comm_ids Base set of used comm_ids to avoid
     * @return The allocated comm_id
     */
    int get_or_allocate(uint64_t signature, const std::set<int>& base_used_comm_ids);

    /**
     * Release a comm_id when a communicator is destroyed.
     *
     * @param comm_id The comm_id to release
     */
    void release(int comm_id);

    /**
     * Check if a comm_id is currently allocated.
     */
    bool is_allocated(int comm_id) const;

    /**
     * Get all currently allocated comm_ids.
     */
    std::set<int> get_allocated_comm_ids() const;

private:
    split_comm_id_registry() = default;
    ~split_comm_id_registry() = default;
    split_comm_id_registry(const split_comm_id_registry&) = delete;
    split_comm_id_registry& operator=(const split_comm_id_registry&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, int> signature_to_comm_id_;
    std::set<int> allocated_comm_ids_;
};

} // namespace ccl

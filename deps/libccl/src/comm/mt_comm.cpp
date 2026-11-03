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

#include <climits>
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.h"
#include "exec/exec.hpp"
#include "coll/coll.hpp"
#include "coll/attr/ccl_common_op_attrs.hpp"
#include "comm/comm.hpp"
#include "comm/comm_impl.hpp"
#include "common/global/global.hpp"
#include "common/event/impls/host_event.hpp"
#include "common/request/request.hpp"
#include "MT/shared_resource.hpp"
#include "sched/sched.hpp"
#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/kvs.hpp"
#include "oneapi/ccl/comm_split_attr_ids.hpp"
#include "oneapi/ccl/comm_split_attr_ids_traits.hpp"
#include "oneapi/ccl/comm_split_attr.hpp"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs/ikvs_wrapper.h"
#include "kvs_impl.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

// this file is created only to support multi threading  functionality

namespace ccl {
namespace v1 {

struct impl_dispatch {
    template <class Object>
    const typename Object::impl_value_t& operator()(const Object& obj) {
        return obj.get_impl();
    }
};

}; // namespace v1
}; // namespace ccl

namespace {

// Allocate group_buffers slot for a given global_id, sized for `size` ranks.
// Caller must invoke this from rank 0 only; followed by a barrier so the
// other ranks see the populated slot.
void allocate_group_buffers(int global_id, int size) {
    auto& shared_data = ccl::global_data::get().shared_data;
    std::lock_guard<std::mutex> lock(shared_data->allocation_mutex);
    if (static_cast<int>(shared_data->group_buffers.size()) <= global_id) {
        shared_data->group_buffers.reserve(global_id * 2);
        shared_data->group_buffers.resize(global_id + 1);
    }
    shared_data->group_buffers[global_id].reserve(size);
    for (int i = 0; i < size; i++) {
        shared_data->group_buffers[global_id].emplace_back(
            std::vector<ccl::group_buffer_entry>(size));
    }
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
// Initialize the sycl-kernels path for an MT communicator: build a SYCL stream
// over the comm's device and run coll_initExt.
void init_sycl_kernels_path(ccl_comm* comm) {
    auto device = comm->get_device();
    if (!ccl::global_data::env().enable_sycl_kernels || device == NULL) {
        return;
    }
    sycl::queue q(device->get_native());
    if (q.get_context().get_backend() != sycl::backend::ext_oneapi_level_zero) {
        return;
    }
    ccl::stream op_stream = ccl::create_stream(q);
    ccl::impl_dispatch disp;
    ccl_stream* cclstream = get_stream_ptr(disp(op_stream));
    coll_initExt(comm, ccl::global_data::get().shared_data->hash_table, cclstream);
}
#endif

} // namespace

ccl_internal_comm::ccl_internal_comm(int comm_id, int rank, int size)
        : m_dtree(size, rank)
#ifdef CCL_ENABLE_SYCL
          ,
          m_barrier_data(rank, size),
          m_flag_data(rank, size),
          m_pair_flag_data(rank, size)
#endif // CCL_ENABLE_SYCL
{
    reset(rank, size);
}

ccl_comm::ccl_comm(int size, int rank) {
    comm_rank = rank;
    comm_size = size;

    comm_impl = std::unique_ptr<ccl_internal_comm>(new ccl_internal_comm(0, comm_rank, comm_size));
}

ccl_comm* ccl_comm::createExt(device_t device,
                              context_t context,
                              int size,
                              int rank,
                              ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                              ccl::ccl_comm_attr_impl& attr) {
    // TODO: handle attr in this case
    return new ccl_comm(device, context, {}, true, size, rank, kvs->get_id());
}

ccl_comm* ccl_comm::createExt(int size,
                              int rank,
                              ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                              ccl::ccl_comm_attr_impl& attr) {
    return new ccl_comm(size, rank, get_kvs_wrapper(kvs), attr);
}

ccl_comm* ccl_comm::createExt(int size,
                              ccl::shared_ptr_class<ccl::kvs_interface> kvs,
                              ccl::ccl_comm_attr_impl& attr) {
    return new ccl_comm(size, get_kvs_wrapper(kvs), attr);
}

void ccl_comm::create_topo_subcommsExt(int size, int rank) {
    // TODO: create r2r properly
    r2r_comm = std::shared_ptr<ccl_comm>(create_subcommExt(1, rank));

    node_comm = std::shared_ptr<ccl_comm>(create_subcommExt(size, rank));
    even_comm = std::shared_ptr<ccl_comm>(create_subcommExt(
        topo_manager.get_inter_card_colors(), rank, topo_manager.get_intra_card_color(rank)));
    pair_comm = std::shared_ptr<ccl_comm>(create_subcommExt(
        topo_manager.get_intra_card_colors(), rank, topo_manager.get_inter_card_color(rank) % 2));

    node_comm->is_uniform = true;
    r2r_comm->is_uniform = true;
}

ccl_comm* ccl_comm::create_subcommExt(const std::vector<int>& colors, int rank, int key) const {
    // Group ranks by color
    std::map<int, std::vector<std::pair<int, int>>> color_to_ranks;
    for (size_t i = 0; i < colors.size(); ++i) {
        color_to_ranks[colors[i]].push_back({ i, key }); // Assuming key is provided for each rank
    }

    // Sort each color group by key to determine rank within the subcommunicator
    for (auto& pair : color_to_ranks) {
        std::sort(pair.second.begin(),
                  pair.second.end(),
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                      return a.second < b.second;
                  });
    }

    // Find the subcommunicator for the given rank and create a new ccl_comm with the appropriate size and rank
    int new_rank = -1;
    int new_size = color_to_ranks[colors[rank]].size();
    for (size_t i = 0; i < color_to_ranks[colors[rank]].size(); ++i) {
        if (color_to_ranks[colors[rank]][i].first == rank) {
            new_rank = i;
            break;
        }
    }

    if (new_rank == -1) {
        // Rank was not found in the subcommunicator, which should not happen
        throw std::runtime_error("Rank not found in subcommunicator");
    }

    ccl_comm* comm = new ccl_comm(new_size, new_rank);
    comm->set_parent_comm(const_cast<ccl_comm*>(this));
    LOG_DEBUG("new subcomm Ext: size: ", new_size, ", rank ", new_rank);
    return comm;
}

ccl_comm* ccl_comm::create_subcommExt(int size, int rank) const {
    ccl_comm* comm = new ccl_comm(size, rank);
    comm->set_parent_comm(const_cast<ccl_comm*>(this));
    LOG_DEBUG("new subcomm Ext: size: ", size, ", rank ", rank);
    return comm;
}

ccl_comm* ccl_comm::create_subcomm_split_mt(int color, int key) const {
    int parent_size = this->size();
    int parent_rank = this->rank();

    auto& shared_data = ccl::global_data::get().shared_data;

    static std::mutex split_mutex;
    static std::unordered_map<int, std::vector<std::pair<int, int>>> split_data;
    // Per-color group: the global_current_id that all threads in that group will share
    static std::unordered_map<int, std::map<int, int>> split_group_ids;

    int comm_id = this->global_current_id;

    // Phase 1: Publish this thread's color and key
    {
        std::lock_guard<std::mutex> lock(split_mutex);
        if (split_data.find(comm_id) == split_data.end()) {
            split_data[comm_id].resize(parent_size, {-1, -1});
        }
        split_data[comm_id][parent_rank] = {color, key};
    }

    // Wait for all parent threads to publish their color/key
    pthread_barrier_wait(shared_data->get_barrier(comm_id));

    // Phase 2: Read colors, compute split group membership and sizes in one pass
    std::vector<int> colors;
    std::map<int, int> color_sizes;
    {
        std::lock_guard<std::mutex> lock(split_mutex);
        for (const auto& pair : split_data[comm_id]) {
            colors.push_back(pair.first);
            color_sizes[pair.first]++;
        }
    }

    int my_color = colors[parent_rank];

    // NOCOLOR early-out: the plugin remaps ONECCL_SPLIT_NOCOLOR to INT_MAX.
    // These threads only participate in parent barriers, then return a minimal
    // comm that the plugin will immediately discard.
    if (my_color == INT_MAX) {
        ccl_comm* dummy = new ccl_comm(1, 0);
        dummy->enable_multi_thread_instance = true;

        // Final parent barrier — all parent threads must sync before cleanup
        pthread_barrier_wait(shared_data->get_barrier(comm_id));

        {
            std::lock_guard<std::mutex> lock(split_mutex);
            if (split_data.find(comm_id) != split_data.end()) {
                split_data.erase(comm_id);
                split_group_ids.erase(comm_id);
            }
        }

        return dummy;
    }

    int split_size = color_sizes[my_color];

    // Allocate a unique global_current_id for each color group.
    // The first thread per color to arrive allocates the ID.
    // All threads in the same color group must share the same ID
    // so their barriers and hash_table entries are coordinated.
    int split_global_id = 0;
    {
        std::lock_guard<std::mutex> lock(split_mutex);
        auto& group_ids = split_group_ids[comm_id];
        if (group_ids.find(my_color) == group_ids.end()) {
            group_ids[my_color] = ccl::kvs::allocate_id();
        }
        split_global_id = group_ids[my_color];
    }

    // Compute split_rank: position of this thread among same-color ranks, ordered by key
    std::vector<std::pair<int, int>> my_group; // {key, parent_rank}
    for (int i = 0; i < parent_size; i++) {
        if (colors[i] == my_color) {
            my_group.push_back({split_data[comm_id][i].second, i});
        }
    }
    std::sort(my_group.begin(), my_group.end());
    int split_rank = 0;
    for (size_t i = 0; i < my_group.size(); i++) {
        if (my_group[i].second == parent_rank) {
            split_rank = static_cast<int>(i);
            break;
        }
    }

    ccl_comm* new_comm = new ccl_comm(split_size, split_rank);
    new_comm->device_ptr = this->device_ptr;
    new_comm->context_ptr = this->context_ptr;
    new_comm->enable_multi_thread_instance = true;
    new_comm->global_current_id = split_global_id;

    // Initialize barrier for the split group (sized to split_size, not parent_size)
    shared_data->init_barrier_wait(split_size, split_global_id);

    // Allocate group_buffers for the split comm (needed by group operations)
    if (split_rank == 0) {
        allocate_group_buffers(split_global_id, split_size);
    }
    pthread_barrier_wait(shared_data->get_barrier(split_global_id));

    // Inherit and filter parent's topology
    std::vector<int> split_parent_ranks;
    for (size_t i = 0; i < colors.size(); ++i) {
        if (colors[i] == my_color) {
            split_parent_ranks.push_back(i);
        }
    }

    auto copy_topo_scalars = [](ccl_comm* dst, const ccl_comm* src, bool single_card) {
        dst->topo_manager.is_single_node = src->topo_manager.is_single_node;
        dst->topo_manager.is_single_card = single_card;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        dst->topo_manager.set_p2p_access(src->topo_manager.has_p2p_access());
        dst->topo_manager.set_p2p_atomics(src->topo_manager.has_p2p_atomics());
        dst->topo_manager.set_all_vertices_connected(src->topo_manager.has_all_vertices_connected());
#endif
    };

    copy_topo_scalars(new_comm, this, split_size <= this->topo_manager.max_ranks_per_card);

    const auto& parent_intra_colors = this->topo_manager.get_intra_card_colors();
    const auto& parent_inter_colors = this->topo_manager.get_inter_card_colors();

    std::vector<int> filtered_intra_colors;
    std::vector<int> filtered_inter_colors;
    for (int pr : split_parent_ranks) {
        filtered_intra_colors.push_back(parent_intra_colors[pr]);
        filtered_inter_colors.push_back(parent_inter_colors[pr]);
    }

    new_comm->r2r_comm = std::shared_ptr<ccl_comm>(new_comm->create_subcommExt(1, split_rank));
    new_comm->node_comm = std::shared_ptr<ccl_comm>(new_comm->create_subcommExt(split_size, split_rank));
    new_comm->even_comm = std::shared_ptr<ccl_comm>(
        new_comm->create_subcommExt(filtered_inter_colors, split_rank, filtered_intra_colors[split_rank]));
    new_comm->pair_comm = std::shared_ptr<ccl_comm>(
        new_comm->create_subcommExt(filtered_intra_colors, split_rank, filtered_inter_colors[split_rank] % 2));

    // Create env with real device so sycl-kernels path is enabled
    new_comm->env = std::make_shared<ccl_comm_env>(new_comm->device_ptr);

    auto init_topo_subcomm = [&](std::shared_ptr<ccl_comm>& subcomm) {
        if (subcomm) {
            subcomm->enable_multi_thread_instance = true;
            subcomm->device_ptr = new_comm->device_ptr;
            subcomm->context_ptr = new_comm->context_ptr;
            subcomm->env = new_comm->env;
            subcomm->global_current_id = split_global_id;
            copy_topo_scalars(subcomm.get(), new_comm, new_comm->topo_manager.is_single_card);
        }
    };

    init_topo_subcomm(new_comm->r2r_comm);
    init_topo_subcomm(new_comm->node_comm);
    init_topo_subcomm(new_comm->even_comm);
    init_topo_subcomm(new_comm->pair_comm);

    new_comm->node_comm->is_uniform = true;
    new_comm->r2r_comm->is_uniform = true;

    // Initialize SYCL kernel shared buffers for the split comm.
    // Each split group has its own global_current_id with barriers sized to split_size,
    // so only threads in this group participate in coll_initExt barriers.
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    init_sycl_kernels_path(new_comm);
#endif

    // Barrier across all parent threads before cleanup
    pthread_barrier_wait(shared_data->get_barrier(comm_id));

    {
        std::lock_guard<std::mutex> lock(split_mutex);
        if (split_data.find(comm_id) != split_data.end()) {
            split_data.erase(comm_id);
            split_group_ids.erase(comm_id);
        }
    }

    return new_comm;
}

void ccl_comm::initExt(int size,
                       int rank,
                       int comm_id,
                       std::shared_ptr<atl_base_comm> atl_comm,
                       bool share_resources,
                       bool is_sub_communicator,
                       int group_id) {
    if (group_impl::is_group_active()) {
        LOG_WARN("Creating communicator inside group operation");
    }

    enable_multi_thread_instance = true;
    // Set the flag in shared_data for group operations
    ccl::global_data::get().shared_data->is_multi_thread_instance = true;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ESIMD)
    // TODO: choose more correct place for falling back
    CCL_THROW_IF_NOT(ccl::global_data::env().sycl_esimd == 0,
                     "esimd kernels are not support by multi-threading case");
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ESIMD
    comm_rank = rank;
    comm_size = size;

    ccl::global_data::get().shared_data->init_barrier_wait(size, group_id);

    if (comm_rank >= comm_size || comm_size <= 0) {
        throw ccl::exception("incorrect rank or size when creating \
                             communicator: rank: " +
                             std::to_string(comm_rank) + ", size: " + std::to_string(comm_size));
    }

    // Potential race: multiple threads may overwrite this variable simultaneously that's why use barrier
    global_current_id = group_id;
    pthread_barrier_wait(ccl::global_data::get().shared_data->get_barrier(group_id));

    if (comm_rank == 0) {
        allocate_group_buffers(global_current_id, comm_size);
    }

    pthread_barrier_wait(ccl::global_data::get().shared_data->get_barrier(group_id));

    comm_impl =
        std::unique_ptr<ccl_internal_comm>(new ccl_internal_comm(comm_id, comm_rank, comm_size));

    if (!share_resources) {
        allocate_resources();
    }

    if (!is_sub_communicator) {
        // warning: color::fixed is hardcoded, todo: implement support for other color modes
        ccl::global_data::env().topo_color = ccl::topo_color_mode::fixed;
        topo_manager.init(size, rank, global_current_id, device_ptr, context_ptr);
        if (!comm_rank && device_ptr) {
            LOG_INFO("topo_manager:", topo_manager.to_string());
        }
        create_topo_subcommsExt(size, rank);
    }

    std::vector<int> rank2rank_map;
    rank2rank_map.resize(size);
    for (int i = 0; i < size; i++) {
        rank2rank_map[i] = i;
    }

    local2global_map = rank2rank_map;
    pthread_barrier_wait(ccl::global_data::get().shared_data->get_barrier(group_id));

    env = std::make_shared<ccl_comm_env>(device_ptr);

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    init_sycl_kernels_path(this);
#endif
}


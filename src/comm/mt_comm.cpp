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
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.h"
#include "exec/exec.hpp"
#include "coll/coll.hpp"
#include "coll/attr/ccl_common_op_attrs.hpp"
#include "comm/comm.hpp"
#include "comm/comm_impl.hpp"
#include "common/global/global.hpp"
#include "common/event/impls/host_event.hpp"
#include "common/request/request.hpp"
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

ccl_internal_comm::ccl_internal_comm(int comm_id, int rank, int size)
        : m_dtree(size, rank)
#ifdef CCL_ENABLE_SYCL
          ,
          m_barrier_data(rank, size)
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

void ccl_comm::initExt(int size,
                       int rank,
                       int comm_id,
                       std::shared_ptr<atl_base_comm> atl_comm,
                       bool share_resources,
                       bool is_sub_communicator,
                       int group_id) {
    if (group_impl::is_group_active) {
        LOG_WARN("Creating communicator inside group operation");
    }

    enable_multi_thread_instance = true;
#ifdef CCL_ENABLE_SYCL
    // TODO: choose more correct place for falling back
    CCL_THROW_IF_NOT(ccl::global_data::env().sycl_esimd == 0,
                     "esimd kernels are not support by multi-threading case");
#endif // CCL_ENABLE_SYCL
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
    pthread_barrier_wait(&ccl::global_data::get().shared_data->barrier_waits[group_id]);

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
    pthread_barrier_wait(&ccl::global_data::get().shared_data->barrier_waits[global_current_id]);

    env = std::make_shared<ccl_comm_env>(device_ptr);
}

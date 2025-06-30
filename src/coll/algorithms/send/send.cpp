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

/*
*
*  (C) 2001 by Argonne National Laboratory.
*      See COPYRIGHT in top-level directory.
*/

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll_util.hpp"
#include "common/utils/utils.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/entry/ack_accept_entry.hpp"
#include "sched/entry/ack_report_entry.hpp"

ccl::status ccl_coll_build_direct_send(ccl_sched* sched,
                                       ccl_buffer buf,
                                       size_t count,
                                       const ccl_datatype& dtype,
                                       int peer_rank,
                                       ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    LOG_DEBUG("build direct SEND: ", comm->rank(), ", count: ", count, ", peer_rank: ", peer_rank);
    CCL_THROW_IF_NOT(peer_rank > CCL_INVALID_PEER_RANK_IDX && peer_rank < comm->size(),
                     "invalid peer_rank: ",
                     peer_rank);
    entry_factory::create<send_entry>(sched, buf, count, dtype, peer_rank, comm);

    return status;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
ccl::status ccl_coll_build_topo_send(ccl_sched* sched,
                                     ccl_buffer buf,
                                     size_t count,
                                     const ccl_datatype& dtype,
                                     int peer_rank,
                                     ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    ccl_comm* node_comm = comm->get_node_comm().get();
    auto node_peer_rank = node_comm->get_rank_from_global(peer_rank);
    auto node_curr_rank = node_comm->rank();

    LOG_DEBUG("build topo SEND buf: ", buf.get_ptr(), " and peer_rank: ", node_peer_rank);
    CCL_THROW_IF_NOT(
        node_peer_rank > CCL_INVALID_PEER_RANK_IDX && node_peer_rank < node_comm->size(),
        "invalid peer_rank: ",
        node_peer_rank,
        " for send op");

    const std::vector<ze_handle_exchange_entry::mem_desc_t> buffer{
        { buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0 idx
    };

    std::vector<ze_event_handle_t> wait_events{};
    ze_event_handle_t out_event{};

    ccl::utils::pt2pt_handle_exchange_info info{ node_peer_rank,
                                                 ccl::utils::pt2pt_handle_exchange_role::sender };
    if (!ccl::global_data::env().ze_pt2pt_read) {
        info.role = ccl::utils::pt2pt_handle_exchange_role::receiver;
    }

    ccl::add_handle_exchange(sched,
                             node_comm,
                             wait_events,
                             out_event,
                             buffer,
                             ccl_comm::invalid_rank /*skip_rank*/,
                             nullptr,
                             0,
                             info);
    ccl::utils::clear_and_push_back(wait_events, out_event);
    LOG_DEBUG("build SEND: add_handle_exchange is done");

    ccl_sched_id_t pt2pt_ack_tag = node_comm->get_atl_comm()->tag_creator->get_pt2pt_ack_tag();

    if (!ccl::global_data::env().ze_pt2pt_read) {
        LOG_DEBUG("build SEND: write mode is enabled");
        entry_factory::create<copy_entry>(
            sched,
            buf,
            ccl_buffer(),
            count,
            dtype,
            copy_attr(node_peer_rank, 0, copy_direction::d2d, true /*pt2pt_op*/));
        LOG_DEBUG("build SEND: copy_entry is created");

        entry_factory::create<ack_report_entry>(
            sched, pt2pt_ack_tag, node_peer_rank, node_curr_rank, node_comm);
        LOG_DEBUG("build SEND: ack_report_entry is created");
    }
    else {
        LOG_DEBUG("build SEND: read mode is enabled");
        entry_factory::create<ack_accept_entry>(sched, pt2pt_ack_tag, node_peer_rank, node_comm);
        LOG_DEBUG("build SEND: ack_accept_entry is created");
    }

    entry_factory::create<ze_execute_cmdlists_on_init_entry>(sched);
    return status;
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

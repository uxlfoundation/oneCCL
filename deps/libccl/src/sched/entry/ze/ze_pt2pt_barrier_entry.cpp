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

#include "sched/entry/coll/direct/base_coll_entry.hpp"
#include "sched/entry/ze/ze_pt2pt_barrier_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#include <string>
#include <tuple>

using namespace ccl;
using namespace ccl::ze;

ze_pt2pt_barrier_entry::ze_pt2pt_barrier_entry(ccl_sched* sched, ccl_comm* comm, int peer_rank)
        : sched_entry(sched),
          comm(comm),
          peer_rank(peer_rank) {}

void ze_pt2pt_barrier_entry::start() {
    LOG_DEBUG("ze_pt2pt_barrier_entry is strated");
    status = ccl_sched_entry_status_started;
}

void ze_pt2pt_barrier_entry::update() {
    ccl_sched_id_t pt2pt_ack_first, pt2pt_ack_second;
    std::tie(pt2pt_ack_first, pt2pt_ack_second) =
        comm->get_atl_comm()->tag_creator->get_pt2pt_sync_tags();

    if (sched->coll_param.ctype == ccl_coll_recv) {
        uint64_t ack_tag_first = comm->get_atl_comm()->tag_creator->create(
            comm->rank(), comm->get_comm_id(), pt2pt_ack_first, sched->get_op_id());
        uint64_t ack_tag_second = comm->get_atl_comm()->tag_creator->create(
            peer_rank, comm->get_comm_id(), pt2pt_ack_second, sched->get_op_id());

        ccl::utils::send_ack_to_peer(comm->get_atl_comm(), ack_tag_first, peer_rank);
        ccl::utils::recv_ack_from_peer(comm->get_atl_comm(), ack_tag_second, peer_rank);
        LOG_DEBUG("recv side: first_tag: ", ack_tag_first, ", second_tag: ", ack_tag_second);
    }
    if (sched->coll_param.ctype == ccl_coll_send) {
        uint64_t ack_tag_first = comm->get_atl_comm()->tag_creator->create(
            peer_rank, comm->get_comm_id(), pt2pt_ack_first, sched->get_op_id());
        uint64_t ack_tag_second = comm->get_atl_comm()->tag_creator->create(
            comm->rank(), comm->get_comm_id(), pt2pt_ack_second, sched->get_op_id());

        ccl::utils::recv_ack_from_peer(comm->get_atl_comm(), ack_tag_first, peer_rank);
        ccl::utils::send_ack_to_peer(comm->get_atl_comm(), ack_tag_second, peer_rank);
        LOG_DEBUG("send side: first_tag: ", ack_tag_first, ", second_tag: ", ack_tag_second);
    }
    LOG_DEBUG("ze_pt2pt_barrier_entry is complete");
    status = ccl_sched_entry_status_complete;
}

std::string ze_pt2pt_barrier_entry::name_ext() const {
    std::stringstream out;
    out << name() << "rank:" << comm->rank() << ", peer_rank: " << peer_rank;
    return out.str();
}

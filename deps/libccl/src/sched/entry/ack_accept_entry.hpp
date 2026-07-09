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

#include "sched/entry/entry.hpp"
#include "common/utils/utils.hpp"
#include "atl_def.h"

class ack_accept_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ACCEPT_ACK";
    }

    ack_accept_entry() = delete;
    ack_accept_entry(ccl_sched* sched,
                     ccl_sched_id_t pt2pt_ack_tag,
                     int node_peer_rank,
                     ccl_comm* node_comm)
            : sched_entry(sched, true /* is_barrier */),
              pt2pt_ack_tag(pt2pt_ack_tag),
              node_comm(node_comm),
              node_peer_rank(node_peer_rank) {}

    void start() override {
        ack_tag = node_comm->get_atl_comm()->tag_creator->create(
            node_peer_rank, node_comm->get_comm_id(), pt2pt_ack_tag, sched->get_op_id());
        status = ccl_sched_entry_status_started;
        sched->set_submitted_to_gpu(true);
        req = ccl::utils::recv_ack_from_peer(
            node_comm->get_atl_comm(), ack_tag, node_peer_rank, sched->coll_attr.synchronous);
    }

    void update() override {
        bool is_completed = sched->coll_attr.synchronous;
        if (!is_completed) {
            is_completed = ccl::utils::check_async(node_comm->get_atl_comm(), req);
        }
        if (is_completed) {
            LOG_DEBUG("recv_ack_from_peer is done with tag: ",
                      ack_tag,
                      ", comm_rank: ",
                      node_comm->rank(),
                      ", peer_rank: ",
                      node_peer_rank);
            status = ccl_sched_entry_status_complete;
        }
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(
            str, "node_peer_rank ", node_peer_rank, ", pt2pt_ack_tag ", pt2pt_ack_tag, "\n");
    }

private:
    ccl_sched_id_t pt2pt_ack_tag{};
    ccl_comm* node_comm{};
    int node_peer_rank{ ccl::utils::invalid_rank };
    uint64_t ack_tag{};
    atl_req_t req{};
};

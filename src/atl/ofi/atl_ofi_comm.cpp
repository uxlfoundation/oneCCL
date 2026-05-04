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
#include "atl/ofi/atl_ofi_comm.hpp"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable_simple.h"
#include "atl/util/pm/pmi_rt/pmi_simple.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable_simple_internal.h"
#include "atl/ofi/atl_ofi.hpp"
#include "exec/exec.hpp"

atl_ofi_comm::atl_ofi_comm() {
    pmi = std::shared_ptr<ipmi>(new pmi_simple());
    CCL_THROW_IF_NOT(init_transport(true) == ATL_STATUS_SUCCESS, "init transport failed");
}

atl_ofi_comm::atl_ofi_comm(std::shared_ptr<ikvs_wrapper> k) {
    pmi = std::shared_ptr<ipmi>(new pmi_simple());
    CCL_THROW_IF_NOT(init_transport(true) == ATL_STATUS_SUCCESS, "init transport failed");
}

atl_ofi_comm::atl_ofi_comm(int comm_size,
                           const std::vector<int>& ranks,
                           std::shared_ptr<ikvs_wrapper> k) {
    std::shared_ptr<internal_kvs> kvs;
    if ((kvs = std::dynamic_pointer_cast<internal_kvs>(k)) != nullptr) {
        pmi = std::shared_ptr<ipmi>(new pmi_resizable_simple_internal(comm_size, ranks, kvs));
    }
    else {
        pmi = std::shared_ptr<ipmi>(new pmi_resizable_simple(comm_size, ranks, k));
    }

    CCL_THROW_IF_NOT(init_transport(true) == ATL_STATUS_SUCCESS, "init transport failed");
}

atl_status_t atl_ofi_comm::allgatherv(size_t ep_idx,
                                      const void* send_buf,
                                      size_t send_len,
                                      void* recv_buf,
                                      const size_t* recv_lens,
                                      const size_t* offsets,
                                      atl_req_t& req) {
    std::vector<atl_req> send_reqs(size - 1);
    std::vector<atl_req> recv_reqs(size - 1);

    int tag_comm_id = (comm_id != atl_comm_id_storage::invalid_comm_id)
                          ? comm_id
                          : atl_comm_id_storage::max_comm_id;

    LOG_DEBUG("ofi_allgatherv: comm_rank: ",
              rank,
              ", comm_size: ",
              size,
              ", send_len: ",
              send_len,
              ", comm_id: ",
              comm_id,
              ", tag_comm_id: ",
              tag_comm_id,
              ", tag_counter: ",
              tag_counter);

    for (int peer = 0, req_idx = 0; peer < size; peer++) {
        if (peer == rank)
            continue;

        uint64_t op_tag = tag_creator->create(rank, tag_comm_id, tag_counter);
        // LOG_DEBUG("ofi_allgatherv: send: rank: ", rank,
        //     ", peer: ", peer,
        //     ", comm_id: ", comm_id,
        //     ", tag_comm_id: ", tag_comm_id,
        //     ", tag_counter: ", tag_counter,
        //     ", op_tag: ", op_tag);

        atl_status_t ret;

        do {
            ret = send(ep_idx, send_buf, send_len, peer, op_tag, send_reqs[req_idx]);
            CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "send failed");
            if (ret == ATL_STATUS_AGAIN) {
                ccl_yield(ccl::global_data::env().yield_type);
            }
        } while (ret == ATL_STATUS_AGAIN);

        op_tag = tag_creator->create(peer, tag_comm_id, tag_counter);
        // LOG_DEBUG("ofi_allgatherv: recv: rank: ", rank,
        //     ", peer: ", peer,
        //     ", comm_id: ", comm_id,
        //     ", tag_comm_id: ", tag_comm_id,
        //     ", tag_counter: ", tag_counter,
        //     ", op_tag: ", op_tag);

        do {
            ret = recv(ep_idx,
                       (char*)recv_buf + offsets[peer],
                       recv_lens[peer],
                       peer,
                       op_tag,
                       recv_reqs[req_idx]);
            CCL_THROW_IF_NOT(ret != ATL_STATUS_FAILURE, "recv failed");
            if (ret == ATL_STATUS_AGAIN) {
                ccl_yield(ccl::global_data::env().yield_type);
            }
        } while (ret == ATL_STATUS_AGAIN);

        req_idx++;
    }

    if ((char*)recv_buf + offsets[rank] != send_buf) {
        memcpy((char*)recv_buf + offsets[rank], send_buf, recv_lens[rank]);
    }

    bool is_completed = false;
    while (!is_completed) {
        is_completed = true;
        poll(ep_idx);
        for (size_t i = 0; i < send_reqs.size(); i++) {
            if (!send_reqs[i].is_completed) {
                CCL_THROW_IF_NOT(check(ep_idx, send_reqs[i]) != ATL_STATUS_FAILURE,
                                 "check send failed");
                is_completed = false;
                break;
            }
            if (!recv_reqs[i].is_completed) {
                CCL_THROW_IF_NOT(check(ep_idx, recv_reqs[i]) != ATL_STATUS_FAILURE,
                                 "check recv failed");
                is_completed = false;
                break;
            }
        }
    }

    // to let user complete this operation through wait(req)
    req.is_completed = false;

    atl_ofi_req_t* ofi_req = ((atl_ofi_req_t*)req.internal);
    ofi_req->comp_state = ATL_OFI_COMP_COMPLETED;

    tag_counter++;

    return ATL_STATUS_SUCCESS;
}

std::shared_ptr<atl_base_comm> atl_ofi_comm::comm_split(int color, int key) {
    return std::shared_ptr<atl_base_comm>(new atl_ofi_comm(this, color));
}

atl_ofi_comm::atl_ofi_comm(atl_ofi_comm* parent, int color) {
    eps = parent->eps;
    parent_size = parent->size;
    parent_rank = parent->rank;
    pmi = parent->pmi;

    coord.hostname_hash = transport->get_proc_coord().hostname_hash;
    coord.local_idx = 0;
    coord.local_count = 0;

    std::vector<rank_info_t> ranks_info(parent_size);
    rank_info_t rank_info{ color, parent_rank, coord.hostname_hash };
    std::vector<size_t> recv_lens(parent_size, sizeof(rank_info));
    std::vector<size_t> offsets(parent_size);
    offsets[0] = 0;
    for (size_t i = 1; i < offsets.size(); i++) {
        offsets[i] = offsets[i - 1] + recv_lens[i];
    }

    atl_req req{};
    atl_status_t status = parent->allgatherv(0 /* ep_idx */,
                                             &rank_info,
                                             sizeof(rank_info),
                                             ranks_info.data(),
                                             recv_lens.data(),
                                             offsets.data(),
                                             req);
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "allgatherv failed with status: ", status);

    status = wait(0, req);
    CCL_THROW_IF_NOT(status == ATL_STATUS_SUCCESS, "wait failed with status: ", status);

    CCL_THROW_IF_NOT(rank2proc_map.empty());
    CCL_THROW_IF_NOT(rank2rank_map.empty());

    size = 0;

    for (auto& it : ranks_info) {
        int recv_color;
        int recv_rank;
        size_t recv_hash;
        std::tie(recv_color, recv_rank, recv_hash) = it;
        if (recv_color == color) {
            rank2proc_map.push_back(parent->rank2proc_map[recv_rank]);
            rank2rank_map.push_back(recv_rank);

            if (recv_hash == coord.hostname_hash) {
                coord.local_count++;
            }

            if (recv_rank == parent_rank) {
                coord.global_idx = rank = rank2proc_map.size() - 1;
                coord.local_idx = (coord.local_count - 1);
            }
            size++;
        }
    }
    coord.global_count = size;

    LOG_DEBUG("color: ",
              color,
              ", ",
              to_string(coord),
              ", rank2proc_map: ",
              ccl::utils::vec_to_string(rank2proc_map),
              ", parent rank2proc_map: ",
              ccl::utils::vec_to_string(parent->rank2proc_map));

    coord.validate(rank, size);

    CCL_THROW_IF_NOT(init_transport(false) == ATL_STATUS_SUCCESS, "init transport failed");
}

atl_status_t atl_ofi_comm::init_transport(bool is_new) {
    LOG_DEBUG("init atl, requested ep_count ", attr.in.ep_count);

    if (is_new) {
        ATL_CHECK_STATUS(pmi->pmrt_init(), "pmi init failed");
        static std::mutex memory_mutex;
        {
            std::lock_guard<std::mutex> lock(memory_mutex);
            if (!transport) {
                transport = new atl_ofi();
            }
            if (!transport->is_inited()) {
                CCL_THROW_IF_NOT(
                    transport->init(nullptr, nullptr, &attr, nullptr, pmi) == ATL_STATUS_SUCCESS,
                    "failed to initialize ATL");

                if (pmi->get_rank() == 0) {
                    LOG_INFO(transport->to_string());
                    LOG_INFO(to_string(attr));
                }
            }
        }
        eps = transport->get_eps();

        parent_rank = rank = pmi->get_rank();
        parent_size = size = pmi->get_size();

        coord = transport->get_proc_coord();
        coord.validate(rank, size);

        transport->get_rank2proc_map(pmi, rank2proc_map, coord);
        rank2rank_map.resize(size);
        for (int i = 0; i < size; i++) {
            rank2rank_map[i] = i;
        }
    }

    init_tag();

    comm_id = create_comm_id();
    comm_count++;

    update_executor();

    return ATL_STATUS_SUCCESS;
}

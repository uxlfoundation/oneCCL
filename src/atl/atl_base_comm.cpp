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
#ifdef CCL_ENABLE_MPI
#include "atl/mpi/atl_mpi.hpp"
#include "atl/mpi/atl_mpi_comm.hpp"
#endif // CCL_ENABLE_MPI

#include "atl/atl_base_comm.hpp"
#include "atl/ofi/atl_ofi_comm.hpp"
#include "atl/ofi/atl_ofi.hpp"
#include "atl/util/pm/pm_rt.h"
#include "common/utils/utils.hpp"
#include "comm/atl_tag.hpp"
#include "exec/exec.hpp"

atl_attr_t atl_base_comm::attr = {
    /* in */
    {
        0, /* enable_shm */
        0, /* enable_rma */
        0, /* enable_hmem */
        0, /* enable_sync_coll */
        0, /* enable_extra_ep */
        1, /* ep_count */
        ATL_MNIC_NONE, /* mnic_type */
        "", /* mnic_name */
        1, /* mnic_count */
        ATL_MNIC_OFFSET_NONE /* mnic_offset */
    },

    /* out */
    {
        0, /* enable_shm */
        0, /* enable_rma */
        0, /* enable_hmem */
        ATL_MNIC_NONE, /* mnic_type */
        0, /* mnic_count */
        0, /* tag_bits */
        0, /* max_tag */
        0, /* max_order_waw_size */
    }
};

ccl_executor* atl_base_comm::executor{ nullptr };
atl_base_transport* atl_base_comm::transport{ nullptr };
std::atomic<size_t> atl_base_comm::comm_count{ 0 };
ccl_spinlock atl_base_comm::comm_id_storage_guard{};

atl_base_comm::~atl_base_comm() {
    std::lock_guard<ccl_spinlock> lock{ comm_id_storage_guard };
    transport->get_comm_id_storage().release(comm_id);
    tag_creator.reset();
    comm_count--;
    if (comm_count.load() == 0) {
        transport->finalize(rank);
        delete transport;
        transport = nullptr;
    }
}

int atl_base_comm::create_comm_id() {
    int new_comm_id = atl_comm_id_storage::invalid_comm_id;

    std::lock_guard<ccl_spinlock> lock{ comm_id_storage_guard };

    auto comm_id_map = transport->get_comm_id_storage().get_map();
    int send_bytes = sizeof(new_comm_id) * comm_id_map.size();
    std::vector<int> all_comm_id_maps(size * comm_id_map.size(),
                                      atl_comm_id_storage::invalid_comm_id);

    std::vector<size_t> recv_bytes(size, send_bytes);
    std::vector<size_t> offsets(size, 0);
    for (int i = 1; i < size; i++) {
        offsets[i] = offsets[i - 1] + recv_bytes[i - 1];
    }

    LOG_DEBUG("rank2rank_map: ", ccl::utils::vec_to_string(rank2rank_map));
    LOG_DEBUG("rank2proc_map: ", ccl::utils::vec_to_string(rank2proc_map));
    // LOG_DEBUG("comm_id_map: ", ccl::utils::vec_to_string(comm_id_map));

    if (std::any_of(comm_id_map.begin(), comm_id_map.end(), [](int id) {
            return id == atl_comm_id_storage::invalid_comm_id;
        })) {
        CCL_THROW("comm_id_map contains invalid value: ", ccl::utils::vec_to_string(comm_id_map));
    }

    atl_req_t req{};
    allgatherv(0 /* ep_idx */,
               comm_id_map.data(),
               send_bytes,
               all_comm_id_maps.data(),
               recv_bytes.data(),
               offsets.data(),
               req);
    wait(0, req);

    // LOG_DEBUG("all_comm_id_maps: ", ccl::utils::vec_to_string(all_comm_id_maps));

    if (std::any_of(all_comm_id_maps.begin(), all_comm_id_maps.end(), [](int id) {
            return id == atl_comm_id_storage::invalid_comm_id;
        })) {
        CCL_THROW("all_comm_id_maps contains invalid value: ",
                  ccl::utils::vec_to_string(all_comm_id_maps));
    }

    for (size_t free_comm_id = 0; free_comm_id < comm_id_map.size(); free_comm_id++) {
        new_comm_id = free_comm_id;
        for (int rank_idx = 0; rank_idx < size; rank_idx++) {
            if (all_comm_id_maps[rank_idx * comm_id_map.size() + free_comm_id] != 1) {
                new_comm_id = atl_comm_id_storage::invalid_comm_id;
                break;
            }
        }
        if (new_comm_id != atl_comm_id_storage::invalid_comm_id) {
            break;
        }
    }

    LOG_DEBUG("new_comm_id ", new_comm_id);
    CCL_THROW_IF_NOT(new_comm_id != atl_comm_id_storage::invalid_comm_id, "unexpected comm_id");

    transport->get_comm_id_storage().acquire(new_comm_id);

    return new_comm_id;
}

void atl_base_comm::init_tag() {
    auto transport_type = ccl::global_data::env().atl_transport;

    size_t tag_bits = attr.out.tag_bits;
    size_t max_tag = attr.out.max_tag;

    switch (transport_type) {
        case ccl_atl_ofi:
            if (tag_bits == tag_layout::cxi) {
                tag_creator = std::shared_ptr<ccl_atl_tag>(
                    new ccl_atl_tag_impl<ofi_cxi_tag_layout>(tag_bits, max_tag));
            }
            else {
                tag_creator = std::shared_ptr<ccl_atl_tag>(
                    new ccl_atl_tag_impl<common_tag_layout>(tag_bits, max_tag));
            }
            break;
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi:
            CCL_THROW_IF_NOT(max_tag >= mpi_tag_layout::op_id_mask + mpi_tag_layout::sched_id_mask,
                             "sched_id and op_id have to be encoded uncut for MPI case");
            tag_creator = std::shared_ptr<ccl_atl_tag>(
                new ccl_atl_tag_impl<mpi_tag_layout>(tag_bits, max_tag));
            break;
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("unsupported tag type"); break;
    }
    if (rank == 0) {
        LOG_DEBUG("atl tag: ", tag_creator->to_string());
    }
}

void atl_base_comm::update_executor() {
    if (!executor->are_workers_started()) {
        if (rank < coord.local_count)
            LOG_INFO(
                "start workers for local process [", coord.local_idx, ":", coord.local_count, "]");
        executor->start_workers(coord);
    }
}

std::shared_ptr<atl_base_comm> atl_comm_manager::create() {
    std::shared_ptr<atl_base_comm> atl_comm;

    auto transport_type = ccl::global_data::env().atl_transport;

    switch (transport_type) {
        case ccl_atl_ofi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_ofi_comm()); break;
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_mpi_comm()); break;
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("unsupported yet"); break;
    }
    return atl_comm;
}

std::shared_ptr<atl_base_comm> atl_comm_manager::create(ccl::ccl_comm_attr_impl& attr) {
    std::shared_ptr<atl_base_comm> atl_comm;
    if (attr.blocking >= 0) {
        atl_comm.get()->attr.in.enable_sync_coll = (attr.blocking != 0);
    }

    auto transport_type = ccl::global_data::env().atl_transport;

    switch (transport_type) {
        case ccl_atl_ofi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_ofi_comm()); break;
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_mpi_comm()); break;
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("unsupported yet"); break;
    }
    return atl_comm;
}

std::shared_ptr<atl_base_comm> atl_comm_manager::create(std::shared_ptr<ikvs_wrapper> k,
                                                        ccl::ccl_comm_attr_impl& attr) {
    std::shared_ptr<atl_base_comm> atl_comm;
    if (attr.blocking >= 0) {
        atl_comm.get()->attr.in.enable_sync_coll = (attr.blocking != 0);
    }

    auto transport_type = ccl::global_data::env().atl_transport;

    switch (transport_type) {
        case ccl_atl_ofi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_ofi_comm(k)); break;
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi: atl_comm = std::shared_ptr<atl_base_comm>(new atl_mpi_comm(k)); break;
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("unsupported yet"); break;
    }
    return atl_comm;
}

std::shared_ptr<atl_base_comm> atl_comm_manager::create(int comm_size,
                                                        const std::vector<int>& ranks,
                                                        std::shared_ptr<ikvs_wrapper> k,
                                                        ccl::ccl_comm_attr_impl& attr) {
    std::shared_ptr<atl_base_comm> atl_comm;
    if (attr.blocking >= 0) {
        atl_comm.get()->attr.in.enable_sync_coll = (attr.blocking != 0);
    }

    auto transport_type = ccl::global_data::env().atl_transport;

    switch (transport_type) {
        case ccl_atl_ofi:
            atl_comm = std::shared_ptr<atl_base_comm>(new atl_ofi_comm(comm_size, ranks, k));
            break;
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi:
            atl_comm = std::shared_ptr<atl_base_comm>(new atl_mpi_comm(comm_size, ranks, k));
            break;
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("unsupported yet"); break;
    }
    return atl_comm;
}

std::shared_ptr<atl_base_comm> atl_comm_manager::create_with_id(
    const std::shared_ptr<atl_base_comm> base_comm,
    int comm_id) {
    std::shared_ptr<atl_base_comm> atl_comm;

    auto transport_type = ccl::global_data::env().atl_transport;

    switch (transport_type) {
        case ccl_atl_ofi: {
            std::shared_ptr<atl_ofi_comm> ofi_base_comm =
                std::dynamic_pointer_cast<atl_ofi_comm>(base_comm);
            atl_comm = std::shared_ptr<atl_base_comm>(new atl_ofi_comm(*ofi_base_comm.get()));
            break;
        }
#ifdef CCL_ENABLE_MPI
        case ccl_atl_mpi: {
            std::shared_ptr<atl_mpi_comm> mpi_base_comm =
                std::dynamic_pointer_cast<atl_mpi_comm>(base_comm);
            atl_comm = std::shared_ptr<atl_base_comm>(new atl_mpi_comm(*mpi_base_comm.get()));
            break;
        }
#endif // CCL_ENABLE_MPI
        default: LOG_ERROR("unsupported yet"); break;
    }

    atl_comm->comm_id = comm_id;
    atl_base_comm::comm_count++;

    return atl_comm;
}

void atl_comm_manager::set_internal_env(const atl_attr_t& attr) {
    auto transport_type = ccl::global_data::env().atl_transport;
    atl_base_comm::attr = attr;

    if (transport_type == ccl_atl_ofi)
        atl_ofi::set_env(attr);
#ifdef CCL_ENABLE_MPI
    else if (transport_type == ccl_atl_mpi)
        atl_mpi::set_env(attr);
#endif // CCL_ENABLE_MPI
}

void atl_comm_manager::set_executor(ccl_executor* exec) {
    atl_base_comm::executor = exec;
}

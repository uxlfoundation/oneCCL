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

#ifdef CCL_ENABLE_MPI

#include "atl/mpi/atl_mpi_comm.hpp"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable_simple.h"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable_simple_internal.h"
#include "exec/exec.hpp"

atl_mpi_comm::atl_mpi_comm() {
    init_transport(true /* new communicator */);
}

atl_mpi_comm::atl_mpi_comm(std::shared_ptr<ikvs_wrapper> k) : atl_mpi_comm() {
    (void)k;
}

atl_mpi_comm::atl_mpi_comm(int comm_size,
                           const std::vector<int>& comm_ranks,
                           std::shared_ptr<ikvs_wrapper> k) {
    std::shared_ptr<internal_kvs> kvs;
    if ((kvs = std::dynamic_pointer_cast<internal_kvs>(k)) != nullptr) {
        pmi = std::shared_ptr<ipmi>(new pmi_resizable_simple_internal(comm_size, comm_ranks, kvs));
    }
    else {
        pmi = std::shared_ptr<ipmi>(new pmi_resizable_simple(comm_size, comm_ranks, k));
    }

    init_transport(true /* new communicator */, comm_size, comm_ranks);
}

std::shared_ptr<atl_base_comm> atl_mpi_comm::comm_split(int color, int key) {
    return std::shared_ptr<atl_base_comm>(new atl_mpi_comm(this, color, key));
}

atl_mpi_comm::atl_mpi_comm(atl_mpi_comm* parent, int color, int key) {
    parent_rank = parent->rank;
    parent_size = parent->size;

    std::vector<atl_ep_t>& parent_eps = parent->eps;
    transport->comm_split(parent_eps,
                          eps,
                          color,
                          key,
                          parent_eps[0].coord.local_idx,
                          parent_eps[0].coord.local_count);

    coord = transport->create_proc_coord(eps[0]);
    rank = coord.global_idx;
    size = coord.global_count;

    init_transport(false /* is_new */);

    atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)eps[0].internal);

    rank2rank_map.resize(size);
    MPI_Allgather(&parent_rank, 1, MPI_INT, rank2rank_map.data(), 1, MPI_INT, mpi_ep->mpi_comm);
}

void atl_mpi_comm::update_eps() {
    for (auto& ep : eps) {
        ep.coord = coord;
    }
}

atl_status_t atl_mpi_comm::init_transport(bool is_new,
                                          int comm_size,
                                          const std::vector<int>& comm_ranks) {
    LOG_DEBUG("init atl, requested ep_count ", attr.in.ep_count);

    if (is_new) {
        MPI_Comm global_comm = MPI_COMM_WORLD;
        static std::mutex memory_mutex;
        {
            if (pmi) {
                ATL_CHECK_STATUS(pmi->pmrt_init(), "pmi init failed");
                global_comm = MPI_COMM_NULL;
            }

            std::lock_guard<std::mutex> lock(memory_mutex);

            if (!transport) {
                transport = new atl_mpi();
            }

            if (!transport->is_inited()) {
                CCL_THROW_IF_NOT(
                    transport->init(nullptr, nullptr, &attr, nullptr, pmi) == ATL_STATUS_SUCCESS,
                    "failed to initialize ATL");
            }

            if (global_comm == MPI_COMM_NULL) {
                atl_mpi* mpi_transport = (atl_mpi*)transport;
                ATL_CHECK_STATUS(
                    mpi_transport->comm_create(comm_size, comm_ranks, pmi, &global_comm),
                    "comm_create error");
            }

            int mpi_rank;
            MPI_Comm_rank(global_comm, &mpi_rank);
            if (mpi_rank == 0) {
                LOG_INFO(transport->to_string());
                LOG_INFO(to_string(attr));
            }
        }

        atl_mpi* mpi_transport = (atl_mpi*)transport;
        coord = mpi_transport->create_proc_coord(global_comm);
        mpi_transport->ep_init(eps, global_comm, coord.local_idx, coord.local_count);
        parent_rank = rank = coord.global_idx;
        parent_size = size = coord.global_count;

        rank2rank_map.resize(size);
        for (int i = 0; i < size; i++) {
            rank2rank_map[i] = i;
        }
    }

    init_tag();

    update_eps(); // TODO: align with atl_ofi_comm?
    comm_id = create_comm_id();
    comm_count++;

    update_executor();

    return ATL_STATUS_SUCCESS;
}

#endif //CCL_ENABLE_MPI

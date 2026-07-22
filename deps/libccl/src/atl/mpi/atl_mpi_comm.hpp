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

#ifdef CCL_ENABLE_MPI

#include "atl/atl_base_comm.hpp"
#include "atl/mpi/atl_mpi.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"

class atl_mpi_comm : public atl_base_comm {
public:
    ~atl_mpi_comm() = default;

    atl_mpi_comm();
    atl_mpi_comm(std::shared_ptr<ikvs_wrapper> k);
    atl_mpi_comm(int comm_size,
                 const std::vector<int>& local_ranks,
                 std::shared_ptr<ikvs_wrapper> k);

    atl_status_t finalize() override {
        transport->comms_free(eps);
        return ATL_STATUS_SUCCESS;
    }

    std::shared_ptr<atl_base_comm> comm_split(int color, int key) override;

    int get_mpi_comm() override {
        atl_mpi_ep_t* mpi_ep = ((atl_mpi_ep_t*)eps[0].internal);
        return (int)(mpi_ep->mpi_comm);
    }

private:
    friend atl_comm_manager;
    atl_mpi_comm(atl_mpi_comm* parent, int color, int key);
    void update_eps();
    atl_status_t init_transport(bool is_new,
                                int comm_size = 0,
                                const std::vector<int>& comm_ranks = {});
};

#endif //CCL_ENABLE_MPI

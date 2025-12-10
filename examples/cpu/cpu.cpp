/*
 Copyright 2016-2025 Intel Corporation
 
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

// examples/simple/simple.cpp
#include "oneapi/ccl.h"
#include "oneapi/ccl/v2/types.h"
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <vector>

int main() {
    int rank = 0;
    int world_size = 0;
    int version = 0;

    onecclComm_t comm = nullptr;
    onecclResult_t result = onecclSuccess;
    onecclUniqueId uid;

    // setenv("CCL_PLUGIN", "ONECCL_LEGACY_CPU", 0);
    onecclGetVersion(&version);

    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        onecclGetUniqueId(&uid);
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    result = onecclCommInitRank(&comm, world_size, uid, rank);
    if (result != onecclSuccess) {
        std::cerr << "Failed to initialize communicator.\n";
        return 1;
    }

    std::vector<int> data = {1, 2, 3, 4};
    std::vector<int> recv_data(data.size());

    result = onecclAllReduce(data.data(), recv_data.data(), data.size(),
                             onecclInt, onecclSum, comm, nullptr);
    if (result != onecclSuccess) {
        std::cerr << "AllReduce operation failed.\n";
        return 1;
    }

    result = onecclAllReduce(data.data(), recv_data.data(), data.size(),
                             onecclInt, onecclSum, comm, nullptr);
    if (result != onecclSuccess) {
        std::cerr << "AllReduce operation failed.\n";
        return 1;
    }

    // Output the result
    for (int val : recv_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    result = onecclCommDestroy(comm);
    if (result != onecclSuccess) {
        std::cerr << "Failed to destroy communicator.\n";
        return 1;
    }

    MPI_Finalize();

    return 0;
}

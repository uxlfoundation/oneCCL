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

#include "base.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    const size_t count = 10 * 1024;

    ccl::init();

    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "problem occurred during MPI init" };
    }

    int size = 0;
    int rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    if (size != 4) {
        if (rank == 0) {
            std::cerr << "FAILED: This test requires exactly 4 ranks." << std::endl;
        }
        return -1;
    }

    int scenario = 0; // Default scenario
    if (argc > 1) {
        scenario = std::atoi(argv[1]);
    }

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(size, rank, kvs);

    // Assign color and key for communicator splitting
    int color = 0;
    int key = rank % 3;

    switch (scenario) {
        case 1: color = (rank < 3) ? 1 : 0; break;
        case 2: color = 1; break;
        case 3: color = 0; break;
        case 4: color = (rank < 2) ? 2 : 1; break;
        default: color = (rank % 2 == 0) ? 0 : 1; break;
    }

    auto new_comm = ccl::split_communicator(comm, color, key);

    std::vector<float> send_buf(count, static_cast<float>(rank + 1));
    std::vector<float> recv_buf(count, 0);

    ccl::allreduce(send_buf.data(), recv_buf.data(), count, ccl::reduction::sum, new_comm).wait();

    float expected_sum = 0;

    switch (scenario) {
        case 1: expected_sum = (color == 1) ? 6 : 4; break;
        case 2:
        case 3: expected_sum = 10; break;
        case 4: expected_sum = (color == 2) ? 3 : 7; break;
        default: expected_sum = (color == 0) ? 4 : 6; break;
    }

    if (std::any_of(recv_buf.begin(), recv_buf.end(), [expected_sum](float value) {
            return value != expected_sum;
        })) {
        std::cerr << "FAILED" << std::endl;
        return -1;
    }

    if (rank == 0) {
        std::cout << "PASSED" << std::endl;
    }

    return 0;
}

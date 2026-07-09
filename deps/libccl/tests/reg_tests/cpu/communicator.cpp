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

#include <vector>

#include "base.hpp"

void check_allreduce_on_comm(const ccl::communicator& comm) {
    const int count = 1000;

    std::vector<int> send_buf(count);
    std::vector<int> recv_buf(count, 0);
    std::vector<int> expected_buf(count);

    for (size_t i = 0; i < count; i++) {
        send_buf.at(i) = i;
        expected_buf.at(i) = i * comm.size();
    }

    auto req = ccl::allreduce(send_buf.data(), recv_buf.data(), count, ccl::reduction::sum, comm);
    req.wait();

    for (size_t i = 0; i < count; i++) {
        if (recv_buf.at(i) != expected_buf.at(i)) {
            printf("FAILED (allreduce test)\n");
            throw std::runtime_error("recv_buf[" + std::to_string(i) +
                                     "]= " + std::to_string(recv_buf.at(i)) + ", expected " +
                                     std::to_string(expected_buf.at(i)));
        }
    }
}

void check_max_comm_number(const ccl::communicator& comm,
                           std::shared_ptr<ccl::kvs> kvs_instance,
                           int mpi_size,
                           int mpi_rank) {
    size_t user_comms = 0;
    std::vector<ccl::communicator> communicators;

    do {
        try {
            auto new_comm = ccl::create_communicator(mpi_size, mpi_rank, kvs_instance);
            PRINT_BY_ROOT(comm, "created %zu comm ", user_comms);
            ++user_comms;
            check_allreduce_on_comm(new_comm);
            communicators.push_back(std::move(new_comm));
        }
        catch (...) {
            break;
        }
    } while (true);

    PRINT_BY_ROOT(comm, "created %zu communicators", user_comms);
}

bool isPowerOfTwo(unsigned int x) {
    return x && !(x & (x - 1));
}

void check_comm_split_by_color(ccl::communicator& comm) {
    if (!isPowerOfTwo(comm.size())) {
        PRINT_BY_ROOT(
            comm,
            "split comm by color: number of processes should be a power of 2 for test purpose");
        return;
    }

    for (int split_by = 2; split_by <= comm.size(); split_by *= 2) {
        int color = comm.rank() % split_by;
        auto attr = ccl::preview::create_comm_split_attr(
            ccl::attr_val<ccl::comm_split_attr_id::color>(color));
        auto new_comm = comm.split(color, 0);

        int comm_size = comm.size();
        int new_comm_size = new_comm.size();
        int comm_rank = comm.rank();
        int new_comm_rank = new_comm.rank();

        int expected_new_comm_size = comm_size / split_by;

        if (new_comm_size != expected_new_comm_size) {
            printf("FAILED (split)\n");

            throw std::runtime_error("mismatch in size, expected " +
                                     std::to_string(expected_new_comm_size) + " received " +
                                     std::to_string(new_comm_size));
        }

        PRINT_BY_ROOT(comm,
                      "base comm: rank = %d, size = %d; "
                      "new comm: rank = %d, size = %d",
                      comm_rank,
                      comm_size,
                      new_comm_rank,
                      new_comm_size);

        PRINT_BY_ROOT(comm, " - allreduce test on a new communicator");
        check_allreduce_on_comm(new_comm);
    }
}

void check_comm_split_identical(ccl::communicator& comm) {
    if (!isPowerOfTwo(comm.size())) {
        PRINT_BY_ROOT(
            comm,
            "split comm by color: number of processes should be a power of 2 for test purpose");
        return;
    }

    for (int split_by = 2; split_by <= comm.size(); split_by *= 2) {
        int color = comm.rank() % split_by;
        auto attr = ccl::preview::create_comm_split_attr(
            ccl::attr_val<ccl::comm_split_attr_id::color>(color));
        auto new_comm1 = comm.split(color, 0);
        auto new_comm2 = comm.split(color, 0);

        if (new_comm1.size() != new_comm2.size()) {
            printf("FAILED (split)\n");

            throw std::runtime_error("the sizes of new communicators are not equal. Comm #1 size " +
                                     std::to_string(new_comm1.size()) + " Comm #2 size " +
                                     std::to_string(new_comm2.size()));
        }

        if (new_comm1.rank() != new_comm2.rank()) {
            printf("FAILED (split)\n");

            throw std::runtime_error("the sizes of new communicators are not equal. Comm #1 rank " +
                                     std::to_string(new_comm1.rank()) + " Comm #2 rank " +
                                     std::to_string(new_comm2.rank()));
        }

        PRINT_BY_ROOT(comm,
                      "comm #1: rank = %d, size = %d; "
                      "comm #2: rank = %d, size = %d",
                      new_comm1.rank(),
                      new_comm1.size(),
                      new_comm2.rank(),
                      new_comm2.size());
    }
}

void check_comm_split_identical_color(ccl::communicator& comm) {
    auto attr =
        ccl::preview::create_comm_split_attr(ccl::attr_val<ccl::comm_split_attr_id::color>(123));
    auto new_comm = comm.split(123, 0);

    if (new_comm.size() != comm.size()) {
        printf("FAILED (split)\n");

        throw std::runtime_error(
            "the sizes of new communicator and base communicator are not equal. New comm size " +
            std::to_string(new_comm.size()) + " Base comm size " + std::to_string(comm.size()));
    }

    if (new_comm.rank() != comm.rank()) {
        printf("FAILED (split)\n");

        throw std::runtime_error(
            "the sizes of new communicator and base communicator are not equal. New comm rank " +
            std::to_string(new_comm.rank()) + " Base comm rank " + std::to_string(comm.rank()));
    }

    PRINT_BY_ROOT(comm,
                  "base comm: rank = %d, size = %d; "
                  "new comm: rank = %d, size = %d",
                  comm.rank(),
                  new_comm.size(),
                  comm.rank(),
                  new_comm.size());

    PRINT_BY_ROOT(comm, " - allreduce test on a new communicator");
    check_allreduce_on_comm(new_comm);
}

int main() {
    ccl::init();

    int mpi_size, mpi_rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    atexit(mpi_finalize);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (mpi_rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(mpi_size, mpi_rank, kvs);

    PRINT_BY_ROOT(comm, "\n- Basic communicator allreduce test");
    check_allreduce_on_comm(comm);
    PRINT_BY_ROOT(comm, "PASSED");

    // PRINT_BY_ROOT(comm, "\n- Create max number of communicators");
    // check_max_comm_number(comm, kvs, mpi_size, mpi_rank);
    // PRINT_BY_ROOT(comm, "PASSED");

    PRINT_BY_ROOT(comm, "\n- Communicator split test");
    check_comm_split_by_color(comm);
    PRINT_BY_ROOT(comm, "PASSED");

    PRINT_BY_ROOT(comm, "\n- Communicator identical split test");
    check_comm_split_identical(comm);
    PRINT_BY_ROOT(comm, "PASSED");

    PRINT_BY_ROOT(comm, "\n- Communicator identical color split test");
    check_comm_split_identical_color(comm);
    PRINT_BY_ROOT(comm, "PASSED");

    return 0;
}

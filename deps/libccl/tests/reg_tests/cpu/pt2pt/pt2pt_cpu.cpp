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

#include <unordered_map>

#include "base.hpp"
#include "oneapi/ccl.hpp"

struct pt2pt_test_args {
    int argc;
    char** argv;
    uint32_t test_case;
    size_t count;
    size_t iter;
    uint32_t print_type;
    std::vector<int> peers;

    pt2pt_test_args(int argc, char* argv[])
            : argc(argc),
              argv(argv),
              test_case(1),
              count(10 * 1024 * 1024),
              iter(5),
              print_type(0) {
        // max peer number for all the test cases is 3
        peers.reserve(3);

        // filling out with the default values
        peers.push_back(0);
        peers.push_back(1);

        // this peer is intented for test_case 2
        // -1 value prevents us from interfering with test_case 1
        // when it comes to the buffer check
        peers.push_back(-1);
    }
    void print(int rank) const {
        if (rank == 0) {
            std::cout << "test_case: " << test_case << "\n"
                      << "count: " << count << "\n"
                      << "iter: " << iter << "\n"
                      << "peers: ";
            std::copy(peers.begin(), peers.end(), std::ostream_iterator<int>(std::cout, ", "));
            std::cout << "\nprint: " << print_type << "\n\n";
        }
    }
};

inline void parse_args(pt2pt_test_args& args) {
    std::unordered_map<std::string, std::string> args_map;
    // parse command line arguments
    for (int i = 1; i < args.argc; ++i) {
        std::string arg = args.argv[i];
        if (arg.substr(0, 2) == "--") {
            arg = arg.substr(2);
            if (i + 1 < args.argc && args.argv[i + 1][0] != '-') {
                args_map[arg] = args.argv[i + 1];
                ++i;
            }
            else {
                args_map[arg] = "";
            }
        }
    }

    if (args_map.find("test_case") != args_map.end()) {
        std::string serialize_str = args_map["test_case"];
        args.test_case = std::stoi(serialize_str);
    }

    if (args_map.find("count") != args_map.end()) {
        std::string count_str = args_map["count"];
        args.count = std::stoi(count_str);
    }

    if (args_map.find("iter") != args_map.end()) {
        std::string count_str = args_map["iter"];
        args.iter = std::stoi(count_str);
    }

    if (args_map.find("print") != args_map.end()) {
        std::string print_str = args_map["print"];
        args.print_type = std::stoi(print_str);
    }

    if (args_map.find("peers") != args_map.end()) {
        std::string peers_str = args_map["peers"];
        size_t pos = peers_str.find(",");
        if (pos == std::string::npos || pos == 0 || pos == peers_str.length() - 1) {
            std::cout << "This test requires a list of peer ranks divided by comma:" << std::endl;
            std::cout << "test_case 1 needs 2 ranks: --peers 0,1" << std::endl;
            std::cout << "test_case 2 needs 3 ranks: --peers 0,1,2" << std::endl;
            exit(0);
        }

        size_t i = 0;
        size_t start_pos = 0;
        while (pos != std::string::npos && pos != start_pos && i < 2) {
            args.peers[i] = std::stoi(peers_str.substr(start_pos, pos - start_pos));
            start_pos = pos + 1;
            pos = peers_str.find(",", start_pos);
            i++;
        }

        if (i == 1 && args.test_case == 2) {
            std::cout << "test_case 2 requires 3 peer ranks, not enough ranks were provided"
                      << std::endl;
            exit(0);
        }

        args.peers[i] = std::stoi(peers_str.substr(start_pos));
    }
    else if (args.test_case == 2) {
        args.peers[2] = 2; // default value
    }
}

int case_envs_2_ranks(const pt2pt_test_args& args) {
    int size = 0;
    int rank = 0;

    size_t count = args.count;
    size_t iter = args.iter;

    ccl::init();

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    if (args.print_type) {
        args.print(rank);
    }

    if (size < 2) {
        std::cout << "This test requires at least two ranks" << std::endl;
        exit(0);
    }

    /* create kvs */
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

    /* create communicator */
    auto comm = ccl::create_communicator(size, rank, kvs);

    /* create buffer */
    std::vector<int> buf(count);
    std::vector<int> check_buf(count);

    bool failed = false;

    for (size_t i = 0; i < iter; i++) {
        if (rank == args.peers[0]) {
            /* init the buffer */
            for (size_t id = 0; id < count; id++) {
                buf[id] = id + i;
            };

            ccl::send(buf.data(), count, ccl::datatype::int32, args.peers[1], comm).wait();
        }
        else if (rank == args.peers[1]) {
            /* init the buffer */
            for (size_t id = 0; id < count; id++) {
                buf[id] = -1;
            }

            ccl::recv(buf.data(), count, ccl::datatype::int32, args.peers[0], comm).wait();
        }

        ccl::barrier(comm);

        if (std::find(args.peers.begin(), args.peers.end(), rank) != args.peers.end()) {
            for (size_t id = 0; id < count; id++) {
                if (buf[id] != static_cast<int>(id + i)) {
                    check_buf[id] = -1;
                }
            }

            for (size_t j = 0; j < count; j++) {
                if (check_buf[j] == -1) {
                    failed = true;
                    break;
                }
            }
        }

        if (failed) {
            break;
        }
    }

    if (failed) {
        std::cout << "FAILED\n";
    }
    else {
        std::cout << "PASSED\n";
    }

    return 0;
}

int case_envs_3_ranks(const pt2pt_test_args& args) {
    int size = 0;
    int rank = 0;

    size_t count = args.count;
    size_t iter = args.iter;

    ccl::init();

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    if (args.print_type) {
        args.print(rank);
    }

    if (size < 3) {
        std::cout << "This test requires at least 3 ranks" << std::endl;
        exit(0);
    }

    /* create kvs */
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

    /* create communicator */
    auto comm = ccl::create_communicator(size, rank, kvs);

    // buffer for rank0 send -> rank1 recv
    std::vector<int> buf_send(count);
    std::vector<int> buf_recv(count);

    // buffer for rank1 send -> rank2 recv
    std::vector<int> buf_send2(count);
    std::vector<int> buf_recv2(count);

    std::vector<int> check_buf(count);
    std::vector<int> check_buf2(count);

    bool failed = false, failed2 = false;

    for (size_t i = 0; i < iter; i++) {
        // rank0 (send) -> rank1
        if (rank == args.peers[0]) {
            /* init the buffer */
            for (size_t id = 0; id < count; id++) {
                buf_send[id] = id + i;
            }

            ccl::send(buf_send.data(), count, ccl::datatype::int32, args.peers[1], comm).wait();
        }
        // rank0 -> rank1(recv)  rank1(send) -> rank2
        else if (rank == args.peers[1]) {
            /* init the buffers */
            for (size_t id = 0; id < count; id++) {
                buf_recv[id] = -1;
                buf_send2[id] = id + i;
            }

            // step 1: rank0 -> rank1(recv)
            ccl::recv(buf_recv.data(), count, ccl::datatype::int32, args.peers[0], comm).wait();

            // step 2: rank1(send) -> rank2
            ccl::send(buf_send2.data(), count, ccl::datatype::int32, args.peers[2], comm).wait();
        }

        // rank1 -> rank2 (recv)
        else if (rank == args.peers[2]) {
            /* init the buffer */
            for (size_t id = 0; id < count; id++) {
                buf_recv2[id] = -1;
            }

            ccl::recv(buf_recv2.data(), count, ccl::datatype::int32, args.peers[1], comm).wait();
        }

        if (std::find(args.peers.begin(), args.peers.end(), rank) != args.peers.end()) {
            for (size_t id = 0; id < count; id++) {
                if (buf_recv[id] != static_cast<int>(id + i)) {
                    check_buf[id] = -1;
                }
                if (buf_recv2[id] != static_cast<int>(id + i)) {
                    check_buf2[id] = -1;
                }
            }

            for (size_t j = 0; j < count; j++) {
                if (check_buf[j] == -1) {
                    failed = true;
                    break;
                }
                if (check_buf2[j] == -1) {
                    failed2 = true;
                    break;
                }
            }

            if (failed || failed2) {
                break;
            }
        }
    }

    if (rank == args.peers[1]) {
        // whether rank1 has received data from rank0?
        if (failed) {
            std::cout << "FAILED: rank: " << rank << " send-recv 0-1 FAILED\n";
        }
        if (!failed) {
            std::cout << "PASSED\n";
        }
    }

    if (rank == args.peers[2]) {
        // whether rank2 has received data from rank1?
        if (failed2) {
            std::cout << "FAILED: rank: " << rank << " send-recv 1-2 FAILED\n";
        }
        if (!failed2) {
            std::cout << "PASSED\n";
        }
    }

    return 0;
}

int case_envs_pt2pt_coll(const pt2pt_test_args& args) {
    int size = 0;
    int rank = 0;

    size_t count = args.count;
    size_t iter_num = args.iter;

    ccl::init();

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    if (args.print_type) {
        args.print(rank);
    }

    if (size < 4) {
        std::cout << "This test requires at least 4 ranks" << std::endl;
        exit(0);
    }

    // create kvs
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

    // create communicator
    auto comm = ccl::create_communicator(size, rank, kvs);

    std::vector<int> buf_send(count);
    std::vector<int> buf_recv(count);

    std::vector<int> buf_send1(count);
    std::vector<int> buf_recv1(count);

    std::vector<int> buf_send2(count);
    std::vector<int> buf_recv2(count);

    std::vector<int> check_buf(count);
    std::vector<int> check_buf1(count);
    std::vector<int> check_buf2(count);

    bool failed = false, failed1 = false, failed2 = false;

    // for allgatherv create buffers
    std::vector<int> coll_send_buf(count);
    std::vector<int> expected_buf(count * size);
    std::vector<int> coll_recv_buf(count * size);

    std::vector<size_t> recv_counts(size, count);

    // for allreduce
    std::vector<int> send_buf(count);
    std::vector<int> recv_buf(count);

    for (size_t iter_idx = 0; iter_idx < iter_num; iter_idx++) {
        // for allreduce begin
        for (size_t i = 0; i < count; i++) {
            send_buf[i] = rank + 1;
            recv_buf[i] = -1;
        }

        ccl::allreduce(send_buf.data(), recv_buf.data(), count, ccl::reduction::sum, comm).wait();

        for (size_t id = 0; id < count; id++) {
            if (recv_buf[id] != size * (size + 1) / 2) {
                recv_buf[id] = -1;
            }
        }

        // print out the result
        size_t i;
        for (i = 0; i < count; i++) {
            if (recv_buf[i] == -1) {
                std::cout << "FAILED (allreduce)\n";
                break;
            }
        }
        if (i == count) {
            std::cout << "PASSED (allreduce)\n";
        }
        // allreduce end

        // pt2pt begin
        // rank0 (send) -> rank1
        if (rank == 0) {
            /* init the buffer */
            for (size_t id = 0; id < count; id++) {
                buf_send[id] = id + iter_idx;
            }

            ccl::send(buf_send.data(), count, ccl::datatype::int32, 1, comm).wait();
        }
        // rank1 -> rank2(recv), rank1 <- rank0(send)
        else if (rank == 1) {
            /* init the buffer */
            for (size_t id = 0; id < count; id++) {
                buf_recv[id] = -1;
                buf_send1[id] = id + iter_idx;
            }

            ccl::recv(buf_recv.data(), count, ccl::datatype::int32, 0, comm).wait();

            ccl::send(buf_send1.data(), count, ccl::datatype::int32, 2, comm).wait();
        }
        // rank2 -> rank3 (recv), rank2 <- rank1(send)
        else if (rank == 2) {
            /* init the buffer */
            for (size_t id = 0; id < count; id++) {
                buf_recv1[id] = -1;
                buf_send2[id] = id + iter_idx;
            }

            ccl::recv(buf_recv1.data(), count, ccl::datatype::int32, 1, comm).wait();

            ccl::send(buf_send2.data(), count, ccl::datatype::int32, 3, comm).wait();
        }
        // rank2 -> rank3 (recv), rank3 <- rank2(send)
        else if (rank == 3) {
            /* init the buffer */
            for (size_t id = 0; id < count; id++) {
                buf_recv2[id] = -1;
            }

            ccl::recv(buf_recv2.data(), count, ccl::datatype::int32, 2, comm).wait();
        }

        ccl::barrier(comm);

        if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
            for (size_t id = 0; id < count; id++) {
                if (buf_recv[id] != static_cast<int>(id + iter_idx)) {
                    check_buf[id] = -1;
                }
                if (buf_recv1[id] != static_cast<int>(id + iter_idx)) {
                    check_buf1[id] = -1;
                }
                if (buf_recv2[id] != static_cast<int>(id + iter_idx)) {
                    check_buf2[id] = -1;
                }
            }

            for (size_t j = 0; j < count; j++) {
                if (check_buf[j] == -1) {
                    failed = true;
                    break;
                }
                if (check_buf1[j] == -1) {
                    failed1 = true;
                    break;
                }
                if (check_buf2[j] == -1) {
                    failed2 = true;
                    break;
                }
            }
        }
        else {
            for (size_t id = 0; id < count; id++) {
                check_buf[id] = 0;
                check_buf1[id] = 0;
                check_buf2[id] = 0;
            }
        }

        if (rank == 1) {
            if (failed) {
                std::cerr << "Rank " << rank << " test FAILED (rank0 -> rank1)" << std::endl;
            }
            else {
                std::cout << "Rank " << rank << " test PASSED (rank0 -> rank1)" << std::endl;
            }
        }
        else if (rank == 2) {
            if (failed1) {
                std::cerr << "Rank " << rank << " test FAILED (rank1 -> rank2)" << std::endl;
            }
            else {
                std::cout << "Rank " << rank << " test PASSED (rank1 -> rank2)" << std::endl;
            }
        }
        else if (rank == 3) {
            if (failed2) {
                std::cerr << "Rank " << rank << " test FAILED (rank2 -> rank3)" << std::endl;
            }
            else {
                std::cout << "Rank " << rank << " test PASSED (rank2 -> rank3)" << std::endl;
            }
        }
        // pt2pt end

        // allgatherv begin
        for (size_t j = 0; j < count; j++) {
            coll_send_buf[j] = rank + 1;
        }
        for (size_t j = 0; j < count * size; j++) {
            coll_recv_buf[j] = -1;
        }
        for (int j = 0; j < size; j++) {
            for (size_t k = 0; k < count; k++) {
                expected_buf[j * count + k] = j + 1;
            }
        }

        // invoke allgatherv
        ccl::allgatherv(coll_send_buf.data(), count, coll_recv_buf.data(), recv_counts, comm)
            .wait();

        // check recv_buf correctness
        for (size_t id = 0; id < size * count; id++) {
            if (coll_recv_buf[id] != expected_buf[id]) {
                coll_recv_buf[id] = -1;
            }
        }

        // print out the result
        size_t j;
        for (j = 0; j < size * count; j++) {
            if (coll_recv_buf[j] == -1) {
                std::cout << "FAILED (allgatherv)\n";
                break;
            }
        }
        if (j == size * count) {
            std::cout << "PASSED (allgatherv)\n";
        }
    }

    return 0;
}

int main(int argc, char* argv[]) {
    pt2pt_test_args args(argc, argv);
    parse_args(args);

    auto test_case = args.test_case;
    switch (test_case) {
        case 1: case_envs_2_ranks(args); break;
        case 2: case_envs_3_ranks(args); break;
        case 3: case_envs_pt2pt_coll(args); break;
        default: throw std::runtime_error("unexpected test case: " + std::to_string(test_case));
    }

    return 0;
}

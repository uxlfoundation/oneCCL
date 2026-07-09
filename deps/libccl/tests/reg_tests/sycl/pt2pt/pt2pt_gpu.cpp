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
#include "oneapi/ccl.hpp"
#include "sycl_base.hpp"

#include <csignal>
#include <cstdlib>

struct pt2pt_test_args {
    int argc;
    char** argv;
    uint32_t test_case;
    uint32_t cache_type;
    size_t count;
    size_t iter;
    uint32_t print_type;
    std::vector<int> peers;
    uint32_t serialize_type;
    uint32_t queue_type;
    uint32_t wait_type;

    pt2pt_test_args(int argc, char* argv[])
            : argc(argc),
              argv(argv),
              test_case(1),
              cache_type(0),
              count(10 * 1024 * 1024),
              iter(5),
              print_type(0),
              serialize_type(0),
              queue_type(0),
              wait_type(0) {
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
                      << "cache: " << cache_type << "\n"
                      << "count: " << count << "\n"
                      << "iter: " << iter << "\n"
                      << "peers: ";
            std::copy(peers.begin(), peers.end(), std::ostream_iterator<int>(std::cout, ", "));
            std::cout << "\nprint: " << print_type << "\n"
                      << "serialize: " << serialize_type << "\n"
                      << "queue_type: " << queue_type << "\n"
                      << "wait_type: " << wait_type << "\n\n";
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

    if (args_map.find("cache") != args_map.end()) {
        std::string cache_str = args_map["cache"];
        args.cache_type = std::stoi(cache_str);
    }

    if (args_map.find("count") != args_map.end()) {
        std::string count_str = args_map["count"];
        args.count = std::stoi(count_str);
    }

    if (args_map.find("iter") != args_map.end()) {
        std::string count_str = args_map["iter"];
        args.iter = std::stoi(count_str);
    }

    if (args_map.find("serialize") != args_map.end()) {
        std::string serialize_str = args_map["serialize"];
        args.serialize_type = std::stoi(serialize_str);
    }

    if (args_map.find("queue") != args_map.end()) {
        std::string type_str = args_map["queue"];
        args.queue_type = std::stoi(type_str);
    }
    if (args_map.find("wait") != args_map.end()) {
        std::string wait_str = args_map["wait"];
        args.wait_type = std::stoi(wait_str);
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

auto create_attr(const uint32_t is_cache, const int count, const std::string& match_id_suffix) {
    auto attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
    if (is_cache) {
        std::string matchId = "_len_" + std::to_string(count) + match_id_suffix;
        attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(matchId));
        attr.set<ccl::operation_attr_id::to_cache>(true);
    }
    return attr;
}

int case_envs_2_ranks(const pt2pt_test_args& args) {
    int size = 0;
    int rank = 0;

    size_t count = args.count;
    size_t iter = args.iter;
    uint32_t cache_type = args.cache_type;
    uint32_t serialize_type = args.serialize_type;
    uint32_t queue_type = args.queue_type;
    uint32_t wait_type = args.wait_type;

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

    sycl::property_list props{};
    if (queue_type) {
        props = { sycl::property::queue::in_order{} };
    }
    sycl::queue q;
    if (!create_sycl_queue("gpu", rank, q, props)) {
        return -1;
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
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* create buffer */
    auto buf_send = sycl::malloc_device<int>(count, q);
    auto buf_recv = sycl::malloc_device<int>(count, q);

    sycl::buffer<int> check_buf(count);
    sycl::buffer<int> check_buf1(count);
    std::vector<ccl::event> ccl_events;

    auto attr = create_attr(cache_type, count, std::to_string(0));
    auto attr1 = create_attr(cache_type, count, std::to_string(1));

    bool failed = false, failed1 = false;

    for (size_t i = 0; i < iter; i++) {
        if (rank == args.peers[0]) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send[id] = id + i;
                    if (serialize_type) {
                        buf_recv[id] = -1;
                    }
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto send_event =
                ccl::send(buf_send, count, ccl::datatype::int32, args.peers[1], comm, stream, attr);
            if (wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(send_event));

            if (serialize_type) {
                auto recv_event = ccl::recv(
                    buf_recv, count, ccl::datatype::int32, args.peers[1], comm, stream, attr1);
                if (wait_type) {
                    recv_event.wait();
                }
                ccl_events.emplace_back(std::move(recv_event));
            }
        }
        else if (rank == args.peers[1]) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    if (serialize_type) {
                        buf_send[id] = id + i;
                        buf_recv[id] = -1;
                    }
                    else {
                        buf_recv[id] = -1;
                        buf_send[id] = buf_recv[i];
                    }
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto buf = buf_recv;
            if (!serialize_type) {
                buf = buf_send;
            }

            auto recv_event =
                ccl::recv(buf, count, ccl::datatype::int32, args.peers[0], comm, stream, attr);
            if (wait_type) {
                recv_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));

            if (serialize_type) {
                auto send_event = ccl::send(
                    buf_send, count, ccl::datatype::int32, args.peers[0], comm, stream, attr1);
                if (wait_type) {
                    send_event.wait();
                }
                ccl_events.emplace_back(std::move(send_event));
            }
        }

        ccl::barrier(comm);

        if (std::find(args.peers.begin(), args.peers.end(), rank) != args.peers.end()) {
            auto e = q.submit([&](auto& h) {
                sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
                sycl::accessor check_buf_acc1(check_buf1, h, sycl::write_only);
                if (!queue_type && !wait_type) {
                    h.depends_on(ccl_events.back().get_native());
                }
                h.parallel_for(count, [=](auto id) {
                    if (buf_send[id] != static_cast<int>(id + i)) {
                        check_buf_acc[id] = -1;
                    }
                    if (serialize_type && (buf_recv[id] != static_cast<int>(id + i))) {
                        check_buf_acc1[id] = -1;
                    }
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            {
                sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
                sycl::host_accessor check_buf_acc1(check_buf1, sycl::read_only);
                for (size_t j = 0; j < count; j++) {
                    if (check_buf_acc[j] == -1) {
                        failed = true;
                        break;
                    }
                    if (serialize_type && (check_buf_acc1[j] == -1)) {
                        failed1 = true;
                        break;
                    }
                }
            }
        }

        if (failed || (failed1 && serialize_type)) {
            break;
        }
    }

    if (failed || (failed1 && serialize_type)) {
        std::cout << "FAILED\n";
    }
    else {
        std::cout << "PASSED\n";
    }

    sycl::free(buf_send, q);
    sycl::free(buf_recv, q);

    return 0;
}

int case_envs_3_ranks(const pt2pt_test_args& args) {
    int size = 0;
    int rank = 0;

    size_t count = args.count;
    size_t iter = args.iter;
    uint32_t wait_type = args.wait_type;
    uint32_t queue_type = args.queue_type;

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

    sycl::property_list props{};
    if (queue_type) {
        props = { sycl::property::queue::in_order{} };
    }

    sycl::queue q;
    if (!create_sycl_queue("gpu", rank, q, props)) {
        return -1;
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
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    // buffer for rank0 send -> rank1 recv
    auto buf_send = sycl::malloc_device<int>(count, q);
    auto buf_recv = sycl::malloc_device<int>(count, q);

    // buffer for rank1 send -> rank2 recv
    auto buf_send2 = sycl::malloc_device<int>(count, q);
    auto buf_recv2 = sycl::malloc_device<int>(count, q);

    sycl::buffer<int> check_buf(count);
    sycl::buffer<int> check_buf2(count);
    std::vector<ccl::event> ccl_events;

    auto attr = ccl::create_operation_attr<ccl::pt2pt_attr>();
    std::string match_id{};

    bool failed = false, failed2 = false;

    for (size_t i = 0; i < iter; i++) {
        // rank0 (send) -> rank1
        if (rank == args.peers[0]) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send[id] = id + i;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto send_event =
                ccl::send(buf_send, count, ccl::datatype::int32, args.peers[1], comm, stream, attr);
            if (wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(send_event));
        }
        // rank0 -> rank1(recv)  rank1(send) -> rank2
        else if (rank == args.peers[1]) {
            // step 1: rank0 -> rank1(recv)
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_recv[id] = -1;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto recv_event =
                ccl::recv(buf_recv, count, ccl::datatype::int32, args.peers[0], comm, stream, attr);
            if (wait_type) {
                recv_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));

            // step 2: rank1(send) -> rank2
            /* init the buffer */
            e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send2[id] = id + i;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto send_event = ccl::send(
                buf_send2, count, ccl::datatype::int32, args.peers[2], comm, stream, attr);
            if (wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(send_event));
        }

        // rank1 -> rank2 (recv)
        else if (rank == args.peers[2]) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_recv2[id] = -1;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto recv_event = ccl::recv(
                buf_recv2, count, ccl::datatype::int32, args.peers[1], comm, stream, attr);
            if (wait_type) {
                recv_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));
        }

        if (std::find(args.peers.begin(), args.peers.end(), rank) != args.peers.end()) {
            auto e = q.submit([&](auto& h) {
                sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
                sycl::accessor check_buf_acc2(check_buf2, h, sycl::write_only);
                if (!args.queue_type && !wait_type) {
                    h.depends_on(ccl_events.back().get_native());
                }
                h.parallel_for(count, [=](auto id) {
                    if (buf_recv[id] != static_cast<int>(id + i)) {
                        check_buf_acc[id] = -1;
                    }
                    if (buf_recv2[id] != static_cast<int>(id + i)) {
                        check_buf_acc2[id] = -1;
                    }
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            {
                sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
                sycl::host_accessor check_buf_acc2(check_buf2, sycl::read_only);
                for (size_t j = 0; j < count; j++) {
                    if (check_buf_acc[j] == -1) {
                        failed = true;
                        break;
                    }
                    if (check_buf_acc2[j] == -1) {
                        failed2 = true;
                        break;
                    }
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

    sycl::free(buf_send, q);
    sycl::free(buf_recv, q);

    sycl::free(buf_send2, q);
    sycl::free(buf_recv2, q);

    return 0;
}

int case_envs_8_ranks(const pt2pt_test_args& args) {
    int size = 0;
    int rank = 0;

    size_t count = args.count;
    size_t iter = args.iter;
    uint32_t wait_type = args.wait_type;

    ccl::init();

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    if (args.print_type) {
        args.print(rank);
    }

    if (size < 8) {
        std::cout << "This test requires at least 8 ranks" << std::endl;
        exit(0);
    }

    sycl::property_list props{};
    if (args.queue_type) {
        props = { sycl::property::queue::in_order{} };
    }
    sycl::queue q;
    if (!create_sycl_queue("gpu", rank, q)) {
        return -1;
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
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    // buffer for rank1 send -> rank3 recv
    auto buf_send = sycl::malloc_device<int>(count, q);
    auto buf_recv = sycl::malloc_device<int>(count, q);

    // buffer for rank3 send -> rank5 recv
    auto buf_send1 = sycl::malloc_device<int>(count, q);
    auto buf_recv1 = sycl::malloc_device<int>(count, q);

    // buffer for rank5 send -> rank7 recv
    auto buf_send2 = sycl::malloc_device<int>(count, q);
    auto buf_recv2 = sycl::malloc_device<int>(count, q);

    sycl::buffer<int> check_buf(count);
    sycl::buffer<int> check_buf1(count);
    sycl::buffer<int> check_buf2(count);
    std::vector<ccl::event> ccl_events;

    auto attr = create_attr(args.cache_type, count, std::to_string(0));
    auto attr1 = create_attr(args.cache_type, count, std::to_string(1));
    auto attr2 = create_attr(args.cache_type, count, std::to_string(2));

    bool failed = false, failed1 = false, failed2 = false;

    for (size_t i = 0; i < iter; i++) {
        // rank1 (send) -> rank3
        if (rank == 1) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send[id] = id + i;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto send_event =
                ccl::send(buf_send, count, ccl::datatype::int32, 3, comm, stream, attr);
            if (wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(send_event));
        }
        // rank1 -> rank3(recv)
        else if (rank == 3) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_recv[id] = -1;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto recv_event =
                ccl::recv(buf_recv, count, ccl::datatype::int32, 1, comm, stream, attr);
            if (wait_type) {
                recv_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));

            /* init the buffer */
            e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send1[id] = id + i;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto send_event =
                ccl::send(buf_send1, count, ccl::datatype::int32, 5, comm, stream, attr1);
            if (wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(send_event));
        }
        // rank3 -> rank5 (recv)
        else if (rank == 5) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_recv1[id] = -1;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto recv_event =
                ccl::recv(buf_recv1, count, ccl::datatype::int32, 3, comm, stream, attr1);
            if (wait_type) {
                recv_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));

            /* init the buffer */
            e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send2[id] = id + i;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto send_event =
                ccl::send(buf_send2, count, ccl::datatype::int32, 7, comm, stream, attr2);
            if (wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));
        }
        // rank15 -> rank7 (recv)
        else if (rank == 7) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_recv2[id] = -1;
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            auto recv_event =
                ccl::recv(buf_recv2, count, ccl::datatype::int32, 5, comm, stream, attr2);
            if (wait_type) {
                recv_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));
        }

        if (rank == 1 || rank == 3 || rank == 5 || rank == 7) {
            auto e = q.submit([&](auto& h) {
                sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
                sycl::accessor check_buf_acc2(check_buf1, h, sycl::write_only);
                sycl::accessor check_buf_acc3(check_buf2, h, sycl::write_only);
                if (!args.queue_type && !wait_type) {
                    h.depends_on(ccl_events.back().get_native());
                }
                h.parallel_for(count, [=](auto id) {
                    if (buf_recv[id] != static_cast<int>(id + i)) {
                        check_buf_acc[id] = -1;
                    }
                    if (buf_recv1[id] != static_cast<int>(id + i)) {
                        check_buf_acc2[id] = -1;
                    }
                    if (buf_recv2[id] != static_cast<int>(id + i)) {
                        check_buf_acc3[id] = -1;
                    }
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            {
                sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
                sycl::host_accessor check_buf_acc1(check_buf1, sycl::read_only);
                sycl::host_accessor check_buf_acc2(check_buf2, sycl::read_only);
                for (size_t j = 0; j < count; j++) {
                    if (check_buf_acc[j] == -1) {
                        failed = true;
                        break;
                    }
                    if (check_buf_acc1[j] == -1) {
                        failed1 = true;
                        break;
                    }
                    if (check_buf_acc2[j] == -1) {
                        failed2 = true;
                        break;
                    }
                }
            }
        }
        else {
            q.submit([&](auto& h) {
                sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
                sycl::accessor check_buf_acc1(check_buf1, h, sycl::write_only);
                sycl::accessor check_buf_acc2(check_buf2, h, sycl::write_only);
                if (!args.queue_type && !wait_type) {
                    h.depends_on(ccl_events.back().get_native());
                }
                h.parallel_for(count, [=](auto id) {
                    check_buf_acc[id] = 0;
                    check_buf_acc1[id] = 0;
                    check_buf_acc2[id] = 0;
                });
            });
        }
    }

    if (rank == 3) {
        if (failed) {
            std::cerr << "Rank " << rank << " test FAILED (rank1 -> rank3)" << std::endl;
        }
        else {
            std::cout << "Rank " << rank << " test PASSED (rank1 -> rank3)" << std::endl;
        }
    }
    else if (rank == 5) {
        if (failed1) {
            std::cerr << "Rank " << rank << " test FAILED (rank3 -> rank5)" << std::endl;
        }
        else {
            std::cout << "Rank " << rank << " test PASSED (rank3 -> rank5)" << std::endl;
        }
    }
    else if (rank == 7) {
        if (failed2) {
            std::cerr << "Rank " << rank << " test FAILED (rank5 -> rank7)" << std::endl;
        }
        else {
            std::cout << "Rank " << rank << " test PASSED (rank5 -> rank7)" << std::endl;
        }
    }

    sycl::free(buf_send, q);
    sycl::free(buf_recv, q);
    sycl::free(buf_send1, q);
    sycl::free(buf_recv1, q);
    sycl::free(buf_send2, q);
    sycl::free(buf_recv2, q);

    return 0;
}

int case_envs_pt2pt_coll(const pt2pt_test_args& args) {
    int size = 0;
    int rank = 0;

    size_t count = args.count;
    size_t iter_num = args.iter;
    uint32_t wait_type = args.wait_type;

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

    sycl::property_list props{};
    if (args.queue_type) {
        props = { sycl::property::queue::in_order{} };
    }
    sycl::queue q;
    if (!create_sycl_queue("gpu", rank, q)) {
        return -1;
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
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    // create stream
    auto stream = ccl::create_stream(q);

    auto buf_send = sycl::malloc_device<int>(count, q);
    auto buf_recv = sycl::malloc_device<int>(count, q);

    auto buf_send1 = sycl::malloc_device<int>(count, q);
    auto buf_recv1 = sycl::malloc_device<int>(count, q);

    auto buf_send2 = sycl::malloc_device<int>(count, q);
    auto buf_recv2 = sycl::malloc_device<int>(count, q);

    sycl::buffer<int> check_buf(count);
    sycl::buffer<int> check_buf1(count);
    sycl::buffer<int> check_buf2(count);
    std::vector<ccl::event> ccl_events;

    auto attr = create_attr(args.cache_type, count, std::to_string(0));
    auto attr1 = create_attr(args.cache_type, count, std::to_string(1));
    auto attr2 = create_attr(args.cache_type, count, std::to_string(2));

    bool failed = false, failed1 = false, failed2 = false;

    // for allgatherv create buffers
    sycl::buffer<int> coll_send_buf(count);
    sycl::buffer<int> expected_buf(count * size);
    sycl::buffer<int> coll_recv_buf(count * size);

    std::vector<size_t> recv_counts(size, count);

    // for allreduce
    sycl::buffer<int> send_buf(count);
    sycl::buffer<int> recv_buf(count);

    for (size_t iter_idx = 0; iter_idx < iter_num; iter_idx++) {
        // for allreduce begin
        {
            // open buffers and initialize them on the host side
            sycl::host_accessor send_buf_acc(send_buf, sycl::write_only);
            sycl::host_accessor recv_buf_acc(recv_buf, sycl::write_only);
            for (size_t i = 0; i < count; i++) {
                send_buf_acc[i] = rank;
                recv_buf_acc[i] = -1;
            }
        }

        // open send_buf and modify it on the device side
        q.submit([&](auto& h) {
            sycl::accessor send_buf_acc(send_buf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                send_buf_acc[id] += 1;
            });
        });

        // invoke allreduce
        ccl::allreduce(send_buf, recv_buf, count, ccl::reduction::sum, comm, stream).wait();

        q.submit([&](auto& h) {
            sycl::accessor recv_buf_acc(recv_buf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                if (recv_buf_acc[id] != size * (size + 1) / 2) {
                    recv_buf_acc[id] = -1;
                }
            });
        });

        if (!handle_exception(q))
            return -1;

        // print out the result of the test on the host side
        {
            sycl::host_accessor recv_buf_acc(recv_buf, sycl::read_only);
            size_t i;
            for (i = 0; i < count; i++) {
                if (recv_buf_acc[i] == -1) {
                    std::cout << "FAILED (allreduce)\n";
                    break;
                }
            }
            if (i == count) {
                std::cout << "PASSED (allreduce)\n";
            }
        }
        // allreduce end

        // pt2pt begin
        // rank0 (send) -> rank1
        if (rank == 0) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send[id] = id + iter_idx;
                });
            });

            if (args.wait_type) {
                e.wait_and_throw();
            }

            auto send_event =
                ccl::send(buf_send, count, ccl::datatype::int32, 1, comm, stream, attr);
            if (args.wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(send_event));
        }
        // rank1 -> rank2(recv), rank1 <- rank0(send)
        else if (rank == 1) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_recv[id] = -1;
                });
            });

            if (args.wait_type) {
                e.wait_and_throw();
            }

            auto recv_event =
                ccl::recv(buf_recv, count, ccl::datatype::int32, 0, comm, stream, attr);
            if (args.wait_type) {
                recv_event.wait();
            }

            ccl_events.emplace_back(std::move(recv_event));

            /* init the buffer */
            e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send1[id] = id + iter_idx;
                });
            });

            if (args.wait_type) {
                e.wait_and_throw();
            }

            auto send_event =
                ccl::send(buf_send1, count, ccl::datatype::int32, 2, comm, stream, attr1);
            if (args.wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(send_event));
        }
        // rank1 -> rank2 (recv), rank2 <- rank1(send)
        else if (rank == 2) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_recv1[id] = -1;
                });
            });

            if (args.wait_type) {
                e.wait_and_throw();
            }

            auto recv_event =
                ccl::recv(buf_recv1, count, ccl::datatype::int32, 1, comm, stream, attr1);
            if (args.wait_type) {
                recv_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));

            /* init the buffer */
            e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_send2[id] = id + iter_idx;
                });
            });

            if (args.wait_type) {
                e.wait_and_throw();
            }

            auto send_event =
                ccl::send(buf_send2, count, ccl::datatype::int32, 3, comm, stream, attr2);
            if (args.wait_type) {
                send_event.wait();
            }
            ccl_events.emplace_back(std::move(send_event));
        }
        // rank2 -> rank3 (recv), rank3 <- rank2(send)
        else if (rank == 3) {
            /* init the buffer */
            auto e = q.submit([&](auto& h) {
                h.parallel_for(count, [=](auto id) {
                    buf_recv2[id] = -1;
                });
            });

            if (args.wait_type) {
                e.wait_and_throw();
            }

            auto recv_event =
                ccl::recv(buf_recv2, count, ccl::datatype::int32, 2, comm, stream, attr2);
            if (args.wait_type) {
                recv_event.wait();
            }
            ccl_events.emplace_back(std::move(recv_event));
        }

        ccl::barrier(comm);

        if (rank == 0 || rank == 1 || rank == 2 || rank == 3) {
            auto e = q.submit([&](auto& h) {
                sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
                sycl::accessor check_buf_acc2(check_buf1, h, sycl::write_only);
                sycl::accessor check_buf_acc3(check_buf2, h, sycl::write_only);
                if (!args.queue_type && !wait_type) {
                    h.depends_on(ccl_events.back().get_native());
                }
                h.parallel_for(count, [=](auto id) {
                    if (buf_recv[id] != static_cast<int>(id + iter_idx)) {
                        check_buf_acc[id] = -1;
                    }
                    if (buf_recv1[id] != static_cast<int>(id + iter_idx)) {
                        check_buf_acc2[id] = -1;
                    }
                    if (buf_recv2[id] != static_cast<int>(id + iter_idx)) {
                        check_buf_acc3[id] = -1;
                    }
                });
            });

            if (wait_type) {
                e.wait_and_throw();
            }

            {
                sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
                sycl::host_accessor check_buf_acc1(check_buf1, sycl::read_only);
                sycl::host_accessor check_buf_acc2(check_buf2, sycl::read_only);
                for (size_t j = 0; j < count; j++) {
                    if (check_buf_acc[j] == -1) {
                        failed = true;
                        break;
                    }
                    if (check_buf_acc1[j] == -1) {
                        failed1 = true;
                        break;
                    }
                    if (check_buf_acc2[j] == -1) {
                        failed2 = true;
                        break;
                    }
                }
            }
        }
        else {
            q.submit([&](auto& h) {
                sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
                sycl::accessor check_buf_acc1(check_buf1, h, sycl::write_only);
                sycl::accessor check_buf_acc2(check_buf2, h, sycl::write_only);
                if (!args.queue_type && !wait_type) {
                    h.depends_on(ccl_events.back().get_native());
                }
                h.parallel_for(count, [=](auto id) {
                    check_buf_acc[id] = 0;
                    check_buf_acc1[id] = 0;
                    check_buf_acc2[id] = 0;
                });
            });
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
        {
            /* open buffers and initialize them on the host side */
            sycl::host_accessor send_buf_acc(coll_send_buf, sycl::write_only);
            sycl::host_accessor recv_buf_acc(coll_recv_buf, sycl::write_only);
            sycl::host_accessor expected_acc_buf(expected_buf, sycl::write_only);

            for (size_t i = 0; i < count; i++) {
                send_buf_acc[i] = rank;
            }
            for (size_t i = 0; i < count * size; i++) {
                recv_buf_acc[i] = -1;
            }
            for (int i = 0; i < size; i++) {
                for (size_t j = 0; j < count; j++) {
                    expected_acc_buf[i * count + j] = i + 1;
                }
            }
        }

        /* open send_buf and modify it on the device side */
        q.submit([&](auto& h) {
            sycl::accessor send_buf_acc(coll_send_buf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                send_buf_acc[id] += 1;
            });
        });

        if (!handle_exception(q))
            return -1;

        // invoke allgatherv
        ccl::allgatherv(coll_send_buf, count, coll_recv_buf, recv_counts, comm, stream).wait();

        // open recv_buf and check its correctness on the device side
        q.submit([&](auto& h) {
            sycl::accessor recv_buf_acc(coll_recv_buf, h, sycl::write_only);
            sycl::accessor expected_buf_acc(expected_buf, h, sycl::read_only);
            h.parallel_for(size * count, [=](auto id) {
                if (recv_buf_acc[id] != expected_buf_acc[id]) {
                    recv_buf_acc[id] = -1;
                }
            });
        });

        if (!handle_exception(q))
            return -1;

        // print out the result of the test on the host side
        {
            sycl::host_accessor recv_buf_acc(coll_recv_buf, sycl::read_only);
            size_t j;
            for (j = 0; j < size * count; j++) {
                if (recv_buf_acc[j] == -1) {
                    std::cout << "FAILED (allgatherv)\n";
                    break;
                }
            }
            if (j == size * count) {
                std::cout << "PASSED (allgatherv)\n";
            }
        }
    }

    sycl::free(buf_send, q);
    sycl::free(buf_recv, q);
    sycl::free(buf_send1, q);
    sycl::free(buf_recv1, q);
    sycl::free(buf_send2, q);
    sycl::free(buf_recv2, q);

    return 0;
}

int case_scatter_envs(const pt2pt_test_args& args) {
    constexpr int ROOT = 0;
    int world_size = 0, rank = 0;

    size_t COUNT_PER_RANK = args.count;
    uint32_t wait_type = args.wait_type;

    ccl::init();

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    if (args.print_type) {
        args.print(rank);
    }

    if (world_size < 2) {
        std::cout << "This test requires at least 4 ranks" << std::endl;
        exit(0);
    }

    sycl::property_list props{};
    if (args.queue_type) {
        props = { sycl::property::queue::in_order{} };
    }
    sycl::queue q;
    if (!create_sycl_queue("gpu", rank, q, props)) {
        return -1;
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
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(world_size, rank, dev, ctx, kvs);
    auto stream = ccl::create_stream(q);

    int* send_buf = nullptr;
    if (rank == ROOT) {
        send_buf = sycl::malloc_device<int>(COUNT_PER_RANK * world_size, q);
    }
    int* recv_buf = sycl::malloc_device<int>(COUNT_PER_RANK, q);

    if (rank == ROOT) {
        q.parallel_for(sycl::range<1>(COUNT_PER_RANK * world_size), [=](sycl::id<1> idx) {
             send_buf[idx] = static_cast<int>(idx);
         }).wait();
    }

    if (rank == ROOT) {
        std::vector<int> host_send(COUNT_PER_RANK * world_size);
        q.memcpy(host_send.data(), send_buf, host_send.size() * sizeof(int)).wait();

        // std::cout << "[root] slice for root (rank 0):\n";
        for (std::size_t i = 0; i < COUNT_PER_RANK; ++i) {
            int v = host_send[ROOT * COUNT_PER_RANK + i];
            // std::cout << "  send_buf[" << i << "] = " << v << "\n";
        }

        /* copy local slice for the root directly */
        q.memcpy(recv_buf, send_buf + ROOT * COUNT_PER_RANK, COUNT_PER_RANK * sizeof(int)).wait();

        /* send everybody else their slice */
        for (int r = 0; r < world_size; ++r) {
            if (r == ROOT)
                continue;
            auto send_event = ccl::send(send_buf + r * COUNT_PER_RANK,
                                        COUNT_PER_RANK,
                                        ccl::datatype::int32,
                                        r,
                                        comm,
                                        stream);
            if (wait_type) {
                send_event.wait();
            }
        }
    }
    else {
        /* non-root ranks: receive your chunk */
        auto recv_event =
            ccl::recv(recv_buf, COUNT_PER_RANK, ccl::datatype::int32, ROOT, comm, stream);
        if (wait_type) {
            recv_event.wait();
        }
    }

    bool failed = false;
    {
        std::vector<int> host_recv(COUNT_PER_RANK);
        q.memcpy(host_recv.data(), recv_buf, COUNT_PER_RANK * sizeof(int)).wait();

        for (std::size_t i = 0; i < COUNT_PER_RANK; ++i) {
            int expected = static_cast<int>(rank * COUNT_PER_RANK + i);
            if (host_recv[i] != expected) {
                failed = true;
                std::cout << "Rank " << rank << " mismatch at index " << i << " expected "
                          << expected << " got " << host_recv[i] << std::endl;
                break;
            }
        }
    }

    if (failed) {
        std::cout << "FAILED\n";
    }
    else {
        std::cout << "PASSED\n";
    }

    if (send_buf) {
        sycl::free(send_buf, q);
    }
    sycl::free(recv_buf, q);

    return 0;
}

void signal_handler(int signal) {
    std::cerr << "FAILED: Caught signal " << signal << " (possible segmentation fault)\n";
    std::exit(EXIT_FAILURE);
}

int case_envs_pt2pt_without_init(const pt2pt_test_args& args) {
    std::signal(SIGSEGV, signal_handler);
    std::signal(SIGABRT, signal_handler);

    try {
        ccl::group_start();
        ccl::group_end();
        std::cout << "PASSED \n";
    }
    catch (const std::exception& e) {
        std::cout << "FAILED: Caught std::exception: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cout << "FAILED: Caught unknown exception\n";
        return EXIT_FAILURE;
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
        case 3:
            case_envs_8_ranks(args); // TODO: include in testing, when sclale-out is enabled
            break;
        case 4: case_envs_pt2pt_coll(args); break;
        case 5: case_envs_pt2pt_without_init(args); break;
        case 6: case_scatter_envs(args); break;
        default: throw std::runtime_error("unexpected test case: " + std::to_string(test_case));
    }

    return 0;
}

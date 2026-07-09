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

#include <algorithm>
#include <cstdlib>
#include <ctime>

#include "oneapi/ccl.hpp"
#include "store.hpp"

#define ELEM_COUNT        (256 * 1024)
#define ITER_COUNT        10
#define REINIT_COUNT      10
#define STORE_TIMEOUT_SEC 120
#define MAX_SLEEP_MSEC    500

#define KVS_BASE_PORT         10000
#define KVS_IP_PORT_DELIMETER "_"

#define KVS_CREATE_SUCCESS 0
#define KVS_CREATE_FAILURE -1

enum class kvs_mode { store, ip_port };

std::map<kvs_mode, std::string> kvs_mode_names = { { kvs_mode::store, "store" },
                                                   { kvs_mode::ip_port, "ip_port" } };

void run_collective(const char* cmd_name,
                    const std::vector<float>& send_buf,
                    std::vector<float>& recv_buf,
                    const ccl::communicator& comm) {
    std::chrono::system_clock::duration exec_time{ 0 };
    float expected = (comm.size() - 1) * (static_cast<float>(comm.size()) / 2);

    ccl::barrier(comm);

    for (size_t idx = 0; idx < ITER_COUNT; ++idx) {
        auto start = std::chrono::system_clock::now();
        ccl::allreduce(send_buf.data(), recv_buf.data(), recv_buf.size(), ccl::reduction::sum, comm)
            .wait();
        exec_time += std::chrono::system_clock::now() - start;
    }

    for (size_t idx = 0; idx < recv_buf.size(); idx++) {
        if (recv_buf[idx] != expected) {
            fprintf(stderr, "idx %zu, expected %4.4f, got %4.4f", idx, expected, recv_buf[idx]);

            std::cout << "FAILED" << std::endl;
            std::terminate();
        }
    }

    ccl::barrier(comm);

    std::cout << "avg time of " << cmd_name << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count() /
                     ITER_COUNT
              << " ms" << std::endl;
}

void print_help() {
    std::cout << "specify: [size] [rank] [kvs_mode] [kvs_param]" << std::endl;
}

int create_kvs_by_store(std::shared_ptr<file_store> store,
                        int rank,
                        ccl::shared_ptr_class<ccl::kvs>& kvs) {
    std::chrono::system_clock::duration exec_time{ 0 };
    auto start = std::chrono::system_clock::now();
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        exec_time = std::chrono::system_clock::now() - start;
        std::cout << "main kvs create time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count()
                  << " ms" << std::endl;

        start = std::chrono::system_clock::now();
        if (store->write((void*)main_addr.data(), main_addr.size()) < 0) {
            printf("error occurred during write attempt\n");
            kvs.reset();
            return KVS_CREATE_FAILURE;
        }
        exec_time = std::chrono::system_clock::now() - start;
        std::cout << "write to store time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count()
                  << " ms" << std::endl;
    }
    else {
        if (store->read((void*)main_addr.data(), main_addr.size()) < 0) {
            printf("error occurred during read attempt\n");
            kvs.reset();
            return KVS_CREATE_FAILURE;
        }
        exec_time = std::chrono::system_clock::now() - start;
        std::cout << "read from store time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count()
                  << " ms" << std::endl;

        start = std::chrono::system_clock::now();
        kvs = ccl::create_kvs(main_addr);
        exec_time = std::chrono::system_clock::now() - start;
        std::cout << "kvs create time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count()
                  << " ms" << std::endl;
    }
    return KVS_CREATE_SUCCESS;
}

int create_kvs_by_attr(ccl::kvs_attr attr, ccl::shared_ptr_class<ccl::kvs>& kvs) {
    std::chrono::system_clock::duration exec_time{ 0 };
    auto start = std::chrono::system_clock::now();
    kvs = ccl::create_main_kvs(attr);
    exec_time = std::chrono::system_clock::now() - start;
    std::cout << "main kvs create time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count() << " ms"
              << std::endl;
    return KVS_CREATE_SUCCESS;
}

int main(int argc, char** argv) {
    int ret = 0;
    int size, rank;
    kvs_mode mode;
    char* kvs_param;
    std::shared_ptr<file_store> store;

    if (argc == 5) {
        size = std::atoi(argv[1]);
        rank = std::atoi(argv[2]);

        bool found_kvs_mode = false;
        std::string kvs_mode_str(argv[3]);
        std::transform(kvs_mode_str.begin(), kvs_mode_str.end(), kvs_mode_str.begin(), ::tolower);
        for (const auto& pair : kvs_mode_names) {
            if (!kvs_mode_str.compare(pair.second)) {
                mode = pair.first;
                found_kvs_mode = true;
                break;
            }
        }

        if (!found_kvs_mode) {
            std::vector<std::string> values;
            std::transform(kvs_mode_names.begin(),
                           kvs_mode_names.end(),
                           std::back_inserter(values),
                           [](const typename std::map<kvs_mode, std::string>::value_type& pair) {
                               return pair.second;
                           });

            std::string expected_values;
            for (size_t idx = 0; idx < values.size(); idx++) {
                expected_values += values[idx];
                if (idx != values.size() - 1)
                    expected_values += ", ";
            }

            std::cout << "unexpected kvs mode: " << kvs_mode_str
                      << ", expected values: " << expected_values;
            return -1;
        }

        kvs_param = argv[4];

        std::cout << "args: "
                  << "size = " << size << ", rank = " << rank
                  << ", kvs mode = " << kvs_mode_names[mode] << ", kvs param = " << kvs_param
                  << std::endl;
    }
    else {
        print_help();
        return -1;
    }

    ccl::init();

    for (int i = 0; i < REINIT_COUNT; ++i) {
        std::cout << "========== started iter " << i << " ==========" << std::endl;

        ccl::shared_ptr_class<ccl::kvs> kvs;

        if (mode == kvs_mode::store) {
            store = std::make_shared<file_store>(
                kvs_param, rank, std::chrono::seconds(STORE_TIMEOUT_SEC));
            if (create_kvs_by_store(store, rank, kvs) != KVS_CREATE_SUCCESS) {
                std::cout << "can not create kvs by store" << std::endl;
                return -1;
            }
        }
        else if (mode == kvs_mode::ip_port) {
            std::string ip_port(std::string(kvs_param) + KVS_IP_PORT_DELIMETER +
                                std::to_string(KVS_BASE_PORT + i));
            auto attr = ccl::create_kvs_attr();
            attr.set<ccl::kvs_attr_id::ip_port>(ccl::string_class(ip_port));
            if (create_kvs_by_attr(attr, kvs) != KVS_CREATE_SUCCESS) {
                std::cout << "can not create kvs by attr" << std::endl;
                return -1;
            }
        }
        else {
            std::cout << "unexpected kvs mode" << std::endl;
            return -1;
        }

        std::chrono::system_clock::duration exec_time{ 0 };
        auto start = std::chrono::system_clock::now();
        auto comm = ccl::create_communicator(size, rank, kvs);
        exec_time = std::chrono::system_clock::now() - start;
        std::cout << "communicator create time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count()
                  << " ms" << std::endl;

        start = std::chrono::system_clock::now();
        std::vector<float> send_buf(ELEM_COUNT, static_cast<float>(comm.rank()));
        std::vector<float> recv_buf(ELEM_COUNT);
        run_collective("allreduce", send_buf, recv_buf, comm);
        exec_time = std::chrono::system_clock::now() - start;
        std::cout << "total collective time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(exec_time).count()
                  << " ms" << std::endl;

        ccl::barrier(comm);

        if (rank == 0)
            store.reset();

        ccl::barrier(comm);

        /* imitate non-simultaneous destruction of KVS */
        int slow_rank = i % size;
        int sleep_ms = (rank == slow_rank) ? MAX_SLEEP_MSEC : 0;
        std::cout << "sleep for " << sleep_ms << " ms" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));

        std::cout << "========== completed iter " << i << " ==========" << std::endl << std::endl;
    }

    std::cout << "PASSED" << std::endl;

    return ret;
}

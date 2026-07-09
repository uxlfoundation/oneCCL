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

#include "occp_server.h"
#include "occp_client.h"

#include "utils.h" // for VERIFY, LOG_HCL_ERR

#include <random>
#include <thread>
#include <chrono>

static inline uint64_t get_thread_id() {
    // return thread id as uint64_t
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

static inline void busy(int max_seconds = 10) {
    // Pause for a random duration between 1 and max_seconds
    // This is just to simulate some delay in the main thread

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dis(1, max_seconds);

    SLEEP_FOR(dis(gen) * 1000); // Convert seconds to milliseconds
}

void comm_init_rank(uint32_t comm_size, rank_t rank, const sockaddr_t& srv_addr) {
    OCCL_INFO("Initializing communication for rank ", rank);
    occp_client_t client(rank, srv_addr);

    ranks_infos_t comm_info(comm_size); // <-- memory for recv
    rank_info_t my_info("rank_host_" + std::to_string(rank));

    if (!client.exchange_rank_info(rank, comm_size, my_info, comm_info)) {
        VERIFY(false, "Failed to exchange rank info");
        return;
    }

    for (const auto& info : comm_info) {
        OCCL_OUT("Received rank info for ", info.hostname);
    }

    ofi_endpoints_t ofi_configs(comm_size);
    ofi_endpoint_t my_config; // Example EFI configuration, can be customized

    if (!client.exchange_ofi_config(rank, comm_size, my_config, ofi_configs)) {
        VERIFY(false, "Failed to exchange EFI config");
        return;
    }

    // FOR_I(comm_size)
    // {
    //     std::cout << ofi_configs[i].size << " bytes of EFI config for rank " << i << std::endl;
    // }

    busy(); // Simulate some processing delay

    if (!client.rendezvous(rank, comm_size)) {
        VERIFY(false, "Failed to rendezvous with server");
        return;
    }

    if (!client.exchange_rank_info(rank, comm_size, my_info, comm_info)) {
        VERIFY(false, "Failed to exchange rank info");
        return;
    }

    for (const auto& info : comm_info) {
        OCCL_OUT("Received rank info for ", info.hostname);
    }

    if (!client.exchange_ofi_config(rank, comm_size, my_config, ofi_configs)) {
        VERIFY(false, "Failed to exchange EFI config");
        return;
    }

    if (!client.rendezvous(rank, comm_size)) {
        VERIFY(false, "Failed to rendezvous with server");
        return;
    }

    OCCL_OUT("COMM INIT FINISHED.", rank);
}

int start_server(const char* addr_str, uint16_t port) {
    // Create and start the server
    occp_server_t server({ addr_str, port }, 4, 8, 120, 4);
    std::cout << "Server started on " << addr_str << ":" << port << " Listening for connections..."
              << std::endl;

    getchar(); // Wait for user input to stop the server
    return 0;
}

int start_clients(uint32_t start_client,
                  uint32_t num_clients,
                  uint32_t comm_size,
                  const char* addr_str,
                  uint16_t port) {
    std::vector<std::thread> clients;

    sockaddr_t srv_addr(addr_str, port);

    FOR_I(num_clients) {
        clients.emplace_back(comm_init_rank, comm_size, start_client + i, srv_addr);
    }

    for (auto& client : clients) {
        client.join();
    }

    return 0;
}

// occp --server <addr> <port>
// occp --clients <start_rank> <count> <comm_size> <srv_addr> <port>
int main(int argc, char* argv[]) {
    if (argc >= 4) {
        if (std::string(argv[1]) == "--server") {
            const char* srv_addr = argv[2];
            uint16_t srv_port = static_cast<uint16_t>(std::stoi(argv[3]));

            return start_server(srv_addr, srv_port);
        }
        else if (std::string(argv[1]) == "--client" && argc >= 6) {
            std::vector<std::thread> clients;
            uint32_t start_client = static_cast<uint32_t>(std::stoi(argv[2]));
            uint32_t num_clients = static_cast<uint32_t>(std::stoi(argv[3]));
            uint32_t comm_size = static_cast<uint32_t>(std::stoi(argv[4]));
            const char* srv_addr = argv[5];
            uint16_t srv_port = static_cast<uint16_t>(std::stoi(argv[6]));

            return start_clients(start_client, num_clients, comm_size, srv_addr, srv_port);
        }
    }

    std::cerr
        << "Usage: " << argv[0]
        << " --server [<addr> <port>] | --client <start_rank> <count> <comm_size> <srv_addr> <port>"
        << std::endl;
    return 1;
}

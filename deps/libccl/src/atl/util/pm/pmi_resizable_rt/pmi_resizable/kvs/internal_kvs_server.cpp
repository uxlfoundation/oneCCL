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
#include <arpa/inet.h>
#include <cerrno>
#include <list>
#include <map>
#include <mutex>
#include <poll.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <memory>

#include "common/log/log.hpp"
#include "internal_kvs_server.hpp"

class server {
public:
    server(server_args_t* server_args) : args(server_args) {}
    kvs_status_t run();
    kvs_status_t check_finalize(size_t& to_finalize);
    kvs_status_t make_client_request(int& socket);
    kvs_status_t try_to_connect_new();

private:
    struct clients_info {
        int socket;
        bool in_barrier;
        clients_info(int sock, bool barrier) : socket(sock), in_barrier(barrier) {}
    };
    struct proc_info {
        std::string rank;
        size_t rank_count;
        std::string thread_id;
    };
    struct socket_info {
        int socket;
        std::string proc_id;
        proc_info process_info;
    };
    struct comm_info {
        size_t global_size = 0;
        size_t local_size = 0;
        std::list<socket_info> sockets;
        std::map<std::string, std::list<proc_info>> processes;
    };
    struct barrier_info {
        size_t global_size = 0;
        size_t local_size = 0;
        std::list<std::shared_ptr<clients_info>> clients;
    };
    enum fd_indexes { FDI_LISTENER = 0, FDI_CONTROL = 1, FDI_LAST = 2 };

    kvs_request_t request{};
    size_t count{};
    size_t client_count = 0;
    const size_t max_client_queue_size = 300;
    const size_t client_count_increase = 300;
    std::map<std::string, barrier_info> barriers;
    std::map<std::string, comm_info> communicators;
    std::mutex server_memory_mutex;
    std::map<std::string, std::map<std::string, std::string>> requests;
    const int free_socket = -1;
    std::vector<struct pollfd> poll_fds;

    sa_family_t address_family{ AF_UNSPEC };
    std::unique_ptr<server_args_t> args;
};

kvs_status_t server::try_to_connect_new() {
    if (poll_fds[FDI_LISTENER].revents != 0) {
        std::shared_ptr<isockaddr> addr;

        if (address_family == AF_INET) {
            addr = std::shared_ptr<isockaddr>(new sockaddr_v4());
        }
        else {
            addr = std::shared_ptr<isockaddr>(new sockaddr_v6());
        }

        int new_socket;
        bool socket_found = false;
        socklen_t peer_addr_size = addr->size();
        if ((new_socket = accept(poll_fds[FDI_LISTENER].fd,
                                 addr->get_sock_addr_ptr(),
                                 (socklen_t*)&peer_addr_size)) < 0) {
            LOG_ERROR("server_listen_sock accept:", strerror(errno));
            return KVS_STATUS_FAILURE;
        }
        for (size_t i = FDI_LAST; i < poll_fds.size(); i++) {
            if (poll_fds[i].fd == free_socket) {
                poll_fds[i].fd = new_socket;
                socket_found = true;
                break;
            }
        }
        if (!socket_found) {
            // the code is written in a way that there should always be a free socket available
            // if no socket is found, this means that there is an error in the code
            // or that out of memory exception occurred while resizing the poll_fds vector
            // and it was not properly handled in the layers above internal_kvs_server
            LOG_ERROR("free socket not found; this indicates programmer's error");
            if (close(new_socket)) {
                // we are already returning failure, there is not much we can do
                // except for logging the exact error that occurred
                LOG_ERROR("error closing a socket: ", strerror(errno));
            }
            return KVS_STATUS_FAILURE;
        }
        client_count++;
        if (poll_fds.size() - FDI_LAST == client_count) {
            size_t old_size = poll_fds.size();
            poll_fds.resize(old_size + client_count_increase);
            for (size_t i = old_size; i < poll_fds.size(); i++) {
                poll_fds[i].fd = free_socket;
                poll_fds[i].events = POLLIN;
            }
        }
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t server::make_client_request(int& socket) {
    KVS_CHECK_STATUS(request.get(socket, server_memory_mutex), "server: get command from client");

    switch (request.mode) {
        case AM_CLOSE: {
            close(socket);
            socket = free_socket;
            client_count--;
            break;
        }
        case AM_PUT: {
            auto& req = requests[request.name];
            req[request.key] = request.val;
            break;
        }
        case AM_REMOVE: {
            requests[request.name].erase(request.key);
            break;
        }
        case AM_GET_VAL: {
            count = 0;
            auto it_name = requests.find(request.name);
            if (it_name != requests.end()) {
                auto it_key = it_name->second.find(request.key);
                if (it_key != it_name->second.end()) {
                    count = 1;
                    KVS_CHECK_STATUS(request.put(socket, server_memory_mutex, count),
                                     "server: put value write size");

                    KVS_CHECK_STATUS(request.put(socket, server_memory_mutex, it_key->second),
                                     "server: put value write data");
                    break;
                }
            }
            KVS_CHECK_STATUS(request.put(socket, server_memory_mutex, count),
                             "server: put value write size");
            break;
        }
        case AM_GET_COUNT: {
            count = 0;
            auto it = requests.find(request.name);
            if (it != requests.end()) {
                count = it->second.size();
            }
            KVS_CHECK_STATUS(request.put(socket, server_memory_mutex, count), "server: put count");
            break;
        }
        case AM_GET_REPLICA: {
            char* replica_size_str = getenv(CCL_WORLD_SIZE_ENV);
            count = client_count;
            if (replica_size_str != nullptr) {
                KVS_CHECK_STATUS(safe_strtol(replica_size_str, count), "failed to convert count");
            }
            KVS_CHECK_STATUS(request.put(socket, server_memory_mutex, count),
                             "server: get_replica");
            break;
        }
        case AM_GET_KEYS_VALUES: {
            std::vector<kvs_request_t> answers;
            count = 0;
            auto it_name = requests.find(request.name);
            if (it_name != requests.end()) {
                count = it_name->second.size();
            }

            KVS_CHECK_STATUS(request.put(socket, server_memory_mutex, count),
                             "server: put keys_values count");
            if (count == 0)
                break;

            KVS_CHECK_STATUS(request.put(socket, server_memory_mutex, it_name->second),
                             "server: put keys_values write data");
            break;
        }
        case AM_BARRIER: {
            auto& barrier_list = barriers[request.name];
            auto& clients = barrier_list.clients;
            auto client_it = std::find_if(std::begin(clients),
                                          std::end(clients),
                                          [&](std::shared_ptr<server::clients_info> v) {
                                              return v->socket == socket;
                                          });
            if (client_it == clients.end()) {
                // TODO: Look deeper to fix this error
                LOG_ERROR("Server error: Unregister Barrier request!");
                return KVS_STATUS_FAILURE;
            }
            auto client_inf = client_it->get();
            client_inf->in_barrier = true;

            if (barrier_list.global_size == barrier_list.local_size) {
                bool is_barrier_stop = true;
                for (const auto& client : clients) {
                    if (!client->in_barrier) {
                        is_barrier_stop = false;
                        break;
                    }
                }
                if (is_barrier_stop) {
                    size_t is_done = 1;
                    for (const auto& client : clients) {
                        client->in_barrier = false;
                        KVS_CHECK_STATUS(request.put(client->socket, server_memory_mutex, is_done),
                                         "server: barrier");
                    }
                }
            }
            break;
        }
        case AM_BARRIER_REGISTER: {
            char* glob_size = request.val;
            char* local_size = strstr(glob_size, "_");
            auto& barrier = barriers[request.name];
            if (local_size == nullptr) {
                barrier.local_size++;
            }
            else {
                local_size[0] = '\0';
                local_size++;
                size_t local_size_tmp;

                KVS_CHECK_STATUS(safe_strtol(local_size, local_size_tmp),
                                 "failed to convert local_size");
                barrier.local_size += local_size_tmp;
            }
            KVS_CHECK_STATUS(safe_strtol(glob_size, barrier.global_size),
                             "failed to convert global_size");

            barrier.clients.push_back(
                std::shared_ptr<clients_info>(new clients_info(socket, false)));
            break;
        }
        case AM_SET_SIZE: {
            char* glob_size = request.val;
            KVS_CHECK_STATUS(safe_strtol(glob_size, communicators[request.key].global_size),
                             "failed to convert global_size");

            break;
        }
        case AM_INTERNAL_REGISTER: {
            auto& communicator = communicators[request.key];
            char* rank_count_str = request.val;
            char* rank = strstr(rank_count_str, "_");
            rank[0] = '\0';
            rank++;
            char* proc_id = strstr(rank, "_");
            proc_id[0] = '\0';
            proc_id++;
            char* thread_id = strstr(proc_id, "_");
            thread_id[0] = '\0';
            thread_id++;
            size_t rank_count;
            KVS_CHECK_STATUS(safe_strtol(rank_count_str, rank_count),
                             "failed to convert rank_count");
            communicators[request.key].local_size += rank_count;
            socket_info sock_info{ socket, proc_id, { rank, rank_count, thread_id } };
            communicator.processes[proc_id].push_back(sock_info.process_info);
            communicator.sockets.push_back(sock_info);
            if (communicator.local_size == communicator.global_size) {
                std::string proc_count_str = std::to_string(communicator.processes.size());
                for (auto& it : communicator.processes) {
                    it.second.sort([](proc_info a, proc_info b) {
                        return a.rank < b.rank;
                    });
                }
                std::string put_buf;
                for (auto& it : communicator.sockets) {
                    std::string thread_num;
                    int i = 0;
                    size_t proc_rank_count = 0;
                    auto process_info = communicator.processes[it.proc_id];
                    std::string threads_count = std::to_string(process_info.size());
                    for (auto& proc_info_it : process_info) {
                        if (it.process_info.rank == proc_info_it.rank) {
                            thread_num = std::to_string(i);
                            break;
                        }
                        i++;
                    }
                    for (auto& proc_info_it : process_info) {
                        proc_rank_count += proc_info_it.rank_count;
                    }
                    /*return string: %PROC_COUNT%_%RANK_NUM%_%PROCESS_RANK_COUNT%_%THREADS_COUNT%_%THREAD_NUM% */
                    put_buf = proc_count_str + "_" + it.process_info.rank + "_" +
                              std::to_string(proc_rank_count) + "_" + threads_count + "_" +
                              thread_num;
                    KVS_CHECK_STATUS(request.put(it.socket, server_memory_mutex, put_buf),
                                     "server: register write data");
                }
            }
            break;
        }
        default: {
            if (request.name[0] == '\0')
                return KVS_STATUS_SUCCESS;
            LOG_ERROR("unknown request mode: ", request.mode);
            return KVS_STATUS_FAILURE;
        }
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t server::check_finalize(size_t& to_finalize) {
    to_finalize = false;
    if (poll_fds[FDI_CONTROL].revents != 0) {
        KVS_CHECK_STATUS(request.get(poll_fds[FDI_CONTROL].fd, server_memory_mutex),
                         "server: get control msg from client");
        if (request.mode != AM_FINALIZE) {
            LOG_ERROR("invalid access mode for local socket\n");
            return KVS_STATUS_FAILURE;
        }
        to_finalize = true;
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t server::run() {
    size_t should_stop = false;

    int so_reuse = 1;
#ifdef SO_REUSEPORT
    int reuse_optname = SO_REUSEPORT;
#else
    int reuse_optname = SO_REUSEADDR;
#endif

    poll_fds.resize(client_count_increase);
    for (auto& it : poll_fds) {
        it.fd = free_socket;
        it.events = POLLIN;
    }
    poll_fds[FDI_LISTENER].fd = args->sock_listener;
    address_family = args->args->sin_family();

    if (setsockopt(
            poll_fds[FDI_LISTENER].fd, SOL_SOCKET, reuse_optname, &so_reuse, sizeof(so_reuse))) {
        LOG_ERROR("server_listen_sock setsockopt(%s)", strerror(errno));
        return KVS_STATUS_FAILURE;
    }

    if (listen(poll_fds[FDI_LISTENER].fd, max_client_queue_size) < 0) {
        LOG_ERROR("server_listen_sock listen(%s)", strerror(errno));
        return KVS_STATUS_FAILURE;
    }

    if ((poll_fds[FDI_CONTROL].fd = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("server_control_sock init(%s)", strerror(errno));
        return KVS_STATUS_FAILURE;
    }

    while (connect(poll_fds[FDI_CONTROL].fd, args->args->get_sock_addr_ptr(), args->args->size()) <
           0) {
    }
    while (!should_stop || client_count > 0) {
        if (poll(poll_fds.data(), poll_fds.size(), -1) < 0) {
            if (errno != EINTR) {
                LOG_ERROR("poll(%s)", strerror(errno));
                return KVS_STATUS_FAILURE;
            }
            else {
                /* restart select */
                continue;
            }
        }

        for (size_t i = FDI_LAST; i < poll_fds.size(); i++) {
            if (poll_fds[i].fd != free_socket && poll_fds[i].revents != 0) {
                KVS_CHECK_STATUS(make_client_request(poll_fds[i].fd), "failed to make request");
            }
        }
        KVS_CHECK_STATUS(try_to_connect_new(), "failed to connect new");
        if (!should_stop) {
            KVS_CHECK_STATUS(check_finalize(should_stop), "failed to check finalize");
        }
    }

    if (poll_fds[FDI_CONTROL].fd != free_socket) {
        KVS_CHECK_STATUS(request.put(poll_fds[FDI_CONTROL].fd, server_memory_mutex, should_stop),
                         "server: put control msg to client");
    }

    close(poll_fds[FDI_CONTROL].fd);
    poll_fds[FDI_CONTROL].fd = free_socket;

    for (size_t i = FDI_LAST; i < poll_fds.size(); i++) {
        if (poll_fds[i].fd != free_socket) {
            close(poll_fds[i].fd);
            poll_fds[i].fd = free_socket;
        }
    }

    close(poll_fds[FDI_LISTENER].fd);
    poll_fds[FDI_LISTENER].fd = free_socket;
    return KVS_STATUS_SUCCESS;
}

void* kvs_server_init(void* args) {
    server s(reinterpret_cast<server_args_t*>(args));

    if (s.run() != KVS_STATUS_SUCCESS) {
        LOG_ERROR("failed");
    }
    return nullptr;
}

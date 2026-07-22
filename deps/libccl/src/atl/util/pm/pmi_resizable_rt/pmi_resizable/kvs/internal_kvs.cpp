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

#include <arpa/inet.h>
#include <errno.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <netinet/in.h>
#include <mutex>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "internal_kvs.h"
#include "internal_kvs_server.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"

bool can_use_internal_kvs() {
    return ccl::global_data::env().kvs_init_mode == ccl::kvs_mode::pmi ||
           (ccl::global_data::env().atl_transport == ccl_atl_ofi &&
            (ccl::global_data::env().kvs_init_mode != ccl::kvs_mode::pmix_ofi ||
             ccl::global_data::env().kvs_init_mode != ccl::kvs_mode::pmix_ofi_shm) &&
            ccl::global_data::env().process_launcher != process_launcher_mode::pmix);
}

#define assert_throw_can_use_internal_kvs() \
    do { \
        CCL_THROW_IF_NOT( \
            can_use_internal_kvs(), \
            "internal kvs should be used with pmi kvs mode or ofi transport with pmi kvs mode and pmix launcher"); \
    } while (0)

internal_kvs::internal_kvs() : CONNECTION_TIMEOUT(ccl::global_data::env().kvs_connection_timeout) {}

kvs_status_t internal_kvs::kvs_set_value(const std::string& kvs_name,
                                         const std::string& kvs_key,
                                         const std::string& kvs_val) {
    assert_throw_can_use_internal_kvs();
    kvs_request_t request;
    KVS_CHECK_STATUS(
        request.put(client_op_sock, AM_PUT, client_memory_mutex, kvs_name, kvs_key, kvs_val),
        "client: put_key_value");

    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_set_size(const std::string& kvs_name,
                                        const std::string& kvs_key,
                                        const std::string& kvs_val) {
    assert_throw_can_use_internal_kvs();
    kvs_request_t request;
    KVS_CHECK_STATUS(
        request.put(client_op_sock, AM_SET_SIZE, client_memory_mutex, kvs_name, kvs_key, kvs_val),
        "client: set_size");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_barrier_register(const std::string& kvs_name,
                                                const std::string& kvs_key,
                                                const std::string& kvs_val) {
    assert_throw_can_use_internal_kvs();
    kvs_request_t request;
    KVS_CHECK_STATUS(
        request.put(
            client_op_sock, AM_BARRIER_REGISTER, client_memory_mutex, kvs_name, kvs_key, kvs_val),
        "client: barrier_register");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_barrier(const std::string& kvs_name,
                                       const std::string& kvs_key,
                                       const std::string& kvs_val) {
    assert_throw_can_use_internal_kvs();
    kvs_request_t request;
    size_t is_done;
    KVS_CHECK_STATUS(
        request.put(client_op_sock, AM_BARRIER, client_memory_mutex, kvs_name, kvs_key, kvs_val),
        "client: barrier");

    KVS_CHECK_STATUS(request.get(client_op_sock, client_memory_mutex, is_done),
                     "client: barrier read data");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_remove_name_key(const std::string& kvs_name,
                                               const std::string& kvs_key) {
    assert_throw_can_use_internal_kvs();
    kvs_request_t request;
    KVS_CHECK_STATUS(request.put(client_op_sock, AM_REMOVE, client_memory_mutex, kvs_name, kvs_key),
                     "client: remove_key");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_register(const std::string& kvs_name,
                                        const std::string& kvs_key,
                                        std::string& kvs_val) {
    assert_throw_can_use_internal_kvs();
    kvs_request_t request;
    KVS_CHECK_STATUS(
        request.put(
            client_op_sock, AM_INTERNAL_REGISTER, client_memory_mutex, kvs_name, kvs_key, kvs_val),
        "client: register");

    KVS_CHECK_STATUS(request.get(client_op_sock, client_memory_mutex, kvs_val),
                     "client: register read data");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_get_value_by_name_key(const std::string& kvs_name,
                                                     const std::string& kvs_key,
                                                     std::string& kvs_val) {
    assert_throw_can_use_internal_kvs();
    kvs_request_t request;
    size_t is_exist = 0;
    KVS_CHECK_STATUS(
        request.put(client_op_sock, AM_GET_VAL, client_memory_mutex, kvs_name, kvs_key),
        "client: get_value");

    KVS_CHECK_STATUS(request.get(client_op_sock, client_memory_mutex, is_exist),
                     "client: get_value is_exist");

    if (is_exist) {
        KVS_CHECK_STATUS(request.get(client_op_sock, client_memory_mutex, kvs_val),
                         "client: get_value read data");
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_get_count_names(const std::string& kvs_name, size_t& count_names) {
    assert_throw_can_use_internal_kvs();
    count_names = 0;
    kvs_request_t request;
    KVS_CHECK_STATUS(request.put(client_op_sock, AM_GET_COUNT, client_memory_mutex, kvs_name),
                     "client: get_count");

    KVS_CHECK_STATUS(request.get(client_op_sock, client_memory_mutex, count_names),
                     "client: get_count read data");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_get_keys_values_by_name(const std::string& kvs_name,
                                                       std::vector<std::string>& kvs_keys,
                                                       std::vector<std::string>& kvs_values,
                                                       size_t& count) {
    assert_throw_can_use_internal_kvs();
    count = 0;
    kvs_request_t request;
    KVS_CHECK_STATUS(request.put(client_op_sock, AM_GET_KEYS_VALUES, client_memory_mutex, kvs_name),
                     "client: get_keys_values");

    KVS_CHECK_STATUS(request.get(client_op_sock, client_memory_mutex, count),
                     "client: get_keys_values read size");
    if (count == 0) {
        return KVS_STATUS_SUCCESS;
    }
    KVS_CHECK_STATUS(request.get(client_op_sock, client_memory_mutex, count, kvs_keys, kvs_values),
                     "client: get_keys_values read data");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_get_replica_size(size_t& replica_size) {
    assert_throw_can_use_internal_kvs();
    replica_size = 0;
    kvs_request_t request;
    KVS_CHECK_STATUS(request.put(client_op_sock, AM_GET_REPLICA, client_memory_mutex),
                     "client: get_replica");

    KVS_CHECK_STATUS(request.get(client_op_sock, client_memory_mutex, replica_size),
                     "client: get_replica read size");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::init_main_server_by_env() {
    char* port = nullptr;

    const char* tmp_host_ip = (!server_address.empty()) ? server_address.c_str()
                                                        : std::getenv(CCL_KVS_IP_PORT_ENV.c_str());

    if (tmp_host_ip == nullptr) {
        LOG_ERROR("specify ", CCL_KVS_IP_PORT_ENV);
        return KVS_STATUS_FAILURE;
    }

    memset(main_host_ip, 0, CCL_IP_LEN);
    kvs_str_copy(main_host_ip, tmp_host_ip, CCL_IP_LEN);
    if ((port = strstr(main_host_ip, "_")) == nullptr) {
        if ((port = strstr(main_host_ip, ":")) == nullptr) {
            LOG_ERROR("set ", CCL_KVS_IP_PORT_ENV, " in format <ip>_<port>\n");
            return KVS_STATUS_FAILURE;
        }
    }
    port[0] = '\0';
    port++;

    KVS_CHECK_STATUS(safe_strtol(port, main_port), "failed to convert main_port");
    main_server_address->set_sin_port(main_port);
    KVS_CHECK_STATUS(main_server_address->set_sin_addr(main_host_ip), "failed to set main_ip");
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::init_main_server_by_string(const char* main_addr) {
    char* port = nullptr;
    KVS_CHECK_STATUS(local_server_address->set_sin_addr(local_host_ip), "failed to set main_ip");

    CCL_ASSERT(server_listen_sock ==
               INVALID_SOCKET); // make sure the socket is not initialized twice
    if ((server_listen_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("server_listen_sock init");
        return KVS_STATUS_FAILURE;
    }

    size_t sin_port = local_server_address->get_sin_port();
    while (bind(server_listen_sock,
                local_server_address->get_sock_addr_ptr(),
                local_server_address->size()) < 0) {
        sin_port++;
        local_server_address->set_sin_port(sin_port);
    }

    memset(main_host_ip, 0, CCL_IP_LEN);
    kvs_str_copy(main_host_ip, main_addr, CCL_IP_LEN);

    if ((port = strstr(main_host_ip, "_")) == nullptr) {
        if ((port = strstr(main_host_ip, ":")) == nullptr) {
            LOG_ERROR("set ", CCL_KVS_IP_PORT_ENV, " in format <ip>_<port>");
            return KVS_STATUS_FAILURE;
        }
    }
    port[0] = '\0';
    port++;

    // check if main_addr has root rank
    if (!can_use_internal_kvs()) {
        // main_addr format : ip_port_root
        char* root_rank_str = nullptr;
        if ((root_rank_str = strstr(port, "_")) == nullptr) {
            LOG_ERROR("set ", CCL_KVS_IP_PORT_ENV, " in format <ip>_<port>_<root_rank>");
            return KVS_STATUS_FAILURE;
        }
        root_rank_str[0] = '\0';
    }

    KVS_CHECK_STATUS(safe_strtol(port, main_port), "failed to convert main_port");
    main_server_address->set_sin_port(main_port);
    KVS_CHECK_STATUS(main_server_address->set_sin_addr(main_host_ip), "failed to set main_ip");

    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::fill_local_host_ip() {
    struct ifaddrs *ifaddr, *ifa;
    int family = AF_UNSPEC;
    char local_ip[CCL_IP_LEN];
    bool is_supported_iface = false;
    if (getifaddrs(&ifaddr) < 0) {
        LOG_ERROR("can not get host IP");
        return KVS_STATUS_FAILURE;
    }

    const char iface_name[] = "lo";
    char* iface_name_env = std::getenv(CCL_KVS_IFACE_ENV.c_str());
    local_host_ips.clear();
    local_host_ipv6s.clear();
    local_host_ipv4s.clear();

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;
        if (iface_name_env) {
            is_supported_iface = strstr(ifa->ifa_name, iface_name_env);
        }
        else {
            is_supported_iface = strstr(ifa->ifa_name, iface_name) == NULL;
        }
        if (is_supported_iface) {
            family = ifa->ifa_addr->sa_family;
            if (family == AF_INET || family == AF_INET6) {
                memset(local_ip, 0, CCL_IP_LEN);
                int res = getnameinfo(
                    ifa->ifa_addr,
                    (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6),
                    local_ip,
                    CCL_IP_LEN,
                    NULL,
                    0,
                    NI_NUMERICHOST);
                if (res != 0) {
                    std::string s("getnameinfo error > ");
                    s.append(gai_strerror(res));
                    LOG_ERROR(s.c_str());
                    return KVS_STATUS_FAILURE;
                }

                local_host_ips.push_back(local_ip);
                if (family == AF_INET6) {
                    char* scope_id_ptr = nullptr;
                    if ((scope_id_ptr = strchr(local_ip, SCOPE_ID_DELIM))) {
                        uint32_t scope_id = ((struct sockaddr_in6*)(ifa->ifa_addr))->sin6_scope_id;
                        sprintf(scope_id_ptr + 1, "%u", scope_id);
                    }
                    local_host_ipv6s.push_back(local_ip);
                }
                else {
                    local_host_ipv4s.push_back(local_ip);
                }
            }
        }
    }
    if (local_host_ips.empty()) {
        LOG_ERROR("can't find interface ", iface_name_env ? iface_name_env : "", " to get host IP");
        return KVS_STATUS_FAILURE;
    }

    memset(local_host_ip, 0, CCL_IP_LEN);

    char* kvs_prefer_ipv6 = std::getenv(CCL_KVS_PREFER_IPV6_ENV.c_str());
    size_t is_kvs_prefer_ipv6 = 0;
    if (kvs_prefer_ipv6) {
        KVS_CHECK_STATUS(safe_strtol(kvs_prefer_ipv6, is_kvs_prefer_ipv6),
                         "failed to set prefer_ip6");
    }

    if (is_kvs_prefer_ipv6) {
        if (!local_host_ipv6s.empty()) {
            address_family = AF_INET6;
        }
        else {
            LOG_WARN("ipv6 addresses are not found, fallback to ipv4");
            address_family = AF_INET;
        }
    }
    else {
        address_family = (!local_host_ipv4s.empty()) ? AF_INET : AF_INET6;
    }

    if (address_family == AF_INET) {
        main_server_address = std::shared_ptr<isockaddr>(new sockaddr_v4());
        local_server_address = std::shared_ptr<isockaddr>(new sockaddr_v4());
        kvs_str_copy(local_host_ip, local_host_ipv4s.front().c_str(), CCL_IP_LEN);
    }
    else {
        main_server_address = std::shared_ptr<isockaddr>(new sockaddr_v6());
        local_server_address = std::shared_ptr<isockaddr>(new sockaddr_v6());
        kvs_str_copy(local_host_ip, local_host_ipv6s.front().c_str(), CCL_IP_LEN);
    }
    LOG_DEBUG("use ", address_family == AF_INET ? "ipv4" : "ipv6", ": ", local_host_ip);

    freeifaddrs(ifaddr);
    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_main_server_address_reserve(char* main_address) {
    if (!server_address.empty())
        return KVS_STATUS_SUCCESS;

    KVS_CHECK_STATUS(fill_local_host_ip(), "failed to get local host IP");

    CCL_ASSERT(server_listen_sock ==
               INVALID_SOCKET); // make sure the socket is not initialized twice
    if ((server_listen_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("server_listen_sock init");
        return KVS_STATUS_FAILURE;
    }

    KVS_CHECK_STATUS(main_server_address->set_sin_addr(local_host_ip), "failed to set local_ip");
    KVS_CHECK_STATUS(local_server_address->set_sin_addr(local_host_ip), "failed to set local_ip");
    size_t sin_port = main_server_address->get_sin_port();

    while (bind(server_listen_sock,
                main_server_address->get_sock_addr_ptr(),
                main_server_address->size()) < 0) {
        sin_port++;
        main_server_address->set_sin_port(sin_port);
    }
    local_server_address->set_sin_port(main_server_address->get_sin_port());

    memset(main_address, '\0', CCL_IP_LEN);
    snprintf(main_address, CCL_IP_LEN, "%s", local_host_ip);
    snprintf(main_address + strlen(local_host_ip),
             INT_STR_SIZE + 1,
             "_%d",
             main_server_address->get_sin_port());
    // add rank to main_address
    if (!can_use_internal_kvs()) {
#ifdef CCL_ENABLE_MPI
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int str_len = strnlen(main_address, ccl::v1::kvs::address_max_size);
        sprintf(main_address + str_len, "_%d", rank);
#else
        CCL_THROW("MPI is required when not using internal KVS");
#endif
    }

    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::init_main_server_address(const char* main_addr) {
    char* ip_getting_type = std::getenv(CCL_KVS_IP_EXCHANGE_ENV.c_str());

    if (local_host_ips.empty()) {
        KVS_CHECK_STATUS(fill_local_host_ip(), "failed to get local host ip");
    }

    if (ip_getting_type) {
        if (ip_getting_type == CCL_KVS_IP_EXCHANGE_VAL_ENV) {
            ip_getting_mode = IGT_ENV;
        }
        else {
            LOG_ERROR("unknown ", CCL_KVS_IP_EXCHANGE_ENV, ": ", ip_getting_type);
            return KVS_STATUS_FAILURE;
        }
    }

    if (server_address.empty()) {
        if (main_addr != NULL) {
            // get root_rank from main_addr
            if (!can_use_internal_kvs()) {
                // main_addr format : ip_port_root
                std::string main_addr_str(main_addr);
                size_t pos_1 = std::string::npos;
                size_t pos_2 = std::string::npos;
                pos_1 = main_addr_str.find("_");
                if (pos_1 != std::string::npos) {
                    pos_2 = main_addr_str.find("_", pos_1 + 1);
                    if (pos_2 != std::string::npos) {
                        root_rank = std::stoi(main_addr_str.substr(pos_2 + 1));
                    }
                }
                if (pos_1 == std::string::npos || pos_2 == std::string::npos) {
                    LOG_ERROR("failed to find root_rank in ", main_addr);
                }
            }

            ip_getting_mode = IGT_ENV;
            if (server_listen_sock ==
                INVALID_SOCKET) { // make sure the socket is not initialized twice
                KVS_CHECK_STATUS(init_main_server_by_string(main_addr),
                                 "failed to init main server");
            }
            return KVS_STATUS_SUCCESS;
        }
    }
    else {
        ip_getting_mode = IGT_ENV;
    }

    KVS_CHECK_STATUS(local_server_address->set_sin_addr(local_host_ip), "failed to set local_ip");

    CCL_ASSERT(server_listen_sock ==
               INVALID_SOCKET); // make sure the socket is not initialized twice
    if ((server_listen_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("server_listen_sock init");
        return KVS_STATUS_FAILURE;
    }

    switch (ip_getting_mode) {
        case IGT_ENV: {
            int is_master_node = 0;

            KVS_CHECK_STATUS(init_main_server_by_env(), "failed to init_main_server_by_env");

            if (strstr(local_host_ip, main_host_ip)) {
                is_master_node = 1;
            }
            else {
                auto main_node_ip =
                    std::find(local_host_ips.begin(), local_host_ips.end(), main_host_ip);
                if (main_node_ip != local_host_ips.end()) {
                    is_master_node = 1;
                    memset(local_host_ip, 0, CCL_IP_LEN);
                    kvs_str_copy_known_sizes(local_host_ip, main_host_ip, CCL_IP_LEN);
                    KVS_CHECK_STATUS(local_server_address->set_sin_addr(local_host_ip),
                                     "get sin add failed");
                }
            }
            if (is_master_node) {
                if (bind(server_listen_sock,
                         main_server_address->get_sock_addr_ptr(),
                         main_server_address->size()) < 0) {
                    LOG_WARN("port [",
                             main_server_address->get_sin_port(),
                             "] is busy, connecting as client");
                    local_port = local_server_address->get_sin_port();
                    while (bind(server_listen_sock,
                                local_server_address->get_sock_addr_ptr(),
                                local_server_address->size()) < 0) {
                        local_port++;
                        local_server_address->set_sin_port(local_port);
                    }
                }
                else {
                    local_port = main_server_address->get_sin_port();
                }
            }
            else {
                local_port = local_server_address->get_sin_port();
                while (bind(server_listen_sock,
                            local_server_address->get_sock_addr_ptr(),
                            local_server_address->size()) < 0) {
                    local_port++;
                    local_server_address->set_sin_port(local_port);
                }
            }

            return KVS_STATUS_SUCCESS;
        }
        default: {
            LOG_ERROR("unknown ", CCL_KVS_IP_EXCHANGE_ENV);
            return KVS_STATUS_FAILURE;
        }
    }
}

kvs_status_t internal_kvs::kvs_init(const char* main_addr) {
    int err;
    socklen_t len = 0;
    std::shared_ptr<isockaddr> addr;

    time_t start_time;
    time_t connection_time = 0;

    if (init_main_server_address(main_addr) != KVS_STATUS_SUCCESS) {
        LOG_ERROR("init main server address error");
        close(client_op_sock);
        close(server_control_sock);
        client_op_sock = 0;
        server_control_sock = 0;
        return KVS_STATUS_FAILURE;
    }

    if (address_family == AF_INET) {
        addr = std::shared_ptr<isockaddr>(new sockaddr_v4());
        KVS_CHECK_STATUS(addr->set_sin_addr("127.0.0.1"), "failed to set sin_addr(\"127.0.0.1\"");
    }
    else {
        addr = std::shared_ptr<isockaddr>(new sockaddr_v6());
        KVS_CHECK_STATUS(addr->set_sin_addr("::1"), "failed to set sin_addr(\"::1\"");
    }

    if ((client_op_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("client_op_sock init");
        return KVS_STATUS_FAILURE;
    }

    if ((server_control_sock = socket(address_family, SOCK_STREAM, 0)) < 0) {
        LOG_ERROR("server_control_sock init");
        return KVS_STATUS_FAILURE;
    }

    size_t sin_port = addr->get_sin_port();
    while (bind(server_control_sock, addr->get_sock_addr_ptr(), addr->size()) < 0) {
        sin_port++;
        addr->set_sin_port(sin_port);
    }

    if (listen(server_control_sock, 1) < 0) {
        LOG_ERROR("server_control_sock listen");
        return KVS_STATUS_FAILURE;
    }

    if (getsockname(server_control_sock, addr->get_sock_addr_ptr(), &len)) {
        LOG_ERROR("server_control_sock getsockname");
        return KVS_STATUS_FAILURE;
    }

    server_args* args = new server_args();
    args->args = std::move(addr);
    args->sock_listener = server_listen_sock;
    err = pthread_create(&kvs_thread, nullptr, kvs_server_init, args);
    if (err) {
        delete args;
        LOG_ERROR("failed to create kvs server thread, pthread_create returns ", err);
        return KVS_STATUS_FAILURE;
    }

    if ((client_control_sock = accept(server_control_sock, nullptr, nullptr)) < 0) {
        LOG_ERROR("server_control_sock accept");
        return KVS_STATUS_FAILURE;
    }

    // socket connection is only used with internal kvs and not with other modes
    // say for example with mpi kvs mode, mpi calls are used for communication.
    if (can_use_internal_kvs()) {
        /* Wait connection to master */
        start_time = time(nullptr);
        do {
            err = connect(client_op_sock,
                          main_server_address->get_sock_addr_ptr(),
                          main_server_address->size());
            connection_time = time(nullptr) - start_time;
        } while ((err < 0) && (connection_time < CONNECTION_TIMEOUT));

        if (connection_time >= CONNECTION_TIMEOUT) {
            LOG_ERROR(
                "connection time (", connection_time, ") >= limit (", CONNECTION_TIMEOUT, ")");
            return KVS_STATUS_FAILURE;
        }
    }

    if (strstr(main_host_ip, local_host_ip) && local_port == main_port) {
        is_master = 1;
    }
    is_inited = true;

    return KVS_STATUS_SUCCESS;
}

kvs_status_t internal_kvs::kvs_finalize(void) {
    close(client_op_sock);
    client_op_sock = 0;
    if (kvs_thread != 0) {
        kvs_request_t request;
        void* exit_code;
        size_t is_stop;
        int err;
        KVS_CHECK_STATUS(request.put(client_control_sock, AM_FINALIZE, client_memory_mutex),
                         "client: finalize start");

        KVS_CHECK_STATUS(request.get(client_control_sock, client_memory_mutex, is_stop),
                         "client: finalize complete");

        err = pthread_join(kvs_thread, &exit_code);
        if (err) {
            LOG_ERROR("failed to stop kvs server thread, pthread_join returns ", err);
            return KVS_STATUS_FAILURE;
        }

        kvs_thread = 0;

        close(client_control_sock);
        close(server_control_sock);

        client_control_sock = 0;
        server_control_sock = 0;
    }

    is_inited = false;

    return KVS_STATUS_SUCCESS;
}

internal_kvs::~internal_kvs() {
    if (is_inited) {
        CCL_THROW_IF_NOT(kvs_finalize() == KVS_STATUS_SUCCESS, "failed to finalize kvs");
    }
}

kvs_status_t sockaddr_v4::set_sin_addr(const char* src) {
    int ret = inet_pton(addr.sin_family, src, &(addr.sin_addr));
    if (ret <= 0) {
        if (ret == 0) {
            LOG_ERROR(
                "inet_pton error - invalid network address, af: ", addr.sin_family, ", src: ", src);
        }
        else if (ret < 0) {
            LOG_ERROR("inet_pton error - af: ",
                      addr.sin_family,
                      ", src: ",
                      src,
                      ", error: ",
                      strerror(errno));
        }
        return KVS_STATUS_FAILURE;
    }
    return KVS_STATUS_SUCCESS;
}

kvs_status_t sockaddr_v6::set_sin_addr(const char* src) {
    char src_copy[internal_kvs::CCL_IP_LEN] = { 0 };
    kvs_str_copy(src_copy, src, internal_kvs::CCL_IP_LEN);

    char* scope_id_ptr = nullptr;
    if ((scope_id_ptr = strchr(src_copy, internal_kvs::SCOPE_ID_DELIM))) {
        KVS_CHECK_STATUS(safe_strtol(scope_id_ptr + 1, addr.sin6_scope_id),
                         "failed to ged sin6_id");
        *scope_id_ptr = '\0';
    }

    int ret = inet_pton(addr.sin6_family, src_copy, &(addr.sin6_addr));
    if (ret <= 0) {
        if (ret == 0) {
            LOG_ERROR("inet_pton error - invalid network address, af: ",
                      addr.sin6_family,
                      ", src_copy: ",
                      src_copy);
        }
        else if (ret < 0) {
            LOG_ERROR("inet_pton error - af: ",
                      addr.sin6_family,
                      ", src_copy: ",
                      src_copy,
                      ", error: ",
                      strerror(errno));
        }
        return KVS_STATUS_FAILURE;
    }

    LOG_DEBUG("", src_copy, ", scope_id: ", addr.sin6_scope_id);
    return KVS_STATUS_SUCCESS;
}

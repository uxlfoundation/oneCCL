/*
 Copyright 2016-2020 Intel Corporation
 
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

#include "comm/comm.hpp"
#include "common/utils/exchange_utils.hpp"
#include "common/global/ze/ze_data.hpp"
#include "common/utils/utils.hpp"
#include "sched/entry/entry.hpp"
#include "sched/sched.hpp"
#include "sched/ze/ze_handle_manager.hpp"

#include <poll.h>
#include <sys/un.h>

class ze_handle_exchange_entry : public sched_entry {
public:
    using mem_desc_t = typename std::pair<void*, ccl::ze::ipc_mem_type>;

    static constexpr const char* class_name() noexcept {
        return "ZE_HANDLES";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    ze_handle_exchange_entry() = delete;
    explicit ze_handle_exchange_entry(ccl_sched* sched,
                                      ccl_comm* comm,
                                      const std::vector<mem_desc_t>& in_buffers,
                                      int skip_rank,
                                      // only for pt2pt usage:
                                      ccl::utils::pt2pt_handle_exchange_info pt2pt_info);
    ~ze_handle_exchange_entry();
    ze_handle_exchange_entry& operator=(const ze_handle_exchange_entry&) = delete;
    ze_handle_exchange_entry(const ze_handle_exchange_entry&) = delete;

    void start() override;
    void update() override;

protected:
    void dump_detail(std::stringstream& str) const override;

private:
    static constexpr int poll_expire_err_code = 0;
    static constexpr int timeout_ms = 1;
    static constexpr size_t max_pfds = 1;

    const ccl_comm* comm;

    std::vector<mem_desc_t> in_buffers;
    ccl::ze::ipc_handle_manager::mem_handle_map_t handles;

    const int rank;
    const int comm_size;
    int skip_rank;
    ccl::utils::pt2pt_handle_exchange_info pt2pt_info;

    pid_t current_pid = ccl::utils::invalid_pid;

    int start_buf_idx{};
    int start_peer_idx{};

    std::vector<struct pollfd> poll_fds;

    int right_peer_socket{};
    int left_peer_socket{};
    int left_peer_connect_socket{};

    struct sockaddr_un right_peer_addr {
    }, left_peer_addr{};
    int right_peer_addr_len{}, left_peer_addr_len{};

    std::string right_peer_socket_name;
    std::string left_peer_socket_name;

    std::vector<int> device_fds;
    std::vector<ccl::ze::device_bdf_info> physical_devices;

    struct payload_t {
        int mem_handle{ ccl::utils::invalid_mem_handle };
        ccl::ze::ipc_mem_type mem_type{};
        size_t handle_id{ ccl::utils::initial_handle_id_value };
        pid_t remote_pid{ ccl::utils::invalid_pid };
        size_t mem_offset{};
        void* remote_ptr{};
        uint64_t remote_mem_alloc_id{};
        ssize_t remote_context_id{ ccl::utils::invalid_context_id };
        ssize_t remote_device_id{ ccl::utils::invalid_device_id };
        int device_fd{ ccl::utils::invalid_fd };
    };

    void fill_payload(payload_t& payload, size_t buf_idx);
    void fill_remote_handle(const payload_t& payload,
                            ze_ipc_mem_handle_t ipc_handle,
                            const size_t idx,
                            const size_t buf_idx);

    int ipc_to_mem_handle(const ze_ipc_mem_handle_t& ipc_handle,
                          const int dev_id = ccl::utils::invalid_device_id);

    void create_local_ipc_handles();
    int sockets_mode_exchange();
    void common_fd_mode_exchange();
    void pt2pt_fd_mode_exchange();

    bool is_created{};
    bool is_connected{};
    bool is_accepted{};
    bool skip_first_send{};

    int create_server_socket(const std::string& socket_name,
                             struct sockaddr_un* socket_addr,
                             int* addr_len);
    int create_client_socket(const std::string& left_peer_socket_name,
                             struct sockaddr_un* sockaddr_cli,
                             int* len);

    int accept_call(int connect_socket,
                    struct sockaddr_un* socket_addr,
                    int* addr_len,
                    const std::string& socket_name,
                    int& sock);
    int connect_call(int sock,
                     struct sockaddr_un* socket_addr,
                     int addr_len,
                     const std::string& socket_name);

    using mem_info_t = typename std::pair<void*, size_t>;
    mem_info_t get_mem_info(const void* ptr);

    bool sockets_closed = false;
    std::vector<int> opened_sockets_fds;

    void unlink_sockets();
    void close_sockets();

    uint32_t get_remote_device_id(ccl::ze::device_info& info);
    int get_remote_physical_device_fd(const ssize_t remote_device_id);
    int get_handle_idx(ccl_coll_type ctype, int rank_arg);
};

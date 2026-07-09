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

#pragma once

#include <vector>
#include <unordered_map>

#include "utils.h"
#include "occp_commands.h"
#include "coordinator.h"

class occp_client_t : public coordinator_t {
public:
    occp_client_t(rank_t rank,
                  const sockaddr_t& srv_addr,
                  uint32_t io_threads = 2,
                  uint32_t batch_size = 2,
                  uint32_t op_timeout = 120); // sec

    bool exchange_rank_info(rank_t rank,
                            uint32_t comm_size,
                            const rank_info_t& my_info,
                            const ranks_infos_t& ranks_info);
    bool exchange_ofi_config(rank_t rank,
                             uint32_t comm_size,
                             const ofi_endpoint_t& my_config,
                             const ofi_endpoints_t& ranks_config);
    bool rendezvous(rank_t rank, uint32_t comm_size);

public: // coordinator_t
    virtual void on_command(occp_command_t& cmd, occp_t& connection) override; // specific command
    virtual void on_message(const occp_message_t& msg, occp_t& connection) override;

private:
    bool send_to(const sockaddr_storage& dest, const occp_command_t& cmd);

    void on_occp_rank_data(const occp_cmd_rank_data_t& cmd);
    void on_occp_ofi_config(const occp_cmd_ofi_config_t& cmd);
    void on_occp_sync(const occp_cmd_sync_t& cmd);

    void node_bcast(const sockaddr_storage* caddrs, uint32_t comm_size, const occp_command_t& cmd);

    rank_t rank_ = OCCL_INVALID_RANK;
    sockaddr_t occp_srv_;

    struct {
        uint32_t io_threads = 2; // Will be set in constructor
        uint32_t op_timeout = 120; // Will be set in constructor
    } cfg_;

    // enable no busy wait on command completion
    // see wait_command() macro
    template <typename T>
    class cmd_event_t : public T {
    public:
        using T::T; // inherit constructors
        using T::operator=; // inherit assignment operator

        event_t completed_{ false };
    };

    // commands we will receive in our srv socket
    cmd_event_t<occp_cmd_rank_data_t> cmd_comm_data_;
    cmd_event_t<occp_cmd_ofi_config_t> cmd_ofi_config_;
    cmd_event_t<occp_cmd_sync_t> cmd_sync_;
};

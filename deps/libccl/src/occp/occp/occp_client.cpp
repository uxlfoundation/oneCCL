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

#include "occp_client.h"
#include "utils.h" // for VERIFY, LOG_HCL_ERR

#define CLNT_LOG(FMT, ...) OCCL_COORD_LOG("rank: {} " FMT, rank_, ##__VA_ARGS__)
#define CLNT_ERR(FMT, ...) OCCL_COORD_ERR("rank: {} " FMT, rank_, ##__VA_ARGS__)
#define CLNT_INF(FMT, ...) OCCL_COORD_INF("rank: {} " FMT, rank_, ##__VA_ARGS__)
#define CLNT_CRT(FMT, ...) OCCL_COORD_CRT("rank: {} " FMT, rank_, ##__VA_ARGS__)
#define CLNT_WRN(FMT, ...) OCCL_COORD_WRN("rank: {} " FMT, rank_, ##__VA_ARGS__)
#define CLNT_DBG(FMT, ...) OCCL_COORD_DBG("rank: {} " FMT, rank_, ##__VA_ARGS__)

#define wait_command(_CMD, _FMT, ...) \
    if (!_CMD.completed_.wait(cfg_.op_timeout * 1000)) { \
        CLNT_ERR("timeout while waiting for: " _FMT, ##__VA_ARGS__); \
        return false; \
    }

occp_client_t::occp_client_t(rank_t rank,
                             const sockaddr_t& srv_addr,
                             uint32_t io_threads,
                             uint32_t batch_size,
                             uint32_t op_timeout)
        : rank_(rank),
          occp_srv_(srv_addr) {
    comm_id_ = occp_srv_.port(); // use port as comm_id for logging

    // Store config values from constructor parameters
    cfg_.io_threads = io_threads;
    cfg_.op_timeout = op_timeout;

    if (!start(cfg_.io_threads, batch_size)) {
        VERIFY(false, "cannot start occp client");
        return;
    }

    CLNT_INF("{} {} occp_srv: {}. op timeout: {} sec.",
             this,
             srv_.local_addr.str(),
             occp_srv_.str(),
             cfg_.op_timeout);
}

void occp_client_t::node_bcast(const sockaddr_storage* caddrs,
                               uint32_t comm_size,
                               const occp_command_t& cmd) {
    CLNT_LOG("comm_size: {}", comm_size);

    auto const& this_rank_addr = caddrs[rank_];

    FOR_I((int)comm_size) {
        CLNT_LOG("check rank: {} addr: {}", i, sockaddr_t(caddrs[i]).str());
        if (rank_ == i)
            continue;

        if (same_address(caddrs[i], this_rank_addr)) {
            CLNT_LOG("-->{}:{}", i, cmd);
            send_to(caddrs[i], cmd);
        }
    }
}

void occp_client_t::on_occp_rank_data(const occp_cmd_rank_data_t& cmd) {
    if (cmd.param_.rank == OCCP_SERVER) {
        cmd_comm_data_.param_.rank = rank_;
        node_bcast(cmd.payload_.caddrs(), cmd.param_.comm_size, cmd_comm_data_);
    }

    cmd_comm_data_.completed_ = true;
}

void occp_client_t::on_occp_ofi_config(const occp_cmd_ofi_config_t& cmd) {
    if (cmd.param_.rank == OCCP_SERVER) {
        cmd_ofi_config_.param_.rank = rank_;
        node_bcast(cmd.payload_.caddrs(), cmd.param_.comm_size, cmd_ofi_config_);
    }

    cmd_ofi_config_.completed_ = true;
}

void occp_client_t::on_occp_sync(const occp_cmd_sync_t& cmd) {
    if (cmd.param_.rank == OCCP_SERVER) {
        cmd_sync_.param_.rank = rank_;
        node_bcast(cmd.payload_.caddrs(), cmd.param_.comm_size, cmd_sync_);
    }

    cmd_sync_.completed_ = true;
}

bool occp_client_t::rendezvous(rank_t rank, uint32_t comm_size) {
    rank_ = rank;

    CLNT_INF("comm_size: {}", comm_size);

    cmd_sync_.payload_ = { comm_size, false }; // allocate space for caddrs only
    cmd_sync_.completed_ = false;

    occp_cmd_sync_t cmd(occp_param_t<>(rank, comm_size, srv_.local_addr.port()));

    RET_ON_FALSE(send_to(occp_srv_, cmd));

    wait_command(cmd_sync_, "rendezvous");

    CLNT_INF("completed");

    return true;
}

bool occp_client_t::exchange_ofi_config(rank_t rank,
                                        uint32_t comm_size,
                                        const ofi_endpoint_t& my_config,
                                        const ofi_endpoints_t& ranks_config) {
    VERIFY(comm_size == ranks_config.size(), "comm size must be equal to ranks_config size");

    rank_ = rank;

    // we will receive ofi_endpoint_t of all ranks from the server into cmd_ofi_config_
    cmd_ofi_config_.payload_ = { ranks_config.data(), comm_size };
    cmd_ofi_config_.completed_ = false;

    CLNT_INF("comm_size: {}", comm_size);

    // send our rank info to the server and wait for info of all ranks
    occp_cmd_ofi_config_t cmd({ rank, comm_size, srv_.local_addr.port(), my_config });

    RET_ON_FALSE(send_to(occp_srv_, cmd));

    wait_command(cmd_ofi_config_, "receive efi config");

    CLNT_INF("completed");

    return true;
}

bool occp_client_t::exchange_rank_info(rank_t rank,
                                       uint32_t comm_size,
                                       const rank_info_t& my_info,
                                       const ranks_infos_t& ranks_info) {
    VERIFY(comm_size == ranks_info.size(), "comm_size must be equal to ranks_info size");

    rank_ = rank;

    // we will receive ranks_info of all ranks from the server into cmd_comm_data_
    cmd_comm_data_.payload_ = { ranks_info.data(), comm_size };
    cmd_comm_data_.completed_ = false;

    // send our rank info to the server and wait for info of all ranks
    occp_cmd_rank_data_t cmd({ rank, comm_size, srv_.local_addr.port(), my_info });

    CLNT_INF("occp_port: {} comm_size:{} host: {}",
             cmd.param_.port,
             cmd.param_.comm_size,
             my_info.hostname);

    RET_ON_FALSE(send_to(occp_srv_, cmd));

    wait_command(cmd_comm_data_, "receive comm data");

    CLNT_INF("completed");

    return true;
}

bool occp_client_t::send_to(const sockaddr_storage& dest, const occp_command_t& cmd) {
    socket_t socket;

    RET_ON_FALSE(socket.connect(dest, cfg_.op_timeout));

    occp_t occp(socket);

    RET_ON_FALSE(occp.send_command(cmd, cfg_.op_timeout));

    CLNT_LOG("{}", cmd);

    return occp.recv_ack();
}

void occp_client_t::on_command(occp_command_t& cmd, occp_t& connection) {
    close_connection(connection);

    const auto* param = reinterpret_cast<const occp_base_param_t*>(cmd.param());

    CLNT_LOG("rank: {} {}", param->rank, cmd);

    switch (cmd.id()) {
        OCCP_CMD_HANDLER(OCCP_RANK_DATA, occp_cmd_rank_data_t, on_occp_rank_data);
        OCCP_CMD_HANDLER(OCCP_OFI_CONFIG, occp_cmd_ofi_config_t, on_occp_ofi_config);
        OCCP_CMD_HANDLER(OCCP_SYNC, occp_cmd_sync_t, on_occp_sync);
    }
}

void occp_client_t::on_message(const occp_message_t& msg, occp_t& connection) {
    CLNT_INF("{}", msg);

    switch (msg.id) {
        OCCP_MSG_CMD_PAYLOAD_HANDLER(OCCP_RANK_DATA, cmd_comm_data_);
        OCCP_MSG_CMD_PAYLOAD_HANDLER(OCCP_OFI_CONFIG, cmd_ofi_config_);
        OCCP_MSG_CMD_PAYLOAD_HANDLER(OCCP_SYNC, cmd_sync_);

        default:
            CLNT_ERR("Unknown message id: {} from: {}", msg.id, connection->str());
            drop_connection(connection);
            break;
    }
}

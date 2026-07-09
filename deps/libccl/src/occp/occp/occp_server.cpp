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
#include "utils.h"
#include <cmath>
#include <numeric>

#define SRV_LOG OCCL_COORD_LOG
#define SRV_ERR OCCL_COORD_ERR
#define SRV_INF OCCL_COORD_INF
#define SRV_CRT OCCL_COORD_CRT
#define SRV_WRN OCCL_COORD_WRN
#define SRV_DBG OCCL_COORD_DBG

// command factory
template <cmdid_t id, typename... Args>
sp_occp_cmd_t occp_cmd(Args&&... args);

template <>
sp_occp_cmd_t occp_cmd<OCCP_RANK_DATA>(occp_param_t<rank_info_t>&& arg1,
                                       occp_cmd_payload_t<rank_info_t>&& arg2) {
    return std::make_shared<occp_cmd_rank_data_t>(
        std::forward<occp_param_t<rank_info_t>>(arg1),
        std::forward<occp_cmd_payload_t<rank_info_t>>(arg2));
}

template <>
sp_occp_cmd_t occp_cmd<OCCP_SYNC>(occp_param_t<>&& arg1, occp_cmd_payload_t<>&& arg2) {
    return std::make_shared<occp_cmd_sync_t>(std::forward<occp_param_t<>>(arg1),
                                             std::forward<occp_cmd_payload_t<>>(arg2));
}

template <>
sp_occp_cmd_t occp_cmd<OCCP_OFI_CONFIG>(occp_param_t<ofi_endpoint_t>&& arg1,
                                        occp_cmd_payload_t<ofi_endpoint_t>&& arg2) {
    return std::make_shared<occp_cmd_ofi_config_t>(
        std::forward<occp_param_t<ofi_endpoint_t>>(arg1),
        std::forward<occp_cmd_payload_t<ofi_endpoint_t>>(arg2));
}

occp_server_t::occp_server_t(const sockaddr_t& ipaddr,
                             uint32_t io_threads,
                             uint32_t batch_size,
                             uint32_t op_timeout,
                             uint32_t ranks_per_send_thread) {
    // Initialize atomic counter
    cnt_ranks_.store(0);

    // Store config values from constructor parameters
    cfg_.io_threads = io_threads;
    cfg_.op_timeout = op_timeout;
    cfg_.ranks_per_send_thread = ranks_per_send_thread;

    if (!start(io_threads, batch_size, ipaddr)) {
        SRV_CRT("Failed to start coordinator server on {}. ({}: {})",
                ipaddr.str(),
                errno,
                strerror(errno));

        VERIFY(false, "Creating coordinator server ({}) failed", ipaddr.str());
    }

    comm_id_ = srv_.local_addr.port();

    SRV_INF(
        "occp server created at {} (this: {}) io threads: {}, op timeout: {} sec, ranks per send thread: {}",
        srv_.local_addr.str(),
        this,
        cfg_.io_threads,
        cfg_.op_timeout,
        cfg_.ranks_per_send_thread);
}

void occp_server_t::start_session(cmdid_t cmdid, uint32_t comm_size) {
    VERIFY(comm_size != 0, "zero comm group size specified");

    comm_size_ = comm_size;
    cnt_ranks_ = 0;

    //
    // reply's payload always contains array of client addresses
    //
    // allocate all memory here to avoid dynamic allocations later
    //
    // caddrs_ will point inside reply_'s payload, which exists until last send thread exits
    //
    switch (cmdid) {
        case OCCP_RANK_DATA:
            reply_ = occp_cmd<OCCP_RANK_DATA>(occp_param_t<rank_info_t>(comm_size),
                                              occp_cmd_payload_t<rank_info_t>(comm_size));
            caddrs_ = static_cast<occp_cmd_rank_data_t&>(*reply_).payload_.caddrs();
            /* code */
            break;
        case OCCP_OFI_CONFIG:
            reply_ = occp_cmd<OCCP_OFI_CONFIG>(occp_param_t<ofi_endpoint_t>(comm_size),
                                               occp_cmd_payload_t<ofi_endpoint_t>(comm_size));
            caddrs_ = static_cast<occp_cmd_ofi_config_t&>(*reply_).payload_.caddrs();
            /* code */
            break;
        case OCCP_SYNC:
            reply_ = occp_cmd<OCCP_SYNC>(occp_param_t<>(comm_size),
                                         occp_cmd_payload_t<>(comm_size, false));
            caddrs_ = static_cast<occp_cmd_sync_t&>(*reply_).payload_.caddrs();
            break;

        default: break;
    }

    forwarders_.reserve(comm_size_);

    SRV_INF("{}, comm_size: {}", cmdid, comm_size_);

    cmd_id_ = cmdid;
}

void occp_server_t::end_session() {
    init_flag_.reset(new std::once_flag());
    comm_size_ = 0;
    caddrs_ = nullptr;

    forwarders_.clear();

    reply_.reset();

    SRV_INF("{}", cmd_id_);

    cmd_id_ = OCCP_CMD_INVALID;
}

bool occp_server_t::send_to(const sockaddr_storage& addr, const occp_command_t& cmd) {
    sockaddr_t rank_addr(addr);

    socket_t socket;

    RET_ON_FALSE(socket.connect(rank_addr, cfg_.op_timeout));

    occp_t occp(socket);

    RET_ON_FALSE(occp.send_command(cmd, cfg_.op_timeout));

    SRV_INF("--> {}: {}", rank_addr.str(), cmd);

    return occp.recv_ack();
}

// runs in separate thread(s)
void occp_server_t::send_cmd(caddrs_ptr addrs,
                             uint64_t start_index,
                             uint64_t count,
                             sp_occp_cmd_t sp_cmd) {
    SRV_INF("start: {}. count: {} cmd: {}", start_index, count, *sp_cmd);

    while (count--) {
        send_to((*addrs)[start_index++], *sp_cmd);
    }
}

void occp_server_t::waterfall() {
    caddrs_ptr dest_ranks = std::make_shared<caddrs_t>(std::move(forwarders_.caddrs()));

    auto send_threads = static_cast<uint32_t>(
        ceil((double)dest_ranks->size() / (double)cfg_.ranks_per_send_thread));

    auto base = dest_ranks->size() / send_threads;
    auto remainder = dest_ranks->size() % send_threads;

    VERIFY(base > 0,
           "base is zero, dest_ranks: {}, send_threads: {}",
           dest_ranks->size(),
           send_threads);

    SRV_INF("{} to {} ranks using {} threads", *reply_, dest_ranks->size(), send_threads);

    sp_occp_cmd_t sp_cmd = reply_; // keep shared_ptr alive until all threads exit

    end_session();

    // !!!  --- no use of members after this point --- !!!

    uint64_t start_index = 0;

    FOR_I(send_threads) {
        uint64_t ranks_in_thread = i < remainder ? base + 1 : base;

        std::thread([=]() {
            send_cmd(dest_ranks, start_index, ranks_in_thread, sp_cmd);
        }).detach();

        start_index += ranks_in_thread;
    }
}

void occp_server_t::on_occp_rank_data(const occp_cmd_rank_data_t& cmd) {
    *static_cast<occp_cmd_rank_data_t&>(*reply_).payload_.data(cmd.param_.rank) = cmd.param_.data;
}

void occp_server_t::on_occp_ofi_config(const occp_cmd_ofi_config_t& cmd) {
    *static_cast<occp_cmd_ofi_config_t&>(*reply_).payload_.data(cmd.param_.rank) = cmd.param_.data;
}

void occp_server_t::on_command(occp_command_t& cmd, occp_t& connection) {
    SRV_LOG("{}", cmd);

    sockaddr_t remote_addr = connection->remote_addr;
    close_connection(connection);

    const auto* param = reinterpret_cast<const occp_base_param_t*>(cmd.param());

    remote_addr.port(param->port);

    caddrs_[param->rank] = remote_addr;

    if (forwarders_.add_node(remote_addr)) {
        SRV_INF("added new forwarder: rank:{} ({})", param->rank, remote_addr.str());
    }

    uint64_t done = ++cnt_ranks_;

    SRV_INF("rank:{} ({}/{})", param->rank, done, comm_size_);

    switch (cmd.id()) {
        OCCP_CMD_HANDLER(OCCP_RANK_DATA, occp_cmd_rank_data_t, on_occp_rank_data);
        OCCP_CMD_HANDLER(OCCP_OFI_CONFIG, occp_cmd_ofi_config_t, on_occp_ofi_config);
    }

    if (done == comm_size_) {
        SRV_INF("all ranks have sent their data. (comm_size: {})", comm_size_);

        waterfall();
    }
}

void occp_server_t::on_message(const occp_message_t& msg, occp_t& connection) {
    SRV_LOG("{}", msg);

    const auto* param = reinterpret_cast<const occp_base_param_t*>(msg.param);
    if (cmd_id_ == OCCP_CMD_INVALID) // no current session
    {
        std::call_once(*init_flag_, [this, param, &msg]() {
            start_session(msg.id, param->comm_size);
        });
    }

    VERIFY(msg.id == cmd_id_ && param->comm_size == comm_size_ && param->rank < (int)comm_size_,
           "msg_id: {}({}) comm_size: {}({}) rank:{}({})",
           msg.id,
           cmd_id_,
           param->comm_size,
           comm_size_,
           param->rank,
           connection->remote_addr.str());

    switch (msg.id) {
        OCCP_MSG_HANDLER(OCCP_RANK_DATA, occp_cmd_rank_data_t);
        OCCP_MSG_HANDLER(OCCP_OFI_CONFIG, occp_cmd_ofi_config_t);
        OCCP_MSG_HANDLER(OCCP_SYNC, occp_cmd_sync_t);

        default:
            SRV_ERR("unknown msg id:{} remote:{} ", msg.id, connection->remote_addr.str());
            drop_connection(connection);
            break;
    }
}

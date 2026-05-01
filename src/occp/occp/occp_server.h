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

#include <vector>
#include <unordered_set>
#include <mutex>
#include <functional>

#include "futex.h"
#include "coordinator.h"
#include "occp_commands.h"

using sp_occp_cmd_t = std::shared_ptr<occp_command_t>;

using lock_t = std::mutex;
using locker_t = std::unique_lock<lock_t>;

//
// Node - ranks with the same IP
//
// Clients → Server: OCCP_RANK_DATA (with IP address)
//
// first rank to contact server from each IP - forwarder
//
// R0 ──┐
// R1 ──┤
// R2 ──┤
// R3 ──┤ Node A [10.0.0.2] → Server identifies R0 as forwarder
// R4 ──┤
// R5 ──┤
// R6 ──┤
// R7 ──┘
//
// R8 ──┐
// R9 ──┤
// R10 ─┤
// R11 ─┤ Node B [10.0.0.3] → Server identifies R10 as forwarder
// R12 ─┤
// R13 ─┤
// R14 ─┤
// R15 ─┘
//
// R16 ─┐
// R17 ─┤
// R18 ─┤
// R19 ─┤ Node C [10.0.0.4] → Server identifies R20 as forwarder
// R20 ─┤
// R21 ─┤
// R22 ─┤
// R23 ─┘
//
//  Server talks only to forwarders
//
//  when a rank(forwarder) receives reply from server after the handshake, it contains all the ranks addresses.
//  so it can deduce other ranks in the same node and use this to forward messages to them.
//
//  assumption: send data is identical for all ranks.
//
//                   "Waterfall"
//
//                     SERVER
//                       │
//                  [broadcast()]
//                       │
//     ┌─────────────────┼─────────────────┐
//     │                 │                 │
//     v                 v                 v
// FORWARDER R0      FORWARDER R10     FORWARDER R20
// Node A [10.0.0.2] Node B [10.0.0.3] Node C [10.0.0.4]
//     │                 │                 │
//     ├─► R1            ├─► R8            ├─► R16
//     ├─► R2            ├─► R9            ├─► R17
//     ├─► R3            ├─► R11           ├─► R18
//     ├─► R4            ├─► R12           ├─► R19
//     ├─► R5            ├─► R13           ├─► R21
//     ├─► R6            ├─► R14           ├─► R22
//     └─► R7            └─► R15           └─► R23
//
//
// Server only sends to forwarders (R0, R10, R20)
// Each forwarder distributes to other ranks on the same node
//
// TODO: after the handshake, client to server messages can be routed through forwarders (aggregate all the node data)
//       less connections but more data - does it make sense to implement?
//
//
class occp_server_t : public coordinator_t {
private:
    struct {
        uint32_t io_threads = 4; // Will be set in constructor
        uint32_t op_timeout = 120; // Will be set in constructor
        uint32_t ranks_per_send_thread = 8; // Will be set in constructor
    } cfg_;

    using once_flag_ptr = std::unique_ptr<std::once_flag>;

    once_flag_ptr init_flag_ = once_flag_ptr(new std::once_flag());
    cmdid_t cmd_id_ = OCCP_CMD_INVALID; // current command id being processed
    uint32_t comm_size_ = 0;
    counter_t cnt_ranks_; // atomic counter. number of ranks that have sent current command
        // (when cnt_ranks_ == comm_size_ => no more connections expected)

    sockaddr_storage* caddrs_ =
        nullptr; // array of comm_size_ client addresses (points  inside reply_ payload)
    sp_occp_cmd_t reply_;

    class {
    private:
        struct sockaddr_storage_hash {
            std::size_t operator()(const sockaddr_storage& addr) const {
                const unsigned char* data = reinterpret_cast<const unsigned char*>(&addr);
                std::size_t hash = 0;
                for (size_t i = 0; i < sizeof(sockaddr_storage); ++i) {
                    hash = hash * 31 + data[i];
                }
                return hash;
            }
        };

        using saddrs_t = std::unordered_set<sockaddr_storage, sockaddr_storage_hash>;

        saddrs_t set_;
        caddrs_t caddrs_;
        lock_t lock_;

    public:
        caddrs_t& caddrs() {
            return caddrs_;
        }

        void clear() {
            set_.clear();
            caddrs_.clear();
        }

        void reserve(size_t n) {
            caddrs_.reserve(n);
            set_.reserve(n);
        }

        bool add_node(const sockaddr_storage& addr) {
            sockaddr_t node_addr = addr;
            node_addr.port(0); // ignore port

            locker_t locker(lock_);

            if (set_.insert(node_addr).second) {
                caddrs_.push_back(addr);
                return true;
            }

            return false;
        }

    } forwarders_;

    void start_session(cmdid_t cmd, uint32_t comm_size);
    void end_session();

    bool send_to(const sockaddr_storage& addr, const occp_command_t& cmd);

    using caddrs_ptr = std::shared_ptr<caddrs_t>;

    void send_cmd(caddrs_ptr addrs, uint64_t start_index, uint64_t count, sp_occp_cmd_t sp_cmd);

    void waterfall();

protected: // command handlers
    void on_occp_rank_data(const occp_cmd_rank_data_t& cmd);
    void on_occp_ofi_config(const occp_cmd_ofi_config_t& cmd);
    void on_occp_sync(const occp_cmd_sync_t& cmd);

public:
    occp_server_t(const sockaddr_t& addr = {},
                  uint32_t io_threads = 4,
                  uint32_t batch_size = 8,
                  uint32_t op_timeout = 120,
                  uint32_t ranks_per_send_thread = 8);

    virtual void on_command(occp_command_t& cmd, occp_t& connection) override; // specific command
    virtual void on_message(const occp_message_t& msg, occp_t& connection) override;
};

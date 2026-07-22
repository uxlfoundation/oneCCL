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

#include "protocol.h"
#include "payload.h"
#include "types.h"

const char* cmd2str(cmdid_t id);

constexpr cmdid_t OCCP_CMD_INVALID = 0;
constexpr cmdid_t OCCP_BASE_CMD_ID = 100;

// Unified payload container. Template parameter T is the element type when treating
// the first mem_block as a typed object/array. For composite command payloads that
// manually push multiple blocks (like address arrays + data arrays) we still expose
// add() via protected access for derived helpers.
template <typename T = uint8_t>
class occp_payload_t {
protected:
    payload_t iov_;
    size_t total_size_ = 0;

    template <typename... Args>
    const mem_block_t& add(Args&&... args) {
        iov_.emplace_back(std::forward<Args>(args)...);
        total_size_ += iov_.back().size();
        return iov_.back();
    }

public:
    occp_payload_t() = default;
    occp_payload_t(const occp_payload_t&) = default;
    occp_payload_t& operator=(const occp_payload_t&) = default;
    occp_payload_t(occp_payload_t&&) = default;
    occp_payload_t& operator=(occp_payload_t&&) = default;

    explicit occp_payload_t(size_t size) {
        add(size);
    }

    occp_payload_t& operator=(size_t size) {
        iov_.clear();
        total_size_ = 0;
        add(size);
        return *this;
    }

    void* ptr() const {
        return iov_.empty() ? nullptr : iov_[0].ptr();
    }

    operator const payload_t&() const {
        return iov_;
    }
    operator T*() const {
        return static_cast<T*>(ptr());
    }
    operator const T*() const {
        return static_cast<const T*>(ptr());
    }
    operator T&() const {
        return *static_cast<T*>(ptr());
    }
    operator const T&() const {
        return *static_cast<const T*>(ptr());
    }

    size_t size() const {
        return total_size_;
    }
};

template <cmdid_t ID, class PARAM = rank_t, class PAYLOAD = occp_payload_t<>>
class _occp_command_t : public occp_command_t {
public:
    using occp_command_t::operator=;

    _occp_command_t() = default;

    explicit _occp_command_t(const PARAM& p) : param_(p) {}
    // Removed unused overloads that accepted only PAYLOAD (lvalue/rvalue) and (PARAM,const PAYLOAD&)
    // to minimize constructor set. Keep only PARAM+PAYLOAD&& which is used by server factory.

    explicit _occp_command_t(const PARAM& p, PAYLOAD&& payload)
            : param_(p),
              payload_(std::move(payload)) {}
    // Message-based construction handled via default ctor + operator=(occp_message_t&)

    static_assert(sizeof(PARAM) <= OCCP_MAX_PARAM_SIZE, "PARAM size exceeds OCCP_MAX_PARAM_SIZE");

    virtual const char* name() const override {
        return cmd2str(ID);
    }

    virtual cmdid_t id() const override {
        return ID;
    }
    virtual void* param() const override {
        return (void*)&param_;
    }
    virtual size_t param_size() const override {
        return sizeof(PARAM);
    }
    virtual size_t payload_size() const override {
        return payload_.size();
    }

    virtual const payload_t& payload() const override {
        return payload_;
    }

    PARAM param_;
    PAYLOAD payload_;
};

// commands and params definitions
#pragma pack(push)
#pragma pack(1)

struct occp_base_param_t {
    rank_t rank = OCCL_INVALID_RANK;
    uint32_t comm_size = 0; // communicator size
    uint16_t port = 0; // port for incoming connections

    occp_base_param_t() = default;
    occp_base_param_t(rank_t rank, uint32_t comm_size, uint16_t port)
            : rank(rank),
              comm_size(comm_size),
              port(port) {}
};

template <typename T = bool>
struct occp_param_t : public occp_base_param_t {
    T data = {};

    occp_param_t() = default;
    occp_param_t(uint32_t comm_size) : occp_base_param_t(OCCL_INVALID_RANK, comm_size, 0) {}
    occp_param_t(rank_t rank, uint32_t comm_size, uint16_t port, const T& data = {})
            : occp_base_param_t(rank, comm_size, port),
              data(data) {}
};
#pragma pack(pop)

template <class P = uint8_t>
class occp_cmd_payload_t : public occp_payload_t<> {
public:
    using occp_payload_t<>::occp_payload_t;
    using occp_payload_t<>::operator=;
    // Removed unused default constructor: all sites provide either (count,bool) or (ptr,count).

    occp_cmd_payload_t(const P* data, uint32_t count) {
        add(count * sizeof(sockaddr_storage));
        add(data, count * sizeof(P));
    }

    occp_cmd_payload_t(uint32_t count, bool data = true) {
        add(count * sizeof(sockaddr_storage));
        if (data) {
            add(count * sizeof(P));
        }
    }

    sockaddr_storage* caddrs() const {
        return iov_.size() > 0 ? static_cast<sockaddr_storage*>(iov_[0].ptr()) : nullptr;
    }
    P* data(uint64_t index = 0) const {
        return iov_.size() > 1 ? &static_cast<P*>(iov_[1].ptr())[index] : nullptr;
    }
};

constexpr auto OCCP_RANK_DATA = OCCP_BASE_CMD_ID + 10;
using occp_cmd_rank_data_t =
    _occp_command_t<OCCP_RANK_DATA, occp_param_t<rank_info_t>, occp_cmd_payload_t<rank_info_t>>;

// sync (rendezvous)
constexpr auto OCCP_SYNC = OCCP_BASE_CMD_ID + 20;
using occp_cmd_sync_t = _occp_command_t<OCCP_SYNC, occp_param_t<>, occp_cmd_payload_t<>>;

// efi endpoint configuration
constexpr auto OCCP_OFI_CONFIG = OCCP_BASE_CMD_ID + 30;
using occp_cmd_ofi_config_t = _occp_command_t<OCCP_OFI_CONFIG,
                                              occp_param_t<ofi_endpoint_t>,
                                              occp_cmd_payload_t<ofi_endpoint_t>>;

//
// To add a new command:
//
// 1) Add the command ID, for example, XXX:
//
//       constexpr cmdid_t OCCP_XXX = OCCP_BASE_CMD_ID + 80;
//
// 1.1) update cmd2str() in occp_commands.cpp
//
//
//
// 2) If the command has a parameter (less than 512 bytes), define a struct for it and use it in _occp_command_t:
//
//       struct occp_xxx_param_t
//       {
//          ...
//       };
//
//       using occp_cmd_xxx_t = _occp_command_t<OCCP_XXX, occp_xxx_param_t>;
//
//
// 3) Define a handler for the command in occp_server_t and/or occp_client_t:
//
//       void on_occp_xxx(occp_cmd_xxx_t& cmd) {}
//
//
// 4) Add a call to the handler in the on_command() method in occp_server_t and/or occp_client_t:
//
//       OCCP_CMD_HANDLER(OCCP_XXX, occp_cmd_xxx_t, on_occp_xxx);
//
//
// 5.1) If the command is WITHOUT PAYLOAD, add the following to on_message():
//
//       OCCP_MSG_HANDLER(OCCP_XXX, occp_cmd_xxx_t);
//
//
// 5.2) If the command is WITH PAYLOAD and it's size is unknown at compile time add the following to on_message():
//
//      OCCP_MSG_PAYLOAD_HANDLER(OCCP_XXX, occp_cmd_xxx_t);
//
//      Remember to delete the command in the handler:
//      void on_occp_xxx(occp_cmd_xxx_t& cmd)
//      {
//         ...
//         delete &cmd;
//      }
//
//
// 5.3) Add call to handle new message id in occp_client_t::on_message() case switch.
//

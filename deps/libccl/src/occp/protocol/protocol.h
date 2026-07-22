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

#include <cstdint>
#include <cstddef>
#include <array>
#include "utils.h"
#include "payload.h"

// OneCCL Coordinator Protocol (OCCP) definitions

constexpr uint32_t OCCP_VERSION = 0xA0010001;
constexpr uint32_t OCCP_FOOTER = 0xCC0CC0CC;

// packet types
constexpr uint32_t OCCP_PKT_ACK = 0x0CCC001A;
constexpr uint32_t OCCP_PKT_DATA = 0x0CCC002D;

using magic_t = std::array<char, 4>;
#define OCCP_MAGIC \
    { 'O', 'C', 'C', 'P' }

#pragma pack(push, 1) // ensure 1-byte alignment for network packets
struct occp_header_t {
    char magic[4] = OCCP_MAGIC;
    uint32_t version = OCCP_VERSION;
    uint32_t type = OCCP_PKT_DATA;
    uint32_t footer = OCCP_FOOTER;
};

constexpr uint32_t OCCP_MAX_PARAM_SIZE = 512; //bytes

using cmdid_t = uint32_t;

struct occp_message_t {
    cmdid_t id = 0;
    uint8_t param[OCCP_MAX_PARAM_SIZE] = {};
    uint64_t payload_size = 0;
};

class occp_command_t;
struct occp_packet_t {
    occp_header_t hdr;
    occp_message_t msg;

    // build packet from user supplied command
    occp_packet_t& operator=(const occp_command_t& cmd);
};

#pragma pack(pop)

// command prototype
class occp_command_t {
protected:
    occp_command_t() = default;

public:
    virtual cmdid_t id() const {
        return 0;
    };
    virtual void* param() const {
        return nullptr;
    }
    virtual size_t param_size() const {
        return 0;
    }

    virtual const payload_t& payload() const {
        static payload_t p;
        return p;
    };
    virtual size_t payload_size() const {
        return 0;
    };

    // build command from network message
    occp_command_t& operator=(const occp_message_t& msg);

    virtual const char* name() const {
        return "empty";
    };
    virtual ~occp_command_t() = default;
};

#include <ostream>
std::ostream& operator<<(std::ostream& out, const occp_header_t& hdr);
HLLOG_DEFINE_OSTREAM_FORMATTER(occp_header_t);
std::ostream& operator<<(std::ostream& out, const occp_message_t& msg);
HLLOG_DEFINE_OSTREAM_FORMATTER(occp_message_t);
std::ostream& operator<<(std::ostream& out, const occp_packet_t& p);
HLLOG_DEFINE_OSTREAM_FORMATTER(occp_packet_t);
std::ostream& operator<<(std::ostream& out, const occp_command_t& c);
HLLOG_DEFINE_OSTREAM_FORMATTER(occp_command_t);

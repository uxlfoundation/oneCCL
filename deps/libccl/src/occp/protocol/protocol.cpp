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

#include "protocol.h"
#include "utils.h" // for VERIFY
#include <cstring>

// build network packet from user supplied buffer  i.e. send
occp_packet_t& occp_packet_t::operator=(const occp_command_t& cmd) {
    hdr.type = OCCP_PKT_DATA;
    msg.id = cmd.id();

    std::memcpy(msg.param, cmd.param(), cmd.param_size());
    msg.payload_size = cmd.payload_size();

    return (*this);
}

// from network packet to user supplied buffer i.e. recv
occp_command_t& occp_command_t::operator=(const occp_message_t& msg) {
    VERIFY(msg.id == id(), "invalid msg.id: {} != {}", msg.id, id());

    std::memcpy(param(), msg.param, param_size());
    return (*this);
}

std::ostream& operator<<(std::ostream& out, const occp_header_t& hdr) {
    const magic_t& magic = *(magic_t*)hdr.magic;

    out << "header_t[" << magic[0] << magic[1] << magic[2] << magic[3];

    out << std::hex << std::uppercase;
    out << " " << hdr.version << " " << hdr.type << " " << hdr.footer << "]";
    out << std::dec << std::nouppercase;

    return out;
}

std::ostream& operator<<(std::ostream& out, const occp_message_t& msg) {
    return out << "msg[" << msg.id << ", " << msg.payload_size << "]";
}

std::ostream& operator<<(std::ostream& out, const occp_packet_t& p) {
    return out << "packet(" << p.hdr << " " << p.msg << ")";
}

std::ostream& operator<<(std::ostream& out, const occp_command_t& c) {
    out << "cmd[" << c.name() << "(" << c.id() << "), " << c.param_size();
    if (c.payload_size()) {
        out << ", payload: " << c.payload_size() << " @ " << c.payload()[0].ptr();
    }
    out << "]";
    return out;
}

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
#include "occp.h"

#define OCCP_LOG COORD_LOG

void occp_t::set_transport(socket_io_t& s) {
    OCCP_LOG("{}", s);

    transport_ = &s;
    s.io_notify_ = this;
}

bool occp_t::send_command(const occp_command_t& cmd) {
    tx_ = cmd;
    return send_header();
}

bool occp_t::send_command(const occp_command_t& cmd, uint32_t timeout_sec) {
    RET_ON_FALSE(send_command(cmd));

    return wait_condition(COND(tx_.completed), timeout_sec, cmd.name());
}

bool occp_t::send_header() {
    tx_.state = header;
    return transport_->send(tx_, sizeof(occp_packet_t));
}

bool occp_t::send_payload() {
    tx_.state = payload;
    return transport_->send(tx_.payload.ptr(), tx_.payload.size());
}

void occp_t::on_send(const packet_t&, socket_base_t&) {
    if (tx_.state == header) // header send complete
    {
        if (tx_.packet.msg.payload_size > 0) {
            tx_.payload.next(true);

            send_payload();
            return;
        }
    }
    else if (tx_.state == payload && tx_.payload.next(false)) {
        send_payload();
        return;
    }

    tx_.payload.reset(); // reset payload for next command
    tx_.completed = true;
}

bool occp_t::send_ack() {
    OCCP_LOG("");

    tx_.state = ack;
    tx_.completed = false;
    tx_.packet.hdr.type = OCCP_PKT_ACK;

    return transport_->send(tx_, sizeof(occp_header_t));
}

bool occp_t::recv_ack() {
    OCCP_LOG("");

    rx_.cmd = nullptr;
    rx_.state = ack;

    return transport_->recv(rx_, sizeof(occp_header_t));
}

bool occp_t::recv_header() {
    OCCP_LOG("");

    rx_.state = header;

    return transport_->recv(rx_, sizeof(occp_packet_t));
}

bool occp_t::recv_payload() {
    OCCP_LOG("{}: {}", rx_.cmd->id(), transport_->str());

    VERIFY(rx_.payload.ptr(), "null payload");

    rx_.state = payload;

    return transport_->recv(rx_.payload.ptr(), rx_.payload.size());
}

bool occp_t::receive_command(occp_command_t& cmd) {
    OCCP_LOG("{}: {}", cmd, transport_->str());

    rx_ = cmd;

    return recv_header();
}

bool occp_t::receive() {
    OCCP_LOG("{}", transport_->str());

    rx_.cmd = nullptr;

    return recv_header();
}

bool occp_t::receive_payload(occp_command_t& cmd) {
    OCCP_LOG("{}", cmd.id());

    rx_ = cmd;

    rx_.payload.next(true);

    return recv_payload();
}

bool occp_t::inspect_header() {
    static constexpr magic_t occp_magic = magic_t(OCCP_MAGIC);

    const auto& hdr = rx_.packet.hdr;
    const auto& magic = *(magic_t*)hdr.magic;
    //
    // check packet signature
    //
    return (magic == occp_magic) && (hdr.version <= OCCP_VERSION) && (hdr.footer == OCCP_FOOTER) &&
           (rx_.state != ack || (hdr.type == OCCP_PKT_ACK));
}

void occp_t::on_recv(const packet_t&, socket_base_t& s) {
    OCCP_LOG("{} {} state: {} {}", s, rx_.packet, rx_.state, rx_.cmd ? rx_.cmd->name() : "");

    if (rx_.state == payload) {
        if (rx_.payload.next(false)) {
            // payload is not complete, wait for more data
            recv_payload();
            return;
        }

        // received full command with payload (header + message + payload)
        rx_.payload.reset(); // reset payload for next command
        notify_->on_command(*rx_.cmd, (*this));
        return;
    }

    // state  == header, ack
    if (!inspect_header()) {
        notify_->on_error(rx_.cmd, rx_.packet, (*this), "invalid header");
        return;
    }

    if (rx_.state == ack)
        return;

    // state == header
    if (!rx_.cmd) {
        // no user supplied cmd, so protocol was requested to recv any
        notify_->on_message(rx_.packet.msg, (*this));
        return;
    }

    // user did asked for specific command
    if (rx_.cmd->id() != rx_.packet.msg.id) {
        // but different one has arrived
        notify_->on_error(rx_.cmd, rx_.packet, (*this), "unexpected message id");
        return;
    }

    // copy command's param to user's buffer
    *(rx_.cmd) = rx_.packet.msg;

    bool error = rx_.packet.msg.payload_size > rx_.cmd->payload_size();
    if (rx_.payload.size() > 0 && !error) {
        rx_.payload.next(true);
        // we can receive payload
        recv_payload();
        return;
    }

    // no payload expected. received full command
    const char* reason = "payload size mismatch";
    if (error) {
        // network packet payload has more data than user expects
        notify_->print_error(rx_.cmd, rx_.packet, (*this)->str(), reason);
    }
    notify_->on_command(*rx_.cmd, (*this));
    if (error) {
        notify_->on_error(rx_.cmd, rx_.packet, (*this), reason);
    }
}

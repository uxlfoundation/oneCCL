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
#include "protocol.h"
#include "socket.h"

class occp_t;

class occp_notify_t {
public:
    // any requested command i.e. when you call connection.receive() it ends here
    virtual void on_message(const occp_message_t& /*msg*/, occp_t& /*connection*/) _DEF_IMPL_;

    // command without payload     connection.receive_command(cmd) ends here
    virtual void on_command(occp_command_t& /*cmd*/, occp_t& /*connection*/) _DEF_IMPL_;

    // accepted new connection (start handshake etc...)
    virtual void on_connect(occp_t& /*connection*/) _DEF_IMPL_;

    // protocol errors
    virtual void on_error(occp_command_t*, const occp_packet_t&, occp_t&, const char* /*reason*/)
        _DEF_IMPL_;

    virtual void print_error(occp_command_t*,
                             const occp_packet_t&,
                             const std::string&,
                             const char* /*reason*/) _DEF_IMPL_;
};

// protocol endpoint. will fire notify events on network packet parsing.

class occp_t : public socket_io_notify_t, public occp_notify_t {
protected:
    enum state_e { header, payload, ack };

    class payload_manager_t {
    private:
        payload_t iov_;
        int32_t current_ = -1;
        size_t total_size_ = 0;

        bool in_progress() const noexcept {
            return current_ >= 0 && current_ < static_cast<int>(iov_.size());
        }

    public:
        void reset() {
            current_ = -1;
            total_size_ = 0;
            iov_.clear();
        }

        bool next(bool start = false) {
            if (total_size_ == 0)
                return false;

            start ? current_ = 0 : current_++;
            return (current_ < (int)iov_.size());
        }

        void* ptr() const {
            if (total_size_ == 0)
                return nullptr;

            return in_progress() ? iov_[current_] : iov_[0];
        }

        size_t size() const {
            return in_progress() ? (size_t)iov_[current_] : total_size_;
        }

        payload_manager_t(const payload_t& payload, size_t total_size)
                : iov_(payload),
                  total_size_(total_size) {}
        payload_manager_t(payload_t&& payload, size_t total_size)
                : iov_(std::move(payload)),
                  total_size_(total_size) {}

        payload_manager_t() = default;
        //payload_manager_t() = default;

        //auto& operator=(const payload_t& payload)
    };

    struct operation_descriptor // operation (tx/rx) descriptor
    {
        state_e state; // what we are expecting/sending (header - payload - ack)
        occp_packet_t packet; // packet being sent/received
        payload_manager_t payload; // payload being sent/received

        operator void*() {
            return &packet;
        }
    };

    struct tx_descriptor_t : public operation_descriptor // TX - current transmit data
    {
        bool completed = false;
        tx_descriptor_t& operator=(const occp_command_t& _cmd) {
            completed = false;
            packet = _cmd;
            payload = { _cmd.payload(), _cmd.payload_size() };
            return (*this);
        }
    } tx_;

    struct rx_descriptor_t : public operation_descriptor // RX - current receive data
    {
        occp_command_t* cmd;
        rx_descriptor_t& operator=(occp_command_t& _cmd) {
            cmd = &_cmd;
            payload = { _cmd.payload(), _cmd.payload_size() }; // copy payload descriptor
            return (*this);
        }
    } rx_;

protected:
    socket_io_t* transport_ = nullptr;

public: // socket io notify
    virtual void on_recv(const packet_t& p, socket_base_t& s) override;
    virtual void on_send(const packet_t& p, socket_base_t& s) override;

public:
    operator socket_io_t&() {
        return *transport_;
    }
    socket_io_t* operator->() {
        return transport_;
    }

protected:
    void set_transport(socket_io_t& s);

    bool inspect_header();

    bool send_header();
    bool send_payload();
    bool recv_header();
    bool recv_payload();

public:
    occp_notify_t* notify_ = this;

    occp_t() = default;
    occp_t(socket_io_t& s) {
        set_transport(s);
    }
    occp_t(socket_io_t& s, occp_notify_t& n) : notify_(&n) {
        set_transport(s);
    }

    virtual ~occp_t() = default;

    bool send_command(const occp_command_t& cmd);
    bool send_command(const occp_command_t& cmd, uint32_t timeout_sec);

    bool receive(); // any command

    bool receive_command(occp_command_t& cmd); // recv specific command
    bool receive_payload(occp_command_t& cmd);

    bool send_ack();
    bool recv_ack();
};

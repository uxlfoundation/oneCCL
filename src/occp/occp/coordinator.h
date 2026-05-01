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
#include "acceptor.h"
#include "occp.h"
#include "types.h"

class coordinator_t : public untransferable_t, public socket_op_notify_t, public occp_notify_t {
protected:
    class xsocket_t : public socket_io_t {
    private:
        occp_t* owner_ = nullptr;

    public:
        bool marked = false; // marked for disconnect

        xsocket_t(socketfd_t s, socket_op_notify_t& n, asio_t* a) : socket_io_t(s, n, a) {
            set_non_blocking();
        }
        xsocket_t& operator=(occp_t& p) {
            owner_ = &p;
            return (*this);
        }
        occp_t* operator->() const {
            return owner_;
        }
        operator occp_t&() {
            return *owner_;
        }
    };

protected:
    comm_t comm_id_ = OCCL_INVALID_COMM;

protected:
    asio_t asio_;
    acceptor_t srv_{ *this, &asio_ }; // server socket

protected: // socket_op_notify_t
    virtual void on_error(socket_base_t& s) override;
    virtual void on_accept(socket_base_t& s, int new_socket_fd) override; // srv new connection
    virtual void on_disconnect(socket_base_t& s) override;

protected:
    virtual xsocket_t& create_socket(socketfd_t s) {
        return *(new xsocket_t(s, *this, &asio_));
    }
    virtual void destroy_socket(xsocket_t& s) {
        delete &s;
    }

    virtual occp_t& create_connection(xsocket_t& s) {
        return *(new occp_t(s, *this));
    }
    virtual void destroy_connection(occp_t& c) {
        delete &c;
    }
    virtual void close_connection(occp_t& c); // gracefully
    virtual void drop_connection(occp_t& c);

protected: // occp_notify_t
    virtual void on_connect(occp_t& connection) override {
        connection.receive();
    }
    virtual void on_error(occp_command_t* cmd,
                          const occp_packet_t& packet,
                          occp_t& connection,
                          const char* reason = "") override;

    virtual void print_error(occp_command_t* cmd,
                             const occp_packet_t& packet,
                             const std::string& str,
                             const char* reason) override;

public:
    coordinator_t() = default;
    virtual ~coordinator_t() {
        stop();
    };

    coordinator_t(const coordinator_t&) = delete;
    coordinator_t(coordinator_t&&) = delete;
    coordinator_t& operator=(const coordinator_t&) = delete;
    coordinator_t& operator=(coordinator_t&&) = delete;

    const acceptor_t* operator->() {
        return &srv_;
    }

    bool start(uint32_t io_threads, uint32_t batch_size, const sockaddr_t& addr = sockaddr_t());

    bool stop();
};

// handler for command
#define OCCP_CMD_HANDLER(CMD_ID, CMD_TYPE, HANDLER) \
    case CMD_ID: HANDLER(static_cast<CMD_TYPE&>(cmd)); break

// auto cmd on stack, no payload
#define OCCP_MSG_HANDLER(MSG_ID, CMD_TYPE) \
    case MSG_ID: { \
        CMD_TYPE command; \
        command = msg; \
        on_command(command, connection); \
        break; \
    }

// cmd with payload. cmd variable provided by user (cmd contains already allocated payload )
#define OCCP_MSG_CMD_PAYLOAD_HANDLER(MSG_ID, CMD) \
    case MSG_ID: { \
        CMD = msg; \
        connection.receive_payload(CMD); \
        break; \
    }

#define OCCL_COORD_LOG(FMT, ...) COORD_LOG("[comm: ", comm_id_, "] " FMT, ##__VA_ARGS__)
#define OCCL_COORD_DBG(FMT, ...) COORD_DBG("[comm: ", comm_id_, "] " FMT, ##__VA_ARGS__)
#define OCCL_COORD_INF(FMT, ...) COORD_INF("[comm: ", comm_id_, "] " FMT, ##__VA_ARGS__)
#define OCCL_COORD_WRN(FMT, ...) COORD_WRN("[comm: ", comm_id_, "] " FMT, ##__VA_ARGS__)
#define OCCL_COORD_ERR(FMT, ...) COORD_ERR("[comm: ", comm_id_, "] " FMT, ##__VA_ARGS__)
#define OCCL_COORD_CRT(FMT, ...) COORD_CRT("[comm: ", comm_id_, "] " FMT, ##__VA_ARGS__)

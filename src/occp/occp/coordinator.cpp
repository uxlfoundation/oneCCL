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
#include "coordinator.h"

bool coordinator_t::start(uint32_t io_threads, uint32_t batch_size, const sockaddr_t& addr) {
    RET_ON_FALSE(asio_.start(io_threads, batch_size));
    RET_ON_FALSE(srv_.listen(addr));
    RET_ON_FALSE(asio_.arm_monitor(srv_));

    return true;
}

bool coordinator_t::stop() {
    asio_.stop();
    asio_.remove(srv_);
    srv_.close_socket();

    return true;
}

void coordinator_t::on_error(socket_base_t& s) {
    OCCL_COORD_ERR("({}){}. {}", errno, strerror(errno), s);

    if (s.fd == srv_.fd) {
        return;
    }

    on_disconnect(s);
}

#define occp2sock(c) (static_cast<xsocket_t&>(static_cast<socket_io_t&>(c)))

void coordinator_t::close_connection(occp_t& c) {
    c.send_ack();

    xsocket_t& xs = occp2sock(c);

    xs.marked = true;
    xs.arm_monitor();
}

void coordinator_t::drop_connection(occp_t& c) {
    xsocket_t& xs = occp2sock(c);

    xs.marked = true;
    on_disconnect(xs);
}

void coordinator_t::on_disconnect(socket_base_t& s) {
    OCCL_COORD_LOG("{}", s);

    xsocket_t& xs = static_cast<xsocket_t&>(s);
    occp_t& conn = static_cast<occp_t&>(xs);

    if (!xs.marked) {
        OCCL_COORD_ERR("peer disconnected {}", s);
    }

    asio_.remove(xs);
    xs.close();

    destroy_connection(conn);
    destroy_socket(xs);
}

void coordinator_t::on_accept(socket_base_t& s, int new_socket_fd) {
    xsocket_t& xs = create_socket(new_socket_fd);
    occp_t& conn = create_connection(xs);

    xs = conn; // bind transport with protocol

    OCCL_COORD_LOG("{} accepted {}", s, xs.str());

    conn.notify_->on_connect(conn);
}

void coordinator_t::on_error(occp_command_t* cmd,
                             const occp_packet_t& packet,
                             occp_t& connection,
                             const char* reason) {
    print_error(cmd, packet, connection->str(), reason);

    drop_connection(connection);
}

void coordinator_t::print_error(occp_command_t* cmd,
                                const occp_packet_t& packet,
                                const std::string& str,
                                const char* reason) {
    if (cmd) {
        OCCL_COORD_ERR("expected {} .{} : [{}]. {}", *cmd, packet, str, reason);
    }
    else {
        OCCL_COORD_LOG("{} : [{}]. {}", packet, str, reason);
    }
}

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

#include "socket.h"

#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <fcntl.h>

// ===============================================================================

#define SOCKET_LOG(FMT, ...) COORD_LOG("{} " FMT, str(), ##__VA_ARGS__)

socket_base_t::socket_base_t(int socket_fd) : socket_(socket_fd) {
    get_info();
}

bool socket_base_t::create(sa_family_t domain, int sock_type) {
    if (socket_ != INVALID_SOCKET) {
        return false;
    }

    socket_ = socket(domain, sock_type, 0);
    return socket_ != INVALID_SOCKET;
}

bool socket_base_t::create(const sockaddr_t& addr) {
    return create((sa_family_t)addr);
}

bool socket_base_t::get_info() {
    socklen_t addr_len = (socklen_t)local_;

    RET_ON_ERR(getsockname(socket_, local_, &addr_len));
    RET_ON_ERR(getpeername(socket_, remote_, &addr_len));

    return true;
}

bool socket_base_t::set_linger(bool set, uint32_t seconds) {
    linger so_linger = { set ? 1 : 0, (int)seconds };

    RET_ON_ERR(setsockopt(socket_, SOL_SOCKET, SO_LINGER, &so_linger, sizeof(so_linger)));

    return true;
}

bool socket_base_t::set_non_blocking(bool non_blocking) {
    SOCKET_LOG("({})", non_blocking);

    int flags = fcntl(socket_, F_GETFL, 0);
    if (flags < 0) {
        return false;
    }

    (non_blocking ? flags |= O_NONBLOCK : flags &= ~O_NONBLOCK);

    RET_ON_ERR(fcntl(socket_, F_SETFL, flags));

    return true;
}

bool socket_base_t::close_socket() {
    if (socket_ != INVALID_SOCKET) {
        SYS_FUNC_CALL(::close(socket_));
    }

    SOCKET_LOG("");

    local_.reset();
    remote_.reset();

    socket_ = INVALID_SOCKET;

    return true;
}

#define RX_TX_CLOSE_TIMEOUT 5 // seconds to send/recv on close

bool socket_base_t::close() {
    if (socket_ == INVALID_SOCKET) {
        return true;
    }

    set_linger(true, RX_TX_CLOSE_TIMEOUT);

    SYS_FUNC_CALL(::shutdown(socket_, SHUT_RDWR));

    close_socket();

    return true;
}

std::string socket_base_t::str() const {
    std::stringstream out;
    out << "socket(0x" << std::hex << socket_ << std::dec << ")[" << local_addr.str() << " <-> "
        << remote_addr.str() << "]";

    return out.str();
}

// =================================================================================

//
// for listen socket EPOLLIN means can accept, no EPOLLOUT reported.
// not connected client socket will fire EPOLLOUT on connect.
// connected socket EPOLLIN on recv.
//

//
// When a socket error is detected (i.e. connection closed/refused/timedout),
// epoll will return the registered interest events POLLIN/POLLOUT with POLLERR.
// So epoll_wait() will return POLLOUT|POLLERR if you registered POLLOUT,
// or POLLIN|POLLOUT|POLLERR if POLLIN|POLLOUT was registered.
//
//  EPOLLERR
//           Error condition happened on the associated file descriptor.
//           epoll_wait(2) will always wait for this event; it is not
//           necessary to set it in events.
//
//  EPOLLHUP
//           Hang up happened on the associated file descriptor.
//           epoll_wait(2) will always wait for this event; it is not
//           necessary to set it in events.
//
//  EPOLLRDHUP (since Linux 2.6.17)
//           Stream socket peer closed connection, or shut down writing
//           half of connection.  (This flag is especially useful for
//           writing simple code to detect peer shutdown when using
//           edge-triggered monitoring.)

int socket_io_t::io_event(uint32_t io_events) {
    SOCKET_LOG("{}", io_events);

    if (io_events & EPOLLERR) {
        op_notify_->on_error(*this);
        return IO_NONE;
    }

    if (io_events & EPOLLHUP) {
        op_notify_->on_error(*this);
        return IO_NONE;
    }

    if (io_events & EPOLLRDHUP) {
        op_notify_->on_disconnect(*this);
        return IO_NONE;
    }

    int rc = IO_NONE;

    if (io_events & EPOLLIN) {
        rc |= recv();
    }

    if (io_events & EPOLLOUT) {
        rc |= send();
    }

    return rc;
}

bool socket_io_t::send(const void* data, size_t size) {
    tx_ = packet_t(data, size);

    if (send() == IO_REARM) {
        return arm_monitor();
    }

    return true;
}

bool socket_io_t::recv(void* data, size_t size) {
    rx_ = packet_t(data, size);

    if (recv() == IO_REARM) {
        return arm_monitor();
    }

    return true;
}

void socket_io_t::set_op(bool send, bool on) {
    send ?
         /*send*/ (on ? events_ |= EPOLLOUT : events_ &= ~EPOLLOUT)
         :
         /*recv*/ (on ? events_ |= EPOLLIN : events_ &= ~EPOLLIN);
}

void socket_io_t::op_complete(bool send) {
    SOCKET_LOG("{} {}", send ? "send" : "recv", send ? tx_.packet.size : rx_.packet.size);

    if (send) {
        tx_.active = false;
        set_op(true, false);
        io_notify_->on_send(tx_, *this);
    }
    else // recv
    {
        rx_.active = false;
        set_op(false, false);
        io_notify_->on_recv(rx_, *this);
    }
}

//
//        An application that employs the EPOLLET flag should use
//        nonblocking file descriptors to avoid having a blocking read or
//        write starve a task that is handling multiple file descriptors.
//        The suggested way to use epoll as an edge-triggered (EPOLLET)
//        interface is as follows:
//
//        (1)  with nonblocking file descriptors; and
//
//        (2)  by waiting for an event only after read(2) or write(2)
//             return EAGAIN.
//

int socket_io_t::send() {
    while (true) {
        SOCKET_LOG(" -> {}", (ssize_t)tx_);

        auto sent = ::send(socket_, tx_, tx_, 0);
        if (tx_ == sent) //  all data sent
        {
            op_complete(true);
            return IO_NONE;
        }
        else if (sent > 0) {
            tx_ += sent;
            continue;
        }
        else if ((sent == -1) && would_block()) {
            set_op(true, true);
            return IO_REARM;
        }
        else if (sent == 0) // socket disconnected ?
        {
            op_notify_->on_disconnect(*this);
            return IO_NONE;
        }
        else // sent == -1 and some other error
        {
            op_notify_->on_error(*this);
            return IO_NONE;
        }
    }
}

int socket_io_t::recv() {
    while (true) {
        SOCKET_LOG(" <- {}", (ssize_t)rx_);

        auto received = ::recv(socket_, rx_, rx_, 0);
        if (rx_ == received) // all data received
        {
            op_complete(false);
            return IO_NONE;
        }
        else if (received > 0) // partial data received
        {
            rx_ += received;
            continue;
        }
        else if ((received == -1) &&
                 would_block()) { // no more data in kernel buf, need wait for more
            set_op(false, true);
            return IO_REARM;
        }
        else if (received == 0) // socket disconnected ?
        {
            op_notify_->on_disconnect(*this);
            return IO_NONE;
        }
        else // error
        {
            op_notify_->on_error(*this);
            return IO_NONE;
        }
    }
}

// =======================================================================

bool socket_t::connect(const sockaddr_t& peer,
                       uint32_t /* timeout */ sec,
                       const std::string& if_name) {
    RET_ON_FALSE(create(peer));

    SOCKET_LOG("-> {}", peer.str());

    if (!if_name.empty()) {
        RET_ON_ERR(setsockopt(
            socket_, SOL_SOCKET, SO_BINDTODEVICE, if_name.c_str(), socklen_t(if_name.size() + 1)));
    }

    /* Set the option active */
    int opt_val = 1;
    RET_ON_ERR(setsockopt(socket_, SOL_SOCKET, SO_KEEPALIVE, &opt_val, sizeof(opt_val)));

    RET_ON_FALSE(wait_condition(COND(::connect(socket_, peer, peer) != -1), sec, peer.str()));

    RET_ON_FALSE(get_info());

    SOCKET_LOG("connected.");

    return true;
}

std::ostream& operator<<(std::ostream& out, const socket_base_t& s) {
    return out << s.str();
}

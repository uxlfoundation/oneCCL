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
#include "asio.h"
#include "sockaddr.h"

using socketfd_t = int;

constexpr socketfd_t INVALID_SOCKET = (socketfd_t)-1;

class socket_base_t;
class socket_op_notify_t // socket operations
{
public:
    virtual void on_accept(socket_base_t& /*s*/, socketfd_t /*new_socket*/)
        _DEF_IMPL_; // server only, new connected endpoint
    virtual void on_disconnect(socket_base_t& /*s*/) _DEF_IMPL_;
    virtual void on_error(socket_base_t& /*s*/) _DEF_IMPL_;
};

class socket_base_t : public untransferable_t, public socket_op_notify_t {
public:
    socket_base_t() = default;
    socket_base_t(socketfd_t socket);
    socket_base_t(socket_op_notify_t& n) : op_notify_(&n) {}
    socket_base_t(socketfd_t s, socket_op_notify_t& n) : socket_base_t(s) {
        op_notify_ = &n;
    }

    virtual ~socket_base_t() {
        close();
    }

    virtual bool send(const void* p, size_t s) {
        return ::send(socket_, p, s, 0) != -1;
    };
    virtual bool recv(void* p, size_t s) {
        return ::recv(socket_, p, s, 0) != -1;
    };

    bool close();

    bool close_socket();

    const sockaddr_t& local_addr = local_;
    const sockaddr_t& remote_addr = remote_;
    const socketfd_t& fd = socket_;

    socket_op_notify_t* op_notify_ = this;

    virtual std::string str() const;

    bool set_non_blocking(bool non_blocking = true);

protected:
    bool create(sa_family_t domain, int sock_type = SOCK_STREAM);
    bool create(const sockaddr_t& addr);

    bool get_info();
    bool set_linger(bool set, uint32_t seconds);

    socketfd_t socket_ = INVALID_SOCKET;
    sockaddr_t local_;
    sockaddr_t remote_;
};

class async_socket_t : public socket_base_t, public asio_client_t {
protected:
    using socket_base_t::socket_base_t; // inherit constructors

    async_socket_t(socket_op_notify_t& n, asio_t* a) : socket_base_t(n), asio_client_t(a) {}
    async_socket_t(socketfd_t s, socket_op_notify_t& n, asio_t* a)
            : socket_base_t(s, n),
              asio_client_t(a) {}

public: // asio
    virtual int io_fd() const override {
        return fd;
    };
    virtual uint32_t events() const override {
        return events_;
    };

protected:
    //                    default IO configuration
    // EPOLLONESHOT
    //           Requests one-shot notification for the associated file
    //           descriptor.  This means that after an event notified for
    //           the file descriptor by epoll_wait(2), the file descriptor
    //           is disabled in the interest list and no other events will
    //           be reported by the epoll interface.  The user must call
    //           epoll_ctl() with EPOLL_CTL_MOD to rearm the file
    //           descriptor with a new event mask.
    //
    // EPOLLET
    //           Requests edge-triggered notification for the associated
    //           file descriptor.
    //           One-Time Notification: In edge-triggered mode, the epoll
    //           interface notifies you only when the state of a file
    //           descriptor changes (like when new data arrives on a socket).
    //           It will not notify you again until there's another state change.
    //
    //           When you receive a notification in edge-triggered mode, you must
    //           read all available data until you get EAGAIN/EWOULDBLOCK, or you
    //           might miss data.
    //
    // EPOLLRDHUP
    //           Stream socket peer closed connection, or shut down writing
    //           half of connection.  (This flag is especially useful for
    //           writing simple code to detect peer shutdown when using
    //           edge-triggered monitoring.)

    uint32_t events_ =
        EPOLLONESHOT | EPOLLET | EPOLLRDHUP; // default mode and events of interest for this socket
};

class socket_io_notify_t // read / write notify
{
public:
    struct packet_t {
        const uint8_t* buf = nullptr;
        size_t size = 0;

        packet_t(const void* b = nullptr, size_t s = 0) : buf((uint8_t*)b), size(s) {}
    };

    virtual void on_send(const packet_t&, socket_base_t&) _DEF_IMPL_; // send completed
    virtual void on_recv(const packet_t&, socket_base_t&) _DEF_IMPL_; // recv completed
};

class socket_io_t : public async_socket_t, public socket_io_notify_t {
private:
    struct io_descriptor_t // send/recv descriptor
    {
        bool active = false;
        size_t offset = 0;
        packet_t packet;

        operator void*() {
            return (void*)(packet.buf + offset);
        }
        operator const void*() const {
            return (packet.buf + offset);
        }
        operator const packet_t&() const {
            return packet;
        }
        operator ssize_t() const {
            return packet.size - offset;
        }

        io_descriptor_t& operator+=(size_t _x) {
            offset += _x;
            return *this;
        }

        io_descriptor_t& operator=(const packet_t& p) {
            VERIFY(!active, "operation in progress");

            active = true;
            offset = 0;
            packet = p;

            return *this;
        }

    } rx_, tx_;

public:
    using async_socket_t::async_socket_t;

    socket_io_notify_t* io_notify_ = this;

public: // asio
    virtual int io_event(uint32_t events) override;

public: // socket_base
    virtual bool send(const void* data, size_t size) override;
    virtual bool recv(void* data, size_t size) override;

private:
    void op_complete(bool send);
    void set_op(bool send, bool on); // send:recv on:off

    int send(); // called when socket is ready to send
    int recv(); // called when socket is ready to recv (pending data)
};

class socket_t : public socket_io_t {
public:
    using socket_io_t::socket_io_t;

    bool connect(const sockaddr_t& peer,
                 uint32_t /* timeout */ sec,
                 const std::string& if_name = "");
};

//
// POSIX says that EAGAIN and EWOULDBLOCK may be identical, but also that they may
// be distinct.  Therefore, well-written portable code MUST check for both values
// in some circumstances.
//
// error: logical ‘or’ of equal expressions [-Werror=logical-op]
//
static inline bool would_block() {
#if EAGAIN == EWOULDBLOCK
    return (errno == EAGAIN);
#else
    return ((errno == EAGAIN) || (errno == EWOULDBLOCK));
#endif
}

std::ostream& operator<<(std::ostream& out, const socket_base_t& s);
HLLOG_DEFINE_OSTREAM_FORMATTER(socket_base_t);

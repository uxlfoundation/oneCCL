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

#include "socket.h"

constexpr int MAX_PENDING_ACCEPTS =
    1024; // maximum length to which the queue of pending connections may grow.

// server socket
class acceptor_t : public async_socket_t {
public:
    virtual int io_event(uint32_t events) override; // for accept() only

public:
    acceptor_t(socket_op_notify_t& n, asio_t* a) : async_socket_t(n, a) {
        events_ = EPOLLIN | EPOLLONESHOT;
    } // for accept

    bool listen(const sockaddr_t& addr, int backlog = MAX_PENDING_ACCEPTS);

    virtual std::string str() const override;

protected:
    bool accept();

private:
    virtual bool send(const void*, size_t) override {
        return false;
    }
    virtual bool recv(void*, size_t) override {
        return false;
    }
};

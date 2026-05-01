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

#include <cstdint>
#include <deque>
#include <thread>
#include <atomic>

#include <sys/epoll.h>
#include "futex.h"

#include "utils.h"

//
// async IO
//
// we have a multithreaded IO server (asio_t) and IO clients (asio_client_t).
// IO client is represented with file descriptor ( virtual int io_fd() ) and
// registers events of interest with server, ( virtual uint32_t events() ).
// when event (or error) occurs, worker thread is awoken and calls client's callback
//   virtual int io_event(uint32_t events)
//
// f.e. IO client is a tcp socket and we want to receive data, so the event of interest is "IN", and we register it with
// the server. when data is arrived, thread is awoken and we shall successfully read() data from the socket.
//

class asio_t;

class asio_client_t : public untransferable_t {
    friend asio_t;

public:
    asio_client_t() = default;
    asio_client_t(asio_t* a) : asio_(a) {}

    // file descriptor of io client. can be file, pipe, socket....
    virtual int io_fd() const = 0;

    // events of interest for this client
    virtual uint32_t events() const = 0;

    // callback called by asio_t when event occurs
    // return values: -1: exit loop, 1: rearm event, 0: do nothing (continue loop)
    virtual int io_event(uint32_t events) = 0;

    bool arm_monitor();

    operator int() const {
        return io_fd();
    }
    operator void*() const {
        return (void*)this;
    }

protected:
    asio_t* asio_ = nullptr;
    bool added_ = false; // added to epoll
    uint32_t armed_ = 0; // cached armed events

    void set_armed(bool set = true) {
        armed_ = set ? events() : 0;
    };
    bool is_armed() const {
        return armed_ == events();
    }
};

constexpr int IO_EXIT = -1;
constexpr int IO_REARM = 1;
constexpr int IO_NONE = 0;

using counter_t = std::atomic<uint64_t>;

class asio_t : public asio_client_t {
public:
    virtual ~asio_t() {
        close();
    }

    bool start(uint32_t io_threads, uint32_t batch_size = 8);
    bool add_workers(uint32_t io_threads);
    bool stop();

    using asio_client_t::arm_monitor; // to avoid compiler "hides overloaded virtual function" error
    bool arm_monitor(asio_client_t& ioc);
    bool remove(asio_client_t& ioc);

private:
    int op_mode(asio_client_t& ioc);

private: // asio_client_t for control pipe
    virtual int io_event(uint32_t events) override;
    virtual int io_fd() const override {
        return control_[0];
    };
    virtual uint32_t events() const override {
        return EPOLLIN;
    };

private:
    bool setup();
    bool close();
    void epoll_thread();

    counter_t running_{ 0 }; // number of running threads
    event_t stopped_{ true }; // signaled when all threads are stopped

    int epoll_fd_ = -1;

    enum { READ_END = 0, WRITE_END = 1 };
    // control pipe [read, write]
    int control_[2] = { -1, -1 };

    struct {
        /*
            Selecting the Optimal epoll_wait() Batch Size:

            -----------------------------------------------------------------------------------------------------------
            | Batch Size |                Advantages                      |              Disadvantages                |
            |------------|------------------------------------------------|-------------------------------------------|
            | Small (8)  | Lower memory usage, Potentially lower latency  | More frequent syscalls                    |
            |            | for individual events                          |                                           |
            |------------|------------------------------------------------|-------------------------------------------|
            | Medium (16)| Good balance of memory/CPU efficiency          | May not be optimal for all workloads      |
            |------------|------------------------------------------------|-------------------------------------------|
            | Large (32+)| Fewer syscalls, Better for high-throughput     | Higher memory usage, Potentially higher   |
            |            | systems                                        | latency                                   |
            -----------------------------------------------------------------------------------------------------------

            Determining Your Optimal Value:
            - High connection count, low activity per connection: Use larger batches (32-64)
            - Low connection count, high activity per connection: Smaller batches work well (8-16)
            - Latency-sensitive applications: Smaller batches
            - Throughput-oriented applications: Larger batches
        */

        uint32_t batch_size = 1; // default batch size for epoll_wait

    } cfg_;
};

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

#include "asio.h"
#include <unistd.h>
#include <string>
#include <string.h>
#include <vector>

#define ASIO_LOG(FMT, ...) COORD_LOG("[epoll_fd: 0x{:x}] " FMT, epoll_fd_, ##__VA_ARGS__)
#define ASIO_CRT(FMT, ...) COORD_CRT("[epoll_fd: 0x{:x}] " FMT, epoll_fd_, ##__VA_ARGS__)

std::string events_to_str(uint32_t events) {
    static constexpr struct {
        uint32_t flag;
        const char* name;
    } event_flags[] = { { EPOLLIN, "EPOLLIN " },         { EPOLLPRI, "EPOLLPRI " },
                        { EPOLLOUT, "EPOLLOUT " },       { EPOLLRDNORM, "EPOLLRDNORM " },
                        { EPOLLRDBAND, "EPOLLRDBAND " }, { EPOLLWRNORM, "EPOLLWRNORM " },
                        { EPOLLWRBAND, "EPOLLWRBAND " }, { EPOLLMSG, "EPOLLMSG " },
                        { EPOLLERR, "EPOLLERR " },       { EPOLLHUP, "EPOLLHUP " },
                        { EPOLLRDHUP, "EPOLLRDHUP " },   { EPOLLEXCLUSIVE, "EPOLLEXCLUSIVE " },
                        { EPOLLWAKEUP, "EPOLLWAKEUP " }, { EPOLLONESHOT, "EPOLLONESHOT " },
                        { EPOLLET, "EPOLLET " } };

    std::string result;
    result.reserve(128); // Pre-allocate reasonable space

    for (const auto& event_flag : event_flags) {
        if (events & event_flag.flag)
            result += event_flag.name;
    }

    return result;
}

bool asio_t::setup() {
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ == -1) {
        return false;
    }

    ASIO_LOG("");

    // create pipe to control thread loop (now for exit only)
    RET_ON_ERR(pipe(control_));

    // Add the read end of the pipe to the epoll set
    epoll_event event = { .events = EPOLLIN, .data = { .ptr = this } };

    return epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, control_[READ_END], &event) != -1;
}

bool asio_t::start(uint32_t io_threads, uint32_t batch_size) {
    RET_ON_FALSE(setup());

    cfg_.batch_size = batch_size;

    stopped_ = false;
    add_workers(io_threads);

    while (running_ < io_threads) {
        usleep(1000);
    }

    return true;
}

bool asio_t::add_workers(uint32_t io_threads) {
    ASIO_LOG("{} workers", io_threads);

    FOR_I(io_threads) {
        std::thread([this] {
            epoll_thread();
        }).detach();
    }

    return true;
}

bool asio_t::stop() {
    ASIO_LOG();
    constexpr uint32_t SIG_STOP = 0xC0DE0FF;
    // Send a stop signal through the pipe
    // it will wake all the threads and instruct them to stop
    return write(control_[WRITE_END], &SIG_STOP, sizeof(SIG_STOP)) == sizeof(SIG_STOP);
}

int asio_t::io_event(uint32_t) {
    ASIO_LOG("exit received");
    return IO_EXIT;
}

bool asio_t::close() {
    ASIO_LOG("running threads: {}", running_);

    stop();
    stopped_.wait();

    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, control_[READ_END], nullptr);

    ASIO_LOG("deleted");

    ::close(control_[WRITE_END]);
    ::close(epoll_fd_);
    ::close(control_[READ_END]);

    control_[READ_END] = control_[WRITE_END] = epoll_fd_ = -1;
    return true;
}

bool asio_t::remove(asio_client_t& ioc) {
    ASIO_LOG("io_fd: 0x{:x}", ioc.io_fd());

    ioc.asio_ = nullptr;
    epoll_event event = {};

    return epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, ioc, &event) != -1;
}

bool asio_client_t::arm_monitor() {
    VERIFY(asio_, "asio not set");

    return asio_->arm_monitor(*this);
}

int asio_t::op_mode(asio_client_t& ioc) {
    if (unlikely(!ioc.added_)) {
        ioc.added_ = true;
        return EPOLL_CTL_ADD;
    }

    return EPOLL_CTL_MOD;
}

bool asio_t::arm_monitor(asio_client_t& ioc) {
    if (ioc.is_armed())
        return true;

    int op = op_mode(ioc);

    epoll_event event = { .events = ioc.events(), .data = { .ptr = ioc } }; // use ioc as user data

    ASIO_LOG("[{}], op: {} io_fd: 0x{:x} [{}]",
             event.data.ptr,
             op,
             ioc.io_fd(),
             events_to_str(event.events));

    ioc.set_armed();

    return epoll_ctl(epoll_fd_, op, ioc, &event) != -1;
}

void asio_t::epoll_thread() {
    auto max_events = cfg_.batch_size;
    bool stop = false;

    running_++;

    // Use dynamic allocation to avoid VLA
    std::vector<epoll_event> events(max_events);

    while (!stop) {
        //
        // When successful, epoll_wait() returns the number of file descriptors ready for the requested I/O,
        // or zero if no file descriptor became ready during the requested timeout milliseconds (-1 == infinite).
        // When an error occurs, epoll_wait() returns -1 and errno is set appropriately.
        //
        // Errors
        // ...
        // EINTR
        // The call was interrupted by a signal handler before either any of the requested events occurred or the
        // timeout expired.
        //
        ASIO_LOG("--> epoll_wait({})", max_events);

        auto nfds = epoll_wait(epoll_fd_, events.data(), max_events, -1);
        if (unlikely(nfds == -1)) {
            if (likely(errno == EINTR))
                continue;

            ASIO_CRT("epoll_wait() failed: ({}) {}", errno, strerror(errno));
            break;
        }

        FOR_I(nfds) {
            asio_client_t& ioc = *static_cast<asio_client_t*>(events[i].data.ptr);

            ioc.set_armed(false);

            ASIO_LOG("[{}/{}] epoll_wait()--> [{}] io_fd: 0x{:x} events: 0x{:x} {}",
                     i + 1,
                     nfds,
                     events[i].data.ptr,
                     ioc.io_fd(),
                     events[i].events,
                     events_to_str(events[i].events));

            int rc = ioc.io_event(events[i].events);
            switch (rc) {
                case IO_REARM: arm_monitor(ioc); break;

                case IO_EXIT: // exit loop
                    stop = true;
                    break;
            }
        }
    }

    if (--running_ == 0) // last thread
    {
        stopped_ = true;
    }
}

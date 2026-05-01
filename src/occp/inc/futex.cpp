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
#include "futex.h"

#include "utils.h" // for VERIFY
#include <climits> // for INT_MAX
#include <linux/futex.h> // for FUTEX_WAIT, FUTEX_WAKE
#include <cstring> // for strerror
#include <syscall.h> // for SYS_futex
#include <unistd.h> // for syscall
#include <cerrno> // for errno, EAGAIN

static long futex(const volatile uint32_t* uaddr,
                  int futex_op,
                  uint32_t val,
                  const struct timespec* timeout = nullptr) {
    return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr, 0);
}

// ======================================================================================================================

static timespec* create_timeout_spec(timespec& timeout, int64_t timeout_ms) {
    if (timeout_ms == -1)
        return nullptr;

    timeout.tv_sec = timeout_ms / 1000;
    timeout.tv_nsec = (timeout_ms % 1000) * 1000000;

    return &timeout;
}

// true - SIGNALLED, false - timeout occurred  (-1 == indefinitely)
bool futex_event_t::wait(int64_t msec) const {
    timespec timeout;
    timespec* timeout_ptr = create_timeout_spec(timeout, msec);
    while (true) {
        // FUTEX_WAIT will immediately return if the value at the futex address is not equal to NON_SIGNALLED
        auto result = futex(&futex_, FUTEX_WAIT, NON_SIGNALLED, timeout_ptr);
        if ((result == -1) && (errno == ETIMEDOUT)) {
            return false;
        }
        VERIFY(0 == result || errno == EAGAIN,
               "futex(FUTEX_WAIT) failed with errno({})",
               strerror(errno));
        if (futex_ == SIGNALLED) {
            return true;
        }
    }
}

void futex_event_t::signal() {
    futex_ = SIGNALLED;
    auto result = futex(&futex_, FUTEX_WAKE, INT_MAX);
    VERIFY(-1 != result, "futex(FUTEX_WAKE) failed with errno({})", strerror(errno));
}

void futex_event_t::reset() {
    futex_ = NON_SIGNALLED;
}

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
#include "defines.h"

/**
 * A class that provides a futex-based event mechanism. It allows thread(s) to wait for another thread to signal an
 * event. The waiting thread(s) will be blocked until the event is signaled.
 */
class futex_event_t : public untransferable_t {
public:
    futex_event_t(bool signaled = false) : futex_(signaled ? SIGNALLED : NON_SIGNALLED) {}

    // return: true - SIGNALLED, false - timeout occurred  (-1 == indefinitely)
    bool wait(int64_t msec = -1) const; // if event is in non-signaled state, wait is blocking
    void signal(); // signal the event, unblock all waiting threads
    void reset(); // reset the event to non-signaled state
    bool signaled() const {
        return futex_ == SIGNALLED;
    }

    futex_event_t& operator=(bool signaled) {
        signaled ? signal() : reset();
        return *this;
    }

private:
    enum event_state_t : uint32_t { NON_SIGNALLED = 0, SIGNALLED = 1 };

    volatile uint32_t futex_; // when signaled, wait is NOT blocking
};

using event_t = futex_event_t;

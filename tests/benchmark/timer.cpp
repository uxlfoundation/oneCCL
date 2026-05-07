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

#include "timer.h"
#include <chrono>

namespace {
// Returns the current time in nanoseconds since epoch.
std::uint64_t now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock::now().time_since_epoch())
        .count();
}
} // namespace

// Constructor: records the creation time.
timer::timer() { t0_ = now(); }

// elapsed: returns the time in seconds since the last reset or construction.
double timer::elapsed() const {
    std::uint64_t const end_time = now();
    return 1.e-9 * static_cast<double>(end_time - t0_);
}

// reset: returns the elapsed time and resets the start time.
double timer::reset() {
    std::uint64_t const end_time = now();
    double const elapsed_time = 1.e-9 * static_cast<double>(end_time - t0_);
    t0_ = end_time;
    return elapsed_time;
}

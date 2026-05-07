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

#ifndef ONECCL_BENCHMARK_TIMER_H
#define ONECCL_BENCHMARK_TIMER_H

#include <cstdint>

// A simple timer class for measuring elapsed time.
// This is backend-agnostic and uses std::chrono.
class timer {
private:
    std::uint64_t t0_;
public:
    timer();
    [[nodiscard]] double elapsed() const;
    double reset();
};

#endif // ONECCL_BENCHMARK_TIMER_H

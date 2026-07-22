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

#include <chrono>
#include <string>
#include <functional>
#include <thread>

#include "defines.h"
#include "log.h"

#define NOW std::chrono::steady_clock::now

#define SLEEP_FOR(ms) std::this_thread::sleep_for(std::chrono::milliseconds((ms)))

using condition_t = std::function<bool()>;

#define COND(func) \
    [&, this]() { \
        return (func); \
    }

static inline bool wait_condition(const condition_t& cond,
                                  uint32_t timeout_sec,
                                  const std::string& reason,
                                  uint32_t retry_ms = 100) {
    auto expired = NOW() + std::chrono::seconds(timeout_sec);
    while (!cond()) {
        if (NOW() >= expired) {
            COORD_ERR("operation: {} timeout ({}) expired while waiting for condition",
                      reason,
                      timeout_sec);
            return false;
        }
        SLEEP_FOR(retry_ms);
    }
    return true;
}

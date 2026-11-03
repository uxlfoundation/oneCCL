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
#include <iostream>
#include <ze_api.h>
#include "ze_handle.h"
#include "ze_helpers.h"

enum class ze_engine_t
{
    Unknown = 0,
    Compute,
    Copy,
    count  // Used to determine the number of engines
};

// Add streaming operator for ze_engine_t
static inline std::ostream& operator<<(std::ostream& os, const ze_engine_t& engine)
{
    switch (engine)
    {
        case ze_engine_t::Compute:
            return os << "Compute";
        case ze_engine_t::Copy:
            return os << "Copy";
        case ze_engine_t::Unknown:
            return os << "Unknown";
        default:
            return os << "Invalid ze_engine_t::(" << static_cast<int>(engine) << ")";
    }
}

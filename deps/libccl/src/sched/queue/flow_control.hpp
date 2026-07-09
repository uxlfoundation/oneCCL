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
#include <cstddef>

namespace ccl {

#define CCL_MAX_FLOW_CREDITS 1024

class flow_control {
public:
    flow_control();
    flow_control(const flow_control &other) = delete;
    flow_control &operator=(const flow_control &other) = delete;
    ~flow_control();

    void set_max_credits(size_t value);
    size_t get_max_credits() const;
    size_t get_credits() const;
    bool take_credit();
    void return_credit();

private:
    size_t max_credits;
    size_t min_credits;
    size_t credits;
};

} // namespace ccl

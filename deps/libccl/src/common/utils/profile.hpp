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

#include <map>
#include <string>
#include <cstddef>
#include <vector>

namespace ccl {
namespace profile {

class metrics_counter {
    const char *collective_name;

public:
    std::map<size_t, size_t> nonparallel_calls_per_count, parallel_calls_per_count;
    metrics_counter(const char *collective_name) : collective_name(collective_name){};
    void init();
    ~metrics_counter();
    metrics_counter(const metrics_counter &) = delete;
    metrics_counter &operator=(const metrics_counter &) = delete;
};

class metrics_manager {
public:
    metrics_counter allreduce_pipe{ "allreduce" };
    metrics_counter reduce_pipe{ "reduce" };
    metrics_counter reduce_scatter_pipe{ "reduce_scatter" };
    metrics_counter allgatherv_pipe{ "allgatherv" };

    void init();
};

class timestamp_manager {
    void finalize();
    std::vector<std::pair<std::string, size_t *>> recorded_timestamps;

public:
    void add_timestamp(std::string text, uint64_t *timestamp_ptr);
    void init();

    timestamp_manager() {}
    timestamp_manager(const timestamp_manager &other) = delete;
    timestamp_manager &operator=(const timestamp_manager &other) = delete;
    ~timestamp_manager();
};

} // namespace profile
} // namespace ccl

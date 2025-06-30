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
#include "common/utils/utils.hpp"
#include "common/log/log.hpp"

void ccl::profile::metrics_counter::init() {
    this->nonparallel_calls_per_count.clear();
    this->parallel_calls_per_count.clear();
}

ccl::profile::metrics_counter::~metrics_counter() {
    std::string pipe_metrics;

    for (auto& calls_per_count : this->nonparallel_calls_per_count) {
        pipe_metrics += "nonparallel_calls_per_count[" + std::to_string(calls_per_count.first) +
                        "]=" + std::to_string(calls_per_count.second) + ",\n";
    }

    for (auto& calls_per_count : this->parallel_calls_per_count) {
        pipe_metrics += "   parallel_calls_per_count[" + std::to_string(calls_per_count.first) +
                        "]=" + std::to_string(calls_per_count.second) + ",\n";
    }

    if (!pipe_metrics.empty()) {
        LOG_INFO(this->collective_name, "_pipe_metrics: [\n", pipe_metrics, "]");
    }
}

void ccl::profile::metrics_manager::init() {
    this->allreduce_pipe.init();
    this->reduce_pipe.init();
    this->reduce_scatter_pipe.init();
    this->allgatherv_pipe.init();
}

void ccl::profile::timestamp_manager::add_timestamp(std::string text, uint64_t* timestamp_ptr) {
    recorded_timestamps.emplace_back(text, timestamp_ptr);
}

void ccl::profile::timestamp_manager::init() {
    recorded_timestamps.clear();
}

ccl::profile::timestamp_manager::~timestamp_manager() {
    finalize();
}

void ccl::profile::timestamp_manager::finalize() {
    std::string timestamp_metrics;

    std::vector<std::pair<std::string, size_t*>> sorted_timestamps(recorded_timestamps.begin(),
                                                                   recorded_timestamps.end());
    std::sort(
        sorted_timestamps.begin(),
        sorted_timestamps.end(),
        [](const std::pair<std::string, size_t*>& a, const std::pair<std::string, size_t*>& b) {
            return *(a.second) < *(b.second);
        });

    if (!sorted_timestamps.empty()) {
        size_t first_timestamp = *(sorted_timestamps.front().second);

        for (const auto& pair : sorted_timestamps) {
            std::string key = pair.first;
            size_t offset = *(pair.second) - first_timestamp;
            LOG_DEBUG("|ZE_TIMESTAMP| ", key, " => ", offset);
        }
    }
}

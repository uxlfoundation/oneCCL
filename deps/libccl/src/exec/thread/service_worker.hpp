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

#include "exec/thread/worker.hpp"
#include "fusion/fusion.hpp"
#include "internal_types.hpp"

class ccl_service_worker : public ccl_worker {
public:
    ccl_service_worker(size_t idx,
                       std::unique_ptr<ccl_sched_queue> data_queue,
                       ccl_fusion_manager& fusion_manager);
    ~ccl_service_worker();

    ccl_service_worker(const ccl_service_worker& other) = delete;
    ccl_service_worker& operator=(const ccl_service_worker& other) = delete;

    ccl::status do_work(size_t& processed_count) override;

    bool can_reset() override;

private:
    ccl_fusion_manager& fusion_manager;
};

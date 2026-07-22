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

#include "exec/thread/service_worker.hpp"

ccl_service_worker::ccl_service_worker(size_t idx,
                                       std::unique_ptr<ccl_sched_queue> data_queue,
                                       ccl_fusion_manager& fusion_manager)
        : ccl_worker(idx, std::move(data_queue)),
          fusion_manager(fusion_manager) {}

ccl_service_worker::~ccl_service_worker() {
    fusion_manager.reset();
}

ccl::status ccl_service_worker::do_work(size_t& processed_count) {
    fusion_manager.execute();
    return ccl_worker::do_work(processed_count);
}

bool ccl_service_worker::can_reset() {
    /* skip ATL processing since it may be already destroyed */
    /* make only local processing */
    process_atl = false;

    size_t processed_count;
    do_work(processed_count);
    return fusion_manager.can_reset();
}

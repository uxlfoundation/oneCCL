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

#include "sched/sched.hpp"
#include "internal_types.hpp"

class ccl_parallelizer {
public:
    ccl_parallelizer(size_t max_data_partition_count)
            : max_data_partition_count(max_data_partition_count) {}

    ~ccl_parallelizer() = default;

    ccl_parallelizer(const ccl_parallelizer& other) = delete;
    ccl_parallelizer& operator=(const ccl_parallelizer& other) = delete;

    ccl_parallelizer(ccl_parallelizer&& other) = delete;
    ccl_parallelizer& operator=(ccl_parallelizer&& other) = delete;

    ccl::status process(ccl_sched* sched, bool update_sched_id = true);

private:
    ccl::status process_deps(ccl_sched* sched);

#ifdef CCL_ENABLE_SYCL
    ccl::status process_pre_post_copies(ccl_sched* sched);
    ccl::status process_output_event(ccl_sched* sched);
#endif // CCL_ENABLE_SYCL

    ccl::status process_base(ccl_sched* sched, bool update_sched_id = true);

    size_t max_data_partition_count;
};

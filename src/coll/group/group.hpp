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
#pragma once

#include "coll/algorithms/algorithm_utils.hpp"
#include "common/env/env.hpp"

#include <vector>
#include <functional>
#include <mutex>
#include <thread>
#include <iostream>

class group_impl {
public:
    static void start();
    static void end();
    static void add_operation(ccl_coll_type ctype, std::function<ccl::event()> operation);
    static void add_post_processing_step(std::function<bool(atl_req_t&, bool)> step);
#ifdef CCL_ENABLE_SYCL
    static void set_sycl_queue(sycl::queue q);
#endif // CCL_ENABLE_SYCL

    static thread_local bool is_group_active;
    static thread_local bool first_group_op;
    static thread_local std::vector<std::pair<ccl_coll_type, std::function<ccl::event()>>>
        operation_storage;
    static thread_local std::vector<std::function<bool(atl_req_t&, bool)>> post_processing_steps;
#ifdef CCL_ENABLE_SYCL
    static thread_local sycl::queue sycl_queue;
#endif // CCL_ENABLE_SYCL

private:
    static std::mutex group_mutex;
};

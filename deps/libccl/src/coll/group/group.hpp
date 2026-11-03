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

#include "coll/algorithms/algorithm_utils.hpp"
#include "common/env/env.hpp"

#include <vector>
#include <functional>
#include <mutex>
#include <thread>
#include <iostream>
#include <set>
#include <algorithm>
#include <memory>

#include "comm/comm.hpp"

// Forward declarations
class ccl_comm;
class ccl_stream;

class group_impl {
public:
    struct group_operation {
        ccl_coll_type ctype;
        ccl_comm* comm;
        int peer_rank;
        ccl_stream* stream;
        std::function<ccl::event()> fn;
    };
    static void start();
    static void end();
    static void add_operation(ccl_coll_type ctype,
                              ccl_comm* comm,
                              int peer_rank,
                              ccl_stream* stream,
                              std::function<ccl::event()> operation);
    // group ops run at group end when the caller's deps may be gone,
    // deps are not supported inside a group
    static const std::vector<ccl::event>& safe_deps(const std::vector<ccl::event>& deps);
    static void add_post_processing_step(std::function<bool(atl_req_t&, bool, bool)> step);
    static bool is_group_active();
#ifdef CCL_ENABLE_SYCL
    struct GroupSendRequest {
        std::shared_ptr<ccl_internal_comm::group_send_request> req;
        sycl::queue q;
        int tag;
    };
    struct GroupRecvRequest {
        std::shared_ptr<ccl_internal_comm::group_recv_request> req;
        sycl::queue q;
        int tag;
    };
    static void set_sycl_queue(sycl::queue q);
    // cannot be used from host task, modifies thread_local data
    static void add_group_send_request(std::shared_ptr<ccl_internal_comm::group_send_request> req,
                                       sycl::queue q, int tag);
    // cannot be used from host task, modifies thread_local data
    static void add_group_recv_request(std::shared_ptr<ccl_internal_comm::group_recv_request> req,
                                       sycl::queue q, int tag);
    static void process_group_pt2pt_scale_out_requests();
    static bool use_group_scale_up(ccl_comm* comm, int peer_rank, ccl_stream* stream);
#endif // CCL_ENABLE_SYCL

    static thread_local bool first_group_op;
    static thread_local std::vector<group_operation> operation_storage;
    static thread_local std::vector<std::function<bool(atl_req_t&, bool, bool)>>
        post_processing_steps;
    static thread_local size_t group_counter;
#ifdef CCL_ENABLE_SYCL
    static thread_local std::vector<GroupSendRequest> all_group_send_requests;
    static thread_local std::vector<GroupRecvRequest> all_group_recv_requests;

    static thread_local sycl::queue sycl_queue;
#endif // CCL_ENABLE_SYCL

private:
    static std::mutex group_mutex;
};

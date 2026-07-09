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

#define CCL_UNORDERED_COLL_COORDINATOR (0)

struct ccl_unordered_coll_ctx;
class ccl_comm;

class ccl_unordered_coll_manager {
public:
    ccl_unordered_coll_manager(const ccl_unordered_coll_manager& other) = delete;
    const ccl_unordered_coll_manager& operator=(const ccl_unordered_coll_manager& other) = delete;
    ccl_unordered_coll_manager(ccl_comm& parent_comm);
    ~ccl_unordered_coll_manager();

    std::shared_ptr<ccl_comm> get_comm(const std::string& match_id);
    ccl_request* postpone(ccl_sched* sched);
    void dump(std::ostream& out) const;

private:
    void start_coordination(const std::string& match_id);
    void start_post_coordination_actions(ccl_unordered_coll_ctx* ctx);
    void run_postponed_scheds(const std::string& match_id, ccl_comm* comm);
    void run_sched(ccl_sched* sched, ccl_comm* comm) const;
    void add_comm(const std::string& match_id, std::shared_ptr<ccl_comm> comm);
    void postpone_sched(ccl_sched* sched);
    size_t get_postponed_sched_count(const std::string& match_id);
    void remove_service_scheds();

    std::unique_ptr<ccl_comm> coordination_comm;

    using unresolved_comms_t = std::unordered_map<std::string, int>;
    unresolved_comms_t unresolved_comms{};
    mutable ccl_spinlock unresolved_comms_guard{};

    using match_id_to_comm_map_type = std::unordered_map<std::string, std::shared_ptr<ccl_comm>>;
    match_id_to_comm_map_type match_id_to_comm_map{};
    mutable ccl_spinlock match_id_to_comm_map_guard{};

    using postponed_scheds_t = std::unordered_multimap<std::string, ccl_sched*>;
    postponed_scheds_t postponed_scheds{};
    mutable ccl_spinlock postponed_scheds_guard{};

    using service_scheds_t = std::map<std::string, ccl_sched*>;
    service_scheds_t service_scheds{};
    // TODO - tbb::spin_rw_mutex
    ccl_spinlock service_scheds_guard{};
};

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
#include "common/utils/comm_profile.hpp"
#include <chrono>

#ifdef CCL_ENABLE_PROFILING

comm_profile comm_profiles;

std::chrono::time_point<std::chrono::high_resolution_clock> get_profile_time() {
    auto now = std::chrono::high_resolution_clock::now();
    return now;
}

comm_profile_key get_profile_key(int comm_id,
                                 int rank,
                                 int comm_size,
                                 std::string op_name,
                                 ccl::reduction reduction,
                                 ccl::datatype dtype,
                                 size_t count) {
    comm_profile_key profile_key(comm_id, rank, comm_size, op_name, reduction, dtype, count);
    return profile_key;
}

bool is_profiling_enabled() {
    return ccl::global_data::env().enable_profiling;
}

bool skip_profiling(int rank) {
    bool is_rank_profiled = true; // TODO: add a way to filter out certain ranks

    if (!is_rank_profiled || !is_profiling_enabled()) {
        return true;
    }
    return false;
}

void profiler_record_comm_event_enter(const ccl_comm* comm,
                                      std::string op_name,
                                      ccl::reduction reduction,
                                      ccl::datatype dtype,
                                      size_t count,
                                      comm_session& comm_event_session) {
    if (skip_profiling(comm->rank()))
        return;

    comm_profile_key coll_prof_key = get_profile_key(
        comm->get_comm_id(), comm->rank(), comm->size(), op_name, reduction, dtype, count);
    comm_event_session.set_key(coll_prof_key);

    comm_profiles.record_prof_key(comm_event_session.comm_key);
    comm_profiles.record_prof_info_call_count(comm_event_session.comm_key);
    comm_profiles.record_prof_info_submit_time(comm_event_session.comm_key, get_profile_time());
}

void profiler_record_comm_event_exit(int rank, comm_session& comm_event_session) {
    if (skip_profiling(rank))
        return;

    comm_profiles.record_prof_info_return_time(comm_event_session.comm_key, get_profile_time());
}

#endif //CCL_ENABLE_PROFILING

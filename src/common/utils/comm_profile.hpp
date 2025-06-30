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
#ifdef CCL_ENABLE_PROFILING

#include <map>
#include <string>
#include <sstream>
#include <cstddef>
#include <vector>
#include <ctime>
#include "oneapi/ccl/types.hpp"
#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include <chrono>

std::chrono::time_point<std::chrono::high_resolution_clock> get_profile_time();
bool is_profiling_enabled();
bool skip_profiling(int rank);

class comm_profile_key {
public:
    int comm_id;
    int rank;
    int comm_size;
    std::string op_name; // collective or pt2pt name
    ccl::reduction reduction = ccl::reduction::none; // Note: useful for reduce based collectives
    ccl::datatype dtype;
    size_t count;

    comm_profile_key() {}

    comm_profile_key(std::string op_name, ccl::datatype dtype, size_t count)
            : op_name(op_name),
              dtype(dtype),
              count(count) {}

    // Overload for allreduce
    comm_profile_key(int comm_id,
                     int rank,
                     int comm_size,
                     std::string op_name,
                     ccl::reduction reduction,
                     ccl::datatype dtype,
                     size_t count)
            : comm_id(comm_id),
              rank(rank),
              comm_size(comm_size),
              op_name(op_name),
              reduction(reduction),
              dtype(dtype),
              count(count) {}

    /* Compare operator required for the map */
    bool operator<(const comm_profile_key& other) const {
        if (comm_id != other.comm_id) {
            return comm_id < other.comm_id;
        }
        else if (rank != other.rank) {
            return rank < other.rank;
        }
        else if (comm_size != other.comm_size) {
            return comm_size < other.comm_size;
        }
        else if (op_name != other.op_name) {
            return op_name < other.op_name;
        }
        else if (reduction != other.reduction) {
            return reduction < other.reduction;
        }
        else if (dtype != other.dtype) {
            return dtype < other.dtype;
        }
        else if (count != other.count) {
            return count < other.count;
        }
        return false;
    }

    void print() {
        std::cout << "Key:" << comm_id << "," << rank << "," << comm_size << "," << op_name << ","
                  << static_cast<int>(reduction) << "," << count << "," << static_cast<int>(dtype)
                  << std::endl;
    }

    static std::string get_header() {
        std::stringstream ss;
        ss << "operation, comm_id, rank, comm_size, msg_count, datatype";
        return ss.str();
    }

    std::string get_print() {
        std::stringstream ss;
        std::map<int, std::string> reduction_strings = {
            { static_cast<int>(ccl::reduction::avg), "_avg" },
            { static_cast<int>(ccl::reduction::custom), "_custom" },
            { static_cast<int>(ccl::reduction::max), "_max" },
            { static_cast<int>(ccl::reduction::min), "_min" },
            { static_cast<int>(ccl::reduction::none), "" },
            { static_cast<int>(ccl::reduction::prod), "_prod" },
            { static_cast<int>(ccl::reduction::sum), "_sum" }
        };

        std::string reduction_string = "";
        if (reduction_strings.find(static_cast<int>(reduction)) != reduction_strings.end())
            reduction_string = reduction_strings[static_cast<int>(reduction)];

        std::map<int, std::string> dtype_strings = {
            { static_cast<int>(ccl::datatype::int8), "int8" },
            { static_cast<int>(ccl::datatype::uint8), "uint8" },
            { static_cast<int>(ccl::datatype::int16), "int16" },
            { static_cast<int>(ccl::datatype::uint16), "uint16" },
            { static_cast<int>(ccl::datatype::int32), "int32" },
            { static_cast<int>(ccl::datatype::uint32), "uint32" },
            { static_cast<int>(ccl::datatype::int64), "int64" },
            { static_cast<int>(ccl::datatype::uint64), "uint64" },
            { static_cast<int>(ccl::datatype::float16), "float16" },
            { static_cast<int>(ccl::datatype::float32), "float32" },
            { static_cast<int>(ccl::datatype::float64), "float64" },
            { static_cast<int>(ccl::datatype::bfloat16), "bfloat16" },
        };

        std::string dtype_string = "";
        if (dtype_strings.find(static_cast<int>(dtype)) != dtype_strings.end())
            dtype_string = dtype_strings[static_cast<int>(dtype)];

        ss << op_name << reduction_string << "," << comm_id << "," << rank << "," << comm_size
           << "," << count << "," << dtype_string;
        return ss.str();
    }
};

class comm_session {
public:
    comm_profile_key comm_key;
    comm_session() {}
    void set_key(comm_profile_key coll_prof_key) {
        comm_key = coll_prof_key;
    }
};

class comm_exec_time {
public: //TODO: make private
    std::chrono::time_point<std::chrono::high_resolution_clock> submit_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> return_time;

    static const std::string get_header() {
        return "[duration-us;*]";
    }

    std::string get_print() {
        std::stringstream ss;
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(return_time - submit_time);
        ss << duration.count() << ";";
        return ss.str();
    }

    // Note: alternatively return a vector instead of string
    //       s.t. the report generator can control the formatting
};

class comm_exec_info {
public:
    int count = 0;
    std::vector<comm_exec_time>
        exec_times; // TODO: should we track a call ID to access/update the exec_times element?

    static std::string get_header() {
        std::stringstream ss;
        ss << "op_count, " << comm_exec_time::get_header();
        return ss.str();
    }
    std::string get_print() {
        std::stringstream ss;

        // print member count
        ss << count << ",";
        // print times
        ss << "[";
        for (size_t i = 0; i < exec_times.size(); i++) {
            ss << exec_times[i].get_print();
        }
        ss << "]";
        return ss.str();
    }
};

/* Info per event key */
class comm_profile_info {
public:
    comm_exec_info comm_events_exec_info;
    // Note: add more members as needed, e.g., comm_algo_info for algo related profiling
    comm_profile_info() {}
    static std::string get_header() {
        return comm_exec_info::get_header();
    }
    std::string get_print() {
        return comm_events_exec_info.get_print();
    }
};

class comm_profile {
public:
    //std::set<comm_profile_key> comm_profile_keys;
    std::map<comm_profile_key, comm_profile_info> comm_events_profile_kv;

    void record_prof_key(const comm_profile_key& key) {
        if (skip_profiling(key.rank))
            return;

        if (comm_events_profile_kv.find(key) == comm_events_profile_kv.end()) {
            comm_events_profile_kv.insert(std::make_pair(key, comm_profile_info()));
        }
    }

    void record_prof_info_call_count(comm_profile_key key) {
        this->comm_events_profile_kv[key].comm_events_exec_info.count += 1;
    }

    void record_prof_info_submit_time(
        comm_profile_key key,
        std::chrono::time_point<std::chrono::high_resolution_clock> t) {
        if (skip_profiling(key.rank))
            return;
        // NOTE/assumption: the updated time is for the last element in the time series list
        this->comm_events_profile_kv[key].comm_events_exec_info.exec_times.push_back(
            comm_exec_time());
        int id = this->comm_events_profile_kv[key].comm_events_exec_info.exec_times.size() - 1;
        this->comm_events_profile_kv[key].comm_events_exec_info.exec_times[id].submit_time = t;
    }
    void record_prof_info_return_time(
        comm_profile_key key,
        std::chrono::time_point<std::chrono::high_resolution_clock> t) {
        if (skip_profiling(key.rank))
            return;
        // NOTE/assumption: the updated time is for the last element in the time series list
        int id = this->comm_events_profile_kv[key].comm_events_exec_info.exec_times.size() - 1;
        this->comm_events_profile_kv[key].comm_events_exec_info.exec_times[id].return_time = t;
    }
    void print_profiles() {
        std::cout
            << "Profiling_note: This feature is currently a tech preview. The feature, the information content, and the output format are subject to change in the future releases. If you have any comment, please submit it to the profiling RFC under the oneCCL github repo.\n";
        std::string header = comm_profile_key::get_header() + "," + comm_profile_info::get_header();
        std::cout << "Profiling_Header:, " << header << std::endl;

        for (std::map<comm_profile_key, comm_profile_info>::iterator it =
                 this->comm_events_profile_kv.begin();
             it != comm_events_profile_kv.end();
             ++it) {
            comm_profile_key key = (comm_profile_key)it->first;
            comm_profile_info profile_info = (comm_profile_info)it->second;
            // collect from profile_info
            std::string key_str = key.get_print();
            std::string profile_info_str = profile_info.get_print();
            std::cout << "Profiling_log:, " << key_str << "," << profile_info_str << std::endl;
        }
    }

    ~comm_profile() {
        if (is_profiling_enabled()) {
            print_profiles();
        }
    }
};

extern comm_profile comm_profiles;

comm_profile_key get_profile_key(int comm_id,
                                 int rank,
                                 int comm_size,
                                 std::string op_name,
                                 ccl::reduction reduction,
                                 ccl::datatype dtype,
                                 size_t count);

void profiler_record_comm_event_enter(const ccl_comm* comm,
                                      std::string op_name,
                                      ccl::reduction reduction,
                                      ccl::datatype dtype,
                                      size_t count,
                                      comm_session& comm_event_session);
void profiler_record_comm_event_exit(int rank, comm_session& comm_event_session);

#endif //CCL_ENABLE_PROFILING

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

#include "comp/comp.hpp"

#include <map>
#include <unordered_map>
#include <vector>

class ccl_comm;

enum ccl_cache_key_type {
    ccl_cache_key_full,
    ccl_cache_key_match_id,

    ccl_cache_key_last_value
};

const char* ccl_cache_key_type_to_str(ccl_cache_key_type type);

class ccl_sched_key {
private:
    size_t hasher_result = 0;

public:
    ccl_sched_key() = default;
    ~ccl_sched_key() = default;
    ccl_sched_key(ccl_sched_key&& other) = default;
    ccl_sched_key& operator=(ccl_sched_key&& other) = default;

    ccl_sched_key(const ccl_coll_param& param, const ccl_coll_attr& attr);
    void set(const ccl_coll_param& param, const ccl_coll_attr& attr);
    bool check(const ccl_coll_param& param, const ccl_coll_attr& attr) const;

    size_t get_hasher_result() const {
        return hasher_result;
    }

    void set_hasher_result(size_t value) {
        has_hasher_result = true;
        hasher_result = value;
    }

    bool has_hasher_result = false;

    struct ccl_sched_key_inner_fields {
        ccl_coll_type ctype = ccl_coll_undefined;
        void* buf1 = nullptr; /* non-data buffer which can be used for caching */
        void* buf2 = nullptr; /* non-data buffer which can be used for caching */
        ccl::datatype dtype = ccl::datatype::int8;
        ccl::reduction reduction = ccl::reduction::sum;
        size_t count1 = 0;
        size_t count2 = 0;
        int root = 0;
        int peer_rank = CCL_INVALID_PEER_RANK_IDX;
        int group_id = CCL_INVALID_GROUP_IDX;
        const ccl_comm* comm = nullptr;
        ccl::reduction_fn reduction_fn = nullptr;
    };

    /* inner structure for bit comparison */
    ccl_sched_key_inner_fields f;

    std::vector<size_t> vec1;
    std::vector<size_t> vec2;

    std::string match_id{};

    bool operator==(const ccl_sched_key& k) const;

    void print() const;

    static std::map<ccl_cache_key_type, std::string> key_type_names;
};

class ccl_sched_key_hasher {
public:
    size_t operator()(const ccl_sched_key& k) const;

private:
    std::hash<std::string> string_hasher{};
};

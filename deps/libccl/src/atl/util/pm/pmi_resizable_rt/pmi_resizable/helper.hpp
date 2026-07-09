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

#ifndef HELPER_H_INCLUDED
#define HELPER_H_INCLUDED

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <ctype.h>
#include <iostream>
#include <memory>
#include <utility>

#include "def.h"
#include "shift_list.hpp"
#include "kvs_keeper.hpp"
#include "kvs/ikvs_wrapper.h"

extern int my_rank;
extern size_t count_pods;
extern size_t barrier_num;
extern size_t up_idx;
extern size_t applied;

extern std::list<int> killed_ranks;
extern int killed_ranks_count;

extern std::list<int> new_ranks;
extern int new_ranks_count;

class helper {
public:
    helper() = delete;
    explicit helper(std::shared_ptr<ikvs_wrapper> k) : kvs(std::move(k)){};
    ~helper() = default;

    kvs_status_t get_update_ranks(void);

    kvs_status_t get_replica_size(size_t& replica_size);

    kvs_status_t wait_accept(void);

    kvs_status_t update(const std::list<shift_rank_t>& list,
                        std::list<int>& dead_up_idx,
                        int root_rank);

    kvs_status_t up_pods_count(void);

    void get_shift(std::list<shift_rank_t>& list);

    kvs_status_t reg_rank(void);

    kvs_status_t get_barrier_idx(size_t& barrier_num_out);

    kvs_status_t up_kvs_new_and_dead(void);

    void keep_first_n_up(int prev_new_ranks_count, int prev_killed_ranks_count);

    kvs_status_t get_new_root(int* old_root);

    /*Work with KVS, new*/
    kvs_status_t set_value(const std::string& kvs_name,
                           const std::string& kvs_key,
                           const std::string& kvs_val);

    kvs_status_t remove_name_key(const std::string& kvs_name, const std::string& kvs_key);

    kvs_status_t get_value_by_name_key(const std::string& kvs_name,
                                       const std::string& kvs_key,
                                       std::string& kvs_val);

    size_t init(const char* main_addr);

    kvs_status_t main_server_address_reserve(char* main_addr);

    kvs_status_t get_count_names(const std::string& kvs_name, size_t& count_names);

    kvs_status_t finalize(void);

    kvs_status_t get_keys_values_by_name(const std::string& kvs_name,
                                         std::vector<std::string>& kvs_keys,
                                         std::vector<std::string>& kvs_values,
                                         size_t& count);

    /*Work with KVS, new*/

private:
    kvs_status_t replace_str(char* str, int old_rank, int new_rank);

    kvs_status_t update_ranks(int* old_count, std::list<int>& origin_list, const char* kvs_name);

    kvs_status_t clean_dead_pods_info(std::list<int>& dead_up_idx);

    kvs_status_t accept_new_ranks(const std::list<shift_rank_t>& cur_list);

    kvs_status_t update_kvs_info(int new_rank);

    kvs_status_t move_to_new_rank(int new_rank);

    kvs_status_t update_my_info(const std::list<shift_rank_t>& list);

    kvs_status_t post_my_info(void);

    kvs_status_t get_val_count(const char* name, const char* val, size_t& res);

    kvs_status_t get_occupied_ranks_count(char* rank, size_t& res);

    kvs_status_t get_count_requested_ranks(char* rank, size_t& count_pods_with_my_rank);

    kvs_status_t occupied_rank(char* rank);

    kvs_status_t up_kvs(const char* new_kvs_name, const char* old_kvs_name);
    std::shared_ptr<ikvs_wrapper> kvs;
};
#endif

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

#include <unistd.h>

#include "util/pm/pmi_resizable_rt/pmi_resizable/def.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/kvs_keeper.hpp"
#include "pmi_resizable_simple.h"
#include "util/pm/codec/pm_rt_codec.h"

#define RESIZABLE_PMI_RT_KEY_FORMAT "%s-%d"
#define RANKS_PER_THREAD            "RANKS_PER_THREAD"
#define PROCESS_THREAD_NAME         "PROCESS_THREAD_NAME"

#define REQUESTED_RANK_TO_NAME "REQUESTED_RANK_TO_NAME"
#define GLOBAL_NAME_TO_RANK    "GLOBAL_NAME_TO_RANK"
#define GLOBAL_RANK_TO_NAME    "GLOBAL_RANK_TO_NAME"
#define LOCAL_KVS_ID           "LOCAL_KVS_ID"

pmi_resizable_simple::pmi_resizable_simple(int size,
                                           const std::vector<int>& ranks,
                                           std::shared_ptr<ikvs_wrapper> k,
                                           const char* main_addr)
        : comm_size(size),
          assigned_proc_idx(0),
          ranks(ranks),
          k(k),
          main_addr(main_addr),
          max_keylen(MAX_KVS_KEY_LENGTH),
          max_vallen(MAX_KVS_VAL_LENGTH) {}

int pmi_resizable_simple::is_pm_resize_enabled() {
    return 0;
}

atl_status_t pmi_resizable_simple::pmrt_init() {
    (void)main_addr;

    char* kvs_get_timeout_str = getenv("CCL_KVS_GET_TIMEOUT");
    if (kvs_get_timeout_str) {
        kvs_get_timeout = atoi(kvs_get_timeout_str);
    }

    local_id = 0;
    val_storage = (char*)calloc(1, max_vallen);
    if (!val_storage) {
        LOG_ERROR("mem alloc failed");
        return ATL_STATUS_FAILURE;
    }
    /*TODO: add sort, ranks should increase continiusly*/
    if (ranks[0] == 0) {
        size_t tmp_local_id;
        ATL_CHECK_STATUS(get_local_kvs_id(tmp_local_id), "failed to get local id");
        tmp_local_id++;
        ATL_CHECK_STATUS(set_local_kvs_id(tmp_local_id), "failed to set local id");
    }

    ATL_CHECK_STATUS(make_requested_info(), "failed to make requested info");
    /* extension */
    //    make_map_requested2global();
    /**/
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::make_requested_info() {
    ATL_CHECK_STATUS(register_first_rank_idx_and_rank_count(), "failed to register ranks");
    ATL_CHECK_STATUS(assign_thread_idx_and_fill_ranks_per_thread_map(), "failed to fill map");

    ATL_CHECK_STATUS(get_local_kvs_id(local_id), "failed to get local id");
    ATL_CHECK_STATUS(register_my_proc_name(), "failed to register proc name");
    ATL_CHECK_STATUS(get_my_proc_idx_and_proc_count(), "failed to get proc idx");
    calculate_local_thread_idx();
    ATL_CHECK_STATUS(remove_initial_data(), "failed to remove initial data");
    ATL_CHECK_STATUS(pmrt_barrier_full(), "full barrier failed");
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::pmrt_main_addr_reserve(char* addr) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_resizable_simple::pmrt_set_resize_function(atl_resize_fn_t resize_fn) {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_resizable_simple::pmrt_update() {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_resizable_simple::pmrt_wait_notification() {
    return ATL_STATUS_UNSUPPORTED;
}

atl_status_t pmi_resizable_simple::pmrt_finalize() {
    is_finalized = true;
    free(val_storage);

    if (getenv("CCL_PMI_FORCE_FINALIZE")) {
        LOG_WARN("skip pmi_resizable_simple::pmrt_finalize\n");
        return ATL_STATUS_SUCCESS;
    }

    char kvs_name[MAX_KVS_NAME_LENGTH];
    char kvs_key[MAX_KVS_KEY_LENGTH];
    char kvs_val[MAX_KVS_VAL_LENGTH];

    while (cut_head(kvs_name, kvs_key, kvs_val, ST_CLIENT)) {
        KVS_2_ATL_CHECK_STATUS(k->kvs_remove_name_key(kvs_name, kvs_key), "failed to remove info");
    }
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::pmrt_barrier() {
    size_t min_barrier_num;
    char barrier_num_str[INT_STR_SIZE];

    ATL_SET_STR(barrier_num_str, INT_STR_SIZE, SIZE_T_TEMPLATE, barrier_num);

    ATL_CHECK_STATUS(
        kvs_set_value(KVS_BARRIER, std::to_string(assigned_proc_idx).c_str(), barrier_num_str),
        "failed to set barrier num");

    do {
        ATL_CHECK_STATUS(get_barrier_idx(min_barrier_num), "failed to get barrier num");
    } while (min_barrier_num != barrier_num);

    barrier_num++;
    if (barrier_num > BARRIER_NUM_MAX)
        barrier_num = 0;
    return ATL_STATUS_SUCCESS;
}
atl_status_t pmi_resizable_simple::pmrt_barrier_full() {
    size_t min_barrier_num;
    char barrier_num_str[INT_STR_SIZE];

    ATL_SET_STR(barrier_num_str, INT_STR_SIZE, SIZE_T_TEMPLATE, barrier_num_full);

    ATL_CHECK_STATUS(
        kvs_set_value(
            KVS_BARRIER_FULL, std::to_string(assigned_thread_idx).c_str(), barrier_num_str),
        "failed to set barrier num");

    ATL_CHECK_STATUS(get_barrier_full_idx(min_barrier_num), "failed to get barrier num");
    while (min_barrier_num != barrier_num) {
        ATL_CHECK_STATUS(get_barrier_idx(min_barrier_num), "failed to get barrier num");
    }

    barrier_num_full++;
    if (barrier_num_full > BARRIER_NUM_MAX)
        barrier_num_full = 0;
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::get_barrier_full_idx(size_t& res) {
    res = 0;
    size_t thread_count = ranks_per_thread_map.size();

    ATL_CHECK_STATUS(kvs_get_value(KVS_BARRIER_FULL, std::to_string(0).c_str(), val_storage),
                     "failed to get barrier idx");

    size_t min_barrier_idx = atoi(val_storage);
    size_t barrier_idx;
    for (size_t i = 1; i < thread_count; i++) {
        ATL_CHECK_STATUS(kvs_get_value(KVS_BARRIER_FULL, std::to_string(i).c_str(), val_storage),
                         "failed to get barrier idx");

        barrier_idx = atoi(val_storage);

        if (min_barrier_idx > barrier_idx)
            min_barrier_idx = barrier_idx;
    }
    res = min_barrier_idx;
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::pmrt_kvs_put(char* kvs_key,
                                                int proc_idx,
                                                const void* kvs_val,
                                                size_t kvs_val_len) {
    int ret;
    std::vector<char> key_storage(max_keylen);
    if (kvs_val_len > max_vallen)
        return ATL_STATUS_FAILURE;

    ret = snprintf(
        key_storage.data(), max_keylen - 1, RESIZABLE_PMI_RT_KEY_FORMAT, kvs_key, proc_idx);
    if (ret < 0) {
        LOG_ERROR("sprintf failed");
        return ATL_STATUS_FAILURE;
    }

    ret = encode(kvs_val, kvs_val_len, val_storage, max_vallen);
    if (ret) {
        LOG_ERROR("encode failed");
        return ATL_STATUS_FAILURE;
    }

    ATL_CHECK_STATUS(kvs_set_value(KVS_NAME, key_storage.data(), val_storage), "failed to set val");

    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::pmrt_kvs_get(char* kvs_key,
                                                int proc_idx,
                                                void* kvs_val,
                                                size_t kvs_val_len) {
    int ret;
    std::vector<char> key_storage(max_keylen);

    ret = snprintf(
        key_storage.data(), max_keylen - 1, RESIZABLE_PMI_RT_KEY_FORMAT, kvs_key, proc_idx);
    if (ret < 0) {
        LOG_ERROR("sprintf failed");
        return ATL_STATUS_FAILURE;
    }

    ATL_CHECK_STATUS(kvs_get_value(KVS_NAME, key_storage.data(), val_storage), "failed to get val");

    ret = decode(val_storage, kvs_val, kvs_val_len);
    if (ret) {
        LOG_ERROR("decode failed");
        return ATL_STATUS_FAILURE;
    }

    return ATL_STATUS_SUCCESS;
}

int pmi_resizable_simple::get_size() {
    return threads_per_proc.size();
}

int pmi_resizable_simple::get_rank() {
    return assigned_proc_idx;
}

size_t pmi_resizable_simple::get_local_thread_idx() {
    return local_thread_idx;
}

atl_status_t pmi_resizable_simple::kvs_set_value(const char* kvs_name,
                                                 const char* key,
                                                 const char* value) {
    std::string result_kvs_name = std::string(kvs_name) + std::to_string(local_id);
    put_key(result_kvs_name.c_str(), key, value, ST_CLIENT);

    return (k->kvs_set_value(result_kvs_name.c_str(), key, value) == KVS_STATUS_SUCCESS)
               ? ATL_STATUS_SUCCESS
               : ATL_STATUS_FAILURE;
}

atl_status_t pmi_resizable_simple::kvs_get_value(const char* kvs_name,
                                                 const char* key,
                                                 char* value) {
    std::string result_kvs_name = std::string(kvs_name) + std::to_string(local_id);

    time_t start_time = time(NULL);
    size_t kvs_get_time = 0;
    std::string value_vec;

    do {
        KVS_2_ATL_CHECK_STATUS(k->kvs_get_value_by_name_key(result_kvs_name, key, value_vec),
                               "failed to get value");
        kvs_get_time = time(NULL) - start_time;
    } while (value_vec.empty() && kvs_get_time < kvs_get_timeout);

    if (kvs_get_time >= kvs_get_timeout) {
        LOG_ERROR("KVS get error: timeout limit: ",
                  kvs_get_time,
                  " > ",
                  kvs_get_timeout,
                  ", prefix: ",
                  result_kvs_name.c_str(),
                  ", key: ",
                  key);
        return ATL_STATUS_FAILURE;
    }
    snprintf(value, value_vec.length(), "%s", value_vec.c_str());
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::get_barrier_idx(size_t& barrier_num_out) {
    size_t proc_count = threads_per_proc.size();
    barrier_num_out = 0;

    ATL_CHECK_STATUS(kvs_get_value(KVS_BARRIER, std::to_string(0).c_str(), val_storage),
                     "failed to get barrier");

    size_t min_barrier_idx = atoi(val_storage);
    size_t barrier_idx;
    for (size_t i = 1; i < proc_count; i++) {
        ATL_CHECK_STATUS(kvs_get_value(KVS_BARRIER, std::to_string(i).c_str(), val_storage),
                         "failed to get barrier");
        barrier_idx = atoi(val_storage);

        if (min_barrier_idx > barrier_idx)
            min_barrier_idx = barrier_idx;
    }

    barrier_num_out = min_barrier_idx;
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::register_first_rank_idx_and_rank_count() {
    return kvs_set_value(
        RANKS_PER_THREAD, std::to_string(ranks[0]).c_str(), std::to_string(ranks.size()).c_str());
}

atl_status_t pmi_resizable_simple::assign_thread_idx_and_fill_ranks_per_thread_map() {
    int rank_count = 0;
    int ranks_per_thread;
    while (rank_count < comm_size) {
        if (rank_count == ranks[0]) {
            assigned_thread_idx = ranks_per_thread_map.size();
        }
        ATL_CHECK_STATUS(
            kvs_get_value(RANKS_PER_THREAD, std::to_string(rank_count).c_str(), val_storage),
            "failed to get ranks");

        ranks_per_thread = std::atoi(val_storage);
        ranks_per_thread_map.push_back(ranks_per_thread);
        rank_count += ranks_per_thread;
    }
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::register_my_proc_name() {
    int my_pid = getpid();
    const int hostname_len = 1024;
    char hostname[hostname_len];
    int ret = gethostname(hostname, hostname_len);
    if (ret) {
        LOG_ERROR("gethostname error: ", strerror(errno));
        return ATL_STATUS_FAILURE;
    }
    my_process_name = std::string(hostname) + std::to_string(my_pid);

    return kvs_set_value(
        PROCESS_THREAD_NAME, std::to_string(assigned_thread_idx).c_str(), my_process_name.c_str());
}

atl_status_t pmi_resizable_simple::get_my_proc_idx_and_proc_count() {
    std::map<std::string, int> proc_name_to_rank;
    std::map<std::string, int>::iterator it;
    int rank;
    for (size_t i = 0; i < ranks_per_thread_map.size(); i++) {
        ATL_CHECK_STATUS(kvs_get_value(PROCESS_THREAD_NAME, std::to_string(i).c_str(), val_storage),
                         "failed to get proc name");

        it = proc_name_to_rank.find(val_storage);
        if (it == proc_name_to_rank.end()) {
            rank = threads_per_proc.size();
            if (!my_process_name.compare(val_storage)) {
                assigned_proc_idx = rank;
                if (assigned_thread_idx == i) {
                    ATL_CHECK_STATUS(kvs_set_value(REQUESTED_RANK_TO_NAME,
                                                   std::to_string(assigned_proc_idx).c_str(),
                                                   my_process_name.c_str()),
                                     "failed to set proc name");
                }
            }
            proc_name_to_rank[val_storage] = rank;
            threads_per_proc[rank].push_back(i);
        }
        else {
            threads_per_proc[it->second].push_back(i);
        }
    }
    return ATL_STATUS_SUCCESS;
}

void pmi_resizable_simple::calculate_local_thread_idx() {
    local_thread_idx = 0;
    for (auto it = threads_per_proc[assigned_proc_idx].begin();
         it != threads_per_proc[assigned_proc_idx].end();
         it++) {
        if (assigned_thread_idx == *it)
            break;
        local_thread_idx++;
    }
}
// TODO: uncomment and fix it for collect real request ranks
#if 0
atl_status_t pmi_resizable_simple::kvs_iget_value(const char* kvs_name,
                                                  const char* key,
                                                  char* value) {
    std::string result_kvs_name = std::string(kvs_name) + std::to_string(local_id);
    return k->kvs_get_value_by_name_key(result_kvs_name.c_str(), key, value) == KVS_STATUS_SUCCESS
               ? ATL_STATUS_SUCCESS
               : ATL_STATUS_FAILURE;
}
atl_status_t pmi_resizable_simple::make_map_requested2global() {
    char global_rank_str[MAX_KVS_VAL_LENGTH];
    char process_name[MAX_KVS_VAL_LENGTH];
    size_t size = get_size();
    requested2global.resize(size);
    ATL_CHECK_STATUS(pmrt_barrier_full(), "make_map_requested2global: full barrier failed");
    for (size_t i = 0; i < size; i++) {
        ATL_CHECK_STATUS(
            kvs_get_value(REQUESTED_RANK_TO_NAME, std::to_string(i).c_str(), process_name),
            "make_map_requested2global: failed to get proc name");
        ATL_CHECK_STATUS(kvs_iget_value(GLOBAL_NAME_TO_RANK, process_name, global_rank_str),
                         "make_map_requested2global: failed to get glob rank");
        if (strlen(global_rank_str) == 0) {
            if (!my_process_name.compare(process_name)) {
                int free_glob_rank = 0;
                ATL_CHECK_STATUS(
                    kvs_iget_value(
                        GLOBAL_RANK_TO_NAME, std::to_string(free_glob_rank).c_str(), process_name),
                    "make_map_requested2global: failed to get proc name");
                while (strlen(process_name) != 0) {
                    free_glob_rank++;
                    ATL_CHECK_STATUS(kvs_iget_value(GLOBAL_RANK_TO_NAME,
                                                    std::to_string(free_glob_rank).c_str(),
                                                    process_name),
                                     "make_map_requested2global: failed to get proc name");
                }
// Fixme: need additional sinchronization
                ATL_CHECK_STATUS(kvs_set_value(GLOBAL_RANK_TO_NAME,
                                               std::to_string(free_glob_rank).c_str(),
                                               my_process_name.c_str()),
                                 "make_map_requested2global: failed to set proc name");
                ATL_CHECK_STATUS(kvs_set_value(GLOBAL_NAME_TO_RANK,
                                               my_process_name.c_str(),
                                               std::to_string(free_glob_rank).c_str()),
                                 "make_map_requested2global: failed to set free rank info");
            }
            ATL_CHECK_STATUS(kvs_get_value(GLOBAL_NAME_TO_RANK, process_name, global_rank_str),
                             "make_map_requested2global: failed to get rank info");
        }
        requested2global[i] = atoi(global_rank_str);
    }
    ATL_CHECK_STATUS(pmrt_barrier_full(), "make_map_requested2global: full barrier failed");
    return ATL_STATUS_SUCCESS;
}
#endif

atl_status_t pmi_resizable_simple::get_local_kvs_id(size_t& res) {
    std::string local_kvs_id;
    res = 0;
    /*TODO: change it for collect local_per_rank id, not global*/
    KVS_2_ATL_CHECK_STATUS(k->kvs_get_value_by_name_key(LOCAL_KVS_ID, "ID", local_kvs_id),
                           "failed to get local kvs id");
    res = atoi(local_kvs_id.c_str());
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_resizable_simple::set_local_kvs_id(size_t local_kvs_id) {
    /*TODO: change it for collect local_per_rank id, not global*/
    put_key(LOCAL_KVS_ID, "ID", std::to_string(local_kvs_id).c_str(), ST_CLIENT);
    return (k->kvs_set_value(LOCAL_KVS_ID, "ID", std::to_string(local_kvs_id)) ==
            KVS_STATUS_SUCCESS)
               ? ATL_STATUS_SUCCESS
               : ATL_STATUS_FAILURE;
}
pmi_resizable_simple::~pmi_resizable_simple() {
    if (!is_finalized) {
        CCL_THROW_IF_NOT(pmrt_finalize() == ATL_STATUS_SUCCESS, "~pmi_resizable_simple: failed");
    }
}
atl_status_t pmi_resizable_simple::remove_initial_data() {
    std::string result_kvs_name = std::string(RANKS_PER_THREAD) + std::to_string(0);
    remove_val(result_kvs_name.c_str(), std::to_string(ranks[0]).c_str(), ST_CLIENT);
    return k->kvs_remove_name_key(result_kvs_name, std::to_string(ranks[0])) == KVS_STATUS_SUCCESS
               ? ATL_STATUS_SUCCESS
               : ATL_STATUS_FAILURE;
}

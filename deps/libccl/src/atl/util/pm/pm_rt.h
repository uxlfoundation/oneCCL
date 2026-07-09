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

#ifndef PM_RT_H
#define PM_RT_H

#include "atl_def.h"
#include "common/api_wrapper/pmix_api_wrapper.hpp"

typedef struct pm_rt_desc pm_rt_desc_t;

typedef struct pm_rt_ops {
    void (*finalize)(pm_rt_desc_t *pmrt_desc);
    void (*barrier)(pm_rt_desc_t *pmrt_desc);
    atl_status_t (*update)(int *proc_idx, int *proc_count);
    atl_status_t (*wait_notification)(void);
} pm_rt_ops_t;

typedef struct pm_rt_kvs_ops {
    atl_status_t (*put)(pm_rt_desc_t *pmrt_desc,
                        char *kvs_key,
                        int proc_idx,
                        const void *kvs_val,
                        size_t kvs_val_len);
    atl_status_t (*get)(pm_rt_desc_t *pmrt_desc,
                        char *kvs_key,
                        int proc_idx,
                        void *kvs_val,
                        size_t kvs_val_len);
} pm_rt_kvs_ops_t;

struct pm_rt_desc {
    pm_rt_ops_t *ops;
    pm_rt_kvs_ops_t *kvs_ops;
};

#ifdef __cplusplus
class ipmi {
public:
    virtual ~ipmi() noexcept(false){};

    virtual int is_pm_resize_enabled() = 0;

    virtual atl_status_t pmrt_main_addr_reserve(char *main_addr) = 0;

    virtual atl_status_t pmrt_set_resize_function(atl_resize_fn_t resize_fn) = 0;

    virtual atl_status_t pmrt_update() = 0;

    virtual atl_status_t pmrt_wait_notification() = 0;

    virtual atl_status_t pmrt_finalize() = 0;

    virtual atl_status_t pmrt_barrier() = 0;

    virtual atl_status_t pmrt_kvs_put(char *kvs_key,
                                      int proc_idx,
                                      const void *kvs_val,
                                      size_t kvs_val_len) = 0;

    virtual atl_status_t pmrt_kvs_get(char *kvs_key,
                                      int proc_idx,
                                      void *kvs_val,
                                      size_t kvs_val_len) = 0;

    virtual int get_rank() = 0;

    virtual int get_size() = 0;

    virtual size_t get_local_thread_idx() = 0;

    virtual atl_status_t get_local_kvs_id(size_t &res) = 0;

    virtual atl_status_t set_local_kvs_id(size_t local_kvs_id) = 0;

    virtual size_t get_threads_per_process() = 0;

    virtual size_t get_ranks_per_process() = 0;

    virtual atl_status_t pmrt_init() = 0;
};

// Currently, ccl_pmix relies on the PMI object to obtain rank and size.
// Introducing ccl_pmix serves as a trade-off to encapsulate #ifdefs within an interface, but it is not part of the PMI class.
// Potentially, this class could become a part of the PMI interface. However, this would require restructuring to achieve:
// 1) Implementation of basic functionality such as get_size, get_rank, and pmrt_init.
// 2) Adaptation to enable object creation within atl_ofi_comm.
// 3) Resolving the dependency on get_pmix_local_coord, as it is invoked during the initialization phase to retrieve the namespace_name for fence and get operations.
// 4) Evaluation of whether ccl_pmix should be integrated into ipmi or pmi_resizable_simple.
// and something more...

#ifdef CCL_ENABLE_PMIX
class ccl_pmix {
public:
    static void fence(const std::string &namespace_name,
                      int rank,
                      int group_size,
                      int global_rank = -1) {
        if (namespace_name.empty() || group_size <= 0) {
            CCL_THROW("Invalid arguments: namespace_name is null or group_size is non-positive.");
        }

        pmix_proc_t proc = initialize_proc(namespace_name, rank);

        if (global_rank > -1 && global_rank >= group_size) {
            LOG_WARN("Mismatch detected - global rank (",
                     rank,
                     ") does not match group_size for fence (",
                     group_size,
                     "). This may indicate the creation of a sub-communicator,"
                     " which could lead to unexpected behavior.");
        }

        std::vector<pmix_proc_t> sync_group(group_size);
        for (int i = 0; i < group_size; i++) {
            sync_group[i] = initialize_proc(namespace_name, i);
            sync_group[i].rank = i;
        }

        pmix_info_t *info;
        PMIX_INFO_CREATE(info, 1);
        int flag = 1;
        PMIx_Info_load(info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);

        pmix_status_t status = PMIx_Fence(sync_group.data(), sync_group.size(), info, 1);
        if (status != PMIX_SUCCESS) {
            CCL_THROW("PMIx_Fence failed with status: ", status);
        }

        for (auto &group_proc : sync_group) {
            PMIX_PROC_DESTRUCT(&group_proc);
        }

        PMIX_PROC_DESTRUCT(&proc);
        PMIX_INFO_FREE(info, 1);
    }

    static void get(const std::string &namespace_name,
                    int rank,
                    const std::string &key,
                    pmix_value_t **ret_value) {
        if (namespace_name.empty() || key.empty()) {
            CCL_THROW("Invalid arguments: namespace or key is empty.");
        }

        pmix_proc_t proc = initialize_proc(namespace_name, rank);

        LOG_DEBUG("Calling PMIx_Get rank: ", rank, ", key: ", key);

        pmix_status_t status = PMIx_Get(&proc, key.c_str(), nullptr, 0, ret_value);
        if (status != PMIX_SUCCESS) {
            CCL_THROW("PMIx_Get failed for key: ", key, " with status: ", status);
        }
        else if (!*ret_value) {
            CCL_THROW("PMIx_Get returned null value for key: ", key);
        }

        PMIX_PROC_DESTRUCT(&proc);
    }

    static void put(const std::string &key, pmix_value_t &value) {
        pmix_status_t status = PMIx_Put(PMIX_GLOBAL, key.c_str(), &value);
        if (status != PMIX_SUCCESS) {
            CCL_THROW("PMIx_Put failed for key: ", key, " with status: ", status);
        }
    }

    static void commit() {
        pmix_status_t status = PMIx_Commit();
        if (status != PMIX_SUCCESS) {
            CCL_THROW("PMIx_Commit failed with status: ", status);
        }
    }

    static void value_load(pmix_value_t &value, const void *data, pmix_data_type_t type) {
        pmix_status_t status = PMIx_Value_load(&value, data, type);
        if (status != PMIX_SUCCESS) {
            CCL_THROW("PMIx_Value_load failed with status: ", status);
        }
    }

    static pmix_proc_t initialize_proc(const std::string &namespace_name, int rank) {
        pmix_proc_t proc;
        PMIX_PROC_CONSTRUCT(&proc);

        if (namespace_name.size() >= PMIX_MAX_NSLEN) {
            LOG_WARN(
                "proc.namespace does not have sufficient size to store namespace_name; truncating to ",
                PMIX_MAX_NSLEN - 1,
                " characters.");
        }

        std::string truncated_name = namespace_name.substr(0, PMIX_MAX_NSLEN - 1);
        strncpy(proc.nspace, truncated_name.c_str(), PMIX_MAX_NSLEN - 1);
        proc.nspace[PMIX_MAX_NSLEN - 1] = '\0';
        proc.rank = rank;

        return proc;
    }
};
#else // CCL_ENABLE_PMIX
class ccl_pmix {
public:
    static void fence(const std::string &, int, int, int) {
        CCL_THROW("ccl_pmix::fence is not implemented without CCL_ENABLE_PMIX");
    }

    static void get(const std::string &, int, const std::string &, pmix_value_t **) {
        CCL_THROW("ccl_pmix::get is not implemented without CCL_ENABLE_PMIX");
    }

    static void put(const std::string &, pmix_value_t &) {
        CCL_THROW("ccl_pmix::put is not implemented without CCL_ENABLE_PMIX");
    }

    static void commit() {
        CCL_THROW("ccl_pmix::commit is not implemented without CCL_ENABLE_PMIX");
    }

    static void value_load(pmix_value_t &, const void *, pmix_data_type_t) {
        CCL_THROW("ccl_pmix::value_load is not implemented without CCL_ENABLE_PMIX");
    }
};
#endif // CCL_ENABLE_PMIX

#endif
#endif // PM_RT_H

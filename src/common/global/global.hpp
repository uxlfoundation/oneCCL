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

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "common/global/ze/ze_data.hpp"
#ifdef CCL_ENABLE_UMF
#include "umf/ipc.hpp"
#endif // CCL_ENABLE_UMF
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL
#include "common/utils/utils.hpp"
#include "hwloc/hwloc_wrapper.hpp"
#include "internal_types.hpp"
// TODO: think about the right place
#include "MT/shared_resource.hpp"
#include "coll/reduction/reduction.hpp"

#include <memory>
#include <thread>

class ccl_datatype_storage;
class ccl_executor;
class ccl_sched_cache;
class ccl_parallelizer;
class ccl_fusion_manager;
class ccl_reduction_type_storage;

template <ccl_coll_type... registered_types_id>
class ccl_algorithm_selector_wrapper;

namespace ccl {

class buffer_cache;
class recycle_storage;

struct os_information {
    std::string sysname;
    std::string nodename;
    std::string release;
    std::string version;
    std::string machine;

    std::string to_string();
    void fill();
};

class global_data {
public:
    global_data(const global_data&) = delete;
    global_data(global_data&&) = delete;

    global_data& operator=(const global_data&) = delete;
    global_data& operator=(global_data&&) = delete;

    ~global_data();

    ccl::status init();
    ccl::status reset();

    static global_data& get();
    static env_data& env();
    static os_information& get_os_info();

    /* public methods to have access from listener thread function */
    void init_resize_dependent_objects();
    void reset_resize_dependent_objects();

    std::unique_ptr<ccl_datatype_storage> dtypes;
    std::unique_ptr<ccl_reduction_type_storage> redtype_storage;
    std::unique_ptr<ccl_executor> executor;
    std::unique_ptr<ccl_sched_cache> sched_cache;
    std::unique_ptr<ccl::recycle_storage> recycle_storage;
    std::unique_ptr<ccl::buffer_cache> buffer_cache;
    std::unique_ptr<ccl_parallelizer> parallelizer;
    std::unique_ptr<ccl_fusion_manager> fusion_manager;
    std::unique_ptr<ccl_algorithm_selector_wrapper<CCL_COLL_LIST>> algorithm_selector;
    std::unique_ptr<ccl_hwloc_wrapper> hwloc_wrapper;
    std::unique_ptr<profile::metrics_manager> metrics_profiler;
    std::unique_ptr<profile::timestamp_manager> timestamp_manager;
    std::unique_ptr<shared_resources> shared_data;
    std::unordered_map<int, std::unordered_map<int, std::vector<void*>>> hash_table;

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    std::unique_ptr<ze::global_data_desc> ze_data;
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

    static thread_local bool is_worker_thread;
    bool is_ft_enabled{ false };

    int get_local_proc_idx() const {
        return local_proc_idx;
    }
    int get_local_proc_count() const {
        return local_proc_count;
    }

    void set_local_proc_idx(int local_idx) {
        local_proc_idx = local_idx;
    }
    void set_local_proc_count(int local_count) {
        local_proc_count = local_count;
    }

private:
    global_data();

    void init_resize_independent_objects();
    void reset_resize_independent_objects();

    int local_proc_idx{ ccl_comm::invalid_rank };
    int local_proc_count{ ccl::utils::invalid_err_code };
    void getenv_local_coord(const char* local_proc_idx_env_name,
                            const char* local_proc_count_env_name);
    void set_local_coord();

    env_data env_object{};
    os_information os_info{};
};

} // namespace ccl

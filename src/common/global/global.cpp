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
#include "coll/selection/selection.hpp"
#include "common/api_wrapper/api_wrapper.hpp"
#include "common/api_wrapper/pmix_api_wrapper.hpp"
#include "common/datatype/datatype.hpp"
#include "common/global/global.hpp"
#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/buffer/buffer_cache.hpp"
#include "sched/cache/cache.hpp"
#include "sched/cache/recycle_storage.hpp"

#include <sys/utsname.h>

namespace ccl {

thread_local bool global_data::is_worker_thread = false;

std::string os_information::to_string() {
    std::stringstream ss;
    ss << " { " << sysname << " " << nodename << " " << release << " " << version << " " << machine
       << " }";
    return ss.str();
}

void os_information::fill() {
    struct utsname os;
    uname(&os);
    sysname = os.sysname;
    nodename = os.nodename;
    release = os.release;
    version = os.version;
    machine = os.machine;
}

global_data::global_data() {
    /* create ccl_logger before ccl::global_data
       to ensure static objects construction/destruction rule */
    LOG_INFO("create global_data object");
}

global_data::~global_data() {
    if (recycle_storage) {
        recycle_storage->recycle_events();
        recycle_storage->recycle_requests();
    }
    reset();
}

global_data& global_data::get() {
    static global_data data;
    return data;
}

env_data& global_data::env() {
    return get().env_object;
}

os_information& global_data::get_os_info() {
    return get().os_info;
}

ccl::status global_data::reset() {
    /*
        executor is resize_dependent object but out of regular reset procedure
        executor is responsible for resize logic and has own multi-step reset
     */
    executor.reset();
    reset_resize_dependent_objects();
    reset_resize_independent_objects();

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    ze_data.reset();
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

    pmix_api_fini();

    api_wrappers_fini();

    return ccl::status::success;
}

ccl::status global_data::init() {
    env_object.parse();

    pmix_api_init();

    set_local_coord();
    api_wrappers_init();

    env_object.set_internal_env();

    os_info.fill();
    LOG_INFO("OS info:", os_info.to_string());
    if (os_info.release.find("WSL2") != std::string::npos) {
        env_object.enable_topo_algo = 0;
    }

    recycle_storage.reset(new ccl::recycle_storage());
    shared_data.reset(new shared_resources());

    init_resize_dependent_objects();
    init_resize_independent_objects();

    return ccl::status::success;
}

void global_data::init_resize_dependent_objects() {
    dtypes.reset(new ccl_datatype_storage());

    sched_cache.reset(new ccl_sched_cache());
    buffer_cache.reset(new ccl::buffer_cache(env_object.worker_count));

    if (env_object.enable_fusion) {
        /* create fusion_manager before executor because service_worker uses fusion_manager */
        fusion_manager.reset(new ccl_fusion_manager());
    }

    executor.reset(new ccl_executor());
}

void global_data::init_resize_independent_objects() {
    parallelizer.reset(new ccl_parallelizer(env_object.worker_count));

    algorithm_selector.reset(new ccl_algorithm_selector_wrapper<CCL_COLL_LIST>());
    algorithm_selector->init();

    hwloc_wrapper.reset(new ccl_hwloc_wrapper());

    metrics_profiler.reset(new profile::metrics_manager());
    timestamp_manager.reset(new profile::timestamp_manager());
    metrics_profiler->init();
}

void global_data::reset_resize_dependent_objects() {
    fusion_manager.reset();
    sched_cache.reset();
    buffer_cache.reset();
    dtypes.reset();
}

void global_data::reset_resize_independent_objects() {
    parallelizer.reset();
    algorithm_selector.reset();
    hwloc_wrapper.reset();
    metrics_profiler.reset();
}

void global_data::getenv_local_coord(const char* local_proc_idx_env_name,
                                     const char* local_proc_count_env_name) {
    char* local_idx_env = getenv(local_proc_idx_env_name);
    char* local_count_env = getenv(local_proc_count_env_name);
    if (!(local_idx_env && local_count_env)) {
        LOG_WARN("could not get local_idx/count from environment variables, "
                 "trying to get them from ATL");
        local_proc_idx = CCL_ENV_INT_NOT_SPECIFIED;
        local_proc_count = CCL_ENV_INT_NOT_SPECIFIED;
        ccl::global_data::env().enable_init_hostname_sharing = 1;
        ccl::global_data::env().enable_hostname_sharing = 1;
        return;
    }

    local_proc_idx = std::atoi(local_idx_env);
    local_proc_count = std::atoi(local_count_env);
    CCL_THROW_IF_NOT(
        local_proc_idx != CCL_ENV_INT_NOT_SPECIFIED, "unexpected local_proc_idx ", local_proc_idx);
    CCL_THROW_IF_NOT(local_proc_count != CCL_ENV_INT_NOT_SPECIFIED,
                     "unexpected local_proc_count ",
                     local_proc_count);
}

void global_data::set_local_coord() {
    auto& env = ccl::global_data::env();

    if (env.process_launcher == process_launcher_mode::hydra) {
        getenv_local_coord("MPI_LOCALRANKID", "MPI_LOCALNRANKS");
    }
    else if (env.process_launcher == process_launcher_mode::torchrun) {
        getenv_local_coord("LOCAL_RANK", "LOCAL_WORLD_SIZE");
    }
#ifdef CCL_ENABLE_PMIX
    else if (env.process_launcher == process_launcher_mode::pmix) {
        if (!get_pmix_local_coord(&local_proc_idx, &local_proc_count)) {
            if (local_proc_idx == CCL_ENV_INT_NOT_SPECIFIED ||
                local_proc_count == CCL_ENV_INT_NOT_SPECIFIED) {
                LOG_WARN("could not get local_idx/count from environment variables, "
                         "trying to get them from ATL");
            }
            else {
                CCL_THROW("unexpected behaviour of get_pmix_local_coord local_proc_idx: ",
                          local_proc_idx,
                          ", local_proc_count: ",
                          local_proc_count);
            }
        }
    }
#endif // CCL_ENABLE_PMIX
    else if (env.process_launcher == process_launcher_mode::none) {
        getenv_local_coord("CCL_LOCAL_RANK", "CCL_LOCAL_SIZE");
    }
    else {
        CCL_THROW("unexpected process launcher");
    }
    LOG_INFO("process launcher: ",
             ccl::env_data::process_launcher_names[env.process_launcher],
             ", local_proc_idx: ",
             local_proc_idx,
             ", local_proc_count: ",
             local_proc_count);
}

} // namespace ccl

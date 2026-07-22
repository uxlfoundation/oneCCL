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

#include "common/api_wrapper/api_wrapper.hpp"
#include "common/api_wrapper/pmix_api_wrapper.hpp"

namespace ccl {

#ifdef CCL_ENABLE_PMIX

// this is not assumed to work with threads in parallel due to potential race conditions
pmix_proc_t global_proc;

ccl::lib_info_t pmix_lib_info;
pmix_lib_ops_t pmix_lib_ops;

bool get_pmix_local_coord(int* local_proc_idx, int* local_proc_count) {
    *local_proc_idx = CCL_ENV_INT_NOT_SPECIFIED;
    *local_proc_count = CCL_ENV_INT_NOT_SPECIFIED;

    pmix_status_t rc = PMIX_SUCCESS;
    pmix_value_t* val = NULL;
    pmix_proc_t proc;

    if (PMIX_SUCCESS != (rc = PMIx_Init(&global_proc, NULL, 0))) {
        LOG_WARN("PMIx_Init failed: ", PMIx_Error_string(rc));
        return false;
    }

    PMIx_Proc_construct(&proc);
    memset(proc.nspace, '\0', PMIX_MAX_NSLEN);
    memcpy(proc.nspace, global_proc.nspace, strnlen(global_proc.nspace, PMIX_MAX_NSLEN - 1));
    proc.rank = PMIX_RANK_WILDCARD;

    // number of local ranks on node
    if (PMIX_SUCCESS != (rc = PMIx_Get(&proc, PMIX_LOCAL_SIZE, NULL, 0, &val))) {
        LOG_WARN("PMIx_Get(PMIX_LOCAL_SIZE) failed: ", PMIx_Error_string(rc));
        PMIx_Proc_destruct(&proc);
        return false;
    }
    *local_proc_count = val->data.uint32;
    PMIx_Value_destruct(val);
    pmix_free(val);

    // my local rank on node
    if (PMIX_SUCCESS != (rc = PMIx_Get(&global_proc, PMIX_LOCAL_RANK, NULL, 0, &val))) {
        LOG_WARN("PMIx_Get(PMIX_LOCAL_RANK) failed: ", PMIx_Error_string(rc));
        PMIx_Proc_destruct(&proc);
        return false;
    }
    *local_proc_idx = val->data.uint16;
    PMIx_Value_destruct(val);
    pmix_free(val);
    PMIx_Proc_destruct(&proc);

    LOG_DEBUG("get pmix_local_rank/size - local_proc_idx: ",
              *local_proc_idx,
              ", local_proc_count: ",
              *local_proc_count);
    return true;
}
#endif // CCL_ENABLE_PMIX

void pmix_api_init() {
#ifdef CCL_ENABLE_PMIX
    if (ccl::global_data::env().process_launcher == process_launcher_mode::pmix) {
        pmix_lib_info.ops = &pmix_lib_ops;
        pmix_lib_info.fn_names = pmix_fn_names;

        // lib_path specifies the name and full path to the PMIX library
        // it should be absolute and validated path
        // pointing to desired libpmix library
        pmix_lib_info.path = ccl::global_data::env().pmix_lib_path;

        if (pmix_lib_info.path.empty()) {
            pmix_lib_info.path = "libpmix.so";
        }
        LOG_DEBUG("pmix lib path: ", pmix_lib_info.path);

        int error = load_library(pmix_lib_info);
        if (error != CCL_LOAD_LB_SUCCESS) {
            print_error(error, pmix_lib_info);
        }

        CCL_THROW_IF_NOT(pmix_lib_info.handle != nullptr, "could not initialize PMIX api");
    }
#endif // CCL_ENABLE_PMIX
}

void pmix_api_fini() {
#ifdef CCL_ENABLE_PMIX
    if (ccl::global_data::env().process_launcher == process_launcher_mode::pmix) {
        pmix_status_t rc = PMIX_SUCCESS;
        if (PMIX_SUCCESS != (rc = PMIx_Finalize(NULL, 0))) {
            CCL_THROW("PMIx_Finalize failed: ", PMIx_Error_string(rc));
        }

        LOG_DEBUG("close pmix lib: handle: ", pmix_lib_info.handle);
        close_library(pmix_lib_info);
    }
#endif // CCL_ENABLE_PMIX
}

} //namespace ccl

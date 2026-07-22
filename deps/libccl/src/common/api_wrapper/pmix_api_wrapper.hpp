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

#include "oneapi/ccl/config.h"

#include <string>
#include <vector>

#ifdef CCL_ENABLE_PMIX
#include <pmix.h>
#endif // CCL_ENABLE_PMIX

namespace ccl {

#ifdef CCL_ENABLE_PMIX
typedef struct pmix_lib_ops {
    decltype(::PMIx_Init) *PMIx_Init;
    decltype(::PMIx_Error_string) *PMIx_Error_string;
    decltype(::PMIx_Get) *PMIx_Get;
    decltype(::PMIx_Finalize) *PMIx_Finalize;
    decltype(::PMIx_Value_destruct) *PMIx_Value_destruct;
    decltype(::PMIx_Proc_construct) *PMIx_Proc_construct;
    decltype(::PMIx_Proc_destruct) *PMIx_Proc_destruct;
    decltype(::PMIx_Put) *PMIx_Put;
    decltype(::PMIx_Commit) *PMIx_Commit;
    decltype(::PMIx_Fence) *PMIx_Fence;
    decltype(::PMIx_Value_load) *PMIx_Value_load;
    decltype(::PMIx_Load_procid) *PMIx_Load_procid;
    decltype(::PMIx_Value_construct) *PMIx_Value_construct;
    decltype(::PMIx_Value_free) *PMIx_Value_free;
    decltype(::PMIx_Info_load) *PMIx_Info_load;
    decltype(::PMIx_Info_create) *PMIx_Info_create;
    decltype(::PMIx_Info_free) *PMIx_Info_free;
} pmix_lib_ops_t;

static std::vector<std::string> pmix_fn_names = { "PMIx_Init",
                                                  "PMIx_Error_string",
                                                  "PMIx_Get",
                                                  "PMIx_Finalize",
                                                  "PMIx_Value_destruct",
                                                  "PMIx_Proc_construct",
                                                  "PMIx_Proc_destruct",
                                                  "PMIx_Put",
                                                  "PMIx_Commit",
                                                  "PMIx_Fence",
                                                  "PMIx_Value_load",
                                                  "PMIx_Load_procid",
                                                  "PMIx_Value_construct",
                                                  "PMIx_Value_free",
                                                  "PMIx_Info_load",
                                                  "PMIx_Info_create",
                                                  "PMIx_Info_free" };

extern ccl::pmix_lib_ops_t pmix_lib_ops;

#define PMIx_Init            ccl::pmix_lib_ops.PMIx_Init
#define PMIx_Error_string    ccl::pmix_lib_ops.PMIx_Error_string
#define PMIx_Get             ccl::pmix_lib_ops.PMIx_Get
#define PMIx_Finalize        ccl::pmix_lib_ops.PMIx_Finalize
#define PMIx_Value_destruct  ccl::pmix_lib_ops.PMIx_Value_destruct
#define PMIx_Proc_construct  ccl::pmix_lib_ops.PMIx_Proc_construct
#define PMIx_Proc_destruct   ccl::pmix_lib_ops.PMIx_Proc_destruct
#define PMIx_Put             ccl::pmix_lib_ops.PMIx_Put
#define PMIx_Commit          ccl::pmix_lib_ops.PMIx_Commit
#define PMIx_Fence           ccl::pmix_lib_ops.PMIx_Fence
#define PMIx_Value_load      ccl::pmix_lib_ops.PMIx_Value_load
#define PMIx_Load_procid     ccl::pmix_lib_ops.PMIx_Load_procid
#define PMIx_Value_construct ccl::pmix_lib_ops.PMIx_Value_construct
#define PMIx_Value_free      ccl::pmix_lib_ops.PMIx_Value_free
#define PMIx_Info_load       ccl::pmix_lib_ops.PMIx_Info_load
#define PMIx_Info_create     ccl::pmix_lib_ops.PMIx_Info_create
#define PMIx_Info_free       ccl::pmix_lib_ops.PMIx_Info_free

extern pmix_proc_t global_proc;
bool get_pmix_local_coord(int *local_proc_idx, int *local_proc_count);
#endif // CCL_ENABLE_PMIX

void pmix_api_init();
void pmix_api_fini();

} //namespace ccl

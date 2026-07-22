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

#include "common/log/log.hpp"
#include "pmi_simple.h"
#include "pmi_rt.c"

int pmi_simple::is_pm_resize_enabled() {
    return false;
}

pmi_simple::pmi_simple() {}

atl_status_t pmi_simple::pmrt_init() {
    return pmirt_init(&rank, &size, &pmrt_desc);
}

atl_status_t pmi_simple::pmrt_main_addr_reserve(char *main_addr) {
    LOG_ERROR("Function main_addr_reserv unsupported yet for simple pmi\n");
    return ATL_STATUS_FAILURE;
}

atl_status_t pmi_simple::pmrt_set_resize_function(atl_resize_fn_t resize_fn) {
    LOG_ERROR("Function set_resize_function unsupported yet for simple pmi\n");
    return ATL_STATUS_FAILURE;
}

atl_status_t pmi_simple::pmrt_update() {
    LOG_ERROR("Function update unsupported yet for simple pmi\n");
    return ATL_STATUS_FAILURE;
}

atl_status_t pmi_simple::pmrt_wait_notification() {
    LOG_ERROR("Function wait_notification unsupported yet for simple pmi\n");
    return ATL_STATUS_FAILURE;
}

atl_status_t pmi_simple::pmrt_finalize() {
    is_finalized = true;
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_simple::pmrt_barrier() {
    pmirt_barrier(pmrt_desc);
    return ATL_STATUS_SUCCESS;
}

atl_status_t pmi_simple::pmrt_kvs_put(char *kvs_key,
                                      int proc_idx,
                                      const void *kvs_val,
                                      size_t kvs_val_len) {
    return pmirt_kvs_put(pmrt_desc, kvs_key, proc_idx, kvs_val, kvs_val_len);
}

atl_status_t pmi_simple::pmrt_kvs_get(char *kvs_key,
                                      int proc_idx,
                                      void *kvs_val,
                                      size_t kvs_val_len) {
    return pmirt_kvs_get(pmrt_desc, kvs_key, proc_idx, kvs_val, kvs_val_len);
}

int pmi_simple::get_rank() {
    return rank;
}

int pmi_simple::get_size() {
    return size;
}

size_t pmi_simple::get_local_thread_idx() {
    return 0;
}
atl_status_t pmi_simple::get_local_kvs_id(size_t &res) {
    res = 0;
    return ATL_STATUS_SUCCESS;
}
atl_status_t pmi_simple::set_local_kvs_id(size_t local_kvs_id) {
    return ATL_STATUS_SUCCESS;
}

pmi_simple::~pmi_simple() {
    if (!is_finalized) {
        CCL_THROW_IF_NOT(pmrt_finalize() == ATL_STATUS_SUCCESS, "~pmi_simple: failed");
    }
}

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

#include "atl/atl_def.h"
#include "atl/util/pm/pm_rt.h"

class pmi_simple final : public ipmi {
public:
    pmi_simple();
    pmi_simple(const pmi_simple &other) = delete;
    pmi_simple &operator=(const pmi_simple &other) = delete;
    ~pmi_simple() override;

    int is_pm_resize_enabled() override;

    atl_status_t pmrt_main_addr_reserve(char *main_addr) override;

    atl_status_t pmrt_set_resize_function(atl_resize_fn_t resize_fn) override;

    atl_status_t pmrt_update() override;

    atl_status_t pmrt_wait_notification() override;

    atl_status_t pmrt_finalize() override;

    atl_status_t pmrt_barrier() override;

    atl_status_t pmrt_kvs_put(char *kvs_key,
                              int proc_idx,
                              const void *kvs_val,
                              size_t kvs_val_len) override;

    atl_status_t pmrt_kvs_get(char *kvs_key,
                              int proc_idx,
                              void *kvs_val,
                              size_t kvs_val_len) override;

    int get_rank() override;

    int get_size() override;

    size_t get_local_thread_idx() override;

    atl_status_t get_local_kvs_id(size_t &res) override;

    atl_status_t set_local_kvs_id(size_t local_kvs_id) override;

    size_t get_threads_per_process() override {
        return 1;
    }

    size_t get_ranks_per_process() override {
        return 1;
    }

    atl_status_t pmrt_init() override;

private:
    int rank = -1;
    int size = -1;
    pm_rt_desc_t *pmrt_desc = nullptr;
    bool is_finalized = false;
};

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

#include <list>
#include <map>
#include <memory>
#include <vector>

#include "atl/atl_def.h"
#include "atl/mpi/atl_mpi.hpp"
#include "atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.h"
#include "util/pm/pmi_resizable_rt/pmi_resizable/helper.hpp"
#include "atl/util/pm/pm_rt.h"

#define PMIR_SUCCESS                0
#define PMIR_FAIL                   -1
#define PMIR_ERR_INIT               1
#define PMIR_ERR_NOMEM              2
#define PMIR_ERR_INVALID_ARG        3
#define PMIR_ERR_INVALID_KEY        4
#define PMIR_ERR_INVALID_KEY_LENGTH 5
#define PMIR_ERR_INVALID_VAL        6
#define PMIR_ERR_INVALID_VAL_LENGTH 7
#define PMIR_ERR_INVALID_LENGTH     8
#define PMIR_ERR_INVALID_NUM_ARGS   9
#define PMIR_ERR_INVALID_ARGS       10
#define PMIR_ERR_INVALID_NUM_PARSED 11
#define PMIR_ERR_INVALID_KEYVALP    12
#define PMIR_ERR_INVALID_SIZE       13

class pmi_resizable_simple_internal final : public ipmi {
public:
    pmi_resizable_simple_internal() = delete;
    pmi_resizable_simple_internal(int comm_size,
                                  const std::vector<int>& ranks,
                                  std::shared_ptr<internal_kvs> k,
                                  const char* main_addr = "");

    ~pmi_resizable_simple_internal() override;

    pmi_resizable_simple_internal& operator=(const pmi_resizable_simple_internal&) = delete;

    pmi_resizable_simple_internal(const pmi_resizable_simple_internal&) = delete;

    int is_pm_resize_enabled() override;

    atl_status_t pmrt_main_addr_reserve(char* main_addr) override;

    atl_status_t pmrt_set_resize_function(atl_resize_fn_t resize_fn) override;

    atl_status_t pmrt_update() override;

    atl_status_t pmrt_wait_notification() override;

    atl_status_t pmrt_barrier() override;

    atl_status_t pmrt_kvs_put(char* kvs_key,
                              int proc_idx,
                              const void* kvs_val,
                              size_t kvs_val_len) override;

    atl_status_t pmrt_kvs_get(char* kvs_key,
                              int proc_idx,
                              void* kvs_val,
                              size_t kvs_val_len) override;

    int get_size() override;

    int get_rank() override;

    size_t get_local_thread_idx() override;

    atl_status_t get_local_kvs_id(size_t& res) override;

    atl_status_t set_local_kvs_id(size_t local_kvs_id) override;

    size_t get_threads_per_process() override;

    size_t get_ranks_per_process() override;

    atl_status_t pmrt_finalize() override;

    atl_status_t pmrt_init() override;

private:
    bool is_finalized{ false };

    int kvs_set_value(const std::string& kvs_name,
                      const std::string& key,
                      const std::string& value);
    atl_status_t kvs_get_value(const std::string& kvs_name,
                               const std::string& key,
                               std::string& value);

    atl_status_t pmrt_barrier_full();
    atl_status_t barrier_full_reg();
    atl_status_t barrier_reg();
    atl_status_t registration();

    int proc_count = 0;
    int rank = 0;
    int proc_rank_count = 0;
    int threads_count = 0;
    int thread_num = 0;

    int comm_size;

    std::vector<int> ranks;
    std::shared_ptr<internal_kvs> k;
    std::string main_addr;
    size_t max_keylen;
    size_t max_vallen;
    char* val_storage = nullptr;
    size_t local_id;
    size_t kvs_get_timeout = 60; /* in seconds */
};

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

#include <deque>
#include <list>
#include <memory>

#include "atl/atl_base_comm.hpp"
#include "coll/coll_param.hpp"
#include "comm/atl_tag.hpp"
#include "common/request/request.hpp"
#include "common/utils/buffer.hpp"
#include "sched/buffer/buffer_manager.hpp"
#include "sched/entry/entry.hpp"
#include "sched/sched_group.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "sched/ze/ze_event_manager.hpp"
#include "sched/ze/ze_handle_manager.hpp"
#include "sched/ze/ze_ipc_event_pool_manager.hpp"
#include "sched/ze/ze_list_manager.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

class ccl_sched_queue;
class ccl_sched_bin;
class ccl_request;
class ccl_parallelizer;
class ccl_executor;

enum ccl_sched_type { ccl_sched_regular, ccl_sched_fusion, ccl_sched_unordered_coll };

enum ccl_sched_add_mode {
    ccl_sched_add_front,
    ccl_sched_add_back,

    ccl_sched_add_mode_last_value
};

std::string to_string(ccl_sched_add_mode mode);

struct ccl_sched_memory {
    ccl::buffer_manager buffer_manager;

#ifdef CCL_ENABLE_ZE
    std::unique_ptr<ccl::ze::event_manager> event_manager;
    ccl::ze::ipc_handle_manager handle_manager;
    ccl::ze::ipc_event_pool_manager ipc_event_pool_manager;
    std::shared_ptr<ccl::ze::list_manager> list_manager;
#endif // CCL_ENABLE_ZE

    std::list<atl_mr_t*> mr_list;
};

struct ccl_sched_create_param {
    ccl_sched_type type;
    ccl_sched_id_t id;
    ccl_coll_param coll_param;

    ccl_sched_create_param(ccl_sched_type type, ccl_sched_id_t id, const ccl_coll_param& coll_param)
            : type(type),
              id(id),
              coll_param(coll_param) {}

    ccl_sched_create_param(ccl_sched_type type, const ccl_coll_param& coll_param)
            : ccl_sched_create_param(type, 0, coll_param) {}

    ccl_sched_create_param(ccl_sched_id_t id, const ccl_coll_param& coll_param)
            : ccl_sched_create_param(ccl_sched_regular, id, coll_param) {}
};

static size_t lifo_priority = 0;

struct ccl_sched_base {
    template <ccl_sched_add_mode mode>
    using add_entry_mode_t = std::integral_constant<ccl_sched_add_mode, mode>;

    using add_entry_front_t = add_entry_mode_t<ccl_sched_add_front>;
    using add_entry_back_t = add_entry_mode_t<ccl_sched_add_back>;

    void set_coll_attr(const struct ccl_coll_attr& attr);

    void update_coll_param_and_attr(const struct ccl_coll_param& param,
                                    const struct ccl_coll_attr& attr);

    size_t get_priority() const;

    ccl_buffer alloc_buffer(const ccl::alloc_param& param);
    void dealloc_buffer(const ccl::dealloc_param& param);

    void add_memory_region(atl_mr_t* mr);
    void free_memory_regions();

    void sched_complete_hook();
    void reset_memory_state();
    void clear_memory();

    /* unsupported */
    ccl_buffer update_buffer(ccl_buffer buffer, size_t new_size);
    ccl_buffer find_and_realloc_buffer(void* buffer, size_t new_size, size_t expected_size = 0);

    void get_pre_post_copy_counts(std::vector<size_t>& d2h_counts,
                                  std::vector<size_t>& h2d_counts,
                                  bool& reuse_buffers);

    void alloc_buffers_for_pre_post_copy();

    void set_entry_exec_mode(ccl_sched_entry_exec_mode mode) {
        exec_mode = mode;
    }

    ccl_sched_add_mode get_add_mode() {
        return add_mode;
    }

    void set_add_mode(ccl_sched_add_mode mode) {
        add_mode = mode;
    }

    ccl_sched_memory& get_memory() {
        return memory;
    }

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    std::vector<sched_entry*> ze_entries;
    bool use_single_list{};

    void try_enable_ze_single_list();
    void append_to_ze_entries_list(sched_entry* entry);
    bool check_pt2pt_pre_post_copy_support(const ccl_coll_param& param, bool enable_pt2pt_offload);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    ccl_sched_type sched_type = ccl_sched_regular;

    /* sequence number of the schedule in the communicator */
    ccl_sched_id_t sched_id{};

    ccl_coll_param coll_param{};
    ccl_coll_attr coll_attr{};

    /* TODO: schedule doesn't necessarily map on single algo */
    ccl_coll_algo hint_algo{};

    static size_t get_lifo_priority() noexcept {
        return lifo_priority++;
    }

    std::shared_ptr<sched_group> group;

protected:
    ~ccl_sched_base();

    ccl_sched_base() = delete;

    ccl_sched_base(const ccl_sched_base& other) = delete;
    ccl_sched_base& operator=(const ccl_sched_base& other) = delete;

    ccl_sched_base(const ccl_sched_create_param& param);

    void update_id();

    void dump(std::ostream& out, const char* name) const;

    ccl_sched_memory memory;
    ccl_sched_entry_exec_mode exec_mode = ccl_sched_entry_exec_regular;
    ccl_sched_add_mode add_mode = ccl_sched_add_back;

    bool deps_is_barrier{ true };
};

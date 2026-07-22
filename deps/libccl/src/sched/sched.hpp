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

#include "common/request/request.hpp"
#include "common/utils/sync_object.hpp"
#include "sched/sched_base.hpp"
#include "sched/sched_timer.hpp"
#include "sched/sched_group.hpp"
#include "sched/queue/flow_control.hpp"
#include "internal_types.hpp"

//todo: sequence diagram
//workflow:
//   if(!found in cache) {
//      1a. new ccl_sched()
//      2. set_coll_attr
//      3. alloc_buffers_for_pre_post_copy
//   } else {
//      1b. restart_manager->add_launch_params
//   }
//   3. sched->commit(parallelizer)
//   4. sched->start(executor)
//      4.1 prepare_partial_scheds()
//          4.1.1 update_id()
//          4.1.2 renew()
//      4.2 reset_request()

enum ccl_sched_in_bin_status {
    ccl_sched_in_bin_none,
    ccl_sched_in_bin_added,
    ccl_sched_in_bin_erased
};

class ccl_sched;
typedef ccl::status (*ccl_sched_finalize_fn_t)(ccl_sched*, const void*);

static constexpr int invalid_entry_idx = -1;

class sched_restart_manager;

class ccl_sched_key;
// TODO: after removing duplicate code, the only sched types currently
// in use: extra and master. Need to rework code further for unification
enum class sched_type_t { /* regular , */ master, extra };
class alignas(CACHELINE_SIZE) ccl_sched : public ccl_sched_base {
public:
    static constexpr const char* class_name() {
        return "sched";
    }

    ccl_sched(const ccl_sched_create_param& param, bool top_level_sched = false);
    ccl_sched(const ccl_sched_create_param& param, ccl_sched* master_sched);

    ~ccl_sched();

    ccl_sched(const ccl_sched& src) = delete;
    ccl_sched() = delete;
    ccl_sched& operator=(const ccl_sched& other) = delete;

    void add_subsched(const ccl_coll_param& param, bool update_sched_id = true);
    std::vector<std::shared_ptr<ccl_sched>>& get_subscheds();
    void commit(ccl_parallelizer* parallelizer = nullptr, bool update_sched_id = true);
    // start executing the schedule
    ccl_request* start(ccl_executor* exec,
                       // reset sched's state(e.g. its request)
                       bool reset_sched = true,
                       // generate a new sched id
                       bool update_sched_id = true,
                       // if true - we're restarting the same sched after it's been delayed
                       bool restart = false);

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
    void inherit_ze_managers_from(ccl_sched* sched) {
        CCL_THROW_IF_NOT(entries.empty());
        CCL_THROW_IF_NOT(subscheds.empty());
        CCL_THROW_IF_NOT(sched);

        memory.list_manager = sched->memory.list_manager;
    }
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

    void set_group(std::shared_ptr<sched_group> new_group) {
        CCL_THROW_IF_NOT(entries.empty());
        CCL_THROW_IF_NOT(subscheds.empty());
        CCL_THROW_IF_NOT(new_group.get() != nullptr);
        group = std::move(new_group);
    }

    std::shared_ptr<sched_group> get_group() {
        return group;
    }

    /**
     * Reset completion counter of @b req
     * @return pointer to req that can be used to track completion
     */
    ccl_request* reset_request();
    /**
     * Synchronizes partial schedules on local barrier
     */
    void sync_subscheds();
    void dump(std::ostream& out) const;

    // TODO: wrap into smart-pointer
    using ccl_sched_ptr = ccl_sched*;
    static ccl_sched_ptr create(const ccl_coll_param& param, const ccl_coll_attr& attr);

    bool is_strict_order_satisfied();

    void do_progress();

    /**
     * Called after all the entries have been completed
     */
    void complete();
    bool is_completed() const;

    size_t get_start_idx() const {
        return start_idx;
    }

    void set_op_id(ccl_op_id_t id) {
        op_id = id;
    }

    ccl_op_id_t get_op_id() {
        return op_id;
    }

    void set_scaleout_flag() {
        is_scaleout_subsched = true;
    }

    int get_scaleout_flag() {
        return is_scaleout_subsched;
    }

    void set_in_bin_status(ccl_sched_in_bin_status status) {
        in_bin_status = status;
    }

    ccl_sched_in_bin_status get_in_bin_status() const {
        return in_bin_status;
    }

    /**
     * Reset runtime parameters and all entries
     */
    void renew(bool need_update_id = false, bool reset = false);

    using ccl_sched_base::add_entry_front_t;
    using ccl_sched_base::add_entry_back_t;

    sched_entry* add_entry(std::unique_ptr<sched_entry>&& entry) {
        entry->set_exec_mode(exec_mode);

        sched_entry* raw_ptr = entry.get();
        if (add_mode == ccl_sched_add_back)
            entries.push_back(std::move(entry));
        else if (add_mode == ccl_sched_add_front)
            entries.push_front(std::move(entry));
        else
            CCL_FATAL("unexpected mode ", add_mode);

        return raw_ptr;
    }

    /**
     * Policy-based add_entry
     */
    sched_entry* add_entry(std::unique_ptr<sched_entry>&& entry,
                           add_entry_mode_t<ccl_sched_add_mode_last_value>) {
        return add_entry(std::move(entry));
    }

    sched_entry* add_entry(std::unique_ptr<sched_entry>&& entry, add_entry_front_t) {
        entry->set_exec_mode(exec_mode);

        sched_entry* raw_ptr = entry.get();
        entries.push_front(std::move(entry));
        return raw_ptr;
    }

    sched_entry* add_entry(std::unique_ptr<sched_entry>&& entry, add_entry_back_t) {
        entry->set_exec_mode(exec_mode);

        sched_entry* raw_ptr = entry.get();
        entries.push_back(std::move(entry));
        return raw_ptr;
    }

    /**
     * Require that all previously added entries are completed before subsequent ops
     * may begin execution
     */
    void add_barrier();

    std::vector<ccl::event>& get_deps() const;

    ccl_sched_bin* bin = nullptr; /* valid only during execution */
    ccl_sched_queue* queue = nullptr; /* cached pointer to queue, valid even after execution */
    size_t start_idx = 0; /* index to start */

    /*
      used for unique ATL tag creation in algorithms with multiple parallel sub-schedules
      set once and then used for all entries
    */
    ccl_op_id_t op_id = 0;
    bool is_scaleout_subsched = false;

    /* to track status of schedule wrt execution bin, not atomic as updated by single thread in time */
    ccl_sched_in_bin_status in_bin_status = ccl_sched_in_bin_none;

    using sched_entry_ptr = std::unique_ptr<sched_entry>;
    std::deque<sched_entry_ptr> entries{};

    /* whether sched should be executed in the same order as in user code */
    /* currently applicable for start phase only */
    bool strict_order = false;

    /*
      limits number of active entries
      mostly makes sense for ATL entries
    */
    ccl::flow_control flow_control;

    void set_finalize_fn(ccl_sched_finalize_fn_t fn, void* ctx) {
        finalize_fn = fn;
        finalize_fn_ctx = ctx;
    }

    // pointer to the parent sched, nullptr for type == master
    ccl_sched* parent_sched = nullptr;

    size_t entries_count() const;
    sched_type_t type;

private:
    void reset_state();
    void prepare_subscheds(bool update_sched_id = true);
    std::vector<std::shared_ptr<ccl_sched>> subscheds;
    ccl_sched_finalize_fn_t finalize_fn = nullptr;
    void* finalize_fn_ctx = nullptr;

    ccl::sched_timer timer;

public:
    ccl_request* get_request();
    const ccl_request* get_request() const;

    // check if the sched needs to be restarted
    // to complete delayed requests
    void try_to_restart();

    // cleanup structs related to the request from
    // the schedule
    bool release_request(ccl_request* req);

    void set_submitted_to_gpu(bool submitted_to_gpu);
    bool is_submitted_to_gpu();

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    bool get_ze_commands_bypass_flag();
    void set_ze_commands_bypass_flag(bool bypass);

    ze_event_handle_t get_related_deps_out_event();

    std::shared_ptr<sync_object>& get_init_ze_hook_sync_obj();
    void set_init_ze_hook_sync_obj(std::shared_ptr<sync_object> sync_obj);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    bool is_deps_barrier();
    void set_deps_is_barrier(bool is_barrier);
    bool has_deps_entry();

private:
    void create_sync_event(ccl_request* request);
    void update_active_request(bool use_delayed);
    static void complete_itt(const ccl_stream* stream);

    int calculate_request_count() const;

    // active request that tracks the currently running execution
    ccl_request* req;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    uint32_t ze_commands_submit();
    bool is_ze_commands_bypass{ true };
    std::shared_ptr<sync_object> init_ze_hook_sync_obj;

    const bool use_output_event = false;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    const bool top_level_sched;

    // pointer to the parent sched if this sched is part of a subsched_entry, nullptr otherwise
    ccl_sched* subsched_entry_parent_sched = nullptr;
    std::atomic<bool> volatile submitted_to_gpu{};

    std::unique_ptr<sched_restart_manager> restart_manager;

    friend class sched_restart_manager;
    friend class ze_execute_cmdlists_on_init_entry; // need to call ze_commands_submit();
    friend class ze_execute_cmdlists_on_start_entry; // need to call ze_commands_submit();
    friend class subsched_entry; // need to call ze_commands_submit();
};

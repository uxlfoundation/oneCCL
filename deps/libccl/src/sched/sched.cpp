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

#include "coll/coll_check.hpp"
#include "coll/coll_util.hpp"
#include "coll/selection/selection.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "common/request/request.hpp"
#include "common/utils/sync_object.hpp"
#include "parallelizer/parallelizer.hpp"
#include "sched/cache/cache.hpp"
#include "sched/cache/key.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/sched.hpp"
#include "sched/queue/queue.hpp"
#include "sched/sched_base.hpp"
#include "sched/sched_restart_manager.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"

#ifdef CCL_ENABLE_ZE
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL

ccl_sched::ccl_sched(const ccl_sched_create_param& param, ccl_sched* master_sched)
        : ccl_sched_base(param),
          parent_sched(master_sched),
          req(new ccl_request(*this)),
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
          // only type = "master" sched should use output event
          use_output_event(false),
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
          top_level_sched(false),
          subsched_entry_parent_sched(nullptr),
          restart_manager() {
    type = sched_type_t::extra;
    strict_order = ccl::global_data::env().enable_strict_order;
}

ccl_sched::ccl_sched(const ccl_sched_create_param& param, bool top_level_sched)
        : ccl_sched_base(param),
          subscheds(),
          req(new ccl_request(*this)),
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
          use_output_event(top_level_sched && coll_param.comm->get_env()->get_enable_topo_algo() &&
                           (ccl::utils::should_use_sycl_output_event(coll_param.stream) ||
                            ccl::is_queue_in_order(coll_param.stream))),
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
          top_level_sched(top_level_sched),
          subsched_entry_parent_sched(nullptr),
          restart_manager(std::unique_ptr<sched_restart_manager>(new sched_restart_manager(this))) {
    type = sched_type_t::master;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (use_output_event) {
        ccl::global_data::get().ze_data->dynamic_event_pools.try_emplace(
            coll_param.stream->get_ze_context(), coll_param.stream);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

ccl_sched::~ccl_sched() {
    if (in_bin_status == ccl_sched_in_bin_added)
        LOG_DEBUG("in_bin_status == ccl_sched_in_bin_added");

    if (finalize_fn) {
        finalize_fn(this, finalize_fn_ctx);
    }

    // delete the last associated request
    LOG_DEBUG("deleting sched ", this, " and its req ", req);
    delete req;

    if (type != sched_type_t::master)
        return;

    for (auto& part_sched : subscheds) {
        part_sched.reset();
    }
    if (!memory.mr_list.empty())
        LOG_WARN("memory region list should be empty for master sched");
}

void ccl_sched::commit(ccl_parallelizer* parallelizer, bool update_sched_id) {
    if (ccl::global_data::env().priority_mode == ccl_priority_lifo) {
        coll_attr.priority = ccl_sched_base::get_lifo_priority();
    }

    if (subscheds.empty()) {
        /* single time operations */
        if (update_sched_id) {
            update_id();
        }
        if (parallelizer) {
            parallelizer->process(this, update_sched_id);
            CCL_THROW_IF_NOT(
                !subscheds.empty(),
                "ccl_master_sched must have at least 1 partial sched after parallelized");
        }
    }
    else {
        /* repeated operations, should happen each time to reuse schedule */
        for (size_t idx = 0; idx < subscheds.size(); idx++) {
            subscheds[idx]->coll_attr.priority = coll_attr.priority;
        }
    }

    LOG_DEBUG("sched ",
              this,
              ", sched_id ",
              sched_id,
              ", req ",
              get_request(),
              ", subscheds_count ",
              subscheds.size());
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
uint32_t ccl_sched::ze_commands_submit() {
    uint32_t cmd_counter = 0;
    for (auto& entry : entries) {
        cmd_counter += entry->ze_commands_submit();
    }
    return cmd_counter;
};

bool ccl_sched::get_ze_commands_bypass_flag() {
    return is_ze_commands_bypass;
}

void ccl_sched::set_ze_commands_bypass_flag(bool bypass) {
    if (subsched_entry_parent_sched) {
        subsched_entry_parent_sched->set_ze_commands_bypass_flag(bypass);
    }
    if (parent_sched) {
        parent_sched->set_ze_commands_bypass_flag(bypass);
    }
    is_ze_commands_bypass = bypass;
}

ze_event_handle_t ccl_sched::get_related_deps_out_event() {
    ze_event_handle_t out_event = nullptr;
    if (subsched_entry_parent_sched) {
        CCL_THROW_IF_NOT(subsched_entry_parent_sched->entries.size(),
                         "subsched_entry_parent_sched is empty");
        CCL_THROW_IF_NOT(subsched_entry_parent_sched->entries[0]->is_deps(),
                         "subsched_entry_parent_sched first entry is not deps");
        out_event = ((deps_entry*)(subsched_entry_parent_sched->entries[0].get()))->out_event;
    }
    else {
        CCL_THROW_IF_NOT(entries.size(), "sched is empty");
        CCL_THROW_IF_NOT(entries[0]->is_deps(), "first sched entry is not deps");
        out_event = ((deps_entry*)(entries[0].get()))->out_event;
    }
    CCL_THROW_IF_NOT(out_event, "dependencies out event is not initialized");
    return out_event;
}

std::shared_ptr<sync_object>& ccl_sched::get_init_ze_hook_sync_obj() {
    return init_ze_hook_sync_obj;
}

void ccl_sched::set_init_ze_hook_sync_obj(std::shared_ptr<sync_object> sync_obj) {
    init_ze_hook_sync_obj = std::move(sync_obj);
}
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

void ccl_sched::reset_state() {
    reset_request();
}

// restart parameter indicates that start() is executed from worker thread once the previous
// sched execution is completed. we restart only if there is at least 1 delayed request
ccl_request* ccl_sched::start(ccl_executor* exec,
                              bool reset_sched,
                              bool update_sched_id,
                              bool restart) {
    if (type == sched_type_t::master) {
        auto new_req = restart_manager->preprocess(restart);
        if (new_req)
            return new_req;
    }

    /* sanity check the schedule */
    CCL_THROW_IF_NOT(coll_param.comm);

    LOG_DEBUG("starting schedule ", this, ", type ", ccl_coll_type_to_str(coll_param.ctype));

    prepare_subscheds(update_sched_id);

    // don't reset request on restart, it's counter is already set to correct value
    if (reset_sched && !restart) {
        reset_state();
    }

    if (ccl::global_data::env().sched_dump) {
        std::stringstream ostream;
        dump(ostream);
        ccl_logger::get_instance().info(ostream.str());
    }

    exec->start(this);

    return get_request();
}

int ccl_sched::calculate_request_count() const {
    return std::max(1, static_cast<int>(subscheds.size()));
}

ccl_request* ccl_sched::reset_request() {
    // if we reset the request when restarting the sched, we need to ignore completion check
    // because the request must be non-complete before it starts executing
    get_request()->set_counter(calculate_request_count());
    return get_request();
}

void ccl_sched::add_subsched(const ccl_coll_param& coll_param, bool update_sched_id) {
    ccl_sched_id_t param_sched_id =
        update_sched_id
            ? coll_param.comm->get_sched_id(sched_type != ccl_sched_regular, coll_param.is_pt2pt)
            : this->sched_id;

    ccl_sched_create_param param = { sched_type, param_sched_id, coll_param };

    subscheds.emplace_back(std::make_shared<ccl_sched>(param, this));
}

std::vector<std::shared_ptr<ccl_sched>>& ccl_sched::get_subscheds() {
    return subscheds;
}

void ccl_sched::prepare_subscheds(bool update_sched_id) {
    for (auto& sched : subscheds) {
        sched->renew(update_sched_id, true);
    }
}

void ccl_sched::sync_subscheds() {
    CCL_THROW_IF_NOT(!subscheds.empty(), "no partial schedules");

    bool add_sync_entry = false;

    /* ensure all partial schedules have the same add_mode */
    ccl_sched_add_mode add_mode = subscheds[0]->get_add_mode();
    for (auto& sched : subscheds) {
        CCL_THROW_IF_NOT(sched->get_add_mode() == add_mode,
                         "unexpected add_mode ",
                         sched->get_add_mode(),
                         ", expected ",
                         add_mode);
    }

    /* check whether all partial schedules already have sync_entry at the tail */
    for (auto& sched : subscheds) {
        if (sched->entries.empty()) {
            add_sync_entry = true;
            break;
        }

        /* TODO: add enum field into base entry to distinguish different entry types */
        const char* tail_entry_name = (add_mode == ccl_sched_add_back)
                                          ? sched->entries.back()->name()
                                          : sched->entries.front()->name();

        if (tail_entry_name && strcmp(tail_entry_name, "SYNC")) {
            add_sync_entry = true;
            break;
        }
    }

    /* if at least one partial schedule doesn't have sync entry
       then sync all partial schedules */
    if (add_sync_entry) {
        auto sync_obj = std::make_shared<sync_object>(subscheds.size());
        for (auto& sched : subscheds) {
            entry_factory::create<sync_entry>(sched.get(), sync_obj);
        }
    }
}

void ccl_sched::dump(std::ostream& out) const {
    if (!ccl::global_data::env().sched_dump) {
        return;
    }

    ccl_sched_base::dump(out, class_name());
    ccl_logger::format(out,
                       ", start_idx: ",
                       start_idx,
                       ", req: ",
                       get_request(),
                       ", num_entries: ",
                       entries.size(),
                       ", priority: ",
                       get_priority(),
                       ", max_flow_credits: ",
                       flow_control.get_max_credits(),
                       ", flow_credits: ",
                       flow_control.get_credits(),
                       ", subscheds size: ",
                       subscheds.size(),
                       "\n");

    std::stringstream msg;
    for (size_t i = 0; i < entries.size(); ++i) {
        entries[i]->dump(msg, i);
    }

    for (const auto& sched : subscheds) {
        sched->dump(out);
    }

    out << msg.str();

    ccl_logger::format(out, "--------------------------------\n");
}

ccl_sched::ccl_sched_ptr ccl_sched::create(const ccl_coll_param& param, const ccl_coll_attr& attr) {
    ccl_sched_key key;
    ccl_sched_ptr sched;
    bool is_created = false;
    // WARNING be cautious while modifying the lambda-related code !
    // `param` is captured by reference
    // lifetimes:
    //      - the lambda's lifetime ends at the end of `ccl_sched::create`
    //      - `param`'s lifetime exceeds the end of `ccl_sched::create`
    // `param` outlives the lambda, so the code is ok (there should be no memory issues)
    // C++ does not check lifetimes, this has to be assured by a programmer !
    // optionally, the code might be refactored in the future to use shared_ptr
    auto create_fn = [&param]() -> ccl_sched_ptr {
        return new ccl_sched(
            { ccl_sched_regular, param.comm->get_sched_id(false, param.is_pt2pt), param },
            /* top-level sched */ true);
    };

#ifdef CCL_ENABLE_ITT
    __itt_event sched_create_event = ccl::profile::itt::event_get("SCHED_CREATE");
    ccl::profile::itt::event_start(sched_create_event);
#endif // CCL_ENABLE_ITT

    if (attr.to_cache) {
        key.set(param, attr);
        std::tie(sched, is_created) =
            ccl::global_data::get().sched_cache->find_or_create(std::move(key), create_fn);
    }
    else {
        sched = create_fn();
        is_created = true;
    }

    if (is_created) {
        sched->set_coll_attr(attr);
        sched->alloc_buffers_for_pre_post_copy();
        LOG_DEBUG("didn't find sched, create new one ",
                  sched,
                  ", coll ",
                  ccl_coll_type_to_str(sched->coll_param.ctype));
    }
    else {
        // we can't update params of the cached sched immediately, it might still be executing using
        // old parameters, they'll be updated once it's ready to launch with them.
        sched->restart_manager->add_launch_params({ param, attr });
        LOG_DEBUG(
            "found sched, reuse ", sched, ", type ", ccl_coll_type_to_str(sched->coll_param.ctype));
    }

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::event_end(sched_create_event);
#endif // CCL_ENABLE_ITT

    return sched;
}

void ccl_sched::set_submitted_to_gpu(bool submitted) {
    LOG_DEBUG(
        "sched ", this, " parent_sched ", parent_sched, " set_submitted_to_gpu(", submitted, ")");
    if (parent_sched) {
        parent_sched->set_submitted_to_gpu(submitted);
    }
    else {
        submitted_to_gpu = submitted;
    }
    if (subsched_entry_parent_sched) {
        subsched_entry_parent_sched->set_submitted_to_gpu(submitted);
    }
}

bool ccl_sched::is_submitted_to_gpu() {
    return submitted_to_gpu;
}

bool ccl_sched::is_deps_barrier() {
    if (subsched_entry_parent_sched) {
        return subsched_entry_parent_sched->is_deps_barrier();
    }
    if (parent_sched) {
        return parent_sched->is_deps_barrier();
    }
    return deps_is_barrier;
}

void ccl_sched::set_deps_is_barrier(bool is_barrier) {
    deps_is_barrier = is_barrier;
    if (subsched_entry_parent_sched) {
        subsched_entry_parent_sched->deps_is_barrier = is_barrier;
    }
    if (parent_sched) {
        parent_sched->deps_is_barrier = is_barrier;
    }
}

bool ccl_sched::has_deps_entry() {
    if ((subsched_entry_parent_sched && subsched_entry_parent_sched->entries.size() > 0 &&
         subsched_entry_parent_sched->entries[0]->is_deps()) ||
        (entries.size() > 0 && entries[0]->is_deps())) {
        return true;
    }
    return false;
}

void ccl_sched::do_progress() {
    for (auto entry_idx = start_idx; entry_idx < entries.size(); ++entry_idx) {
        auto& entry = entries[entry_idx];

        if (entry->get_status() == ccl_sched_entry_status_not_started) {
            LOG_DEBUG("starting entry: ",
                      entry.get(),
                      ", name: ",
                      entry->name(),
                      " [",
                      entry_idx,
                      "/",
                      entries.size(),
                      "]");
        }

        entry->do_progress();

        if (entry->get_status() == ccl_sched_entry_status_again) {
            LOG_DEBUG("entry ",
                      entry->name(),
                      " is in again state, stop progressing [",
                      entry_idx,
                      "/",
                      entries.size(),
                      "]");
            break;
        }

        if (entry_idx == start_idx && entry->is_completed()) {
            /* the entry has been completed, increment start_idx */
            ++start_idx;
            LOG_DEBUG("completed entry: ",
                      entry.get(),
                      ", name: ",
                      entry->name(),
                      entry->is_barrier() ? " barrier" : "",
                      " entry [",
                      entry_idx,
                      "/",
                      entries.size(),
                      "], shift start_idx to ",
                      start_idx,
                      ", sched ",
                      this);
        }
        else if (entry->is_barrier() && (!entry->is_completed() || (start_idx != entry_idx + 1))) {
            /* barrier is not completed or completed too early, skip the further progressing */
            break;
        }
    }
}

bool ccl_sched::is_strict_order_satisfied() {
    return std::all_of(entries.begin(), entries.end(), [](const sched_entry_ptr& e) {
        return e->is_strict_order_satisfied();
    });
}

void ccl_sched::complete() {
    CCL_ASSERT(req, "ccl_sched must have req");

    // save sched type because we cannot assume that the sched with type == master
    // is not destroyed after we complete the request
    auto* parent_schedule = this->parent_sched;

    // it's important to do finalization/cleanup before full completion of the request
    // because right after its completion, the request and the sched can be destroyed
    // by a user thread waiting on a corresponding event, so it's hard to do anything
    // with the object at that time. To overcome this issue, for each request we add
    // +1 to its counter, so here we can check that while worker completed the
    // requests, it still have count 1, so we can do finalization/cleanup before
    // completing it one more time setting the counter to 0.
    if (get_request()->complete_counter() == 1) {
        if (ccl::global_data::env().sched_profile) {
            timer.update();
            if (entries.size() > 0) {
                std::stringstream ss;
                ss << "\ncoll:";

                ccl_coll_param* profile_param = &(coll_param);
                ss << ccl_coll_type_to_str(profile_param->ctype);

                if (!profile_param->send_counts.empty()) {
                    ss << " count:" << profile_param->get_send_count();
                }

                ss << " time(usec): sched total:\n" << to_string(timer) << "\n";
                for (size_t idx = 0; idx < entries.size(); ++idx) {
                    ss << "[" << idx << "] " << entries[idx]->name()
                       << ": total: " << to_string(entries[idx]->total_timer);
                    ss << ", update: " << to_string(entries[idx]->update_timer);
                    ss << "\n";
                }
                ss << "-----------------------------";
                ccl_logger::get_instance().info(ss.str());
            }
        }

        sched_complete_hook();

        // now we completed everything related to finalization of the current sched,
        // so we can finally complete it
        bool success = get_request()->complete();
        CCL_THROW_IF_NOT(success, "request was not completed correctly!");

        if (parent_schedule) {
            // after we call try_to_restart() on the parent, its request might be changed
            // so rememeber it here to call complete on it
            auto parent_req = parent_schedule->get_request();
            // check for completed parent request, see comment above for how this works
            if (parent_req->complete_counter() == 1) {
                // itt tracks only top-level sched execution
                if (top_level_sched)
                    complete_itt(parent_schedule->coll_param.stream);
                // if we don't use cache, it doesn't make sense to restart the sched
                // as there are never be any requests to restart
                if (parent_schedule->coll_attr.to_cache) {
                    // current sched execution is completed, always check if we need to
                    // restart it again
                    parent_schedule->try_to_restart();
                }
                parent_req->complete();
            }
        }
    }
}

bool ccl_sched::is_completed() const {
    return get_request()->is_completed();
}

void ccl_sched::renew(bool need_update_id, bool reset) {
    if (need_update_id) {
        update_id();
    }

    start_idx = 0;

    if (ccl::global_data::env().sched_profile) {
        timer.start();
    }

    for (size_t idx = 0; idx < entries.size(); idx++) {
        entries[idx].get()->reset(idx);
    }

    if (reset)
        reset_state();

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (init_ze_hook_sync_obj && ze_entries.empty()) {
        init_ze_hook_sync_obj->visit();
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

void ccl_sched::add_barrier() {
    if (!entries.empty()) {
        if (add_mode == ccl_sched_add_back)
            entries.back()->make_barrier();
        else if (add_mode == ccl_sched_add_front)
            entries.front()->make_barrier();
        else
            CCL_FATAL("unexpected add_mode ", add_mode);
    }
}

std::vector<ccl::event>& ccl_sched::get_deps() const {
    // if parent is not set then we should have own deps
    if (parent_sched)
        return const_cast<ccl_sched*>(parent_sched)->coll_param.deps;
    else
        return const_cast<ccl_sched*>(this)->coll_param.deps;
}

size_t ccl_sched::entries_count() const {
    return entries.size();
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
void ccl_sched::create_sync_event(ccl_request* request) {
    if (!use_output_event) {
        return;
    }

    auto q = coll_param.stream->get_native_stream();
    auto context = q.get_context();
#ifdef CCL_ENABLE_SYCL_INTEROP_EVENT
    if (request->has_sync_event()) {
        LOG_DEBUG("has a sync event, returning");
        return;
    }

    // even when we use cache in non-async case we cannot really guarantee that
    // some ccl event is not used by a user, so we cannot reuse request for another
    // operation, otherwise we could end up with multiple events referring to the
    // same request. So every time we run a schedule, we get a new event, but
    // use the pool to save on event/pool creation.
    auto& pools = ccl::global_data::get().ze_data->dynamic_event_pools;
    auto pool_it = pools.find(coll_param.stream->get_ze_context());
    CCL_THROW_IF_NOT(pool_it != pools.end(), "pool must be initialized for the context");

    ze_event_handle_t ev = pool_it->second.get_event();
    LOG_DEBUG("convert L0 event: ", ev, " into a SYCL event and submit a barrier");

    auto sync_event = ccl::utils::make_event(context, ev);
    request->set_sync_event(std::move(sync_event));

#else // CCL_ENABLE_SYCL_INTEROP_EVENT
    CCL_THROW("interop event functionality is not available with current configuration, "
              "please rebuild oneCCL using ENABLE_SYCL_INTEROP_EVENT option "
              "and a DPCPP compiler that supports that feature");
#endif // CCL_ENABLE_SYCL_INTEROP_EVENT
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

ccl_request* ccl_sched::get_request() {
    return req;
}
const ccl_request* ccl_sched::get_request() const {
    return req;
}

void ccl_sched::try_to_restart() {
    // execute for all scheds to perform a proper finalization
    if (!restart_manager->check_delayed_requests()) {
        return;
    }

    CCL_THROW_IF_NOT(top_level_sched, "only top-level scheds must be restarted");

    LOG_DEBUG("Restarting schedule: ", this);

    // we have at least 1 delayed request, need to restart the schedule
    (void)start(ccl::global_data::get().executor.get(),
                /* reset state */ true,
                /* update sched id */ true,
                /* restart */ true);
}

void ccl_sched::update_active_request(bool use_delayed) {
    // at this point we reset the active request, but it still can
    // be referenced via an event, returned previously to the user.
    // the request object will be destroyed together with the event
    auto* old_req = req;
    req = ((use_delayed) ? restart_manager->get_delayed_request() : new ccl_request(*this));
    LOG_DEBUG("updated req: ", req, ", old: ", old_req, ", use_delayed: ", use_delayed);
}

void ccl_sched::complete_itt(const ccl_stream* stream) {
    (void)stream;
}

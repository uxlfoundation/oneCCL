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

#include "common/global/global.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "comm/comm.hpp"
#include "unordered_coll/unordered_coll.hpp"

#include <cstring>

struct ccl_unordered_coll_ctx {
    int reserved_comm_id;
    size_t match_id_size;
    void* match_id_value;
    ccl_sched* service_sched;
    ccl_unordered_coll_manager* manager;
};

ccl_unordered_coll_manager::ccl_unordered_coll_manager(ccl_comm& parent_comm) {
    coordination_comm =
        std::unique_ptr<ccl_comm>(new ccl_comm(parent_comm.get_atl_comm()->create_comm_id(),
                                               parent_comm.get_atl_comm(),
                                               true /* share_resources */,
                                               true /* is_sub_communicator */));

    CCL_ASSERT(coordination_comm.get(), "coordination_comm is null");

    if (parent_comm.rank() == 0)
        LOG_INFO("created unordered collectives manager");
}

ccl_unordered_coll_manager::~ccl_unordered_coll_manager() {
    coordination_comm.reset();
}

std::shared_ptr<ccl_comm> ccl_unordered_coll_manager::get_comm(const std::string& match_id) {
    /* check if there are completed service scheds and remove them */
    remove_service_scheds();

    std::lock_guard<ccl_spinlock> lock(match_id_to_comm_map_guard);
    auto comm = match_id_to_comm_map.find(match_id);
    if (comm != match_id_to_comm_map.end()) {
        LOG_DEBUG("comm_id ", comm->second->id(), " for match_id ", match_id, " has been found");
        return comm->second;
    }
    LOG_DEBUG("no comm for match_id ", match_id, " has been found");
    return nullptr;
}

ccl_request* ccl_unordered_coll_manager::postpone(ccl_sched* sched) {
    CCL_ASSERT(!sched->coll_attr.match_id.empty(), "invalid match_id");
    const std::string& match_id = sched->coll_attr.match_id;

    ccl_request* req = sched->reset_request();

    /* 1. check whether comm was created between get_comm and postpone calls on user level */
    ccl_comm* comm = nullptr;
    {
        std::lock_guard<ccl_spinlock> lock(match_id_to_comm_map_guard);
        auto m2comm = match_id_to_comm_map.find(match_id);
        if (m2comm != match_id_to_comm_map.end()) {
            comm = m2comm->second.get();
            CCL_ASSERT(comm);
        }
        else {
            /* keep match_id_to_comm_map_guard to postpone before m2c table update */
            CCL_ASSERT(!comm);
            LOG_DEBUG("postpone sched because didn't find comm_id for match_id ", match_id);
            postpone_sched(sched);
            if (get_postponed_sched_count(match_id) > 1) {
                return req;
            }
        }
    }

    if (comm) {
        LOG_INFO(
            "don't postpone sched because found comm_id ", comm->id(), " for match_id ", match_id);
        run_sched(sched, comm);
        return req;
    }

    /* 2. start coordination for match_id */
    start_coordination(match_id);

    /* 3.
        for non-coordinator ranks comm_id can arrive too early before collective with corresponding match_id
        check this and start schedule if we already have the required match_id
    */
    if (coordination_comm->rank() != CCL_UNORDERED_COLL_COORDINATOR) {
        std::unique_lock<ccl_spinlock> lock(unresolved_comms_guard);
        auto unresolved = unresolved_comms.find(match_id);
        if (unresolved != unresolved_comms.end()) {
            LOG_DEBUG("found comm_id ",
                      unresolved->second,
                      " for match_id ",
                      match_id,
                      " in unresolved_comms, ");
            auto comm_id(std::move(unresolved->second));
            unresolved_comms.erase(unresolved);
            lock.unlock();

            CCL_ASSERT(sched->coll_param.comm);
            auto new_comm = sched->coll_param.comm->clone_with_new_id(comm_id);
            add_comm(match_id, new_comm);
            run_sched(sched, new_comm.get());
        }
    }

    return req;
}

void ccl_unordered_coll_manager::dump(std::ostream& out) const {
    std::stringstream s;

    {
        std::lock_guard<ccl_spinlock> lock{ unresolved_comms_guard };
        s << "unresolved_comms: " << std::endl;
        for (auto& comm : unresolved_comms) {
            s << "[" << comm.first << ", " << comm.second << "] " << std::endl;
        }
    }

    {
        std::lock_guard<ccl_spinlock> lock{ match_id_to_comm_map_guard };
        s << "match_id_to_comm_map: " << std::endl;
        for (auto& m2c : match_id_to_comm_map) {
            s << "[" << m2c.first << ", " << m2c.second->id() << "] " << std::endl;
        }
    }

    {
        std::lock_guard<ccl_spinlock> lock{ postponed_scheds_guard };
        s << "postponed_scheds: " << std::endl;
        for (auto& sched : postponed_scheds) {
            s << "[" << sched.first << ", " << sched.second << "] " << std::endl;
        }
    }

    out << s.str();
}

void ccl_unordered_coll_manager::start_coordination(const std::string& match_id) {
    CCL_THROW_IF_NOT(!match_id.empty(), "match_id is empty");

    ccl_coll_param coll_param{};
    coll_param.ctype = ccl_coll_undefined;
    coll_param.dtype = ccl_datatype_int8;
    coll_param.comm = coordination_comm.get();

    ccl_sched* null_sched = nullptr;
    std::unique_ptr<ccl_sched> service_sched(
        new ccl_sched({ ccl_sched_unordered_coll,
                        coordination_comm->get_sched_id(true, coll_param.is_pt2pt),
                        coll_param },
                      null_sched));

    if (ccl::global_data::env().priority_mode == ccl_priority_lifo) {
        service_sched->coll_attr.priority = ccl_sched_base::get_lifo_priority();
    }

    LOG_DEBUG("start coordination for match_id ",
              match_id,
              " (service_sched ",
              service_sched.get(),
              ", req ",
              service_sched->get_request(),
              ")");

    /* 1. broadcast match_id_size */
    auto ctx = static_cast<ccl_unordered_coll_ctx*>(
        service_sched->alloc_buffer(sizeof(ccl_unordered_coll_ctx)).get_ptr());
    ctx->service_sched = service_sched.get();
    ctx->manager = this;

    ctx->reserved_comm_id = coll_param.comm->get_atl_comm()->create_comm_id();

    if (coordination_comm->rank() == CCL_UNORDERED_COLL_COORDINATOR) {
        ctx->match_id_size = match_id.length() + 1;
        ctx->match_id_value = service_sched->alloc_buffer(ctx->match_id_size).get_ptr();
        strncpy(static_cast<char*>(ctx->match_id_value), match_id.c_str(), ctx->match_id_size);
        LOG_DEBUG("coordinator bcasts match_id ",
                  match_id,
                  ", comm_id ",
                  ctx->reserved_comm_id,
                  ", ctx->match_id_size ",
                  ctx->match_id_size);
    }

    ccl_coll_param match_id_size_param{};
    match_id_size_param.ctype = ccl_coll_bcast;
    match_id_size_param.recv_buf = ccl_buffer(&ctx->match_id_size, sizeof(size_t));
    match_id_size_param.count = sizeof(size_t);
    match_id_size_param.dtype = ccl_datatype_int8;
    match_id_size_param.root = CCL_UNORDERED_COLL_COORDINATOR;
    match_id_size_param.comm = coll_param.comm;
    entry_factory::create<coll_entry>(service_sched.get(), match_id_size_param);

    service_sched->add_barrier();

    /* 2. broadcast match_id_value */
    ccl_coll_param match_id_val_param{};
    match_id_val_param.ctype = ccl_coll_bcast;
    match_id_val_param.recv_buf = ccl_buffer();
    match_id_val_param.count = 0;
    match_id_val_param.dtype = ccl_datatype_int8;
    match_id_val_param.root = CCL_UNORDERED_COLL_COORDINATOR;
    match_id_val_param.comm = coll_param.comm;
    auto entry = entry_factory::create<coll_entry>(service_sched.get(), match_id_val_param);

    entry->set_field_fn<ccl_sched_entry_field_recv_buf>(
        [](const void* fn_ctx, void* field_ptr) {
            auto ctx = static_cast<ccl_unordered_coll_ctx*>(const_cast<void*>(fn_ctx));
            if (ctx->service_sched->coll_param.comm->rank() != CCL_UNORDERED_COLL_COORDINATOR) {
                /* coordinator allocates and fills this buffer during schedule creation */
                ctx->match_id_value =
                    ctx->service_sched->alloc_buffer(ctx->match_id_size).get_ptr();
            }
            ccl_buffer* buf_ptr = (ccl_buffer*)field_ptr;
            buf_ptr->set(ctx->match_id_value, ctx->match_id_size);
            return ccl::status::success;
        },
        ctx);

    entry->set_field_fn<ccl_sched_entry_field_cnt>(
        [](const void* fn_ctx, void* field_ptr) -> ccl::status {
            auto ctx = static_cast<ccl_unordered_coll_ctx*>(const_cast<void*>(fn_ctx));
            auto count_ptr = static_cast<size_t*>(field_ptr);
            *count_ptr = ctx->match_id_size;
            return ccl::status::success;
        },
        ctx);

    service_sched->add_barrier();

    /* 4. start post actions (create communicator and start postponed schedules) */
    entry_factory::create<function_entry>(
        service_sched.get(),
        [](const void* func_ctx) -> ccl::status {
            auto ctx = static_cast<ccl_unordered_coll_ctx*>(const_cast<void*>(func_ctx));
            ctx->manager->start_post_coordination_actions(ctx);
            return ccl::status::success;
        },
        ctx);

    LOG_DEBUG("start service_sched ", service_sched.get(), " for match_id ", match_id);
    /* release ownership */
    bool extra_sched = true;
    ccl::global_data::get().executor->start(service_sched.release(), extra_sched);
}

void ccl_unordered_coll_manager::start_post_coordination_actions(ccl_unordered_coll_ctx* ctx) {
    int id = ctx->reserved_comm_id;
    std::string match_id{ static_cast<const char*>(ctx->match_id_value) };

    LOG_DEBUG("creating communicator with id ", id, " for match_id ", match_id);

    /* original comm is required to create new communicator with the same size */
    ccl_comm* original_comm = nullptr;

    {
        std::lock_guard<ccl_spinlock> lock{ postponed_scheds_guard };
        auto sched = postponed_scheds.find(match_id);
        if (sched != postponed_scheds.end()) {
            original_comm = sched->second->coll_param.comm;
            CCL_ASSERT(original_comm);
        }
    }

    if (!original_comm) {
        CCL_THROW_IF_NOT(coordination_comm->rank() != CCL_UNORDERED_COLL_COORDINATOR);
        /*
            coordinator broadcasted match_id
            but other ranks don't have collectives with the same match_id yet
        */
        LOG_DEBUG("can't find postponed sched for match_id ",
                  match_id,
                  ", postpone comm creation, reserved comm_id ",
                  id);
        std::lock_guard<ccl_spinlock> lock{ unresolved_comms_guard };
        unresolved_comms.emplace(match_id, id);
    }
    else {
        auto new_comm = original_comm->clone_with_new_id(id);
        add_comm(match_id, new_comm);
        run_postponed_scheds(match_id, new_comm.get());
    }

    CCL_ASSERT(ctx->service_sched, "service_sched is null");
    std::lock_guard<ccl_spinlock> lock{ service_scheds_guard };
    CCL_THROW_IF_NOT(service_scheds.find(match_id) == service_scheds.end());
    auto emplace_result = service_scheds.emplace(std::move(match_id), ctx->service_sched);
    CCL_ASSERT(emplace_result.second);
}

void ccl_unordered_coll_manager::run_postponed_scheds(const std::string& match_id, ccl_comm* comm) {
    CCL_THROW_IF_NOT(comm, "communicator is null");

    std::vector<ccl_sched*> scheds_to_run;
    std::unique_lock<ccl_spinlock> lock{ postponed_scheds_guard };
    auto scheds = postponed_scheds.equal_range(match_id);
    size_t sched_count = std::distance(scheds.first, scheds.second);
    LOG_DEBUG("found ", sched_count, " scheds for match_id ", match_id);

    scheds_to_run.reserve(sched_count);
    transform(scheds.first,
              scheds.second,
              back_inserter(scheds_to_run),
              [](const std::pair<std::string, ccl_sched*>& element) {
                  return element.second;
              });
    postponed_scheds.erase(scheds.first, scheds.second);
    lock.unlock();

    for (auto& sched : scheds_to_run) {
        run_sched(sched, comm);
    }
}

void ccl_unordered_coll_manager::run_sched(ccl_sched* sched, ccl_comm* comm) const {
    auto& partial_scheds = sched->get_subscheds();
    ccl_sched_key old_key, new_key;
    old_key.set(sched->coll_param, sched->coll_attr);
    sched->coll_param.comm = comm;
    new_key.set(sched->coll_param, sched->coll_attr);

    if (sched->coll_attr.to_cache) {
        ccl::global_data::get().sched_cache->recache(old_key, std::move(new_key));
    }

    for (size_t part_idx = 0; part_idx < partial_scheds.size(); ++part_idx) {
        partial_scheds[part_idx]->coll_param.comm = comm;
        if (ccl::global_data::env().priority_mode == ccl_priority_lifo) {
            /*
                can't use real lifo priority here because it can be unsynchronized between nodes
                so use comm id as synchronized increasing value for priority
            */
            partial_scheds[part_idx]->coll_attr.priority = comm->id();
        }
        partial_scheds[part_idx]->coll_attr.match_id = sched->coll_attr.match_id;
    }

    LOG_DEBUG("running sched ",
              sched,
              ", coll ",
              ccl_coll_type_to_str(sched->coll_param.ctype),
              ",  for match_id ",
              sched->coll_attr.match_id);

    sched->start(ccl::global_data::get().executor.get(), false);
}

void ccl_unordered_coll_manager::add_comm(const std::string& match_id,
                                          std::shared_ptr<ccl_comm> comm) {
    std::lock_guard<ccl_spinlock> lock(match_id_to_comm_map_guard);
    auto emplace_result = match_id_to_comm_map.emplace(match_id, comm);
    CCL_ASSERT(emplace_result.second);
}

void ccl_unordered_coll_manager::postpone_sched(ccl_sched* sched) {
    std::lock_guard<ccl_spinlock> lock{ postponed_scheds_guard };
    size_t sched_count = postponed_scheds.count(sched->coll_attr.match_id);
    LOG_DEBUG("postponed_scheds contains ",
              sched_count,
              " entries for match_id ",
              sched->coll_attr.match_id);
    postponed_scheds.emplace(sched->coll_attr.match_id, sched);
}

size_t ccl_unordered_coll_manager::get_postponed_sched_count(const std::string& match_id) {
    std::lock_guard<ccl_spinlock> lock{ postponed_scheds_guard };
    return postponed_scheds.count(match_id);
}

void ccl_unordered_coll_manager::remove_service_scheds() {
    std::lock_guard<ccl_spinlock> lock{ service_scheds_guard };
    for (auto it = service_scheds.begin(); it != service_scheds.end();) {
        ccl_sched* sched = it->second;
        if (sched->is_completed()) {
            LOG_DEBUG("sched ", sched, ", match_id ", it->first);
            delete sched;
            it = service_scheds.erase(it);
        }
        else {
            ++it;
        }
    }
}

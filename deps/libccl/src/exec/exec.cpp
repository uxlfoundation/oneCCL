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

#include <numeric>

#include "exec/exec.hpp"
#include "exec/thread/service_worker.hpp"
#include "exec/thread/worker.hpp"
#include "common/env/env.hpp"
#include "sched/sched.hpp"

size_t ccl_executor::get_worker_idx_by_sched_id(ccl_sched* sched) {
    if (sched->get_scaleout_flag()) {
        // sched->sched_id is same as master sched id for all sub schedules
        // so use the op_id to get different workers for different sub schedules
        return sched->get_op_id() % workers.size();
    }

    return sched->sched_id % workers.size();
}

size_t ccl_executor::calculate_atl_ep_count(size_t worker_count) {
    size_t ep_count = worker_count;

    if (ccl::global_data::env().priority_mode != ccl_priority_none) {
        ep_count *= CCL_PRIORITY_BUCKET_COUNT;
    }

    return ep_count;
}

atl_attr_t ccl_executor::generate_atl_attr(const ccl::env_data& env) {
    atl_attr_t attr;
    attr.in.enable_shm = env.enable_shm;
    /*
        TODO:
        executor may be destroyed before cached rma-based schedule made memory deregistration
        need to refactor global objects dependencies
        don't use ring_rma till that
    */
    attr.in.enable_rma = 0; // env.enable_rma;
    attr.in.enable_hmem = env.enable_hmem;
    attr.in.enable_sync_coll = env.enable_sync_coll;
    attr.in.enable_extra_ep = env.enable_extra_ep;
    attr.in.ep_count = calculate_atl_ep_count(env.worker_count);
    attr.in.mnic_type = env.mnic_type;
    attr.in.mnic_name = env.mnic_name_raw;
    attr.in.mnic_count = env.mnic_count;
    attr.in.mnic_offset = env.mnic_offset;

    memset(&attr.out, 0, sizeof(attr.out));

    return attr;
}

std::unique_ptr<ccl_sched_queue> ccl_executor::create_sched_queue(size_t idx,
                                                                  size_t ep_per_worker) {
    std::vector<size_t> ep_vec(ep_per_worker);
    std::iota(std::begin(ep_vec), std::end(ep_vec), idx * ep_per_worker);
    std::unique_ptr<ccl_sched_queue> sched_queue{ new ccl_sched_queue(idx, std::move(ep_vec)) };
    return sched_queue;
}

ccl_executor::ccl_executor(const char* main_addr) {
    auto& env = ccl::global_data::env();

    /* generate ATL attr for all future communicators */
    atl_comm_manager::set_internal_env(generate_atl_attr(env));
    atl_comm_manager::set_executor(this);
}

void ccl_executor::start_workers(atl_proc_coord_t& coord) {
    auto& env = ccl::global_data::env();
    auto& global_data = ccl::global_data::get();

    auto worker_count = env.worker_count;
    auto ep_count = calculate_atl_ep_count(worker_count);

    // firstly, check if an end user sets local coordinates.
    // if it is not, take coordinates from ATL coord structure
    if (global_data.get_local_proc_idx() == CCL_ENV_INT_NOT_SPECIFIED ||
        global_data.get_local_proc_count() == CCL_ENV_INT_NOT_SPECIFIED) {
        global_data.set_local_proc_idx(coord.local_idx);
        global_data.set_local_proc_count(coord.local_count);
        LOG_INFO("local_proc_idx: ",
                 global_data.get_local_proc_idx(),
                 ", local_proc_count: ",
                 global_data.get_local_proc_count(),
                 " are set by ATL transport");
    }

    CCL_THROW_IF_NOT(env.env_2_worker_affinity(global_data.get_local_proc_idx(),
                                               global_data.get_local_proc_count()));
    CCL_THROW_IF_NOT(env.env_2_worker_mem_affinity(global_data.get_local_proc_count()));

    if (env.worker_offload) {
        CCL_THROW_IF_NOT(
            env.worker_affinity.size() >= global_data.get_local_proc_count() * worker_count,
            "unexpected worker affinity length ",
            env.worker_affinity.size(),
            ", should be ",
            global_data.get_local_proc_count() * worker_count);
    }

    size_t ep_per_worker = ep_count / worker_count;
    for (size_t idx = 0; idx < worker_count; idx++) {
        if (env.enable_fusion && idx == 0) {
            LOG_DEBUG("create service worker");
            workers.emplace_back(new ccl_service_worker(
                idx, create_sched_queue(idx, ep_per_worker), *global_data.fusion_manager));
        }
        else {
            workers.emplace_back(new ccl_worker(idx, create_sched_queue(idx, ep_per_worker)));
        }

        if (env.worker_offload) {
            size_t cpu_affinity =
                env.worker_affinity[global_data.get_local_proc_idx() * worker_count + idx];
            size_t mem_affinity =
                env.worker_mem_affinity[global_data.get_local_proc_idx() * worker_count + idx];

            CCL_THROW_IF_NOT(
                workers.back()->start(cpu_affinity, mem_affinity) == ccl::status::success,
                "failed to start worker # ",
                idx);

            LOG_DEBUG("started worker: local_proc_idx ",
                      global_data.get_local_proc_idx(),
                      ", worker_idx ",
                      idx,
                      ", cpu: ",
                      cpu_affinity,
                      ", numa: ",
                      mem_affinity);
        }
    }
    workers_started = true;
}

ccl_executor::~ccl_executor() {
    // TODO: Rework to support listener
    //    if (listener) {
    //        listener->stop();
    //        LOG_DEBUG("stopped listener");
    //
    //        lock_workers();
    //        unlock_workers();
    //    }
    //    listener.reset();

    for (size_t idx = 0; idx < workers.size(); idx++) {
        if (ccl::global_data::env().worker_offload) {
            if (workers[idx]->stop() != ccl::status::success) {
                LOG_ERROR("failed to stop worker # ", idx);
            }
            else
                LOG_DEBUG("stopped worker # ", idx);
        }

        while (!workers[idx]->can_reset()) {
            ccl_yield(ccl::global_data::env().yield_type);
        }

        workers[idx].reset();
    }
}

void ccl_executor::lock_workers() {
    size_t idx;
    for (idx = 0; idx < workers.size(); idx++) {
        workers[idx]->should_lock = true;
    }

    idx = 0;
    while (idx < workers.size()) {
        if (workers[idx]->is_locked.load(std::memory_order_relaxed)) {
            idx++;
        }
        else {
            ccl_yield(ccl::global_data::env().yield_type);
        }
    }
}

void ccl_executor::unlock_workers() {
    size_t idx;
    for (idx = 0; idx < workers.size(); idx++) {
        workers[idx]->should_lock = false;
    }
    idx = 0;
    while (idx < workers.size()) {
        if (!workers[idx]->is_locked.load(std::memory_order_relaxed)) {
            idx++;
        }
    }
}

void ccl_executor::update_workers() {
    size_t ep_count = calculate_atl_ep_count(workers.size());
    size_t ep_per_worker = ep_count / workers.size();

    LOG_INFO("atl ep_count ", ep_count);

    for (size_t idx = 0; idx < workers.size(); idx++) {
        workers[idx]->reset_queue(create_sched_queue(idx, ep_per_worker));
    }
}

// TODO: Rework to support listener
//ccl::status ccl_executor::create_listener(ccl_resize_fn_t resize_func) {
//    if (listener) {
//        LOG_ERROR("attempt to create listener twice");
//        return ccl::status::runtime_error;
//    }
//
//    if (resize_func != NULL)
//        ccl::global_data::get().atl->set_resize_function((atl_resize_fn_t)resize_func);
//
//    /* pin listener thread together with first worker thread */
//    auto worker_affinity = ccl::global_data::env().worker_affinity;
//    size_t affinity_idx = get_local_proc_idx() * ccl::global_data::env().worker_count;
//    CCL_THROW_IF_NOT(worker_affinity.size() > affinity_idx);
//    size_t affinity = worker_affinity[affinity_idx];
//
//    listener = std::unique_ptr<ccl_listener>(new ccl_listener());
//    listener->start(affinity);
//
//    LOG_DEBUG("started listener");
//
//    return ccl::status::success;
//}

void ccl_executor::start(ccl_sched* sched, bool extra_sched) {
    if (extra_sched) {
        CCL_ASSERT(sched->sched_type == ccl_sched_unordered_coll,
                   "should be unordered_coll for now");

        sched->get_request()->set_counter(1);
        workers[0]->add(sched);
        return;
    }
    auto& partial_scheds = sched->get_subscheds();
    CCL_ASSERT(!partial_scheds.empty(), "at least one sub-sched should exist");
    /* Assign each partial scheduler to the worker in a round-robin manner.
       Use worker_idx only as a starting point for assignment.
       This approach covers the case when the sched_id maximum value overflows,
       therefore partial schedulers ids are not sequential, which means that several schedulers
       can be assigned to the same worker. This situation can lead to hangs on algos such as barrier. */
    size_t worker_idx = get_worker_idx_by_sched_id(partial_scheds.front().get());
    for (size_t idx = 0; idx < partial_scheds.size(); idx++) {
        LOG_DEBUG(
            "worker idx: ", worker_idx, ", coll: ", ccl_coll_type_to_str(sched->coll_param.ctype));
        workers[worker_idx]->add(partial_scheds[idx].get());
        worker_idx = (worker_idx + 1) % workers.size();
    }
}

void ccl_executor::wait(const ccl_request* req) {
    /* set urgent state for fusion manager */
    req->urgent = true;

    /* wait completion */
    while (!req->is_completed()) {
        do_work();
    }

    req->urgent = false;
}

bool ccl_executor::test(const ccl_request* req) {
    if (!req->is_completed()) {
        req->urgent = true;
        do_work();
        return false;
    }

    req->urgent = false;
    return true;
}

void ccl_executor::do_work() {
    size_t processed_count;
    if (ccl::global_data::env().worker_offload) {
        ccl_yield(ccl::global_data::env().yield_type);
    }
    else {
        for (auto& worker : workers) {
            worker->do_work(processed_count);
        }
    }
}

size_t ccl_executor::get_worker_count() const {
    return workers.size();
}

void ccl_executor::update_wait_condition(size_t idx,
                                         ccl_base_thread::wait_data::update_type type,
                                         size_t delta) {
    CCL_THROW_IF_NOT(idx < workers.size(), "unexpected worker idx ", idx);
    workers[idx]->update_wait_condition(type, delta);
}

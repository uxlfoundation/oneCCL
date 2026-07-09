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
#include "common/log/log.hpp"
#include "exec/exec.hpp"
#include "exec/thread/worker.hpp"

#include "sched/sched_timer.hpp"

#define CCL_WORKER_CHECK_STOP_ITERS     (16384)
#define CCL_WORKER_CHECK_UPDATE_ITERS   (16384)
#define CCL_WORKER_CHECK_AFFINITY_ITERS (16384)
#define CCL_WORKER_PROCESS_ALL_ITERS    (4096)

static void* ccl_worker_func(void* args);

ccl_worker::ccl_worker(size_t idx, std::unique_ptr<ccl_sched_queue> queue)
        : ccl_base_thread(idx, ccl_worker_func),
          should_lock(false),
          is_locked(false),
          process_atl(true),
          strict_sched_queue(std::unique_ptr<ccl_strict_sched_queue>(new ccl_strict_sched_queue())),
          sched_queue(std::move(queue)) {}

void ccl_worker::add(ccl_sched* sched) {
    LOG_DEBUG("add sched ",
              sched,
              ", coll ",
              ccl_coll_type_to_str(sched->coll_param.ctype),
              " bin: ",
              sched->bin);

    CCL_ASSERT(sched);
    CCL_ASSERT(!sched->bin);
    CCL_ASSERT(sched->get_in_bin_status() != ccl_sched_in_bin_added);

    update_wait_condition(ccl_base_thread::wait_data::update_type::increment, 1);

    if (sched->strict_order) {
        /* to keep valid non-completed req until safe releasing */
        sched->get_request()->increase_counter(1);
        strict_sched_queue->add(sched);
    }
    else {
        sched_queue->add(sched);
    }
}

ccl::status ccl_worker::do_work(size_t& processed_count) {
    do_work_counter++;

    auto ret = process_strict_sched_queue();
    if (ret != ccl::status::success)
        return ret;

    ret = process_sched_queue(processed_count,
                              (do_work_counter % CCL_WORKER_PROCESS_ALL_ITERS) ? false : true);
    if (ret != ccl::status::success)
        return ret;

    if ((do_work_counter % (4 * CCL_WORKER_PROCESS_ALL_ITERS) == 0) &&
        ccl::global_data::env().queue_dump) {
        sched_queue->dump(std::cout);
    }

    return ccl::status::success;
}

ccl::status ccl_worker::process_strict_sched_queue() {
    auto& queue = strict_sched_queue->peek();
    if (queue.empty())
        return ccl::status::success;

    size_t erased_scheds = 0;

    /* try to finish previous postponed operations */
    for (auto sched_it = queue.begin(); sched_it != queue.end(); sched_it++) {
        ccl_sched* sched = *sched_it;

        if (sched->get_in_bin_status() == ccl_sched_in_bin_erased) {
            CCL_THROW_IF_NOT(!sched->bin, "erased sched should be without bin");
            erased_scheds++;

            CCL_THROW_IF_NOT(
                erased_scheds == 1,
                "only single sched in active strict queue can be erased since previous call");

            /* now it is safe to release this sched */
            sched->complete();
            continue;
        }

        if (sched->get_in_bin_status() == ccl_sched_in_bin_none) {
            CCL_THROW_IF_NOT(!sched->bin, "unexpected bin ", sched->bin);
            /* here we add sched from strict_queue to regular queue for real execution */
            LOG_DEBUG("add sched ",
                      sched,
                      " from strict_queue to exec_queue, req ",
                      sched->get_request());
            sched_queue->add(sched);
        }

        CCL_THROW_IF_NOT(sched->get_in_bin_status() == ccl_sched_in_bin_added,
                         "sched ",
                         sched,
                         " unexpected in_bin_status ",
                         sched->get_in_bin_status());

        sched->do_progress();

        if (!sched->is_strict_order_satisfied()) {
            /*
                we can't state that current operation is started with strict order
                remove all previous operations from queue, as they were successfully started with strict order
                and return to strict starting for current operation on the next call
            */
            std::vector<ccl_sched*>(sched_it, queue.end()).swap(queue);
            return ccl::status::success;
        }
        else {
            /* now it is safe to release this sched */
            sched->complete();
        }
    }

    queue.clear();

    return ccl::status::success;
}

ccl::status ccl_worker::process_sched_queue(size_t& completed_sched_count, bool process_all) {
    completed_sched_count = 0;
    if (process_all) {
        auto bins = sched_queue->peek_all();

        if (bins.empty())
            return ccl::status::success;

        size_t completed_sched_count_local = 0;
        for (auto& bin : bins) {
            process_sched_bin(bin, completed_sched_count_local);
            completed_sched_count += completed_sched_count_local;
        }

        if (completed_sched_count)
            LOG_DEBUG("process_all, completed_sched_count ", completed_sched_count);

        return ccl::status::success;
    }
    else {
        ccl_sched_bin* bin = sched_queue->peek();
        if (!bin)
            return ccl::status::success;
        return process_sched_bin(bin, completed_sched_count);
    }
}

ccl::status ccl_worker::process_sched_bin(ccl_sched_bin* bin, size_t& completed_sched_count) {
    CCL_ASSERT(bin);
    completed_sched_count = 0;

    size_t bin_size = bin->size();

    if (bin_size == 0)
        return ccl::status::success;

    LOG_TRACE("bin ", bin, ", sched_count ", bin_size);

    /* ensure communication progress */

    if (process_atl) {
        for (size_t sched_idx = 0; sched_idx < 1; sched_idx++) {
            ccl_sched* sched = bin->get(sched_idx);
            ccl_comm* comm = sched->coll_param.comm;
            atl_status_t atl_status = comm->get_atl_comm()->poll(bin->get_atl_ep());
            CCL_THROW_IF_NOT(atl_status == ATL_STATUS_SUCCESS, "bad status ", atl_status);
        }
    }

    //    if (ccl::global_data::get().is_ft_enabled) {
    //        if (atl_status != ATL_STATUS_SUCCESS)
    //            return ccl::status::blocked_due_to_resize;
    //    }
    //    else {
    //        CCL_THROW_IF_NOT(atl_status == ATL_STATUS_SUCCESS, "bad status ", atl_status);
    //    }

    // iterate through the scheds stored in the bin
    for (size_t sched_idx = 0; sched_idx < bin_size;) {
        ccl_sched* sched = bin->get(sched_idx);
        CCL_ASSERT(sched && bin == sched->bin);

        sched->do_progress();

        if (sched->start_idx == sched->entries.size()) {
            // the last entry in the schedule has been completed, clean up the schedule and complete its request
            LOG_DEBUG("complete and dequeue: sched ",
                      sched,
                      ", coll ",
                      ccl_coll_type_to_str(sched->coll_param.ctype),
                      ", req ",
                      sched->get_request(),
                      ", entry_count ",
                      sched->entries.size());

            // remove completed schedule from the bin
            sched_queue->erase(bin, sched_idx);
            CCL_ASSERT(!sched->bin);
            bin_size--;
            LOG_DEBUG("completing request ", sched->get_request(), " for ", sched);
            sched->complete();
            ++completed_sched_count;
        }
        else {
            // this schedule is not completed yet, switch to the next sched in bin scheds list
            // progression of unfinished schedules will be continued in the next call of @ref ccl_bin_progress
            ++sched_idx;
        }
    }

    return ccl::status::success;
}

void ccl_worker::clear_queue() {
    strict_sched_queue->clear();
    sched_queue->clear();
}

void ccl_worker::update_wait_condition(ccl_base_thread::wait_data::update_type type, size_t delta) {
    if (delta == 0)
        return;

    LOG_DEBUG("type ", type, ", delta ", delta);

    if (!ccl::global_data::env().worker_wait)
        return;

    std::unique_lock<std::mutex> lock(wait.mtx);

    if (type == wait_data::update_type::increment) {
        wait.value += delta;
        if (wait.value - delta == 0)
            wait.var.notify_one();
    }
    else if (type == wait_data::update_type::decrement) {
        CCL_THROW_IF_NOT(
            delta <= wait.value, "decrement ", delta, " should be less or equal to ", wait.value);
        wait.value -= delta;
    }

    LOG_DEBUG("type ", type, ", delta ", delta, ", new value ", wait.value);
}

bool ccl_worker::check_wait_condition(size_t iter) {
    if (ccl::global_data::env().worker_wait && (wait.value == 0)) {
        std::unique_lock<std::mutex> lock(wait.mtx);
        wait.var.wait(lock, [this] {
            bool cond = ((wait.value == 0) && (check_stop_condition(0) == false));
            return !cond;
        });
    }
    else {
        ccl_yield(ccl::global_data::env().yield_type);
    }

    return true;
}

bool ccl_worker::check_stop_condition(size_t iter) {
    bool stop_signal = false;

    if ((iter % CCL_WORKER_CHECK_STOP_ITERS) == 0) {
        if (should_stop.load(std::memory_order_acquire))
            stop_signal = true;
    }

    return stop_signal;
}

bool ccl_worker::check_affinity_condition(size_t iter) {
    if ((iter % CCL_WORKER_CHECK_AFFINITY_ITERS) == 0) {
        int start_cpu_affinity = get_start_cpu_affinity();
        int real_cpu_affinity = get_real_cpu_affinity();
        if (start_cpu_affinity != real_cpu_affinity) {
            LOG_ERROR("worker ",
                      get_idx(),
                      " unexpectedly changed CPU affinity from ",
                      start_cpu_affinity,
                      " to ",
                      real_cpu_affinity);
        }
    }

    return true;
}

static void* ccl_worker_func(void* args) {
    auto worker = static_cast<ccl_worker*>(args);

    auto worker_idx = worker->get_idx();

    int cpu_core = worker->get_start_cpu_affinity();
    int numa_node = worker->get_start_mem_affinity();

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::set_thread_name("ccl_" + worker->name() + " " + std::to_string(worker_idx));
#endif // CCL_ENABLE_ITT

    LOG_DEBUG("worker: ",
              "idx: ",
              worker_idx,
              ", cpu: ",
              cpu_core,
              ", numa: ",
              ccl::global_data::get().hwloc_wrapper->get_numa_node(numa_node).to_string());

    ccl::global_data::get().hwloc_wrapper->membind_thread(numa_node);

    size_t iter = 0;
    size_t processed_count = 0;
    size_t max_spin_count = ccl::global_data::env().spin_count;
    size_t spin_count = max_spin_count;

    ccl::global_data::get().is_worker_thread = true;

    worker->started = true;

    do {
        try {
            if (worker->check_stop_condition(iter))
                break;
            worker->check_affinity_condition(iter);

            worker->do_work(processed_count);

            worker->update_wait_condition(ccl_base_thread::wait_data::update_type::decrement,
                                          processed_count);
        }
        catch (ccl::exception& ccl_e) {
            CCL_FATAL("worker ", worker_idx, " caught internal exception: ", ccl_e.what());
        }
        catch (std::exception& e) {
            CCL_FATAL("worker ", worker_idx, " caught exception: ", e.what());
        }
        catch (...) {
            CCL_FATAL("worker ", worker_idx, " caught general exception");
        }

        iter++;

        if (processed_count == 0) {
            spin_count--;
            if (!spin_count) {
                worker->check_wait_condition(iter);
                spin_count = 1;
            }
        }
        else {
            spin_count = max_spin_count;
        }
    } while (true);

    worker->started = false;

    return nullptr;
}

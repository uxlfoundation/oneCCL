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

#include "exec/exec.hpp"
#include "fusion/fusion.hpp"
#include "sched/buffer/buffer_cache.hpp"
#include "sched/cache/cache.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#define CCL_FUSION_CHECK_SCHEDS_ITERS (1024)

ccl::status complete_user_request(const void* ctx) {
    ccl_sched* sched = (ccl_sched*)ctx;
    LOG_DEBUG("complete fusion request: ", sched->get_request());
    sched->complete();
    return ccl::status::success;
}

ccl::status release_fusion_buf(const void* ctx) {
    void* buf = (void*)ctx;

    if (ccl::global_data::get().fusion_manager)
        ccl::global_data::get().fusion_manager->release_buffer(buf);

    return ccl::status::success;
}

ccl::status release_fusion_buf_for_cached_sched(ccl_sched* sched, const void* ctx) {
    return release_fusion_buf(ctx);
}

ccl_fusion_manager::ccl_fusion_manager()
        : bytes_threshold(ccl::global_data::env().fusion_bytes_threshold),
          count_threshold(ccl::global_data::env().fusion_count_threshold),
          buffer_size(bytes_threshold * count_threshold) {
    CCL_THROW_IF_NOT(bytes_threshold >= 1, "unexpected fusion_bytes_threshold ", bytes_threshold);
    CCL_THROW_IF_NOT(count_threshold >= 1, "unexpected fusion_count_threshold ", count_threshold);
    CCL_THROW_IF_NOT(buffer_size >= 1, "unexpected fusion_buffer_size ", buffer_size);

    long cycle_usec = long(ccl::global_data::env().fusion_cycle_ms * 1000.0);
    cycle = std::chrono::microseconds(cycle_usec);
    last_exec_time = std::chrono::steady_clock::now();

    LOG_INFO("created fusion manager, cycle_usec ",
             cycle_usec,
             ", bytes_threshold ",
             bytes_threshold,
             ", count_threshold ",
             count_threshold,
             ", buffer_size ",
             buffer_size);
}

ccl_fusion_manager::~ccl_fusion_manager() {
    LOG_INFO("fused_bytes ",
             stat_fused_bytes,
             ", fused_ops ",
             stat_fused_ops,
             ", empty_exec_calls ",
             stat_empty_exec_calls,
             ", overlapped_exec_calls ",
             stat_overlapped_exec_calls);

    reset();

    CCL_ASSERT(postponed_queue.empty() && exec_queue.empty() && tracked_scheds.empty(),
               "queues are not empty, ",
               postponed_queue.size(),
               " ",
               exec_queue.size(),
               " ",
               tracked_scheds.size());
}

bool ccl_fusion_manager::can_reset() {
    check_tracked_scheds(true);
    return tracked_scheds.empty();
}

void ccl_fusion_manager::reset() {
    while (tracked_scheds.size())
        check_tracked_scheds(true);
}

bool ccl_fusion_manager::can_fuse(ccl_sched* sched) {
    if (atl_base_comm::attr.out.enable_hmem) {
        /* TODO: implement fusion with D2D copies */
        return false;
    }

    if (sched->coll_param.ctype != ccl_coll_allreduce) {
        LOG_DEBUG("can't fuse due to coll ", ccl_coll_type_to_str(sched->coll_param.ctype));
        return false;
    }

    size_t bytes = sched->coll_param.get_send_count() * sched->coll_param.dtype.size();

    if (bytes >= bytes_threshold) {
        LOG_DEBUG("can't fuse due to size ", bytes, ", max ", bytes_threshold);
        return false;
    }

    if (sched->coll_param.deps.size()) {
        LOG_DEBUG("can't fuse due to deps size ", sched->coll_param.deps.size());
        return false;
    }

    if (sched->coll_attr.reduction_fn || sched->coll_attr.synchronous) {
        LOG_DEBUG("can't fuse due to unexpected fields in coll_attr");
        return false;
    }

    LOG_DEBUG("can fuse, bytes ", bytes);
    return true;
}

bool ccl_fusion_manager::add(ccl_sched* sched) {
    if (!can_fuse(sched)) {
        return false;
    }

    CCL_THROW_IF_NOT(sched->is_completed(), "incorrect completion counter");
    sched->get_request()->set_counter(1);

    {
        std::lock_guard<ccl_fusion_manager::lock_t> lock{ guard };
        postponed_queue.push_back(sched);
    }

    return true;
}

ccl_sched* ccl_fusion_manager::build_sched() {
    size_t sum_count = 0, sum_bytes = 0, dtype_size;
    size_t max_priority = 0;
    bool use_cache = true;
    ccl_comm* comm;
    ccl::reduction reduction;
    ccl_coll_type ctype;
    const ccl_stream* stream __attribute__((unused)) = nullptr;
    void* fusion_buf = nullptr;
    bool fill_sched = true;

    CCL_THROW_IF_NOT(exec_queue.size(), "empty queue");

    auto first_sched = exec_queue.front();
    auto last_sched = exec_queue.back();
    const ccl_datatype& dtype = first_sched->coll_param.dtype;
    dtype_size = dtype.size();
    reduction = first_sched->coll_param.reduction;
    comm = first_sched->coll_param.comm;
    ctype = first_sched->coll_param.ctype;
    stream = first_sched->coll_param.stream;
    max_priority = first_sched->coll_attr.priority;

    for (const auto& s : exec_queue) {
        sum_count += s->coll_param.get_send_count();
        if (!s->coll_attr.to_cache) {
            use_cache = false;
        }
        max_priority = std::max(s->coll_attr.priority, max_priority);
    }
    sum_bytes = sum_count * dtype_size;

    LOG_DEBUG("build fused_sched for sum_count ",
              sum_count,
              ", sum_bytes ",
              sum_bytes,
              ", sched_count ",
              exec_queue.size());

    ccl_sched* sched = nullptr;
    auto create_fn = [this, ctype, &fusion_buf, sum_count, dtype, reduction, comm, stream]() {
        ccl_sched* sched = nullptr;
        switch (ctype) {
            case ccl_coll_allreduce: {
                ccl::global_data::get().buffer_cache->get(0, buffer_size, &fusion_buf);
                ccl_coll_attr coll_attr;
                ccl_coll_param coll_param = ccl_coll_param::create_allreduce_param(fusion_buf,
                                                                                   fusion_buf,
                                                                                   sum_count,
                                                                                   dtype.idx(),
                                                                                   reduction,
                                                                                   coll_attr,
                                                                                   comm,
                                                                                   stream);
                sched = new ccl_sched({ ccl_sched_fusion, coll_param });
            } break;
            default: CCL_FATAL("not supported"); break;
        }
        return sched;
    };

    if (use_cache) {
        ccl_sched_key key{};
        key.f.ctype = ctype;
        key.f.count1 = sum_count;
        key.f.count2 = exec_queue.size();
        key.f.dtype = dtype.idx();
        key.f.reduction = reduction;
        key.f.comm = comm;
        key.match_id = first_sched->coll_attr.match_id + last_sched->coll_attr.match_id;
        LOG_DEBUG("key.match_id ", key.match_id);
        bool is_created = false;
        std::tie(sched, is_created) =
            ccl::global_data::get().sched_cache->find_or_create(std::move(key), create_fn);

        fill_sched = is_created;

        if (!is_created) {
            LOG_DEBUG("found fused_sched in cache: ", sched);
            if (!sched->is_completed()) {
                LOG_DEBUG("it is not completed sched");
                stat_overlapped_exec_calls++;
                if (ccl::global_data::get().executor->get_worker_count() > 1) {
                    LOG_DEBUG("found fused_sched in cache, which is not completed yet");
                    ccl::global_data::get().executor->wait(sched->get_request());
                }
                else {
                    CCL_THROW_IF_NOT(sched->is_completed(),
                                     "non completed fused_sched found in cache");
                }
            }
        }
    }
    else {
        sched = create_fn();
    }

    CCL_THROW_IF_NOT(sched);

    {
        std::lock_guard<ccl_fusion_manager::lock_t> lock{ guard };
        tracked_scheds.push_back(sched);
    }

    sched->coll_attr.priority = max_priority;
    sched->coll_attr.to_cache = use_cache;

    stat_fused_bytes += sum_bytes;
    stat_fused_ops += exec_queue.size();

    if (!fill_sched) {
        clear_exec_queue();
        return sched;
    }

    sched->commit(ccl::global_data::get().parallelizer.get());

    size_t exec_queue_size = exec_queue.size();
    size_t part_count = sched->get_subscheds().size();
    std::vector<std::shared_ptr<ccl_sched>>& part_scheds = sched->get_subscheds();
    size_t copies_per_part = exec_queue_size / part_count;
    size_t copies_per_last_part = copies_per_part + exec_queue_size % part_count;

    CCL_THROW_IF_NOT(part_count > 0, "unexpected part_count");

    LOG_DEBUG("part_count ",
              part_count,
              ", sum_count ",
              sum_count,
              ", exec_queue_size ",
              exec_queue_size);

    for (size_t idx = 0; idx < part_count; idx++) {
        part_scheds[idx]->add_barrier();
        part_scheds[idx]->set_add_mode(ccl_sched_add_front);
    }
    sched->sync_subscheds();

    size_t offset = 0;
    for (size_t idx = 0; idx < part_count; idx++) {
        size_t copies_count = (idx < part_count - 1) ? copies_per_part : copies_per_last_part;

        for (size_t copy_idx = 0; copy_idx < copies_count; copy_idx++) {
            size_t global_copy_idx = idx * copies_per_part + copy_idx;
#ifdef CCL_ENABLE_SYCL
            if (stream && stream->is_sycl_device_stream())
                entry_factory::create<copy_entry>(
                    part_scheds[idx].get(),
                    ccl_buffer(
                        exec_queue[global_copy_idx]->coll_param.get_send_buf_ptr(
                            0, ccl_coll_param::buf_type::device),
                        exec_queue[global_copy_idx]->coll_param.get_send_count() * dtype_size,
                        ccl_buffer_type::INDIRECT),
                    ccl_buffer(fusion_buf, buffer_size, offset),
                    exec_queue[global_copy_idx]->coll_param.get_send_count(),
                    dtype,
                    copy_attr(copy_direction::d2h));
            else
#endif // CCL_ENABLE_SYCL
                entry_factory::create<copy_entry>(
                    part_scheds[idx].get(),
                    ccl_buffer(
                        exec_queue[global_copy_idx]->coll_param.get_send_buf_ptr(),
                        exec_queue[global_copy_idx]->coll_param.get_send_count() * dtype_size,
                        ccl_buffer_type::INDIRECT),
                    ccl_buffer(fusion_buf, buffer_size, offset),
                    exec_queue[global_copy_idx]->coll_param.get_send_count(),
                    dtype);

            offset += exec_queue[global_copy_idx]->coll_param.get_send_count() * dtype_size;
        }
    }

    for (size_t idx = 0; idx < part_count; idx++) {
        part_scheds[idx]->set_add_mode(ccl_sched_add_back);
    }
    sched->sync_subscheds();

    offset = 0;
    for (size_t idx = 0; idx < part_count; idx++) {
        size_t copies_count = (idx < part_count - 1) ? copies_per_part : copies_per_last_part;

        for (size_t copy_idx = 0; copy_idx < copies_count; copy_idx++) {
            size_t global_copy_idx = idx * copies_per_part + copy_idx;
#ifdef CCL_ENABLE_SYCL
            if (stream && stream->is_sycl_device_stream())
                entry_factory::create<copy_entry>(
                    part_scheds[idx].get(),
                    ccl_buffer(fusion_buf, buffer_size, offset),
                    ccl_buffer(
                        exec_queue[global_copy_idx]->coll_param.get_recv_buf_ptr(
                            0, ccl_coll_param::buf_type::device),
                        exec_queue[global_copy_idx]->coll_param.get_recv_count() * dtype_size,
                        ccl_buffer_type::INDIRECT),
                    exec_queue[global_copy_idx]->coll_param.get_recv_count(),
                    dtype,
                    copy_attr(copy_direction::h2d));
            else
#endif // CCL_ENABLE_SYCL
                entry_factory::create<copy_entry>(
                    part_scheds[idx].get(),
                    ccl_buffer(fusion_buf, buffer_size, offset),
                    ccl_buffer(
                        exec_queue[global_copy_idx]->coll_param.get_recv_buf_ptr(),
                        exec_queue[global_copy_idx]->coll_param.get_recv_count() * dtype_size,
                        ccl_buffer_type::INDIRECT),
                    exec_queue[global_copy_idx]->coll_param.get_recv_count(),
                    dtype);

            part_scheds[idx]->add_barrier();

            offset += exec_queue[global_copy_idx]->coll_param.get_recv_count() * dtype_size;
            entry_factory::create<function_entry>(
                part_scheds[idx].get(), complete_user_request, exec_queue[global_copy_idx]);
            CCL_THROW_IF_NOT(!exec_queue[global_copy_idx]->is_completed(),
                             "incorrect completion counter");
        }
    }

    sched->sync_subscheds();

    if (use_cache) {
        part_scheds[0]->set_finalize_fn(release_fusion_buf_for_cached_sched, fusion_buf);
    }
    else {
        entry_factory::create<function_entry>(part_scheds[0].get(), release_fusion_buf, fusion_buf);
    }

    clear_exec_queue();

    return sched;
}

void ccl_fusion_manager::execute() {
    auto this_time = std::chrono::steady_clock::now();
    auto diff = (last_exec_time + cycle - this_time);
    if (diff > std::chrono::steady_clock::duration::zero()) {
        /* it is too early, do nothing */
        stat_empty_exec_calls++;
        return;
    }
    last_exec_time = std::chrono::steady_clock::now();

    bool flush_exec_queue = false;
    if (ccl::global_data::env().fusion_check_urgent && !exec_queue.empty()) {
        /* recheck scheds from exec_queue, maybe some of them were marked as urgent since previous call */
        for (auto it = exec_queue.begin(); it != exec_queue.end(); ++it) {
            if ((*it)->get_request()->urgent) {
                LOG_DEBUG("found urgent sched in exec_queue, flush exec_queue");
                flush_exec_queue = true;
                break;
            }
        }
    }

    /* separate block to reduce lock scope */
    {
        std::lock_guard<ccl_fusion_manager::lock_t> lock{ guard };
        if (!postponed_queue.empty()) {
            LOG_DEBUG("postponed_queue size ", postponed_queue.size());

            ccl_sched* first_sched;
            if (!exec_queue.empty()) {
                first_sched = exec_queue.front();
            }
            else {
                first_sched = postponed_queue.front();
                exec_queue.push_back(first_sched);
                postponed_queue.pop_front();
                exec_queue_sum_bytes =
                    first_sched->coll_param.get_send_count() * first_sched->coll_param.dtype.size();
            }

            for (auto it = postponed_queue.begin(); it != postponed_queue.end();) {
                auto s = *it;
                if (s->coll_param.dtype == first_sched->coll_param.dtype &&
                    s->coll_param.comm == first_sched->coll_param.comm &&
                    s->coll_param.ctype == first_sched->coll_param.ctype &&
                    s->coll_param.reduction == first_sched->coll_param.reduction &&
                    s->coll_param.stream == first_sched->coll_param.stream) {
                    size_t size = s->coll_param.get_send_count() * s->coll_param.dtype.size();
                    if (exec_queue_sum_bytes + size > buffer_size) {
                        LOG_DEBUG("too much bytes in buffer, flush exec_queue");
                        flush_exec_queue = true;
                        break;
                    }
                    exec_queue_sum_bytes += size;

                    if (ccl::global_data::env().fusion_check_urgent && !flush_exec_queue &&
                        s->get_request()->urgent) {
                        LOG_DEBUG(
                            "found urgent sched in postponed_queue, flush exec_queue, postponed_queue size ",
                            postponed_queue.size());
                        flush_exec_queue = true;
                    }

                    exec_queue.push_back(s);
                    it = postponed_queue.erase(it);

                    if (exec_queue.size() == count_threshold) {
                        LOG_DEBUG("too many scheds, flush exec_queue");
                        flush_exec_queue = true;
                        break;
                    }
                }
                else {
                    ++it;
                }
            }
        }
    }

    if (flush_exec_queue) {
        LOG_DEBUG("exec_queue size ", exec_queue.size(), ", bytes ", exec_queue_sum_bytes);
        ccl_sched* sched = build_sched();
        sched->start(ccl::global_data::get().executor.get());
    }

    if (stat_fused_ops % CCL_FUSION_CHECK_SCHEDS_ITERS == 0) {
        check_tracked_scheds();
    }
}

void ccl_fusion_manager::release_buffer(void* buf) {
    ccl::global_data::get().buffer_cache->push(0, buffer_size, buf);
}

void ccl_fusion_manager::clear_exec_queue() {
    exec_queue.clear();
    exec_queue_sum_bytes = 0;
}

void ccl_fusion_manager::check_tracked_scheds(bool force_release) {
    std::lock_guard<ccl_fusion_manager::lock_t> lock{ guard };
    for (auto it = tracked_scheds.begin(); it != tracked_scheds.end();) {
        ccl_sched* sched = *it;
        if (sched->is_completed() && (!sched->coll_attr.to_cache || force_release)) {
            ccl_release_sched(sched);
            it = tracked_scheds.erase(it);
        }
        else {
            ++it;
        }
    }
}

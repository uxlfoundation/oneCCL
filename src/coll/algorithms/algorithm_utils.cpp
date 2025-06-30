/*
 Copyright 2016-2020 Intel Corporation
 
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
#include <algorithm>
#include <numeric>
#include <sstream>

#include "coll/algorithms/algorithm_utils.hpp"
#include "common/log/log.hpp"
#include "sched/entry/factory/entry_factory.hpp"

const char* ccl_coll_type_to_str(ccl_coll_type type) {
    auto type_str = "undefined";
    switch (type) {
        case ccl_coll_allgather: return "allgather";
        case ccl_coll_allgatherv: return "allgatherv";
        case ccl_coll_allreduce: return "allreduce";
        case ccl_coll_alltoall: return "alltoall";
        case ccl_coll_alltoallv: return "alltoallv";
        case ccl_coll_barrier: return "barrier";
        case ccl_coll_bcast: return "bcast";
        case ccl_coll_broadcast: return "broadcast";
        case ccl_coll_recv: return "recv";
        case ccl_coll_reduce: return "reduce";
        case ccl_coll_reduce_scatter: return "reduce_scatter";
        case ccl_coll_send: return "send";
        case ccl_coll_partial: return "partial";
        case ccl_coll_undefined: return type_str;
        default: type_str = "unknown";
    }
    return type_str;
}

void ccl_get_segment_sizes(size_t dtype_size,
                           size_t elem_count,
                           size_t requested_seg_size,
                           std::vector<size_t>& seg_sizes) {
    seg_sizes.clear();

    if (dtype_size * elem_count == 0) {
        return;
    }
    else if (dtype_size >= requested_seg_size) {
        seg_sizes.resize(elem_count, 1);
    }
    else {
        size_t seg_size = (requested_seg_size + dtype_size - 1) / dtype_size;
        size_t total_seg_count = std::max((elem_count + seg_size - 1) / seg_size, 1UL);
        size_t regular_seg_size = elem_count / total_seg_count;
        size_t large_seg_size = regular_seg_size + ((elem_count % total_seg_count) != 0);
        size_t regular_seg_count = total_seg_count * large_seg_size - elem_count;

        seg_sizes.resize(total_seg_count, regular_seg_size);
        std::fill(seg_sizes.begin() + regular_seg_count, seg_sizes.end(), large_seg_size);

        size_t sum =
            std::accumulate(seg_sizes.begin(), seg_sizes.end(), ccl::utils::initial_count_value);
        if (sum != elem_count) {
            std::stringstream ss;
            for (size_t idx = 0; idx < seg_sizes.size(); idx++) {
                ss << seg_sizes[idx] << " ";
            }
            CCL_THROW_IF_NOT(false,
                             "unexpected sum of seg_sizes ",
                             sum,
                             ", expected ",
                             elem_count,
                             ", total_seg_count ",
                             total_seg_count,
                             ", regular_seg_count ",
                             regular_seg_count,
                             ", regular_seg_size ",
                             regular_seg_size,
                             ", large_seg_size ",
                             large_seg_size,
                             ", all seg_sizes: ",
                             ss.str());
        }
    }
}

bool ccl_is_ptr_aligned(uintptr_t ptr, size_t alignment) {
    CCL_THROW_IF_NOT(alignment != 0, "memory alignment cannot be 0 by definition");
    return (ptr % alignment) == 0;
}

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)

using entry_iterator = std::deque<std::unique_ptr<sched_entry>>::iterator;

static bool is_reorderable_algo(const char* algo_name) {
    const char* reordable_algo_prefixes[] = {
        "ALLREDUCE_PIPE",
        "ALLGATHERV_PIPE",
        "REDUCE_PIPE",
        "REDUCE_SCATTER_PIPE",
    };
    for (auto& reordable_algo_prefix : reordable_algo_prefixes) {
        if (0 == strncmp(algo_name, reordable_algo_prefix, strlen(reordable_algo_prefix))) {
            return true;
        }
    }
    return false;
}

std::string to_string(ccl_chunking_mode& mode) {
    switch (mode) {
        case ccl_chunking_mode::ccl_pipeline_none: return "ccl_pipeline_none";
        case ccl_chunking_mode::ccl_buffer_implicit: return "ccl_buffer_implicit";
        case ccl_chunking_mode::ccl_offset_explicit: return "ccl_offset_explicit";
        default: return "unknown";
    }
}

// Reorders commands to minimize wait time
//
// Sample case before submission, 3-stage command split into 2 chunks:
//      chunk_0.0 chunk_0.1 chunk_0.2 chunk_1.0 chunk_1.1 chunk_1.2
//
// Problem: each command within chunk depends on its predecessor, serial execution
//
// Submission order within this function:
//      chunk_0.0 chunk_1.0 chunk_0.1 chunk_1.1 chunk_0.2 chunk_1.2
// allows parallel execution of e.g. chunk_0.0 and chunk_1.1
//
// Reordering only happens for entries that have a name that matches `is_reorderable_algo`
//
// Opportunity for micro-optimization and simplification: compare enum instead of strings
uint32_t ccl_submit_ze_commands_in_subsched_entries(ccl_sched* sched) {
    // The structure is mapping group's index onto a vector with pairs of iterators. First iterator in a
    // pair represents an iterator over entries of a single subschedule in the group. Second iterator is
    // equal to `chunk->get_subsched()->entries.end()` and is used to check if the first iterator is finished.
    //
    // Example structure shape:
    // 0-->[<a->entries->begin(), a->entries->end()>, <b->entries->begin(), b->entries->end()>]
    // 1-->[<c->entries->begin(), c->entries->end()>, <d->entries->begin(), d->entries->end()>]
    std::map<size_t, std::vector<std::pair<entry_iterator, entry_iterator>>> groups_iterators_map;
    std::unordered_map<size_t, std::shared_ptr<sched_group>> groups_map;
    for (auto& entry : sched->entries) {
        if (is_reorderable_algo(entry->name())) {
            auto chunk = static_cast<subsched_entry*>(entry.get());
            auto chunk_group = chunk->get_subsched()->get_group();
            auto& group_iterators = groups_iterators_map[chunk_group->get_id()];
            group_iterators.push_back(std::pair(chunk->get_subsched()->entries.begin(),
                                                chunk->get_subsched()->entries.end()));

            groups_map[chunk_group->get_id()] = std::move(chunk_group);
        }
        else {
            LOG_DEBUG("entry: ", entry->name(), "is NOT reorderable algo")
        }
    }

    uint32_t command_count = 0;

    for (auto& [group_id, group_iterators] : groups_iterators_map) {
        auto group = groups_map[group_id];
        LOG_DEBUG("|GROUPS| Submitting group: ",
                  group->get_id(),
                  ", iterators count: ",
                  group_iterators.size());

        while (!group_iterators.empty()) {
            // `group_iterators` is a vector which constists of iterators
            // over all subscheds in a group. Aim of the loop is to submit
            // all entries in these subscheds with regard to their parallel
            // execution.
            for (auto& chunk_iterator : group_iterators) {
                bool is_chunk_partial_submission_finished = false;
                while (is_chunk_partial_submission_finished == false) {
                    // Exit condition for non parallelizable groups and
                    // subsheds with no entries at all
                    if (chunk_iterator.first == chunk_iterator.second) {
                        // If the group is NOT parallelizable we MUST submit
                        // all entries from each subsched sequentially, so
                        // entries from a subsched are NOT interrupted by
                        // the other subscheds. To acheive that here we submit all
                        // the entries from a subsched in one take.
                        // This approach results in following entry pattern:
                        // a[0]->a[1]->a[2]->b[0]->n[1]->b[2]
                        is_chunk_partial_submission_finished = true;
                        break;
                    }

                    auto entry = chunk_iterator.first->get();
                    LOG_DEBUG("|GROUPS| Submitting entry: ", entry->name());
                    command_count += entry->ze_commands_submit();
                    chunk_iterator.first++;

                    // Exit condition for parallelizable groups
                    if (group->parallelizable()) {
                        // If a group can be parallelized we should submit
                        // one entry per subsched in each iteration of the for loop.
                        // The aim is to achieve concurrent execution of the entries, so
                        // having two subscheds `a` and `b` we want submit each subsched in
                        // turns. This approach results in following entry pattern:
                        // a[0]->b[0]->a[1]->b[1]->a[2]->b[2]
                        is_chunk_partial_submission_finished = true;
                    }
                }
            }

            auto new_end =
                std::remove_if(group_iterators.begin(), group_iterators.end(), [](auto e) {
                    // Remove if entries iterator is equal to entries.end()
                    return e.first == e.second;
                });
            group_iterators.erase(new_end, group_iterators.end());
        }
    }

    return command_count;
}

// Attempts to calculate buffer size for pipelining. If the pipeline
// is not possible with such parameters returns std::nullopt
std::optional<size_t> ccl_get_pipe_size(const size_t buf_size,
                                        const size_t dtype_size,
                                        const size_t chunk_count) {
    // Note about cache lines and pipelining: The same cache line must contain
    // a single chunk only.
    //
    // If the same cache line contains two chunks (or more), and we parallelize
    // the instructions required for both chunks, a conflict (race condition)
    // may appear between the copy-out for the scaleout portion and the
    // reduce_scatter phase.
    //
    // The easiest way to avoid that race condition is to require that each
    // cache line contains a single entry. If that is not the case, we must not
    // parallelize the instructions for different chunks.

    bool is_pipe = chunk_count > 1 && ccl::global_data::env().enable_ze_single_list;

    if (!is_pipe) {
        LOG_DEBUG("Pipelining code disabled");
        return std::nullopt;
    }

    // TODO: why does oneCCL have CACHELINE_SIZE *and* CCL_KERNEL_MEM_ALIGN?
    size_t mem_align = ccl::global_data::env().kernel_mem_align;
    CCL_THROW_IF_NOT(mem_align != 0, "memory alignment cannot be zero by definition");

    // First, determine if we need to fallback to non-pipelined algorightm.
    // Such a fallback may happen in cases such as (1) the user requests it,
    // (2) message fits into a cache line, or (3) the cache line size is not
    // divisible by the data type size.

    size_t number_of_cache_lines_per_chunk =
        std::max(mem_align, buf_size / chunk_count) / mem_align;
    size_t main_chunk_size_bytes = mem_align * number_of_cache_lines_per_chunk;

    bool is_dtype_divisible = ((main_chunk_size_bytes % dtype_size) == 0);
    bool is_msg_bigger_than_cache_line = buf_size > main_chunk_size_bytes;
    bool is_singleworker =
        !ccl::global_data::env().ze_multi_workers || (ccl::global_data::env().worker_count <= 1);

    if (!is_dtype_divisible) {
        LOG_INFO("Running without pipelining because datatype size (",
                 dtype_size,
                 ") is not divisible by cache line size (",
                 mem_align,
                 ")");
        return std::nullopt;
    }

    if (!is_msg_bigger_than_cache_line) {
        LOG_INFO("Running without pipelining because message size (",
                 buf_size,
                 ") is smaller than a cache line (",
                 mem_align,
                 ") or than main_chunk_size_bytes (",
                 main_chunk_size_bytes,
                 ")");
        return std::nullopt;
    }

    if (!is_singleworker) {
        LOG_INFO("Running without pipelining because ze_multi_workers was requested with ",
                 ccl::global_data::env().worker_count,
                 " workers, which is more than one worker ");
        return std::nullopt;
    }

    return main_chunk_size_bytes;
}

size_t estimate_tmp_bufs_count(size_t count,
                               size_t pair_comm_size,
                               size_t even_comm_size,
                               size_t comm_size) {
    size_t base_count = count / pair_comm_size;
    base_count += count % pair_comm_size;

    size_t main_block_count = base_count / even_comm_size;
    size_t block_count = main_block_count;
    block_count += base_count % even_comm_size;

    return block_count;
}

// Estimate count of elements needed by temporary buffers
// to execute the operation
size_t estimate_tmp_count(ccl_coll_type coll_type,
                          size_t count,
                          size_t pair_comm_size,
                          size_t even_comm_size,
                          size_t comm_size,
                          bool is_scaleout) {
    size_t result = 0;
    size_t buf_count = 0;
    switch (coll_type) {
        case ccl_coll_reduce_scatter:
            result += estimate_tmp_bufs_count(
                          count * comm_size, pair_comm_size, even_comm_size, comm_size) *
                      even_comm_size;
            buf_count += even_comm_size;
            LOG_DEBUG("|GROUPS| tmp_bufs will utilize: ", result);
            if (is_scaleout) {
                result += count * comm_size;
                buf_count += 1;
                LOG_DEBUG("|GROUPS| scaleout will utilize: ", result);
            }

            // Account for possible alignment
            result += buf_count * ccl::global_data::env().kernel_mem_align;
            LOG_DEBUG("|GROUPS| alignments will utilize: ", result);
            return result;
        default: return count;
    }
}

using sched_group_size_vec = std::vector<std::pair<std::shared_ptr<sched_group>, size_t>>;

sched_group_size_vec serialize_chunks(const sched_group_size_vec& chunks,
                                      size_t memory_scale_factor) {
    // TODO: The code seems to break collectives when outputting count that is less than comm->size()
    if (!ccl::global_data::env().enable_ze_single_list) {
        // Currently only single list mode is supported, so we do not split the collective
        // in any way.
        return chunks;
    }

    LOG_DEBUG("|GROUPS| Serialize operation with memory scale factor: ", memory_scale_factor);

    sched_group_size_vec new_serial_chunks;
    for (auto& [group, bytes] : chunks) {
        size_t mem_align = ccl::global_data::env().kernel_mem_align;
        size_t serial_chunk_bytes = ccl::global_data::env().ze_tmp_buf_size / memory_scale_factor;
        serial_chunk_bytes = (serial_chunk_bytes / mem_align) * mem_align;

        if (serial_chunk_bytes == 0) {
            // We were unable to align `serial_chunk_bytes` to `mem_align`.
            // This is very unlikely, and would require user to set very low
            // `CCL_ZE_TMP_BUF_SIZE`.
            static bool memory_warning_issued = false;
            if (!memory_warning_issued) {
                LOG_WARN("Memory usage might exceed `CCL_ZE_TMP_BUF_SIZE`");
                memory_warning_issued = true;
            }
            serial_chunk_bytes = mem_align;
        }

        size_t half_serial_chunk_bytes = serial_chunk_bytes / 2;
        if (bytes > serial_chunk_bytes) {
            size_t bytes_left = bytes;
            auto cur_group = group;
            while (bytes_left > 0) {
                size_t new_serial_chunk_bytes =
                    bytes_left > serial_chunk_bytes ? serial_chunk_bytes : bytes_left;
                size_t next_serial_chunk_bytes = bytes_left - new_serial_chunk_bytes;

                // If next chunk after currently processed one would be less than half of `serial_chunk_bytes`
                // we should split the bytes_left approximately in half rounding first chunk to cacheline size.
                if (next_serial_chunk_bytes > 0 &&
                    next_serial_chunk_bytes < half_serial_chunk_bytes) {
                    new_serial_chunk_bytes = (bytes_left / 2 / mem_align) * mem_align;

                    if (new_serial_chunk_bytes == 0) {
                        new_serial_chunk_bytes = half_serial_chunk_bytes;
                    }
                    // After the iteration `bytes_left` will be a little more than
                    // current `new_serial_chunk_bytes`
                }

                bytes_left -= new_serial_chunk_bytes;
                new_serial_chunks.emplace_back(cur_group, new_serial_chunk_bytes);

                // Create new `sched_group` with the same
                // `memory_context`, but with ability to create
                // different allocations across the same buffer.
                cur_group = std::make_shared<sched_group>(*cur_group);

                LOG_DEBUG("|GROUPS| New serial chunk: ", new_serial_chunk_bytes);
            }
        }
        else {
            // Operation size is smaller than chunk size, so there will
            // be only one sequential chunk. Since there's only one chunk
            // we do not need barrier after GPU entries from the chunk.
            group->disable_last_chunk_barrier();

            new_serial_chunks.emplace_back(group, bytes);
            LOG_DEBUG("|GROUPS| New serial chunk: ", bytes);
        }
    }

    LOG_DEBUG("|GROUPS| Serialized: ", new_serial_chunks.size())
    return new_serial_chunks;
}

sched_group_size_vec parallelize_chunks(const sched_group_size_vec& chunks,
                                        const size_t dtype_size,
                                        const size_t pipe_chunk_count) {
    sched_group_size_vec new_parallel_chunks;

    for (auto& [group, bytes] : chunks) {
        std::optional<size_t> pipe_chunk_size_opt =
            ccl_get_pipe_size(bytes, dtype_size, pipe_chunk_count);
        size_t pipe_chunk_size = pipe_chunk_size_opt.value_or(bytes);

        size_t bytes_left = bytes;
        while (bytes_left > 0) {
            size_t new_parallel_chunk_count = pipe_chunk_size;
            LOG_DEBUG("|GROUPS| Bytes left: ", bytes_left);
            if (bytes_left - pipe_chunk_size < pipe_chunk_size) {
                // Next chunk would end up smaller than `pipe_chunk_size`,
                // so we make this one bigger.
                new_parallel_chunk_count = bytes_left;
            }

            new_parallel_chunks.emplace_back(group, new_parallel_chunk_count);
            group->increase_chunk_count();
            bytes_left -= new_parallel_chunk_count;

            LOG_DEBUG("|GROUPS| New parallel chunk: ", new_parallel_chunk_count);
        }
    }
    LOG_DEBUG("|GROUPS| Paralellized: ", new_parallel_chunks.size())
    return new_parallel_chunks;
}

ccl::status ccl_build_topo_uniform_buff_size_op(
    ccl_sched* sched,
    ccl_buffer send_buf,
    ccl_buffer recv_buf,
    size_t count,
    size_t dtype_size,
    size_t pipe_nof_chunks,
    const std::string& op_name,
    ccl::profile::metrics_counter& metrics,
    ccl_comm* comm,
    std::function<ccl::status(ccl_sched* sched,
                              ccl_buffer send_buf,
                              ccl_buffer recv_buf,
                              size_t count,
                              size_t offset,
                              size_t combined_count)> fill_op_lambda,
    ccl_chunking_mode mode,
    ccl_coll_type coll) {
    // Allocate buffer used as `memory_context` for all `sched_groups`. This means that
    // subschedules inside one group will share allocations across the buffer and
    // other groups will reuse the same memory but with ability to allocate different buffers
    // over the same allocation. Groups guarantee that only one group is executed at a time on GPU,
    // so we are safe to perform such overlapping allocations because the ownership of each allocation
    // will be exclusive during execution of each group.
    //
    //                    ┌───────────────────┬─────► memory_context_ptr
    //    ┌────────────┐  │   ┌────────────┐  │
    //    │  Group 1   │  │   │  Group 2   │  │
    //    ├────────────┼──┘   ├────────────┼──┘
    //    │Allocation 1│      │Allocation 1│
    //    ├────────────┤      │            │
    //    │Allocation 2│      │            │
    //    │            │      ├────────────┤
    //    │            │      │Allocation 2│
    //    ├────────────┤      │            │
    //    │Allocation 3│      │            │
    //    └┬──────────▲┘      └┬──────────▲┘
    //    ┌▼──────────┼┐      ┌▼──────────┼┐               GPU Timeline
    // ───┴────────────┴──────┴────────────┴───────────────────────────►
    //
    static void* memory_context_ptr = nullptr;
    if (memory_context_ptr == nullptr) {
        auto stream = sched->coll_param.stream;
        auto context = stream->get_ze_context();
        auto device = stream->get_ze_device();
        device_allocate(context,
                        ccl::ze::default_device_mem_alloc_desc,
                        ccl::global_data::env().ze_tmp_buf_size,
                        ccl::global_data::env().kernel_mem_align,
                        device,
                        &memory_context_ptr);
    }

    // This is an initial `sched_group` that is used to group execution of
    // serial chunks and provide them shared resources.
    auto base_group = std::make_shared<sched_group>(
        sched, comm, memory_context_ptr, ccl::global_data::env().ze_tmp_buf_size);

    sched_group_size_vec final_chunk_bytes_per_group;

    if (mode == ccl_chunking_mode::ccl_pipeline_none) {
        // Pipeline disenabled
        final_chunk_bytes_per_group.emplace_back(std::move(base_group), count * dtype_size);
    }
    else if (count > 0) {
        // Normal counts
        sched_group_size_vec initial_chunk_bytes_per_group;
        initial_chunk_bytes_per_group.emplace_back(std::move(base_group), count * dtype_size);

        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        const bool is_scaleout = !topo_manager.is_single_node;

        ccl_comm* pair_comm = comm->get_pair_comm().get();
        ccl_comm* even_comm = comm->get_even_comm().get();

        size_t estimated_tmp_count = estimate_tmp_count(
            coll, count, pair_comm->size(), even_comm->size(), comm->size(), is_scaleout);

        // Divide and round up count of temporary elements by operation count
        size_t memory_scale_factor =
            estimated_tmp_count / count + (estimated_tmp_count % count != 0);

        auto serialized_chunks =
            serialize_chunks(initial_chunk_bytes_per_group, memory_scale_factor);
        auto parallelized_chunks =
            parallelize_chunks(serialized_chunks, dtype_size, pipe_nof_chunks);
        final_chunk_bytes_per_group = std::move(parallelized_chunks);
    }
    else {
        // Zero count
        final_chunk_bytes_per_group.emplace_back(std::move(base_group), 0);
    }

    if (final_chunk_bytes_per_group.size() == 1) {
        // Fall back to topo algorithm without pipelining
        auto group = final_chunk_bytes_per_group[0].first;

        sched->set_group(std::move(group));
        fill_op_lambda(sched, send_buf, recv_buf, count, 0, count);

        entry_factory::create<ze_execute_cmdlists_on_init_entry>(sched);

        return ccl::status::success;
    }

    LOG_DEBUG("build pipe ", op_name);

    sched->try_enable_ze_single_list();
    auto sync_obj = std::make_shared<sync_object>(final_chunk_bytes_per_group.size());
    auto groups_sync_obj = std::make_shared<sync_object>(final_chunk_bytes_per_group.size());
    bool is_parallelizable_chunks = true;

    size_t chunk_idx = 0;
    size_t combined_recv_count = count;
    size_t offset = 0;
    size_t mem_align = ccl::global_data::env().kernel_mem_align;

    for (auto& [group, bytes] : final_chunk_bytes_per_group) {
        ccl_buffer sbuf = send_buf;
        ccl_buffer rbuf = recv_buf;

        if (mode == ccl_chunking_mode::ccl_buffer_implicit) {
            sbuf += offset;
            rbuf += offset;
        }

        LOG_DEBUG("|GROUPS| Creating subsched with group: ",
                  group->get_id(),
                  ", offset: ",
                  offset,
                  ", bytes: ",
                  bytes,
                  ", mode: ",
                  to_string(mode));

        auto moved_group = std::move(group);
        moved_group->set_sync_obj(groups_sync_obj);
        size_t this_chunk_count = bytes / dtype_size;
        LOG_DEBUG("Count: ", this_chunk_count);

        if (this_chunk_count || (count == 0 && chunk_idx == 0)) {
            entry_factory::create<subsched_entry>(
                sched,
                chunk_idx,
                [sched,
                 sbuf,
                 rbuf,
                 this_chunk_count,
                 sync_obj,
                 moved_group,
                 offset,
                 combined_recv_count,
                 fill_op_lambda](ccl_sched* s) {
                    s->inherit_ze_managers_from(sched);
                    s->set_init_ze_hook_sync_obj(sync_obj);
                    s->set_ze_commands_bypass_flag(false);
                    s->set_group(std::move(moved_group));

                    fill_op_lambda(s, sbuf, rbuf, this_chunk_count, offset, combined_recv_count);
                },
                (op_name + "_PIPE" + std::to_string(chunk_idx)).c_str());
        }

        offset += bytes;
        chunk_idx++;

        // WARNING: previous chunk has part of this chunk's first cache
        // line. Cannot use pipelining. However, since this is a
        // "local" decision (i.e., other ranks may decide differently),
        // we still need to apply chunking. However, we will run one
        // chunk at a time, without parallelizing them.
        // Another way to have implemented this would be to link the
        // last task of the prev chunk with the first of this chunk
        // with an event.
        is_parallelizable_chunks &=
            ccl_is_ptr_aligned(reinterpret_cast<uintptr_t>(rbuf.get_ptr()), mem_align);
        if (!is_parallelizable_chunks) {
            moved_group->disable_parallel_execution();
        }
    }

    static bool is_chunk_mem_align_warning_printed{};
    if (!is_parallelizable_chunks && !is_chunk_mem_align_warning_printed) {
        is_chunk_mem_align_warning_printed = true;
        LOG_WARN(
            "[",
            op_name,
            " pipelining]: For best performance, (i) chunk size should be a multiple of a cache line (",
            mem_align,
            " bytes), and (ii) buffers in all ranks should be aligned to ",
            mem_align);
    }

    if (!is_parallelizable_chunks) {
        metrics.nonparallel_calls_per_count[count]++;
    }
    else {
        metrics.parallel_calls_per_count[count]++;
    }

    if (sched->use_single_list) {
        entry_factory::create<ze_execute_cmdlists_on_start_entry>(
            sched, sync_obj, ccl_submit_ze_commands_in_subsched_entries);
    }

    return ccl::status::success;
}

#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

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
#include "coll/coll_util.hpp"
#include "coll/selection/selection.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
#include "sched/entry/ze/ze_event_signal_entry.hpp"
#include "sched/entry/ze/ze_event_wait_entry.hpp"
#include "sched/entry/ze/ze_pt2pt_barrier_entry.hpp"
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

namespace ccl {

void add_coll_entry(ccl_sched* sched, const ccl_coll_param& param) {
    ccl_selector_param selector_param;

    if (param.ctype == ccl_coll_send || param.ctype == ccl_coll_recv) {
        coll_entry::build_sched(sched, param);
    }
    else {
        selector_param.ctype = param.ctype;
        selector_param.count = param.count;
        if (param.ctype == ccl_coll_allgatherv) {
            selector_param.count = param.send_count;
        }
        selector_param.recv_counts =
            const_cast<size_t*>(reinterpret_cast<const size_t*>(param.recv_counts.data()));
        selector_param.dtype = param.dtype;
        selector_param.comm = param.comm;
        selector_param.stream = param.stream;
        selector_param.buf = (param.send_buf) ? param.send_buf.get_ptr() : param.recv_buf.get_ptr();
        selector_param.is_vector_buf = sched->coll_attr.is_vector_buf;
#ifdef CCL_ENABLE_SYCL
        selector_param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
        selector_param.hint_algo = param.hint_algo;
        selector_param.peer_rank = param.peer_rank;
        selector_param.is_scaleout = param.is_scaleout;

        if (ccl_is_device_side_algo(selector_param)) {
            sched->strict_order = true;
        }

        if ((ccl::global_data::env().atl_transport == ccl_atl_mpi) &&
            ccl_is_direct_algo(selector_param)) {
            /* entry directly into schedule due to performance reasons */
            coll_entry::build_sched(sched, param);
        }
        else {
            entry_factory::create<coll_entry>(sched, param);
        }
    }
}

#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)

void add_wait_events(ccl_sched* sched, const std::vector<ze_event_handle_t>& wait_events) {
    if (wait_events.size() > 0) {
        entry_factory::create<ze_event_wait_entry>(sched, wait_events);
        sched->add_barrier();
    }
}

void add_signal_event(ccl_sched* sched, ze_event_handle_t signal_event) {
    if (signal_event) {
        entry_factory::create<ze_event_signal_entry>(sched, signal_event);
        sched->add_barrier();
    }
}

ze_event_handle_t add_signal_event(ccl_sched* sched) {
    auto signal_event = sched->get_memory().event_manager->create();
    add_signal_event(sched, signal_event);
    return signal_event;
}

// This version of add_comm_barrier without events should only be used in this file, as
// it does not deal with ze_events. While developing collectives, use the other version
// of add_comm_barrier instead.
void add_comm_barrier(ccl_sched* sched,
                      ccl_comm* comm,
                      ze_event_pool_handle_t ipc_pool,
                      size_t ipc_event_idx) {
    sched->add_barrier();
    if (ipc_pool && global_data::env().enable_ze_barrier) {
        entry_factory::create<ze_barrier_entry>(sched, comm, ipc_pool, ipc_event_idx);
    }
    else {
        ccl_coll_param barrier_param{};
        barrier_param.ctype = ccl_coll_barrier;
        barrier_param.comm = comm;

        /* TODO: optimize p2p based barrier */
        //barrier_param.hint_algo.barrier = ccl_coll_barrier_ring;

        add_coll_entry(sched, barrier_param);
    }
    sched->add_barrier();
}

void add_comm_barrier(ccl_sched* sched,
                      ccl_comm* comm,
                      const std::vector<ze_event_handle_t>& wait_events,
                      ze_event_handle_t& out_event,
                      ze_event_pool_handle_t ipc_pool,
                      size_t ipc_event_idx) {
    sched->add_barrier();
    out_event = sched->get_memory().event_manager->create();
    if (sched->use_single_list) {
        add_wait_events(sched, wait_events);
    }
    add_comm_barrier(sched, comm, ipc_pool, ipc_event_idx);
    add_signal_event(sched, out_event);
    sched->add_barrier();
}

void add_handle_exchange(ccl_sched* sched,
                         ccl_comm* comm,
                         const std::vector<ze_event_handle_t>& wait_events,
                         ze_event_handle_t& out_event,
                         const std::vector<ze_handle_exchange_entry::mem_desc_t>& in_buffers,
                         int skip_rank,
                         ze_event_pool_handle_t pool,
                         size_t event_idx,
                         const ccl::utils::pt2pt_handle_exchange_info& info) {
    if (!wait_events.empty()) {
        ccl::add_wait_events(sched, wait_events);
    }
    if (sched->coll_attr.to_cache) {
        sched->set_entry_exec_mode(ccl_sched_entry_exec_once);
        entry_factory::create<ze_handle_exchange_entry>(sched, comm, in_buffers, skip_rank, info);
        sched->add_barrier();
        sched->set_entry_exec_mode(ccl_sched_entry_exec_regular);

        if (sched->coll_param.ctype == ccl_coll_recv || sched->coll_param.ctype == ccl_coll_send) {
            // the entry emulates 'alignment' between send and recv op-s
            // the 'alignment' is needed to avoid situations when send or recv
            // can be faster executed (e.g. initialize the send buffer), not
            // waiting the previous pair of op-s to be finished. it must be as
            // entry because 'alignment' must be in the schedule with other
            // entries. such 'alignment' should be exectuted before other
            // entries that's why sched->add_barrier is used.
            entry_factory::create<ze_pt2pt_barrier_entry>(sched, comm, info.peer_rank);
            sched->add_barrier();
        }
        else {
            // TODO: no need barrier for the first iteration where ze_handle_exchange_entry exists
            add_comm_barrier(sched, comm, {}, out_event, pool, event_idx);
        }
    }
    else {
        entry_factory::create<ze_handle_exchange_entry>(sched, comm, in_buffers, skip_rank, info);
        sched->add_barrier();
        out_event = ccl::add_signal_event(sched);
    }
}

ze_event_handle_t add_coll(ccl_sched* sched,
                           const ccl_coll_param& param,
                           const std::vector<ze_event_handle_t>& wait_events) {
    if (sched->use_single_list) {
        ccl::add_wait_events(sched, wait_events);
    }
    if (ccl::global_data::env().ze_multi_workers && !param.is_hmem_enabled) {
        ccl_coll_attr attr{};
        ccl_coll_param coll_param;
        switch (param.ctype) {
            case ccl_coll_allreduce: {
                coll_param = ccl_coll_param::create_allreduce_param(param.send_buf.get_src(),
                                                                    param.recv_buf.get_src(),
                                                                    param.count,
                                                                    param.dtype.idx(),
                                                                    param.reduction,
                                                                    attr,
                                                                    param.comm,
                                                                    param.stream);
                break;
            }
            case ccl_coll_reduce_scatter: {
                coll_param = ccl_coll_param::create_reduce_scatter_param(param.send_buf.get_src(),
                                                                         param.recv_buf.get_src(),
                                                                         param.count,
                                                                         param.dtype.idx(),
                                                                         param.reduction,
                                                                         attr,
                                                                         param.comm,
                                                                         param.stream);
                break;
            }
            case ccl_coll_reduce: {
                coll_param = ccl_coll_param::create_reduce_param(param.send_buf.get_src(),
                                                                 param.recv_buf.get_src(),
                                                                 param.count,
                                                                 param.dtype.idx(),
                                                                 param.reduction,
                                                                 param.root,
                                                                 attr,
                                                                 param.comm,
                                                                 param.stream);
                break;
            }
            case ccl_coll_alltoallv: {
                coll_param = ccl_coll_param::create_alltoallv_param(param.send_buf.get_src(),
                                                                    param.send_counts.data(),
                                                                    param.recv_buf.get_src(),
                                                                    param.recv_counts.data(),
                                                                    param.dtype.idx(),
                                                                    attr,
                                                                    param.comm,
                                                                    param.stream);
                break;
            }
            case ccl_coll_allgatherv: {
                coll_param = ccl_coll_param::create_allgatherv_param(param.send_buf.get_src(),
                                                                     param.send_count,
                                                                     param.recv_buf.get_src(),
                                                                     param.recv_counts.data(),
                                                                     param.dtype.idx(),
                                                                     attr,
                                                                     param.comm,
                                                                     param.stream);
                break;
            }
            default: CCL_THROW("unexpected coll type", ccl_coll_type_to_str(param.ctype));
        }
        LOG_DEBUG("scaleout/multi_workers: created params for: ",
                  ccl_coll_type_to_str(param.ctype),
                  " coll");
        // pass the scale-out selection param through factory
        coll_param.is_scaleout = param.is_scaleout;
        ccl_sched_create_param sched_param(sched->sched_id, coll_param);
        entry_factory::create<subsched_entry>(sched, 0, sched_param, "SCALEOUT");
    }
    else {
        coll_entry::build_sched(sched, param);
    }
    sched->add_barrier();

    auto out_event = ccl::add_signal_event(sched);
    return out_event;
}

ze_event_handle_t add_copy_entry(ccl_buffer src,
                                 ccl_buffer dst,
                                 const size_t count,
                                 const ccl_datatype dtype,
                                 const copy_attr& copy_attr,
                                 ccl_sched* sched,
                                 const std::vector<ze_event_handle_t>& wait_events) {
    LOG_DEBUG("topo/scale_out/intra: use ze_copy_entry");
    auto entry =
        entry_factory::create<ze_copy_entry>(sched, src, dst, count, dtype, copy_attr, wait_events);
    return entry->entry_event;
}

ze_event_handle_t add_copy_entry_with_offset(std::vector<ccl_buffer> bufs,
                                             ccl_buffer buf,
                                             const std::vector<size_t> counts,
                                             ccl_comm* comm,
                                             const ccl_datatype dtype,
                                             const copy_attr& copy_attr,
                                             ccl_sched* sched,
                                             const std::vector<ze_event_handle_t>& wait_events,
                                             bool is_skip_own_rank = false) {
    const size_t counts_size = comm->size();
    CCL_THROW_IF_NOT(bufs.size() == counts_size,
                     "buffers number is different from the number of counts");
    size_t offset = 0;
    std::vector<ze_event_handle_t> out_events;
    // number of not skipped s/r_counts, helps calculate the offset
    for (size_t idx = 0; idx < counts_size; idx++) {
        if (counts[idx] == 0) {
            continue;
        }

        if (!is_skip_own_rank || idx != size_t(comm->rank())) {
            ccl_buffer src = bufs[idx];
            ccl_buffer dst = buf + offset;
            // reverse the function logic (src <=> dst)
            if (copy_attr.direction == copy_direction::h2d) {
                src = buf + offset;
                dst = bufs[idx];
            }
            auto out_event =
                add_copy_entry(src, dst, counts[idx], dtype, copy_attr, sched, wait_events);
            out_events.push_back(out_event);
        }
        offset += counts[idx] * dtype.size();
    }
    LOG_DEBUG("add_copy_entry_with_offset done");

    add_wait_events(sched, out_events);
    return add_signal_event(sched);
}

ccl_buffer alloc_host_buffer(ccl_sched* sched, size_t host_buf_size, void* device_ptr) {
    // host and device buffer are required to be aligned on the same alignment
    // when using compute kernels for H2D and D2H copy. Therefore, use kernel_mem_align as an anchor
    // to find the device pointer alignment and apply the setting to host buffer allocation.
    // Otherwise, bad performance or even performance bugs (MLSL-2628: unexpectedly high execution time) may appear.
    if (ccl::global_data::env().ze_h2d_copy_engine == h2d_copy_engine_mode::none &&
        ccl::global_data::env().ze_enable_ccs_fallback_for_copy) {
        uintptr_t offset_bytes = ccl::utils::get_aligned_offset_byte(
            device_ptr, host_buf_size, ccl::global_data::env().kernel_mem_align);
        if (offset_bytes && offset_bytes != host_buf_size) {
            // 256 minimum alignment space value was choosen as the best performing constant on ccl benchmark
            size_t alignment_space =
                std::max((ssize_t)(256), ccl::global_data::env().kernel_mem_align);
            ccl::alloc_param alloc_param(host_buf_size + alignment_space,
                                         ccl::buffer_type::regular,
                                         ccl::buffer_place::host);
            return sched->alloc_buffer(alloc_param) + alignment_space - offset_bytes;
        }
    }
    ccl::alloc_param alloc_param(host_buf_size, ccl::buffer_type::regular, ccl::buffer_place::host);
    return sched->alloc_buffer(alloc_param);
}

ze_event_handle_t fill_scaleout_coll_param(const ccl_coll_param& in_coll_param,
                                           ccl_coll_param& out_coll_param,
                                           ccl_sched* sched,
                                           const std::vector<ze_event_handle_t>& wait_events) {
    ze_event_handle_t out_event{};
    // general case
    size_t host_buf_size = 0;
    // alltoallv case
    size_t a2av_send_bytes = 0;
    size_t a2av_recv_bytes = 0;
    // for simplicity
    ccl_coll_type ctype = out_coll_param.ctype;
    size_t counts_size = out_coll_param.comm->size();
    size_t dtype_size = out_coll_param.dtype.size();

    // calculate counts
    if (ctype == ccl_coll_alltoallv) {
        size_t a2av_send_count = std::accumulate(out_coll_param.send_counts.begin(),
                                                 out_coll_param.send_counts.end(),
                                                 ccl::utils::initial_count_value);
        size_t a2av_recv_count = std::accumulate(out_coll_param.recv_counts.begin(),
                                                 out_coll_param.recv_counts.end(),
                                                 ccl::utils::initial_count_value);
        a2av_send_bytes = a2av_send_count * dtype_size;
        a2av_recv_bytes = a2av_recv_count * dtype_size;
        if (a2av_send_bytes == 0 || a2av_recv_count == 0) {
            return out_event;
        }
    }
    else if (ctype == ccl_coll_alltoall || ctype == ccl_coll_allgatherv ||
             ctype == ccl_coll_allgather) {
        // assume sum of send_counts and recv_counts are equal
        host_buf_size = std::accumulate(out_coll_param.recv_counts.begin(),
                                        out_coll_param.recv_counts.end(),
                                        ccl::utils::initial_count_value) *
                        dtype_size;
        LOG_DEBUG("alltoall(v)/allgather(v) scale_out host buf size: ", host_buf_size);
    }
    else if (ctype == ccl_coll_reduce_scatter) {
        host_buf_size = out_coll_param.count * counts_size * dtype_size;
    }
    else {
        host_buf_size = out_coll_param.count * dtype_size;
    }

    // allocate receive and send (in out-of-place case) buffers
    if (ctype == ccl_coll_alltoallv) {
        ccl::alloc_param a2av_send_alloc_param(
            a2av_send_bytes, ccl::buffer_type::regular, ccl::buffer_place::host);
        ccl::alloc_param a2av_recv_alloc_param(
            a2av_recv_bytes, ccl::buffer_type::regular, ccl::buffer_place::host);
        out_coll_param.send_buf = sched->alloc_buffer(a2av_send_alloc_param);
        out_coll_param.recv_buf = sched->alloc_buffer(a2av_recv_alloc_param);
    }
    else if (ctype == ccl_coll_reduce_scatter) {
        CCL_THROW_IF_NOT(host_buf_size != invalid_host_buf_size,
                         "unexpected the size of buffer in scaleout phase");
        out_coll_param.send_buf =
            alloc_host_buffer(sched, host_buf_size, in_coll_param.send_buf.get_ptr());
        out_coll_param.recv_buf = out_coll_param.send_buf +
                                  out_coll_param.comm->rank() * out_coll_param.count * dtype_size;
    }
    else {
        if (host_buf_size == 0) {
            return out_event;
        }

        out_coll_param.recv_buf =
            alloc_host_buffer(sched, host_buf_size, in_coll_param.send_buf.get_ptr());
        if (ctype == ccl_coll_allgatherv || ctype == ccl_coll_allgather) {
            size_t offset =
                std::accumulate(out_coll_param.recv_counts.begin(),
                                out_coll_param.recv_counts.begin() + out_coll_param.comm->rank(),
                                ccl::utils::initial_count_value) *
                dtype_size;
            out_coll_param.send_buf = out_coll_param.recv_buf + offset;
        }
        else {
            out_coll_param.send_buf = out_coll_param.recv_buf;
        }
    }

    // transform array of buffers in contiguous buffer with offsets
    if (ctype == ccl_coll_alltoallv) {
        auto send_out_event = add_copy_entry_with_offset(in_coll_param.send_scale_out_bufs,
                                                         out_coll_param.send_buf,
                                                         out_coll_param.send_counts,
                                                         out_coll_param.comm,
                                                         out_coll_param.dtype,
                                                         copy_attr(copy_direction::d2h),
                                                         sched,
                                                         wait_events);
        out_event = add_copy_entry_with_offset(in_coll_param.recv_scale_out_bufs,
                                               out_coll_param.recv_buf,
                                               out_coll_param.recv_counts,
                                               out_coll_param.comm,
                                               out_coll_param.dtype,
                                               copy_attr(copy_direction::d2h),
                                               sched,
                                               std::vector<ze_event_handle_t>{ send_out_event });
    }
    else if (ctype == ccl_coll_alltoall) {
        out_event = add_copy_entry_with_offset(in_coll_param.send_scale_out_bufs,
                                               out_coll_param.send_buf,
                                               out_coll_param.send_counts,
                                               out_coll_param.comm,
                                               out_coll_param.dtype,
                                               copy_attr(copy_direction::d2h),
                                               sched,
                                               wait_events);
    }
    else if (ctype == ccl_coll_allgather) {
        out_event = add_copy_entry(in_coll_param.send_buf,
                                   out_coll_param.send_buf,
                                   out_coll_param.count,
                                   out_coll_param.dtype,
                                   copy_attr(copy_direction::d2h),
                                   sched,
                                   wait_events);
    }
    else if (ctype == ccl_coll_allgatherv) {
        out_event = add_copy_entry(in_coll_param.send_buf,
                                   out_coll_param.send_buf,
                                   out_coll_param.send_count,
                                   out_coll_param.dtype,
                                   copy_attr(copy_direction::d2h),
                                   sched,
                                   wait_events);
    }
    else if (ctype == ccl_coll_reduce_scatter) {
        out_event = add_copy_entry(in_coll_param.send_buf,
                                   out_coll_param.send_buf,
                                   out_coll_param.count * counts_size,
                                   out_coll_param.dtype,
                                   copy_attr(copy_direction::d2h),
                                   sched,
                                   wait_events);
    }
    else {
        out_event =
            add_copy_entry(in_coll_param.send_buf,
                           out_coll_param.send_buf,
                           out_coll_param.count,
                           out_coll_param.dtype,
                           copy_attr(copy_direction::d2h
#if defined(CCL_ENABLE_ZE) && defined(CCL_ENABLE_SYCL)
                                     ,
                                     ccl::global_data::env().allreduce_pipe_chunk_count > 1
                                         ? ccl::ze::queue_group_type::main
                                         : ccl::ze::queue_group_type::unknown // force_queue_type
#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL
                                     ),
                           sched,
                           wait_events);
    }
    return out_event;
}

// Some algorithms for particular collectives may support H2D copy and internal operations overlapping
bool is_copy_overlap_enabled(ccl_sched* sched, const ccl_coll_param& coll_param, bool multi_node) {
    ccl_selector_param selector_param;
    selector_param.ctype = coll_param.ctype;
    selector_param.count = coll_param.count;
    if (coll_param.ctype == ccl_coll_allgatherv) {
        selector_param.count = coll_param.send_count;
    }
    selector_param.recv_counts =
        const_cast<size_t*>(reinterpret_cast<const size_t*>(coll_param.recv_counts.data()));
    selector_param.dtype = coll_param.dtype;
    selector_param.comm = coll_param.comm;
    selector_param.stream = coll_param.stream;
    selector_param.buf =
        (coll_param.send_buf) ? coll_param.send_buf.get_ptr() : coll_param.recv_buf.get_ptr();
    selector_param.is_vector_buf = sched->coll_attr.is_vector_buf;
#ifdef CCL_ENABLE_SYCL
    selector_param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    selector_param.hint_algo = coll_param.hint_algo;
    selector_param.peer_rank = coll_param.peer_rank;
    selector_param.is_scaleout = coll_param.is_scaleout;

    if (coll_param.ctype == ccl_coll_allgatherv) {
        auto algo =
            ccl::global_data::get().algorithm_selector->get<ccl_coll_allgatherv>(selector_param);
        return (algo == ccl_coll_allgatherv_ring) && multi_node &&
               !ccl::global_data::env().ze_multi_workers;
    }
    else if (coll_param.ctype == ccl_coll_allreduce) {
        auto algo =
            ccl::global_data::get().algorithm_selector->get<ccl_coll_allreduce>(selector_param);
        // ring allreduce calls ring allgatherv underneath
        return (algo == ccl_coll_allreduce_ring) && multi_node &&
               !ccl::global_data::env().ze_multi_workers;
    }
    return false;
}

bool use_hmem_with_algo(ccl_sched* sched, const ccl_coll_param& coll_param) {
    ccl_selector_param selector_param;
    selector_param.ctype = coll_param.ctype;
    selector_param.count = coll_param.send_count;
    selector_param.recv_counts =
        const_cast<size_t*>(reinterpret_cast<const size_t*>(coll_param.recv_counts.data()));
    selector_param.hint_algo = coll_param.hint_algo;
    selector_param.dtype = coll_param.dtype;
    selector_param.comm = coll_param.comm;
    selector_param.stream = coll_param.stream;
    selector_param.buf =
        (coll_param.send_buf) ? coll_param.send_buf.get_ptr() : coll_param.recv_buf.get_ptr();
    selector_param.is_vector_buf = sched->coll_attr.is_vector_buf;
#ifdef CCL_ENABLE_SYCL
    selector_param.is_sycl_buf = sched->coll_attr.is_sycl_buf;
#endif // CCL_ENABLE_SYCL
    selector_param.is_scaleout = coll_param.is_scaleout;

    if (ccl_is_direct_algo(selector_param))
        return false;

    return true;
}

void add_scaleout(ccl_sched* sched,
                  const ccl_coll_param& in_coll_param,
                  const bool is_single_node,
                  const std::vector<ze_event_handle_t>& in_wait_events,
                  ze_event_handle_t& out_event,
                  const copy_attr& h2d_copy_attr,
                  ccl_comm* global_comm,
                  ccl_buffer global_recv_buf,
                  int global_root) {
    std::vector<ze_event_handle_t> wait_events{ in_wait_events };
    ccl_coll_param coll_param(in_coll_param);
    out_event = nullptr;

    bool multi_node =
        (!is_single_node && (coll_param.count != 0 || coll_param.recv_counts.size() != 0));
    bool enable_hmem = (ccl::global_data::env().use_hmem && atl_base_comm::attr.out.enable_hmem);
    if (enable_hmem && in_coll_param.ctype == ccl_coll_allgatherv &&
        ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        enable_hmem = use_hmem_with_algo(sched, in_coll_param);
    }

    bool do_h2d_copy =
        ((coll_param.ctype == ccl_coll_allreduce || coll_param.ctype == ccl_coll_reduce_scatter ||
          coll_param.ctype == ccl_coll_alltoallv || coll_param.ctype == ccl_coll_alltoall ||
          coll_param.ctype == ccl_coll_allgatherv || coll_param.ctype == ccl_coll_allgather) &&
         multi_node && !enable_hmem) ||
        (coll_param.ctype == ccl_coll_reduce && coll_param.comm->rank() == coll_param.root);

    if (multi_node) {
        if (!enable_hmem) {
            LOG_DEBUG("topo/scale_out: use host_", ccl_coll_type_to_str(coll_param.ctype));

            // mostly initialize contiguous send/recv buffers from array of input buffers
            out_event = fill_scaleout_coll_param(in_coll_param, coll_param, sched, wait_events);
            if (out_event) {
                utils::clear_and_push_back(wait_events, out_event);
            }
            sched->add_barrier();

            LOG_DEBUG("topo/scale_out: ze_copy_entry of D2H for ",
                      ccl_coll_type_to_str(coll_param.ctype),
                      " done");
        }
        // pass the scale-out selection param directly
        coll_param.is_scaleout = true;
        coll_param.is_hmem_enabled = enable_hmem;
        // do inplace collective
        out_event = ccl::add_coll(sched, coll_param, wait_events);
        utils::clear_and_push_back(wait_events, out_event);
    }

    if ((is_copy_overlap_enabled(sched, coll_param, multi_node) && !enable_hmem) || !do_h2d_copy)
        return;

    ccl_buffer src_copy_buf = coll_param.recv_buf;
    ccl_buffer dst_copy_buf = in_coll_param.recv_buf;

    if (in_coll_param.ctype == ccl_coll_reduce) {
        if (!multi_node)
            src_copy_buf = in_coll_param.recv_buf;
        dst_copy_buf = (global_comm->rank() == global_root) ? global_recv_buf : ccl_buffer();
    }

    if (coll_param.ctype == ccl_coll_alltoallv || coll_param.ctype == ccl_coll_alltoall ||
        coll_param.ctype == ccl_coll_allgatherv || coll_param.ctype == ccl_coll_allgather) {
        out_event = add_copy_entry_with_offset(in_coll_param.recv_scale_out_bufs,
                                               coll_param.recv_buf,
                                               coll_param.recv_counts,
                                               coll_param.comm,
                                               coll_param.dtype,
                                               h2d_copy_attr,
                                               sched,
                                               wait_events,
                                               coll_param.ctype == ccl_coll_allgatherv);
    }
    else {
        out_event = add_copy_entry(src_copy_buf,
                                   dst_copy_buf,
                                   coll_param.count,
                                   coll_param.dtype,
                                   h2d_copy_attr,
                                   sched,
                                   wait_events);
    }
    sched->add_barrier();

    LOG_DEBUG("topo/scale_out: ze_copy_entry of H2D for ",
              ccl_coll_type_to_str(coll_param.ctype),
              " done");
}

bool is_queue_in_order(const ccl_stream* s) {
    return s != nullptr && s->is_sycl_device_stream() && s->get_native_stream().is_in_order();
}

#endif // CCL_ENABLE_ZE && CCL_ENABLE_SYCL

CCL_API bool is_allgatherv_inplace(const void* send_buf,
                                   const size_t send_count,
                                   const void* recv_buf,
                                   const size_t* recv_counts,
                                   const size_t dtype_size,
                                   const size_t rank,
                                   const size_t comm_size) {
    // From MPI 4.1: The "in place" option for intra-communicators is specified
    // by passing the value MPI_IN_PLACE to the argument sendbuf at all MPI
    // processes. sendcount and sendtype are ignored. Then the input data of
    // each MPI process is assumed to be in the area where that MPI process
    // would receive its own contribution to the receive buffer.

    // The next block of code calculates whether or not the MPI_IN_PLACE option
    // should be used (i.e. if the data is "in the area where that MPI process
    // would receive its own contribution to the receive buffer"). It also
    // warns the user if the send_buf is within the recv_buf but at an
    // unexpected location.

    // It is the responsibility of the user to use "in place" option correctly
    // in *all* processes that call atl_mpi::allgatherv, as specified in the
    // MPI standard.

    size_t total_recv_buf_count = 0;
    size_t curr_recv_buf_offset_count = 0;
    for (size_t i = 0; i < comm_size; ++i) {
        if (i == rank) {
            curr_recv_buf_offset_count = total_recv_buf_count;
        }
        total_recv_buf_count += recv_counts[i];
    }

    size_t recv_buf_size = total_recv_buf_count * dtype_size;
    const char* send_buf_ptr = reinterpret_cast<const char*>(send_buf);
    const char* recv_buf_start_ptr = reinterpret_cast<const char*>(recv_buf);
    const char* recv_buf_end_ptr = recv_buf_start_ptr + recv_buf_size;
    bool address_aliasing =
        (send_buf_ptr >= recv_buf_start_ptr) && (send_buf_ptr < recv_buf_end_ptr);

    size_t curr_recv_buf_offset = curr_recv_buf_offset_count * dtype_size;
    const char* curr_recv_buf = recv_buf_start_ptr + curr_recv_buf_offset;
    bool inplace = (send_buf_ptr == nullptr) || (send_buf_ptr == curr_recv_buf);

    CCL_THROW_IF_NOT(send_count == recv_counts[rank],
                     "send_count excpected to be equal to recv_counts[my_rank]");

    LOG_DEBUG("send_count: ",
              send_count,
              ", recv_buf_size: ",
              recv_buf_size,
              ", inplace: ",
              inplace,
              ", send_buf: ",
              send_buf,
              ", recv_buf: ",
              recv_buf);
    if (address_aliasing && !inplace) {
        std::stringstream ss_send, ss_recv;
        ss_send << send_buf;
        ss_recv << recv_buf;

        std::string warning_msg("using allgatherv with a send_buf that is not in-place"
                                " but that overlaps with recv_buf! Did you mean to place your"
                                " send_buf in the appropriate location within recv_buf?");
        LOG_DEBUG("CAUTION: ", warning_msg);
        LOG_DEBUG("   send_buf ", ss_send.str());
        LOG_DEBUG("   recv_buf ", ss_recv.str());
        if (ccl::global_data::env().check_inplace_aliasing) {
            CCL_THROW("ERROR: " + warning_msg + ".\n" +
                      "Set CCL_CHECK_INPLACE_ALIASING=0 to disable this check.");
        }
    }

    return inplace;
}

bool is_reduce_scatter_inplace(const void* send_buf,
                               const void* recv_buf,
                               const size_t recv_count,
                               const size_t dtype_size,
                               const size_t rank,
                               const size_t comm_size) {
    // Contrary to MPI, oneCCL defines "in-place" semantics differently.
    // For reduce-scatter, in-place operations are done when
    // the per-rank recv pointer is located at the rank offset
    // of the global send buffer.

    // It is the responsibility of the user to use "in place" option correctly
    // in *all* processes that call ccl::reduce_scatter,
    // as specified in the comment above.

    size_t recv_buf_size = recv_count * dtype_size;
    size_t send_buf_size = recv_buf_size * comm_size;

    const char* send_buf_start_ptr = reinterpret_cast<const char*>(send_buf);
    const char* send_buf_end_ptr = send_buf_start_ptr + send_buf_size;
    const char* recv_buf_ptr = reinterpret_cast<const char*>(recv_buf);

    bool address_aliasing =
        (recv_buf_ptr >= send_buf_start_ptr) && (recv_buf_ptr < send_buf_end_ptr);

    const char* curr_send_buf = send_buf_start_ptr + rank * recv_buf_size;
    bool inplace = recv_buf_ptr == curr_send_buf;

    LOG_DEBUG("recv_count: ",
              recv_count,
              ", inplace: ",
              inplace,
              ", send_buf: ",
              send_buf,
              ", recv_buf: ",
              recv_buf);
    if (address_aliasing && !inplace) {
        std::stringstream ss_send, ss_recv;
        ss_send << send_buf;
        ss_recv << recv_buf;

        std::string warning_msg("using reduce_scatter with a recv_buf that is not in-place"
                                " but that overlaps with send_buf! Did you mean to place your"
                                " recv_buf in the appropriate location within send_buf?");
        LOG_DEBUG("CAUTION: ", warning_msg);
        LOG_DEBUG("   send_buf ", ss_send.str());
        LOG_DEBUG("   recv_buf ", ss_recv.str());
        if (ccl::global_data::env().check_inplace_aliasing) {
            CCL_THROW("ERROR: " + warning_msg + ".\n" +
                      "Set CCL_CHECK_INPLACE_ALIASING=0 to disable this check.");
        }
    }

    return inplace;
}

// this is needed to enable the direct algo fallback for pt2pt when used inside the Group API
void enable_direct_fallback_for_pt2pt() {
    if (!ccl::global_data::env().fallback_recv) {
        ccl::global_data::env().fallback_recv =
            std::make_shared<ccl_selection_table_t<ccl_coll_recv_algo>>();
    }

    if (!ccl::global_data::env().fallback_send) {
        ccl::global_data::env().fallback_send =
            std::make_shared<ccl_selection_table_t<ccl_coll_send_algo>>();
    }

    ccl_algorithm_selector_base<ccl_coll_recv_algo>::insert(
        *(ccl::global_data::env().fallback_recv),
        0,
        CCL_SELECTION_MAX_COLL_SIZE,
        ccl_coll_recv_direct);
    ccl_algorithm_selector_base<ccl_coll_send_algo>::insert(
        *(ccl::global_data::env().fallback_send),
        0,
        CCL_SELECTION_MAX_COLL_SIZE,
        ccl_coll_send_direct);

    ccl::global_data::env().store_fallback_recv =
        std::make_shared<ccl_selection_table_t<ccl_coll_recv_algo>>(
            *(ccl::global_data::env().fallback_recv));

    ccl::global_data::env().store_fallback_send =
        std::make_shared<ccl_selection_table_t<ccl_coll_send_algo>>(
            *(ccl::global_data::env().fallback_send));
}

// this is needed to preserve general fallback table for pt2pt when used outside the Group API
void restore_pt2pt_fallback_table() {
    CCL_THROW_IF_NOT(ccl::global_data::env().fallback_recv,
                     "CCL_RECV global fallback table is null");
    CCL_THROW_IF_NOT(ccl::global_data::env().fallback_send,
                     "CCL_SEND global fallback table is null");
    CCL_THROW_IF_NOT(ccl::global_data::env().store_fallback_recv,
                     "CCL_RECV fallback table storage is null");
    CCL_THROW_IF_NOT(ccl::global_data::env().store_fallback_send,
                     "CCL_SEND fallback table storage is null");
    *(ccl::global_data::env().fallback_recv) = *(ccl::global_data::env().store_fallback_recv);
    *(ccl::global_data::env().fallback_send) = *(ccl::global_data::env().store_fallback_send);
}

} // namespace ccl

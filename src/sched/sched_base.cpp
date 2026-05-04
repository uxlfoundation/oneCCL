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
#include <numeric>

#include "atl/mpi/atl_mpi_ctx.hpp"
#include "coll/algorithms/algorithm_utils.hpp"
#include "coll/coll_param.hpp"
#include "coll/selection/selection.hpp"
#include "common/global/global.hpp"
#include "comm/comm.hpp"
#include "sched/buffer/buffer_manager.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/sched_base.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

ccl_sched_base::ccl_sched_base(const ccl_sched_create_param& param)
        : sched_type(param.type),
          sched_id(param.id),
          coll_param(param.coll_param) {
    memory.buffer_manager.init(sched_id);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (coll_param.stream &&
        coll_param.stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        memory.event_manager.reset(new ccl::ze::event_manager(coll_param.stream));

        auto node_comm = coll_param.comm->get_node_comm().get();
        memory.handle_manager.init(node_comm, coll_param.stream);

        memory.ipc_event_pool_manager.init(coll_param.stream);
        memory.list_manager.reset(new ccl::ze::list_manager(this, coll_param.stream));
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

std::string to_string(ccl_sched_add_mode mode) {
    auto mode_str = "UNDEFINED";
    switch (mode) {
        case ccl_sched_add_front: return "FRONT";
        case ccl_sched_add_back: return "BACK";
        default: mode_str = "DEFAULT";
    }
    return mode_str;
}

ccl_sched_base::~ccl_sched_base() {
    clear_memory();
}

void ccl_sched_base::set_coll_attr(const ccl_coll_attr& attr) {
    coll_attr = attr;
}

void ccl_sched_base::update_coll_param_and_attr(const ccl_coll_param& param,
                                                const ccl_coll_attr& attr) {
#ifdef CCL_ENABLE_SYCL
    // we already have barrier event in the list, just update coll param
    coll_param.copy_deps(param.deps);
#endif // CCL_ENABLE_SYCL

    bool has_pre_post_copies =
        (!coll_param.send_dev_bufs.empty() || !coll_param.recv_dev_bufs.empty()) ? true : false;

    if (has_pre_post_copies) {
        CCL_THROW_IF_NOT(coll_param.send_dev_bufs.size() == param.send_bufs.size(),
                         "send_bufs sizes mismatch");
        CCL_THROW_IF_NOT(coll_param.recv_dev_bufs.size() == param.recv_bufs.size(),
                         "recv_bufs sizes mismatch");
        coll_param.send_dev_bufs = param.send_bufs;
        coll_param.recv_dev_bufs = param.recv_bufs;
    }
    else {
        CCL_THROW_IF_NOT(coll_param.send_bufs.size() == param.send_bufs.size(),
                         "send_bufs sizes mismatch");
        CCL_THROW_IF_NOT(coll_param.recv_bufs.size() == param.recv_bufs.size(),
                         "recv_bufs sizes mismatch");
        coll_param.send_bufs = param.send_bufs;
        coll_param.recv_bufs = param.recv_bufs;
    }

    coll_param.stream = param.stream;

    int comm_size = coll_param.comm->size();

    if (coll_param.ctype == ccl_coll_allgatherv) {
        if (coll_attr.is_vector_buf)
            CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_bufs.size()) == comm_size);
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_counts.size()) == comm_size);
    }

    if (coll_param.ctype == ccl_coll_alltoallv) {
        if (coll_attr.is_vector_buf)
            CCL_THROW_IF_NOT(static_cast<int>(coll_param.send_bufs.size()) == comm_size);
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.send_counts.size()) == comm_size);

        if (coll_attr.is_vector_buf)
            CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_bufs.size()) == comm_size);
        CCL_THROW_IF_NOT(static_cast<int>(coll_param.recv_counts.size()) == comm_size);
    }

    if (ccl::global_data::env().priority_mode == ccl_priority_direct) {
        coll_attr.priority = attr.priority;
    }
}

size_t ccl_sched_base::get_priority() const {
    size_t priority = 0;

    switch (ccl::global_data::env().priority_mode) {
        case ccl_priority_none: priority = 0; break;
        case ccl_priority_direct:
        case ccl_priority_lifo: priority = coll_attr.priority; break;
        default:
            CCL_FATAL("unexpected priority_mode ", ccl::global_data::env().priority_mode);
            break;
    }

    LOG_DEBUG("sched, ", this, ", priority ", priority);

    return priority;
}

ccl_buffer ccl_sched_base::alloc_buffer(const ccl::alloc_param& user_param) {
    ccl::alloc_param param = user_param;

    if (!param.stream) {
        param.stream = coll_param.stream;
    }

#ifdef CCL_ENABLE_SYCL
    if ((param.buf_type == ccl::buffer_type::unknown) && param.stream && param.hint_ptr) {
        auto ptr_type =
            sycl::get_pointer_type(param.hint_ptr, param.stream->get_native_stream().get_context());
        if (ptr_type == sycl::usm::alloc::device) {
            param.buf_type = ccl::buffer_type::ze;
            param.buf_place = ccl::buffer_place::device;
        }
    }
#endif // CCL_ENABLE_SYCL

    if (param.buf_type == ccl::buffer_type::unknown) {
        param.buf_type = ccl::buffer_type::regular;
        param.buf_place = ccl::buffer_place::host;
    }

    if (group.get() != nullptr && param.buf_place == ccl::buffer_place::device) {
        size_t default_alignment = 8;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        default_alignment = ccl::global_data::env().kernel_mem_align;
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
        auto ptr = group->allocate(user_param.bytes, default_alignment);

        if (ptr != nullptr) {
            return ccl_buffer(ptr, param.bytes);
        }
    }

    return ccl_buffer(memory.buffer_manager.alloc(param), param.bytes);
}

void ccl_sched_base::dealloc_buffer(const ccl::dealloc_param& user_param) {
    ccl::dealloc_param param = user_param;

#ifdef CCL_ENABLE_SYCL
    if (!param.stream) {
        param.stream = coll_param.stream;
    }
#endif // CCL_ENABLE_SYCL

    if (group.get() != nullptr && param.buf_type == ccl::buffer_type::ze &&
        group->is_pointer_within_memory_context(param.ptr)) {
        return;
    }

    memory.buffer_manager.dealloc(param);
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
void ccl_sched_base::try_enable_ze_single_list() {
    CCL_THROW_IF_NOT(ze_entries.empty(),
                     "trying to modify the list mode after ze_entries has already been formed");
    use_single_list = ccl::global_data::env().enable_ze_single_list &&
                      !ccl::global_data::env().kernel_debug &&
                      !ccl::global_data::env().enable_fusion;
    LOG_DEBUG("ze_single_list set to: ", use_single_list);
}

void ccl_sched_base::append_to_ze_entries_list(sched_entry* entry) {
    if (memory.list_manager && memory.list_manager->is_executed()) {
        CCL_THROW("modifying ze_entries during list execution");
    }
    ze_entries.push_back(entry);
}

bool ccl_sched_base::check_pt2pt_pre_post_copy_support(const ccl_coll_param& param,
                                                       bool enable_pt2pt_offload) {
    if ((param.ctype == ccl_coll_send || param.ctype == ccl_coll_recv) &&
        (param.stream && param.stream->is_sycl_device_stream())) {
        bool enable_hmem =
            (ccl::global_data::env().use_hmem && atl_base_comm::attr.out.enable_hmem);
        LOG_DEBUG("value of hmem is: ", enable_hmem);

        if (enable_hmem) {
            LOG_DEBUG("hmem is enabled, no need for pre/post copy");
            return false;
        }
        else if (enable_pt2pt_offload) {
            LOG_DEBUG("offload algo is selected for send-recv, no need for pre/post copy");
            return false;
        }
    }
    return true;
}
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

void ccl_sched_base::sched_complete_hook() {
    /*
     * just reset without destroying
     */
    reset_memory_state();

#ifdef CCL_ENABLE_ZE
    for (auto& ze_entry : ze_entries) {
        /** During the start of the schedule, all entries contained in
         *  ze_entries list were initialized. If this exception is throw,
         *  it means that the entry was initialized but never started.
         *  This is unexpected behavior and can lead to issues and overhead.
         *  Need to make sure that the entry in ze_entries list is really needed
         *  and skip the entry initialization if not.
         */
        ze_base_entry* ptr = reinterpret_cast<ze_base_entry*>(ze_entry);
        CCL_THROW_IF_NOT(ptr->get_status() >= ccl_sched_entry_status_started,
                         "entry ",
                         ptr->name(),
                         " ",
                         ptr,
                         " was initialized but never started");
        CCL_THROW_IF_NOT(ptr->is_finalized || coll_attr.to_cache,
                         "entry ",
                         ptr->name(),
                         " ",
                         ptr,
                         " was not finalized");
    }
#endif // CCL_ENABLE_ZE
}

/* in this function we just reset state without destroy if to_cache=1,
   function is called on sched complete and never be called*/
void ccl_sched_base::reset_memory_state() {
#ifdef CCL_ENABLE_ZE
    if (!ze_entries.empty()) {
        get_memory().event_manager->reset();
        get_memory().list_manager->reset_execution_state();
    }
#endif // CCL_ENABLE_ZE
}

void ccl_sched_base::clear_memory() {
#ifdef CCL_ENABLE_ZE
    if (coll_param.stream &&
        coll_param.stream->get_backend() == ccl::utils::get_level_zero_backend()) {
        if (memory.event_manager) {
            memory.event_manager->clear();
        }

        /* TODO add a barrier; make sure the handles are not used before they are cleared
         * issue with the barrier: we are within ccl_sched_base destructor
         * and we can't freely construct and destroy new scheds here
         * we can't easily omit that since we don't have access to atl_comm here
         * sched->sched_bin seems to be nullptr in some cases and the barrier crashes */

        memory.handle_manager.clear();
        memory.ipc_event_pool_manager.clear();

        // Since list_manager is a shared_ptr, call clear only for the last
        //  reference (when use_count() is 1).
        // In all other cases, it is correct to simply skip calling clear since
        //  other schedules may have inherited the same list_manager entry, and,
        //  therefore, the same lists may still be in use.
        if (memory.list_manager.use_count() == 1) {
            memory.list_manager->clear();
        }
    }
#endif // CCL_ENABLE_ZE
    memory.buffer_manager.clear();
    free_memory_regions();
}

ccl_buffer ccl_sched_base::update_buffer(ccl_buffer buffer, size_t new_size) {
    CCL_THROW("unsupported");
    return ccl_buffer();
}

ccl_buffer ccl_sched_base::find_and_realloc_buffer(void* in_ptr,
                                                   size_t new_size,
                                                   size_t expected_size) {
    CCL_THROW("unsupported");
    return ccl_buffer();
}

void ccl_sched_base::add_memory_region(atl_mr_t* mr) {
    CCL_THROW_IF_NOT(mr);
    memory.mr_list.emplace_back(mr);
}

void ccl_sched_base::free_memory_regions() {
    if (memory.mr_list.empty()) {
        return;
    }

    /* TODO add a barrier; make sure the handles are not used before they are cleared
     * issue with the barrier: we are within ccl_sched_base destructor
     * and we can't freely construct and destroy new scheds here
     * we can't easily omit that since we don't have access to atl_comm here
     * sched->sched_bin seems to be nullptr in some cases and the barrier crashes */

    /* perform deregistration in worker thread */

    ccl_coll_param param{};
    param.ctype = ccl_coll_undefined;
    param.comm = coll_param.comm;
    ccl_sched* sched = nullptr;
    std::unique_ptr<ccl_sched> dereg_sched(
        new ccl_sched({ ccl_sched_regular, sched_id, param }, sched));
    entry_factory::create<deregister_entry>(dereg_sched.get(), memory.mr_list, param.comm);

    if (ccl::global_data::get().is_worker_thread || !ccl::global_data::env().worker_offload) {
        dereg_sched->do_progress();
    }
    else {
        CCL_THROW("unsupported path");
        /* release ownership, because ccl_wait_impl use delete inside */
        // ccl_wait_impl<ccl_extra_sched>(
        //     ccl::global_data::get().executor.get(),
        //     start_subsched(dereg_sched.release()));
    }

    if (!memory.mr_list.empty()) {
        LOG_ERROR("memory region list is not empty after deregister_entry completion");
    }
}

void ccl_sched_base::get_pre_post_copy_counts(std::vector<size_t>& d2h_counts,
                                              std::vector<size_t>& h2d_counts,
                                              bool& reuse_buffers) {
    ccl_coll_param& param = coll_param;

    d2h_counts.clear();
    h2d_counts.clear();
    reuse_buffers = false;

    switch (param.ctype) {
        case ccl_coll_allgather:
            d2h_counts.push_back(param.get_send_count());
            h2d_counts.push_back(param.get_recv_count() * param.comm->size());
            break;
        case ccl_coll_allgatherv:
            d2h_counts.push_back(param.get_send_count());
            if (param.recv_bufs.size() > 1) {
                h2d_counts.insert(
                    h2d_counts.end(), param.recv_counts.begin(), param.recv_counts.end());
            }
            else {
                h2d_counts.push_back(std::accumulate(param.recv_counts.begin(),
                                                     param.recv_counts.end(),
                                                     ccl::utils::initial_count_value));
            }
            break;
        case ccl_coll_allreduce:
            d2h_counts.push_back(param.get_send_count());
            h2d_counts.push_back(param.get_recv_count());
            /* use in-place to avoid allocation of extra staging buffer*/
            reuse_buffers = true;
            break;
        case ccl_coll_alltoall:
            d2h_counts.push_back(param.get_send_count() * param.comm->size());
            h2d_counts.push_back(param.get_recv_count() * param.comm->size());
            break;
        case ccl_coll_alltoallv:
            if (param.recv_bufs.size() > 1) {
                /* expect that is_vector_buf is enabled for send/recv both */
                d2h_counts.insert(
                    d2h_counts.end(), param.send_counts.begin(), param.send_counts.end());
                h2d_counts.insert(
                    h2d_counts.end(), param.recv_counts.begin(), param.recv_counts.end());
            }
            else {
                d2h_counts.push_back(std::accumulate(param.send_counts.begin(),
                                                     param.send_counts.end(),
                                                     ccl::utils::initial_count_value));
                h2d_counts.push_back(std::accumulate(param.recv_counts.begin(),
                                                     param.recv_counts.end(),
                                                     ccl::utils::initial_count_value));
            }
            break;
        case ccl_coll_bcast:
        case ccl_coll_broadcast:
            if (param.comm->rank() == param.root)
                d2h_counts.push_back(param.get_send_count());
            h2d_counts.push_back(param.get_recv_count());
            reuse_buffers = true;
            break;
        case ccl_coll_reduce:
            d2h_counts.push_back(param.get_send_count());
            if (param.comm->rank() == param.root)
                h2d_counts.push_back(param.get_recv_count());
            break;
        case ccl_coll_reduce_scatter:
            d2h_counts.push_back(param.get_send_count());
            h2d_counts.push_back(param.get_recv_count());
            break;
        case ccl_coll_send: d2h_counts.push_back(param.get_send_count()); break;
        case ccl_coll_recv:
            d2h_counts.push_back(param.get_recv_count());
            h2d_counts.push_back(param.get_recv_count());
            break;
        default: break;
    }
}

void ccl_sched_base::alloc_buffers_for_pre_post_copy() {
#ifdef CCL_ENABLE_SYCL
    if (coll_param.ctype == ccl_coll_last_value) {
        return;
    }

    ccl_coll_param& param = coll_param;

    param.send_dev_bufs.clear();
    param.recv_dev_bufs.clear();

    ccl_selector_param selector_param;
    selector_param.ctype = param.ctype;
    selector_param.count = param.get_send_count();
    selector_param.dtype = param.dtype;
    selector_param.comm = param.comm;
    selector_param.stream = param.stream;
    selector_param.is_sycl_buf = coll_attr.is_sycl_buf;
    selector_param.peer_rank = param.peer_rank;
    selector_param.recv_counts = param.recv_counts.data();

    if (!param.stream || !param.stream->is_sycl_device_stream() ||
        ccl_is_device_side_algo(selector_param)) {
        return;
    }

    if (!check_pt2pt_pre_post_copy_support(
            param, (ccl_is_offload_pt2pt_algo(selector_param) && use_pt2pt_offload_algo()))) {
        return;
    }

    bool should_alloc_buffers = true;

    if (!coll_attr.is_sycl_buf) {
        auto bufs = param.get_all_non_zero_bufs();
        if (!bufs.empty()) {
            auto usm_type =
                sycl::get_pointer_type(bufs[0], param.stream->get_native_stream().get_context());
            if ((usm_type == sycl::usm::alloc::host) || (usm_type == sycl::usm::alloc::shared) ||
                ((usm_type == sycl::usm::alloc::device) && ccl::global_data::env().use_hmem &&
                 atl_base_comm::attr.out.enable_hmem)) {
                should_alloc_buffers = false;
            }
        }
    }

    LOG_DEBUG("coll ", ccl_coll_type_to_str(param.ctype), ", alloc_buffers ", should_alloc_buffers);

    if (!should_alloc_buffers) {
        return;
    }

    /*
        move user-supplied pointers into device_* fields
        they will be used further for pre-post copies
    */
    param.send_dev_bufs = param.send_bufs;
    param.recv_dev_bufs = param.recv_bufs;

    std::vector<size_t> d2h_counts;
    std::vector<size_t> h2d_counts;
    bool reuse_buffers;
    get_pre_post_copy_counts(d2h_counts, h2d_counts, reuse_buffers);

    LOG_DEBUG("alloc tmp buffers for D2H and H2D copies, coll ",
              ccl_coll_type_to_str(param.ctype),
              ", dtype_size ",
              param.dtype.size(),
              ", comm_size ",
              param.comm->size(),
              ", d2h_counts_size ",
              d2h_counts.size(),
              ", h2d_counts_size ",
              h2d_counts.size(),
              ", reuse_buffers ",
              reuse_buffers);

    if (reuse_buffers) {
        /* keep only single vector with counts */
        if (d2h_counts.size() < h2d_counts.size())
            d2h_counts = h2d_counts;
        h2d_counts.clear();
    }

    ccl::buffer_type buf_type = ccl::buffer_type::regular;
    if (ccl::global_data::env().staging_buffer == ccl_staging_usm) {
        buf_type = ccl::buffer_type::sycl;
    }
    ccl::alloc_param alloc_param(0, buf_type, ccl::buffer_place::host, true, param.stream);

    for (size_t idx = 0; idx < d2h_counts.size(); idx++) {
        if (d2h_counts[idx]) {
            alloc_param.bytes = d2h_counts[idx] * param.dtype.size();
            param.send_bufs[idx] = alloc_buffer(alloc_param).get_ptr();
        }
        else
            param.send_bufs[idx] = nullptr;
    }

    for (size_t idx = 0; idx < h2d_counts.size(); idx++) {
        if (h2d_counts[idx]) {
            alloc_param.bytes = h2d_counts[idx] * param.dtype.size();
            param.recv_bufs[idx] = alloc_buffer(alloc_param).get_ptr();
        }
        else
            param.recv_bufs[idx] = nullptr;
    }

    if (reuse_buffers) {
        param.recv_bufs = param.send_bufs;
    }

    CCL_THROW_IF_NOT(param.send_bufs.size() == param.send_dev_bufs.size(),
                     "send_bufs.size() mismatch: ",
                     param.send_bufs.size(),
                     " vs ",
                     param.send_dev_bufs.size());

    CCL_THROW_IF_NOT(param.recv_bufs.size() == param.recv_dev_bufs.size(),
                     "recv_bufs.size() mismatch: ",
                     param.recv_bufs.size(),
                     " vs ",
                     param.recv_dev_bufs.size());

#endif // CCL_ENABLE_SYCL
}

void ccl_sched_base::update_id() {
    sched_id = coll_param.comm->get_sched_id(sched_type != ccl_sched_regular, coll_param.is_pt2pt);
}

void ccl_sched_base::dump(std::ostream& out, const char* name) const {
    ccl_logger::format(out, "\n-----------------", name, "---------------\n");
    ccl_logger::format(out,
                       "sched: ",
                       this,
                       ", coll ",
                       ccl_coll_type_to_str(coll_param.ctype),
                       ", comm_id ",
                       std::dec,
                       coll_param.comm->id(),
                       ", sched_id ",
                       sched_id);
}

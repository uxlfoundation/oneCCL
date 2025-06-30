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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/broadcast/sycl/broadcast_sycl.hpp"

namespace ccl {
namespace v1 {

ccl::event broadcast_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      int root,
                                      ccl_comm* comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    ccl::event e;
    done = true;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    uint32_t world = comm->get_node_comm()->size();

    if (world == 1) {
        sycl::event sycl_e;
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: bcast");
            sycl_e = q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: bcast");
            sycl_e = submit_wait_on_events(q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    if (count * ccl_dtype.size() <= ccl::global_data::env().sycl_broadcast_small_threshold) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("broadcast_small", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG(
            "|CCL_SYCL| broadcast selects small kernel, count: ", count, " datatype: ", dtype);
        e = broadcast_small(send_buf, recv_buf, count, dtype, root, comm, global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| broadcast selects small kernel, count: ",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("broadcast_large", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG(
            "|CCL_SYCL| broadcast selects large kernel, count: ", count, " datatype: ", dtype);
        e = broadcast_large(send_buf, recv_buf, count, dtype, root, comm, global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| broadcast selects large kernel, count: ",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    return e;
}

static bool do_fallback_to_scheduler(size_t size) {
    bool is_above_threshold = size > ccl::global_data::env().sycl_broadcast_scaleout_threshold;
    bool exception_cases = ccl::global_data::env().atl_transport == ccl_atl_ofi;
    return is_above_threshold || exception_cases;
}

ccl::event broadcast_sycl_multi_node(sycl::queue& q,
                                     const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     int root,
                                     ccl_comm* global_comm,
                                     ccl_stream* global_stream,
                                     const vector_class<event>& deps,
                                     bool& done) {
    done = true;
    ccl::event ev;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    ccl_comm* node_comm = global_comm->get_node_comm().get();
    ccl_comm* r2r_comm = global_comm->get_r2r_comm().get();

    if (do_fallback_to_scheduler(count * ccl_dtype.size())) {
        LOG_DEBUG("broadcast count size = ",
                  count * ccl_dtype.size(),
                  " is above scaleout SYCL threshold = ",
                  ccl::global_data::env().sycl_broadcast_scaleout_threshold,
                  " or sycl::esimd mode is enabled, or other conditions not met -- falling back");
        done = false;
        return ev;
    }

    LOG_DEBUG("broadcast_sycl multi-node count: ", count);

    // initial dependencies for the loop
    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    std::vector<ccl::event> evs;
    for (int i = 0; i < dep_events.size(); i++) {
        ev = ccl::event::create_from_native(dep_events[i]);
        evs.push_back(std::move(ev));
    }

    // copy to recv_buf or not
    const bool inplace = send_buf == recv_buf;
    // what node from global ranks contains the root
    const int root_node_idx = root / node_comm->size();
    // project global root rank to node_comm rank idx
    const int root_rank_in_node_comm = root - root_node_idx * node_comm->size();

    // chunking buffer size computation
    // host buffer is used in direct algorithm
    const int buf_size = global_comm->get_scaleout_host_buf_size();
    size_t max_iter_count;
    if (count * ccl_dtype.size() <= buf_size) {
        max_iter_count = count;
    }
    else {
        max_iter_count = buf_size / ccl_dtype.size();
    }

    size_t displ = 0, displ_count = 0;
    int nchunks = (count + max_iter_count - 1) / max_iter_count;
    for (int i = 0; i < nchunks; i++) {
        size_t iter_count = i < nchunks - 1 ? max_iter_count : count - displ_count;

        auto send_buf_ptr = ptr_offset(send_buf, displ);
        auto recv_buf_ptr = ptr_offset(recv_buf, displ);

        LOG_DEBUG("broadcast_sycl count: ",
                  count,
                  " root: ",
                  root,
                  " and #chunk: ",
                  i,
                  " of nchunks: ",
                  nchunks,
                  " - broadcast_scaleout_sycl");

        // scale-out phase only bettween projected (global rank => node rank) root ranks
        if (node_comm->rank() == root_rank_in_node_comm) {
            bool original_deps = (i == 0 && !evs.empty());
            ev = broadcast_scaleout_sycl(q,
                                         send_buf_ptr,
                                         recv_buf_ptr,
                                         iter_count,
                                         dtype,
                                         root_node_idx,
                                         r2r_comm,
                                         evs,
                                         original_deps,
                                         done);

            if (!done) {
                LOG_INFO("broadcast_sycl scale-out phase was not done -- falling back");
                // fallback
                return ev;
            }
        }

        // scale-up phase, broadcast from new root
        if (node_comm->size() > 1) {
            // create a dependency with scale-out phase only for the ranks
            // that are actually participated in scale-out phase
            if (node_comm->rank() == root_rank_in_node_comm) {
                evs.clear();
                evs.push_back(std::move(ev));
            }
            // scale-up phase will be inplace
            send_buf_ptr = recv_buf_ptr;

            ev = broadcast_sycl_single_node(q,
                                            send_buf_ptr,
                                            recv_buf_ptr,
                                            iter_count,
                                            dtype,
                                            root_rank_in_node_comm,
                                            node_comm,
                                            global_stream,
                                            evs,
                                            done);

            if (!done) {
                LOG_INFO("broadcast_sycl scale-up phase was not done -- falling back");
                // fallback
                return ev;
            }
        }

        if (i < nchunks - 1) {
            evs.clear();
            evs.push_back(std::move(ev));
            displ_count += iter_count;
            displ += iter_count * ccl_dtype.size();
        }
    }
    return ev;
}

ccl::event broadcast_sycl(sycl::queue& q,
                          const void* send_buf,
                          void* recv_buf,
                          size_t count,
                          ccl::datatype dtype,
                          int root,
                          ccl_comm* global_comm,
                          ccl_stream* global_stream,
                          const broadcast_attr& attr,
                          const vector_class<ccl::event>& deps,
                          bool& done) {
    if (count == 0) {
        done = true;
        auto sycl_deps = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_deps);
        return ccl::event::create_from_native(e);
    }

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node && ccl::global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("is_single_node");
        return broadcast_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, root, global_comm, global_stream, deps, done);
    }

    return broadcast_sycl_multi_node(
        q, send_buf, recv_buf, count, dtype, root, global_comm, global_stream, deps, done);
}

} // namespace v1
} // namespace ccl

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
#include "coll/algorithms/alltoall/sycl/alltoall_sycl.hpp"
#include "coll/algorithms/alltoall/sycl/alltoall_ll256.hpp"

namespace ccl {
namespace v1 {

ccl::event alltoall_sycl_single_node(sycl::queue& q,
                                     const void* send_buf,
                                     void* recv_buf,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl_comm* comm,
                                     ccl_stream* global_stream,
                                     const vector_class<event>& deps,
                                     bool& done) {
    ccl::event e;
    done = true;

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    //
    //     const bool is_single_tile = comm->get_pair_comm()->size() == 1;
    const bool has_all_vertices_connected = comm->get_topo_manager().has_all_vertices_connected();
    LOG_DEBUG("|CCL_SYCL| has_all_vertices_connected", has_all_vertices_connected);
    //
    uint32_t world = comm->get_node_comm()->size();
    int rank = comm->get_node_comm()->rank();

    if (world == 1) {
        sycl::event sycl_e;
        auto sycl_q = global_stream->get_native_stream();
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        if (send_buf != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: alltoall");
            sycl_e = sycl_q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, send_buf, count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: alltoall");
            sycl_e = submit_wait_on_events(sycl_q, dep_events);
        }
        return ccl::event::create_from_native(sycl_e);
    }

    if (is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device())) &&
        ccl::global_data::env().sycl_enable_arc_alltoall_ll) {
        ccl::event e;
        size_t dt_sz = ccl_dtype.size();
        if (((count * dt_sz) % LS_SZ == 0) && ((world & (world - 1)) == 0)) {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("arc_alltoall", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG(
                "|CCL_SYCL| alltoall selects arc_alltoall, count: ", count, " datatype: ", dtype);
            e = arc_alltoall(send_buf, recv_buf, count, dtype, comm, global_stream);
            LOG_DEBUG("|CCL_SYCL| alltoall selects arc_alltoall, count: ",
                      count,
                      " datatype: ",
                      dtype,
                      " done");
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            return e;
        }
    }

    if (ccl::global_data::env().sycl_esimd) {
        LOG_WARN(
            "|CCL_SYCL| sycl ESIMD requested for alltoall collective; ESIMD not supported, falling back to alltoall sycl implementation");
    }

    if (send_buf != recv_buf) {
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_begin("alltoall_large", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
        LOG_DEBUG("|CCL_SYCL| alltoall selects large kernel, count: ", count, " datatype: ", dtype);
        e = alltoall_large(send_buf, recv_buf, count, dtype, comm, global_stream, deps);
        LOG_DEBUG("|CCL_SYCL| alltoall selects large kernel, count: ",
                  count,
                  " datatype: ",
                  dtype,
                  " done");
#ifdef CCL_ENABLE_ITT
        ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    }
    else {
        LOG_WARN(
            "|CCL_SYCL| sycl inplace requested for alltoall collective; inplace not supported, falling back");
        done = false;
    }

    return e;
}

ccl::event alltoall_sycl(sycl::queue q,
                         const void* send_buf,
                         void* recv_buf,
                         size_t count,
                         ccl::datatype dtype,
                         ccl_comm* comm,
                         ccl_stream* op_stream,
                         const alltoall_attr& attr,
                         const vector_class<event>& deps,
                         bool& done) {
    if (count == 0) {
        done = true;
        auto sycl_events = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_events);
        return ccl::event::create_from_native(e);
    }

    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node && ccl::global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("is_single_node");
        return alltoall_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, comm, op_stream, deps, done);
    }

    // multi-node scenario not supported, fallback
    done = false;
    return ccl::event();
}

} // namespace v1
} // namespace ccl

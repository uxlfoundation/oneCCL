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

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/scatter/sycl/scatter_sycl.hpp"

namespace ccl {
namespace v1 {

event scatter_sycl_single_node(sycl::queue& q,
                               const void* send_buf,
                               void* recv_buf,
                               size_t count,
                               datatype dtype,
                               int root,
                               ccl_comm* comm,
                               ccl_stream* global_stream,
                               const vector_class<event>& deps,
                               bool& done) {
    event e;

    auto ccl_dtype = global_data::get().dtypes->get(dtype);
    uint32_t world = comm->get_node_comm()->size();

    if (world == 1) {
        sycl::event sycl_e;
        auto sycl_q = global_stream->get_native_stream();
        std::vector<sycl::event> dep_events = get_sycl_events(deps);
        /* single rank: root copies own block to recv_buf; in-place is a no-op */
        size_t own_offset = static_cast<size_t>(root) * count * ccl_dtype.size();
        const void* own_block = static_cast<const char*>(send_buf) + own_offset;
        if (own_block != recv_buf) {
            LOG_DEBUG("single rank: out-of-place case, coll: scatter");
            sycl_e = sycl_q.submit([=](sycl::handler& h) {
                h.depends_on(dep_events);
                h.memcpy(recv_buf, own_block, count * ccl_dtype.size());
            });
        }
        else {
            LOG_DEBUG("single rank: inplace case, coll: scatter");
            sycl_e = submit_wait_on_events(sycl_q, dep_events);
        }
        done = true;
        return event::create_from_native(sycl_e);
    }

    // TODO (Option B): implement small/large GPU kernel selection.
    // Scatter is the inverse of broadcast: root distributes one block
    // per rank from its send_buf. Reference: broadcast_small/large_sycl.
    //
    //   if (count * ccl_dtype.size() <= global_data::env().sycl_scatter_small_threshold) {
    //       e = scatter_small(send_buf, recv_buf, count, dtype, root, comm, global_stream, deps);
    //   } else {
    //       e = scatter_large(send_buf, recv_buf, count, dtype, root, comm, global_stream, deps);
    //   }

    done = false;
    return e;
}

event scatter_sycl_multi_node(sycl::queue& q,
                              const void* send_buf,
                              void* recv_buf,
                              size_t count,
                              datatype dtype,
                              int root,
                              ccl_comm* global_comm,
                              ccl_stream* global_stream,
                              const vector_class<event>& deps,
                              bool& done) {
    // Multi-node SYCL scatter is out of scope for this PR; always fall back
    // to the scheduler. A future scaleout PR will replace this stub.
    event ev;
    done = false;
    return ev;
}

event scatter_sycl(sycl::queue q,
                   const void* send_buf,
                   void* recv_buf,
                   size_t count,
                   datatype dtype,
                   int root,
                   ccl_comm* global_comm,
                   ccl_stream* global_stream,
                   const scatter_attr& attr,
                   const vector_class<event>& deps,
                   bool& done) {
    if (count == 0) {
        done = true;
        auto sycl_deps = get_sycl_events(deps);
        auto e = submit_wait_on_events(q, sycl_deps);
        return event::create_from_native(e);
    }

    bool is_single_node = false;
    if (global_data::env().backend == backend_mode::native) {
        const topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node && global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("is_single_node");
        return scatter_sycl_single_node(
            q, send_buf, recv_buf, count, dtype, root, global_comm, global_stream, deps, done);
    }

    return scatter_sycl_multi_node(
        q, send_buf, recv_buf, count, dtype, root, global_comm, global_stream, deps, done);
}

} // namespace v1
} // namespace ccl

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
#include "coll/algorithms/broadcast/sycl/broadcast_sycl.hpp"

namespace ccl {
namespace v1 {

ccl::event barrier_sycl_single_node(sycl::queue& q,
                                    ccl_comm* comm,
                                    ccl_stream* global_stream,
                                    const vector_class<event>& deps,
                                    bool& done) {
    done = true;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
    const bool is_cpu_barrier = ccl::global_data::env().sycl_ccl_barrier;

    std::shared_ptr<ccl_comm> node_comm = comm->get_node_comm();
    sycl::event e = invoke_barrier(node_comm, q, dep_events, is_cpu_barrier);

    return ccl::event::create_from_native(e);
}

ccl::event barrier_sycl(sycl::queue& q,
                        ccl_comm* global_comm,
                        ccl_stream* global_stream,
                        const barrier_attr& attr,
                        const vector_class<ccl::event>& deps,
                        bool& done) {
    bool is_single_node = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = global_comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
    }

    if (is_single_node && ccl::global_data::env().sycl_single_node_algorithm) {
        LOG_DEBUG("is_single_node");
        return barrier_sycl_single_node(q, global_comm, global_stream, deps, done);
    }

    done = false;
    ccl::event e;
    return e;
}

} // namespace v1
} // namespace ccl

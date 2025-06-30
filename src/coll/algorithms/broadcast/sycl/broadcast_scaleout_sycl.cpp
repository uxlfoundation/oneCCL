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

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/broadcast/sycl/broadcast_sycl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

ccl::event broadcast_scaleout_sycl_direct(sycl::queue& q,
                                          const void* send_buf,
                                          void* recv_buf,
                                          size_t count,
                                          ccl::datatype dtype,
                                          int root,
                                          ccl_comm* comm,
                                          const ccl::vector_class<ccl::event>& deps,
                                          bool& done,
                                          bool copy_to_host,
                                          bool is_cpu_buffers) {
    sycl::event op_end;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const bool inplace = send_buf == recv_buf;

    void* scaleout_host_buf = const_cast<void*>(send_buf);

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    if (copy_to_host) {
        if (comm->get_scaleout_host_buf_size() < count * ccl_dtype.size()) {
            LOG_WARN("scaleout_host_buf_size is not big enough to handle ",
                     count * ccl_dtype.size(),
                     " bytes. Falling back.");
            done = false;
            ccl::event e;
            return e;
        }

        scaleout_host_buf = comm->get_scaleout_host_buf();
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(scaleout_host_buf, send_buf, count * ccl_dtype.size());
        });
    }
    else if (!is_cpu_buffers) {
        // TODO: check if I_MPI_OFFLOAD is set, then let the scaleout allreduce go through.
        LOG_WARN("copy_to_host=false with a GPU buffer. "
                 "TODO: make sure I_MPI_OFFLOAD is set or GPU RDMA is enabled");
        // TODO: determine whether we want to fallback or not. For now, no.
        // done = false;
        // ccl::event e;
        // return e;
    }

    op_end = q.submit([=](sycl::handler& h) {
        h.depends_on(op_end);
        h.host_task([=]() {
            // call ccl::wrapper for MPI/OFI.
            int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
            atl_req_t req;
            std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
            ATL_CALL_THROW_IF_ERROR(atl_comm->broadcast(
                ep_idx, scaleout_host_buf, scaleout_host_buf, count * ccl_dtype.size(), root, req));

            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
            if (!req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
            }
        });
    });

    if (copy_to_host && (comm->rank() != root || !inplace)) {
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(op_end);
            h.memcpy(recv_buf, scaleout_host_buf, count * ccl_dtype.size());
        });
    }

    done = true;
    return ccl::event::create_from_native(op_end);
}

ccl::event broadcast_scaleout_sycl(sycl::queue& q,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   int root,
                                   ccl_comm* comm,
                                   const ccl::vector_class<ccl::event>& deps,
                                   bool original_deps,
                                   bool& done,
                                   bool is_cpu_buffers) {
    bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
    return broadcast_scaleout_sycl_direct(
        q, send_buf, recv_buf, count, dtype, root, comm, deps, done, copy_to_host, is_cpu_buffers);
}

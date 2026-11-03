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

#pragma once

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/host_task.hpp"

#include "proxy/proxy.hpp"
#include "proxy/proxy_func.hpp"
#include "proxy/proxy_staging.hpp"
#include "proxy/proxy_types.hpp"
#include "proxy/proxy_utils.hpp"

#include "proxy/host_initiated/cmd_persistent_kernel.hpp"

#include "comm/comm.hpp"
#include "sched/entry/ze/ze_primitives.hpp"

// Helper function: Execute allreduce operation on host (common code for host_task and proxy)
inline void allreduce_direct_impl(ccl_comm* comm,
                                  const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  ccl::reduction reduction) {
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
    atl_req_t req;
    ATL_CALL_THROW_IF_ERROR(atl_comm->allreduce(ep_idx,
                                                send_buf,
                                                recv_buf,
                                                count,
                                                ccl_dtype.atl_datatype(),
                                                static_cast<atl_reduction_t>(reduction),
                                                req));

    ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
    if (!req.is_completed) {
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
    }
}

/* ======== Proxy service path ======== */

// Proxy service wrapper for direct allreduce
// Note: direct allreduce doesn't use producer/staging, but signature must match
inline void proxy_allreduce_direct(const ccl_proxy_op& op, ccl_proxy_context& ctx) {
    // Unpack operation descriptor and call the algorithm
    allreduce_direct_impl(op.comm, op.src, op.dst, op.count, op.dtype, op.reduction);
}

inline sycl::event allreduce_scaleout_sycl_simple(sycl::queue& q,
                                                  const void* send_buf,
                                                  void* recv_buf,
                                                  size_t count,
                                                  ccl::datatype dtype,
                                                  ccl::reduction reduction,
                                                  ccl_comm* comm,
                                                  const ccl::vector_class<ccl::event>& deps,
                                                  bool& done,
                                                  bool copy_to_host,
                                                  bool is_cpu_buffers,
                                                  bool use_proxy = false) {
    sycl::event op_end, copy_e;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    
    LOG_DEBUG("allreduce_scaleout_sycl_simple called on comm size: ", comm->size(),
		    " count: ", count, " copy_to_host ", copy_to_host,
		    " send_buf: ", std::hex, send_buf, " recv_buf: ", recv_buf,
		    " use_proxy: ", use_proxy);

    const void* scaleout_send_buf = send_buf;
    void* scaleout_recv_buf = recv_buf;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    if (copy_to_host) {
        if (comm->get_scaleout_host_buf_size() < count * ccl_dtype.size()) {
            LOG_WARN("scaleout_host_buf_size is not big enough to handle ",
                     count * ccl_dtype.size(),
                     " bytes. Falling back. TODO: chunking/pipelining");
            done = false;
            sycl::event e;
            return e;
        }

        scaleout_send_buf = ATL_IN_PLACE;
        scaleout_recv_buf = comm->get_scaleout_host_buf();
        copy_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(scaleout_recv_buf,
                     send_buf == ATL_IN_PLACE ? recv_buf : send_buf,
                     count * ccl_dtype.size());
        });
        dep_events.clear();
        dep_events.push_back(std::move(copy_e));
    }
    else if (!is_cpu_buffers) {
        if (!check_mpi_supports_rdma()) {
            LOG_WARN("copy_to_host=false with a GPU buffer. "
                     "TODO: make sure I_MPI_OFFLOAD is set or GPU RDMA is enabled");
        }
        // TODO: check if I_MPI_OFFLOAD is set, then let the scaleout allreduce go through.
        // TODO: determine whether we want to fallback or not. For now, no.
        // done = false;
        // ccl::event e;
        // return e;
    }

    if (use_proxy) {
        // Use proxy service path with persistent kernel
        ccl_proxy_op op{ ccl_proxy_op_type::allreduce_direct,
                        (void*)scaleout_send_buf,
                        scaleout_recv_buf,
                        count,
                        dtype,
                        reduction,
                        comm };
        // No device compute: the reduction runs on the host via MPI. A single
        // work-group sync-only kernel is enough — it just publishes start and
        // exits via the ring once the proxy releases it.
        op.sync_only = true;

        op_end = submit_and_launch_persistent_kernel(q, op, dep_events);
    }
    else {
        // Use host_task path (direct execution)
        op_end = ccl_host_task(q, dep_events, [=]() {
            allreduce_direct_impl(comm, scaleout_send_buf, scaleout_recv_buf, count, dtype, reduction);
        });
    }

    if (copy_to_host) {
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(op_end);
            h.memcpy(recv_buf, scaleout_recv_buf, count * ccl_dtype.size());
        });
    }

    done = true;
    return op_end;
}

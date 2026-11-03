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

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/reduce/sycl/reduce_sycl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

ccl::event reduce_scaleout_sycl_direct(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        int root,
                                        ccl_comm* comm,
                                        const ccl::vector_class<ccl::event>& deps,
                                        bool& done,
                                        bool copy_to_host,
                                        bool is_cpu_buffers) {
    sycl::event op_end;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const bool inplace = send_buf == recv_buf;

    const void* scaleout_send_buf = send_buf;
    void* scaleout_recv_buf = recv_buf;

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

        scaleout_send_buf = ATL_IN_PLACE;
        scaleout_recv_buf = comm->get_scaleout_host_buf();
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(scaleout_recv_buf,
                     send_buf == ATL_IN_PLACE ? recv_buf : send_buf,
                     count * ccl_dtype.size());
        });
        dep_events.clear();
        dep_events.push_back(std::move(op_end));
    }
    else if (!is_cpu_buffers) {
        // check if I_MPI_OFFLOAD is set, then let the scaleout reduce go through.
        if (!check_mpi_supports_rdma()) {
            LOG_WARN("copy_to_host=false with a GPU buffer. "
                    "Ensure I_MPI_OFFLOAD is set or GPU RDMA is enabled.");

            // fallback.
            done = false;
            ccl::event e;
            return e;
        }
    }

    op_end = q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);
        h.host_task([=]() {
            // call ccl::wrapper for MPI/OFI.
            // use endpoint 0; multi-endpoint selection (atl_ep->idx / sched->bin->get_atl_ep())
            // is not wired through this path yet
            int ep_idx = 0;
            atl_req_t req;
            std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
            ATL_CALL_THROW_IF_ERROR(
                atl_comm->reduce(ep_idx,
                                scaleout_send_buf,
                                scaleout_recv_buf,
                                count,
                                root,
                                ccl_dtype.atl_datatype(),
                                static_cast<atl_reduction_t>(reduction),
                                req));

            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
            if (!req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
            }
        });
    });

    if (copy_to_host && comm->rank() == root) {
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(op_end);
            h.memcpy(recv_buf, scaleout_recv_buf, count * ccl_dtype.size());
        });
    }

    done = true;
    return ccl::event::create_from_native(op_end);
}

ccl::event reduce_scaleout_sycl(sycl::queue& q,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t count,
                                 ccl::datatype dtype,
                                 ccl::reduction reduction,
                                 int root,
                                 ccl_comm* comm,
                                 const ccl::vector_class<ccl::event>& deps,
                                 bool original_deps,
                                 bool& done,
                                 bool is_cpu_buffers) {
    CCL_UNUSED(original_deps);
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (should_disable_rdma(ze_dev) || ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        copy_to_host = true;
    }

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin("reduce_scaleout_sycl_direct",
                                  "send_size",
                                  count * ccl_dtype.size(),
                                  comm->unique_id());
#endif // CCL_ENABLE_ITT
    LOG_DEBUG("|CCL_SYCL| reduce scaleout selects simple (direct) kernel, count:",
              count,
              " datatype: ",
              dtype);
    ccl::event ev = reduce_scaleout_sycl_direct(
        q, send_buf, recv_buf, count, dtype, reduction, root, comm, deps, done, copy_to_host, is_cpu_buffers);
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    return ev;
}

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
#include "atl/mpi/atl_mpi_ctx.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

ccl::event allgatherv_scaleout_sycl_direct(sycl::queue& q,
                                           const void* send_buf,
                                           size_t send_count,
                                           void* recv_buf,
                                           const ccl::vector_class<size_t>& recv_counts,
                                           ccl::datatype dtype,
                                           ccl_comm* comm,
                                           const ccl::vector_class<ccl::event>& deps,
                                           bool& done,
                                           bool copy_to_host,
                                           bool is_cpu_buffers) {
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const void* scaleout_send_buf = send_buf;
    void* scaleout_recv_buf = recv_buf;

    std::vector<size_t> recv_scaleout_bytes(comm->size());
    std::vector<size_t> scaleout_offsets(comm->size());
    size_t total_scaleout_count = recv_counts[0];
    scaleout_offsets[0] = 0;
    recv_scaleout_bytes[0] = recv_counts[0] * ccl_dtype.size();
    for (size_t i = 1; i < comm->size(); i++) {
        scaleout_offsets[i] = scaleout_offsets[i - 1] + recv_counts[i - 1] * ccl_dtype.size();
        recv_scaleout_bytes[i] = recv_counts[i] * ccl_dtype.size();
        total_scaleout_count += recv_counts[i];
    }

    std::vector<sycl::event> sycl_deps = get_sycl_events(deps);
    sycl::event ev;
    if (copy_to_host) {
        if (comm->get_scaleout_host_buf_size() < total_scaleout_count * ccl_dtype.size()) {
            LOG_DEBUG("scaleout_host_buf_size is not big enough to handle ",
                      total_scaleout_count * ccl_dtype.size(),
                      " bytes. Falling back. TODO: chunking/pipelining");
            done = false;
            ccl::event e;
            return e;
        }

        scaleout_send_buf = MPI_IN_PLACE;
        scaleout_recv_buf = comm->get_scaleout_host_buf();
        ev = q.submit([=](sycl::handler& h) {
            h.depends_on(sycl_deps);
            h.memcpy((char*)scaleout_recv_buf + scaleout_offsets[comm->rank()],
                     send_buf,
                     send_count * ccl_dtype.size());
        });
        sycl_deps.clear();
        sycl_deps.push_back(ev);
    }
    else if (!is_cpu_buffers) {
        if (!check_mpi_supports_rdma()) {
            LOG_INFO("copy_to_host=false with a GPU buffer. "
                     "make sure MPI GPU RDMA is enabled");
        }
    }

    auto op_end = q.submit([=](sycl::handler& h) {
        h.depends_on(sycl_deps);
        h.host_task([=]() {
            // call ccl::wrapper for MPI/OFI.
            int ep_idx = 0;
            atl_req_t req;
            std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
            ATL_CALL_THROW_IF_ERROR(atl_comm->allgatherv(ep_idx,
                                                         scaleout_send_buf,
                                                         send_count * ccl_dtype.size(),
                                                         scaleout_recv_buf,
                                                         recv_scaleout_bytes.data(),
                                                         scaleout_offsets.data(),
                                                         req));

            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
            if (!req.is_completed) {
                // We do not want to call check() in a loop (because we would call MPI_Test repeatedly). Call MPI_Wait() instead.
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
            }
            else {
                // The operation was probably blocking, since it finished really quickly
            }
        });
    });

    if (copy_to_host) {
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(op_end);
            h.memcpy(recv_buf, scaleout_recv_buf, total_scaleout_count * ccl_dtype.size());
        });
    }

    done = true;
    return ccl::event::create_from_native(op_end);
}

ccl::event allgatherv_scaleout_sycl(sycl::queue& q,
                                    const void* send_buf,
                                    size_t send_count,
                                    void* recv_buf,
                                    const ccl::vector_class<size_t>& recv_counts,
                                    ccl::datatype dtype,
                                    ccl_comm* comm,
                                    const ccl::vector_class<ccl::event>& deps,
                                    bool original_deps,
                                    bool& done,
                                    bool direct,
                                    bool is_cpu_buffers) {
    bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (should_disable_rdma(ze_dev)) {
        copy_to_host = true;
    }

    return allgatherv_scaleout_sycl_direct(
        q, send_buf, send_count, recv_buf, recv_counts, dtype, comm, deps, done, copy_to_host, is_cpu_buffers);
}

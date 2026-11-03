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

#include "atl/mpi/atl_mpi_ctx.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/host_task.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/allgatherv/sycl/allgatherv_sycl.hpp"
#include "coll/algorithms/allgatherv/sycl/allgatherv_ring.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

#include "proxy/proxy.hpp"

ccl::event allgatherv_scaleout_sycl_direct(sycl::queue& q,
                                           const void* send_buf,
                                           size_t send_count,
                                           void* recv_buf,
                                           const ccl::vector_class<size_t>& recv_counts,
                                           const std::vector<size_t>& recv_offsets,
                                           ccl::datatype dtype,
                                           ccl_comm* comm,
                                           const ccl::vector_class<ccl::event>& deps,
                                           bool& done,
                                           bool copy_to_host,
                                           bool is_cpu_buffers,
                                           void* aux_buf) {
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const void* scaleout_send_buf = send_buf;
    void* scaleout_recv_buf = recv_buf;

    std::vector<size_t> recv_scaleout_bytes(comm->size(), send_count * ccl_dtype.size());
    size_t total_scaleout_count = send_count * comm->size();
    std::vector<size_t> scaleout_offsets;
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

        scaleout_send_buf = ATL_IN_PLACE;
        scaleout_recv_buf = aux_buf ? aux_buf : comm->get_scaleout_host_buf();
        scaleout_offsets.resize(comm->size());
        for (size_t i = 0; i < comm->size(); i++) {
            scaleout_offsets[i] = i * recv_scaleout_bytes[i];
        }

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

    auto op_end = ccl_host_task(q, sycl_deps, [=]() {
        // call ccl::wrapper for MPI/OFI.
        int ep_idx = 0;
        atl_req_t req;
        std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
        ATL_CALL_THROW_IF_ERROR(
            atl_comm->allgatherv(ep_idx,
                                 scaleout_send_buf,
                                 send_count * ccl_dtype.size(),
                                 scaleout_recv_buf,
                                 recv_scaleout_bytes.data(),
                                 copy_to_host ? scaleout_offsets.data() : recv_offsets.data(),
                                 req));

        atl_comm->check_wait_throw(ep_idx, req);
    });

    if (copy_to_host && aux_buf == nullptr) {
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
                                    std::vector<size_t>& recv_offsets,
                                    ccl::datatype dtype,
                                    ccl_comm* comm,
                                    const ccl::vector_class<ccl::event>& deps,
                                    bool original_deps,
                                    bool& done,
                                    sycl_scaleout_tune_attr tune_attr,
                                    bool copy_to_host,
                                    bool is_cpu_buffers,
                                    void* aux_buf) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    // when the proxy service is enabled for allgatherv, the selected algorithm runs its
    // scale-out exchange on the proxy thread instead of in a host task; a service that
    // failed to start (already warned about) leaves the non-proxy implementation
    const bool use_proxy = ccl::global_data::env().sycl_enable_allgatherv_scaleout_proxy &&
                           ccl::global_data::get().proxy_service->is_started();

    switch (tune_attr.algo()) {
        case ccl::sycl_policy::algorithm::direct:
        default: {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                "allgatherv_scaleout_sycl_direct", "send_size", send_count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allgatherv scaleout selects direct kernel, send_count:",
                      send_count,
                      " datatype: ",
                      dtype);
            auto e = allgatherv_scaleout_sycl_direct(q,
                                                     send_buf,
                                                     send_count,
                                                     recv_buf,
                                                     recv_counts,
                                                     recv_offsets,
                                                     dtype,
                                                     comm,
                                                     deps,
                                                     done,
                                                     copy_to_host,
                                                     is_cpu_buffers,
                                                     aux_buf);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            return e;
        }
        case ccl::sycl_policy::algorithm::ring: {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                use_proxy ? "allgatherv_scaleout_sycl_proxy_ring" : "allgatherv_scaleout_sycl_ring",
                "send_size",
                send_count * ccl_dtype.size(),
                comm->unique_id());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allgatherv scaleout selects ring kernel, send_count:",
                      send_count,
                      " datatype: ",
                      dtype,
                      " use_proxy: ",
                      use_proxy);
            sycl::event ev;
            if (use_proxy) {
                ev = allgatherv_scaleout_sycl_proxy_ring(q,
                                                         send_buf,
                                                         send_count,
                                                         recv_buf,
                                                         recv_counts,
                                                         recv_offsets,
                                                         dtype,
                                                         comm,
                                                         deps,
                                                         tune_attr,
                                                         done);
            }
            else {
                ev = allgatherv_scaleout_sycl_ring(q,
                                                   send_buf,
                                                   send_count,
                                                   recv_buf,
                                                   recv_counts,
                                                   recv_offsets,
                                                   dtype,
                                                   comm,
                                                   deps,
                                                   original_deps,
                                                   tune_attr,
                                                   done);
            }
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            return ccl::event::create_from_native(ev);
        }
    }
}

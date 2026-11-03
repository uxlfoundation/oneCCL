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

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/scatter/sycl/scatter_sycl.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

// -----------------------------------------------------------------------------
// r2r-comm scaleout scatter — thin wrapper around atl_comm->scatter
// (MPI_Scatter on Intel-MPI). OFI transport is unsupported here and the
// multi-node SYCL entrypoint falls back to the scheduler in that case.
// Shape mirrors allgatherv_scaleout_sycl.
//
// Two paths:
//   - direct: submit atl_comm->scatter in a host_task on device buffers
//     (requires GPU-aware MPI).
//   - copy_to_host: stage through a pinned host aux buf — root D->H
//     before scatter, non-roots H->D after. Used when MPI is not
//     GPU-aware.
//
// Root layout: send_buf holds `comm->size() * send_count` elements laid
// out [peer_0][peer_1]...[peer_{n-1}]. Every rank returns `send_count`
// elements in recv_buf.
// -----------------------------------------------------------------------------

namespace ccl {
namespace v1 {

ccl::event scatter_scaleout_sycl(sycl::queue& q,
                                 const void* send_buf,
                                 void* recv_buf,
                                 size_t send_count,
                                 ccl::datatype dtype,
                                 int root,
                                 ccl_comm* comm,
                                 const ccl::vector_class<ccl::event>& deps,
                                 bool& done,
                                 bool copy_to_host,
                                 bool is_cpu_buffers,
                                 void* aux_buf) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();
    const size_t send_bytes = send_count * dsize;
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin(
        "scatter_scaleout_sycl", "send_size", send_bytes, comm->unique_id());
    struct itt_scope {
        ~itt_scope() { ccl::profile::itt::task_end(); }
    } _itt_scope;
#endif // CCL_ENABLE_ITT
    const int my_rank = comm->rank();
    const int comm_size = comm->size();

    std::vector<sycl::event> sycl_deps = get_sycl_events(deps);

    // Single-rank comm: pure local copy (or no-op if the buffers alias).
    if (comm_size == 1) {
        sycl::event e;
        if (send_buf && recv_buf != send_buf) {
            e = q.submit([=](sycl::handler& h) {
                h.depends_on(sycl_deps);
                h.memcpy(recv_buf, send_buf, send_bytes);
            });
        }
        else {
            e = submit_wait_on_events(q, sycl_deps);
        }
        done = true;
        return ccl::event::create_from_native(e);
    }

    const void* scaleout_send_buf = send_buf;
    void* scaleout_recv_buf = recv_buf;

    // Host-staged path: root's send_buf goes into aux_buf (shaped as MPI
    // expects — all N peer slices contiguous). Non-root ranks receive
    // their slice into aux_buf then H->D into recv_buf. When the caller
    // does not supply aux_buf we borrow one from the comm's host pool
    // for the duration of the scatter and return it before we exit.
    bool owned_aux_buf = false;
    if (copy_to_host && aux_buf == nullptr) {
        aux_buf = comm->get_scaleout_host_buf();
        owned_aux_buf = true;
    }
    auto release_owned_aux_buf = [&]() {
        if (owned_aux_buf && aux_buf) {
            comm->put_scaleout_host_buf(aux_buf);
            aux_buf = nullptr;
            owned_aux_buf = false;
        }
    };

    if (copy_to_host) {
        const size_t root_slab_bytes = static_cast<size_t>(comm_size) * send_bytes;
        if (comm->get_scaleout_host_buf_size() < root_slab_bytes) {
            LOG_DEBUG("scaleout_host_buf_size (",
                      comm->get_scaleout_host_buf_size(),
                      " B) too small to stage a scatter of ",
                      root_slab_bytes,
                      " B at root. Falling back.");
            release_owned_aux_buf();
            done = false;
            ccl::event e;
            return e;
        }

        if (my_rank == root) {
            // MPI_IN_PLACE at root: send and recv aliased at aux_buf + root*slice.
            // Skip the D->H stage if the caller already parked send_buf in aux_buf.
            if (send_buf != aux_buf) {
                sycl::event stage = q.submit([=](sycl::handler& h) {
                    h.depends_on(sycl_deps);
                    h.memcpy(aux_buf, send_buf, root_slab_bytes);
                });
                sycl_deps.clear();
                sycl_deps.push_back(stage);
            }
            scaleout_send_buf = aux_buf;
            scaleout_recv_buf =
                static_cast<char*>(aux_buf) + static_cast<size_t>(root) * send_bytes;
        }
        else {
            scaleout_send_buf = nullptr;
            scaleout_recv_buf = aux_buf;
        }
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
            int ep_idx = 0;
            atl_req_t req;
            std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
            ATL_CALL_THROW_IF_ERROR(atl_comm->scatter(
                ep_idx, scaleout_send_buf, scaleout_recv_buf, send_bytes, root, req));

            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
            if (!req.is_completed) {
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
            }
        });
    });

    if (copy_to_host) {
        // H->D the freshly-received slice into the caller's recv_buf.
        // Skip if recv_buf already IS the host slot we wrote to.
        const void* my_host_slot =
            (my_rank == root) ? static_cast<const char*>(aux_buf) +
                                    static_cast<size_t>(root) * send_bytes
                              : aux_buf;
        if (recv_buf != my_host_slot) {
            op_end = q.submit([=](sycl::handler& h) {
                h.depends_on(op_end);
                h.memcpy(recv_buf, my_host_slot, send_bytes);
            });
        }
    }

    if (owned_aux_buf) {

        comm->put_scaleout_host_buf(aux_buf);
        owned_aux_buf = false;
    }

    done = true;
    return ccl::event::create_from_native(op_end);
}

} // namespace v1
} // namespace ccl

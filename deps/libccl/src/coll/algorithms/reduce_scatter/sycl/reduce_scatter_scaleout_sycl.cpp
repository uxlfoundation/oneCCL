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
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_ring.hpp"
#include "proxy/proxy.hpp"
#include "coll/algorithms/utils/host_task.hpp"

//  ATL_CALL_THROW_IF_ERROR function is in 'small_refactors' branch. No need to merge to master
#define ATL_CALL_THROW_IF_ERROR(func) \
    do { \
        atl_status_t status = func; \
        if (unlikely(status != ATL_STATUS_SUCCESS)) { \
            CCL_THROW(#func "\n fails with status: ", status); \
        } \
    } while (0)

//#define PRINT_TIMING

// simple algorithm calls MPI reduce-scatter directly
ccl::event reduce_scatter_scaleout_sycl_simple(sycl::queue& q,
                                               const void* send_buf,
                                               void* recv_buf,
                                               size_t recv_count,
                                               ccl::datatype dtype,
                                               ccl::reduction reduction,
                                               ccl_comm* comm,
                                               const ccl::vector_class<ccl::event>& deps,
                                               bool& done,
                                               bool copy_to_host,
                                               bool is_cpu_buffers) {
    sycl::event op_end;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t count = recv_count * comm->size();

    void* scaleout_send_buf = (void*)send_buf;
    void* scaleout_recv_buf = recv_buf;

#ifdef PRINT_TIMING
    q.wait();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    cpu_timer<1> ctimer;
    ctimer.start(0);
#endif // PRINT_TIMING

    std::vector<sycl::event> dep_events = get_sycl_events(deps);
#if 0
    sycl::queue lce_q = get_mce_queue(q);
#else
    sycl::queue lce_q = q; //sycl::queue(q.get_device(), sycl::property::queue::in_order{});
#endif

    if (copy_to_host) {
        if (comm->get_scaleout_host_buf_size() < count * ccl_dtype.size()) {
            LOG_WARN("scaleout_host_buf_size is not big enough to handle ",
                     count * ccl_dtype.size(),
                     " bytes. Falling back. TODO: chunking/pipelining");
            done = false;
            ccl::event e;
            return e;
        }

        scaleout_send_buf = comm->get_scaleout_host_buf();
        scaleout_recv_buf = comm->get_scaleout_host_buf();
        op_end = lce_q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(scaleout_send_buf, send_buf, count * ccl_dtype.size());
        });
    }
    else if (!is_cpu_buffers) {
        if (!check_mpi_supports_rdma()) {
            LOG_WARN("copy_to_host=false with a GPU buffer. "
                     "make sure MPI GPU RDMA is enabled");
        }
    }

#ifdef PRINT_TIMING
    op_end.wait();
    ctimer.stop(0);
    fprintf(stderr,
            "[%d] copy GPU to CPU takes: %f us on %ld bytes (recv_count: %ld)\n",
            rank,
            ctimer.get_us(0),
            count * ccl_dtype.size(),
            recv_count);
    ctimer.start(0);
#endif // PRINT_TIMING

    op_end = ccl_host_task(q, {op_end}, [=]() {
        // call ccl::wrapper for MPI/OFI.
        int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
        atl_req_t req;
        std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
        ATL_CALL_THROW_IF_ERROR(atl_comm->reduce_scatter(ep_idx,
                                                         scaleout_send_buf,
                                                         scaleout_recv_buf,
                                                         recv_count,
                                                         ccl_dtype.atl_datatype(),
                                                         static_cast<atl_reduction_t>(reduction),
                                                         req));

        atl_comm->check_wait_throw(ep_idx, req);
    });

#ifdef PRINT_TIMING
    op_end.wait();
    ctimer.stop(0);
    fprintf(stderr, "[%d] MPI reduce_scatter takes: %f us, recv_count: %ld\n", rank, ctimer.get_us(0), recv_count);
#endif // PRINT_TIMING

    if (copy_to_host) {
#ifdef PRINT_TIMING
        ctimer.start(0);
#endif // PRINT_TIMING
        op_end = lce_q.submit([=](sycl::handler& h) {
            h.depends_on(op_end);
            h.memcpy(recv_buf, scaleout_recv_buf, recv_count * ccl_dtype.size());
        });
#ifdef PRINT_TIMING
        op_end.wait();
        ctimer.stop(0);
        fprintf(stderr,
                "[%d] copy CPU to GPU takes: %f us with %ld bytes\n",
                rank,
                ctimer.get_us(0),
                recv_count * ccl_dtype.size());
#endif // PRINT_TIMING
    }

    done = true;
    return ccl::event::create_from_native(op_end);
}

ccl::event reduce_scatter_scaleout_sycl(sycl::queue& q,
                                        const void* send_buf,
                                        void* recv_buf,
                                        size_t recv_count,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        ccl_comm* comm,
                                        const ccl::vector_class<ccl::event>& deps,
                                        bool original_deps,
                                        sycl_scaleout_tune_attr& tune_attr,
                                        bool& done,
                                        bool is_cpu_buffers) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
    ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (should_disable_rdma(ze_dev) || ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        copy_to_host = true;
    }
    sycl_scaleout_algo algo = tune_attr.algo();

    // when the proxy service is enabled for reduce-scatter, the selected algorithm runs
    // its scale-out exchange on the proxy thread instead of in a host task; a service
    // that failed to start (already warned about) leaves the non-proxy implementation
    const bool use_proxy = ccl::global_data::env().sycl_enable_reduce_scatter_scaleout_proxy &&
                           ccl::global_data::get().proxy_service->is_started();

    switch (algo) {
        case ccl::sycl_policy::algorithm::direct:
        default: {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                "reduce_scatter_scaleout_sycl_simple", "recv_size", recv_count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| reduce_scatter scaleout selects direct kernel, recv_count:",
                      recv_count,
                      " datatype: ",
                      dtype);
            auto e = reduce_scatter_scaleout_sycl_simple(
                q, send_buf, recv_buf, recv_count, dtype, reduction, comm, deps, done, copy_to_host, is_cpu_buffers);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            return e;
        }
        case ccl::sycl_policy::algorithm::ring: {
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                use_proxy ? "reduce_scatter_scaleout_sycl_proxy_ring" : "reduce_scatter_scaleout_sycl_ring",
                "recv_size",
                recv_count * ccl_dtype.size(),
                comm->unique_id());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| reduce_scatter scaleout selects ring kernel, recv_count:",
                      recv_count,
                      " datatype: ",
                      dtype,
                      " use_proxy: ",
                      use_proxy);
            sycl::event e;
            if (use_proxy) {
                e = reduce_scatter_scaleout_sycl_proxy_ring(
                    q, send_buf, recv_buf, recv_count, dtype, reduction, comm, deps, tune_attr, done);
            }
            else {
                e = reduce_scatter_ring(q,
                                        send_buf,
                                        recv_buf,
                                        recv_count,
                                        dtype,
                                        reduction,
                                        comm,
                                        deps,
                                        original_deps,
                                        tune_attr,
                                        done);
            }
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            return ccl::event::create_from_native(e);
        }
    }
}

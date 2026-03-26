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
#include "coll/algorithms/utils/sycl_selection.hpp"

#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
#include "coll/algorithms/alltoall/sycl/alltoall_sycl.hpp"
#include "coll/algorithms/alltoall/sycl/alltoall_pairwise.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

sycl::event alltoall_scaleout_sycl_simple_blocking(sycl::queue& q,
                                                   const void* send_buf,
                                                   void* recv_buf,
                                                   size_t count,
                                                   ccl::datatype dtype,
                                                   ccl_comm* comm,
                                                   const ccl::vector_class<ccl::event>& deps,
                                                   bool original_deps,
                                                   bool& done,
                                                   bool copy_to_host,
                                                   bool is_cpu_buffers) {
    sycl::event op_end, copy_e;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    uint32_t world = comm->size();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    const void* scaleout_send_buf = send_buf;
    void* scaleout_recv_buf = recv_buf;

    if (original_deps)
        q.wait();
    for (auto& dep : deps) {
        auto e = dep.get_native();
        e.wait();
    }

    if (copy_to_host) {
        if (comm->get_scaleout_host_buf_size() < count * ccl_dtype.size() * world) {
            LOG_WARN("scaleout_host_buf_size is not big enough to handle ",
                     count * ccl_dtype.size(),
                     " bytes. Falling back. TODO: chunking/pipelining");
            done = false;
            sycl::event e;
            return e;
        }

        scaleout_recv_buf = comm->get_scaleout_host_buf();
        scaleout_send_buf = scaleout_recv_buf;
        copy_e = q.submit([=](sycl::handler& h) {
            h.memcpy(scaleout_recv_buf,
                     send_buf == MPI_IN_PLACE ? recv_buf : send_buf,
                     count * ccl_dtype.size() * world);
        });
        copy_e.wait();
    }
    else if (!is_cpu_buffers) {
        // TODO: check if I_MPI_OFFLOAD is set, then let the scaleout allreduce go through.
        //LOG_WARN("copy_to_host=false with a GPU buffer. "
        //         "TODO: make sure I_MPI_OFFLOAD is set or GPU RDMA is enabled");
        // TODO: determine whether we want to fallback or not. For now, no.
        // done = false;
        // ccl::event e;
        // return e;
    }

    // call ccl::wrapper for MPI/OFI.
    int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
    atl_req_t req;
    ATL_CALL_THROW_IF_ERROR(atl_comm->alltoall(
        ep_idx, scaleout_send_buf, scaleout_recv_buf, count * ccl_dtype.size(), req));

    ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
    if (!req.is_completed) {
        // We do not want to call check() in a loop (because we would call MPI_Test repeatedly). Call MPI_Wait() instead.
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));

        // TODO: if it is determined that running atl_comm->allreduce from inside allreduce_entry (i.e. the sched) is WAY faster than running it from out here, how about checking how the schedule does progress()?
        //       allreduce_entry::update() does a simple check():     atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);
        //       Experimentally, it doesn't seem to make any difference.
    }
    else {
        // The operation was probably blocking, since it finished really quickly
    }

    if (copy_to_host) {
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(op_end);
            h.memcpy(recv_buf, scaleout_recv_buf, count * ccl_dtype.size() * world);
        });
        op_end.wait();
    }

    done = true;
    return op_end;
}

sycl::event alltoall_scaleout_sycl_simple_nonblocking(sycl::queue& q,
                                                      const void* send_buf,
                                                      void* recv_buf,
                                                      size_t count,
                                                      ccl::datatype dtype,
                                                      ccl_comm* comm,
                                                      const ccl::vector_class<ccl::event>& deps,
                                                      bool original_deps,
                                                      bool& done,
                                                      bool copy_to_host,
                                                      bool is_cpu_buffers) {
    sycl::event op_end, copy_e;
    sycl::context ctx = q.get_context();
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
    uint32_t world = comm->size();
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    size_t size = count * ccl_dtype.size();

    const void* scaleout_send_buf = send_buf;
    void* scaleout_recv_buf = recv_buf;

    std::vector<sycl::event> dep_events = get_sycl_events(deps);

    bool src_do_fast_gdrcopy = false;
    bool dst_do_fast_gdrcopy = false;
    void *mmap_src_buf, *mmap_dst_buf;

    if (copy_to_host) {
        if (comm->get_scaleout_host_buf_size() < size) {
            LOG_WARN("scaleout_host_buf_size is not big enough to handle ",
                     count * ccl_dtype.size(),
                     " bytes. Falling back. TODO: chunking/pipelining");
            done = false;
            sycl::event e;
            return e;
        }

        scaleout_send_buf = MPI_IN_PLACE;
        scaleout_recv_buf = comm->get_scaleout_host_buf();
        copy_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(scaleout_recv_buf,
                     send_buf == MPI_IN_PLACE ? recv_buf : send_buf,
                     count * ccl_dtype.size() * world);
        });
        dep_events.clear();
        dep_events.push_back(std::move(copy_e));
    }
    else if (!is_cpu_buffers) {
        // TODO: check if I_MPI_OFFLOAD is set, then let the scaleout allreduce go through.
        //LOG_WARN("copy_to_host=false with a GPU buffer. "
        //         "TODO: make sure I_MPI_OFFLOAD is set or GPU RDMA is enabled");
        // TODO: determine whether we want to fallback or not. For now, no.
        // done = false;
        // ccl::event e;
        // return e;
    }

    op_end = q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);
        h.host_task([=]() {
            // call ccl::wrapper for MPI/OFI.
            int ep_idx = 0; // TODO: instead of "0", use atl_ep->idx, or sched->bin->get_atl_ep()
            atl_req_t req;
            std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();
            ATL_CALL_THROW_IF_ERROR(atl_comm->alltoall(
                ep_idx, scaleout_send_buf, scaleout_recv_buf, count * ccl_dtype.size(), req));

            ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
            if (!req.is_completed) {
                // We do not want to call check() in a loop (because we would call MPI_Test repeatedly). Call MPI_Wait() instead.
                ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));

                // TODO: if it is determined that running atl_comm->allreduce from inside allreduce_entry (i.e. the sched) is WAY faster than running it from out here, how about checking how the schedule does progress()?
                //       allreduce_entry::update() does a simple check():     atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);
                //       Experimentally, it doesn't seem to make any difference.
            }
            else {
                // The operation was probably blocking, since it finished really quickly
            }
        });
    });

    if (copy_to_host) {
        op_end = q.submit([=](sycl::handler& h) {
            h.depends_on(op_end);
            h.memcpy(recv_buf, scaleout_recv_buf, count * ccl_dtype.size() * world);
        });
    }

    done = true;
    return op_end;
}

inline sycl::event alltoall_scaleout_sycl_direct(sycl::queue& q,
                                                 const void* send_buf,
                                                 void* recv_buf,
                                                 size_t count,
                                                 ccl::datatype dtype,
                                                 ccl_comm* comm,
                                                 const ccl::vector_class<ccl::event>& deps,
                                                 bool original_deps,
                                                 sycl_alltoall_tune_attr tune_attr,
                                                 bool& done,
                                                 bool is_cpu_buffers) {
    bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (should_disable_rdma(ze_dev)) {
        copy_to_host = true;
    }
    if (ccl::global_data::env().enable_op_sync) {
        return alltoall_scaleout_sycl_simple_blocking(q,
                                                      send_buf,
                                                      recv_buf,
                                                      count,
                                                      dtype,
                                                      comm,
                                                      deps,
                                                      original_deps,
                                                      done,
                                                      copy_to_host,
                                                      is_cpu_buffers);
    }
    else {
        return alltoall_scaleout_sycl_simple_nonblocking(q,
                                                         send_buf,
                                                         recv_buf,
                                                         count,
                                                         dtype,
                                                         comm,
                                                         deps,
                                                         original_deps,
                                                         done,
                                                         copy_to_host,
                                                         is_cpu_buffers);
    };
}

ccl::event alltoall_scaleout_sycl(sycl::queue& q,
                                  const void* send_buf,
                                  void* recv_buf,
                                  size_t count,
                                  ccl::datatype dtype,
                                  ccl_comm* comm,
                                  ccl_stream* global_stream,
                                  const ccl::vector_class<ccl::event>& deps,
                                  bool original_deps,
                                  sycl_alltoall_tune_attr tune_attr,
                                  bool& done) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (should_disable_rdma(ze_dev)) {
        copy_to_host = true;
    }
    alltoall_scaleout_algo algo = tune_attr.algo;

    sycl::event ev;
    ccl::event e;
    done = false;

    switch (algo) {
        case alltoall_scaleout_algo::direct:
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                "alltoall_scaleout_sycl_simple", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| alltoall scaleout selects simple (direct) kernel, count:",
                      count,
                      " datatype: ",
                      dtype);
            ev = alltoall_scaleout_sycl_direct(q,
                                               send_buf,
                                               recv_buf,
                                               count,
                                               dtype,
                                               comm,
                                               deps,
                                               original_deps,
                                               tune_attr,
                                               done,
                                               false);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            return ccl::event::create_from_native(ev);
        case alltoall_scaleout_algo::pairwise:
            e = alltoall_sycl_pairwise_rdma(
                q, send_buf, recv_buf, count, dtype, comm, global_stream, deps, done);
            return e;
        case alltoall_scaleout_algo::scatter:
            e = alltoall_sycl_scatter_rdma(
                q, send_buf, recv_buf, count, dtype, comm, global_stream, deps, done);
            return e;
        case alltoall_scaleout_algo::gdr_only_pairwise:
            e = alltoall_sycl_global_pairwise_rdma(
                q, send_buf, recv_buf, count, dtype, comm, global_stream, deps, done);
            return e;
        case alltoall_scaleout_algo::numa_gdr_only:
            e = alltoall_sycl_numa_pairwise_rdma_oneshot(q,
                                                         send_buf,
                                                         recv_buf,
                                                         count,
                                                         dtype,
                                                         comm,
                                                         global_stream,
                                                         deps,
                                                         true /* batch_node */,
                                                         done);
            return e;
        case alltoall_scaleout_algo::numa_gdr_split:
            e = alltoall_sycl_numa_pairwise_rdma_oneshot_split(q,
                                                               send_buf,
                                                               recv_buf,
                                                               count,
                                                               dtype,
                                                               comm,
                                                               global_stream,
                                                               deps,
                                                               true /* batch_node */,
                                                               done);
            return e;
        case alltoall_scaleout_algo::fallback:
        default: done = false; return e;
    }
}

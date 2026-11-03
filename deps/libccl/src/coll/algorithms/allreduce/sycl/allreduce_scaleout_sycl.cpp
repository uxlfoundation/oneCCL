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
#include "coll/algorithms/allreduce/sycl/allreduce_sycl.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_direct.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_ring.hpp"
#include "coll/algorithms/allreduce/sycl/allreduce_rabenseifner.hpp"
#endif // defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)

ccl::event allreduce_scaleout_sycl(sycl::queue& q,
                                   const void* send_buf,
                                   void* recv_buf,
                                   size_t count,
                                   ccl::datatype dtype,
                                   ccl::reduction reduction,
                                   ccl_comm* comm,
                                   const ccl::vector_class<ccl::event>& deps,
                                   bool original_deps,
                                   sycl_scaleout_tune_attr tune_attr,
                                   bool& done,
                                   bool is_cpu_buffers) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
    ze_device_handle_t ze_dev =
        sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    if (should_disable_rdma(ze_dev) || ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        copy_to_host = true;
    }
    sycl_scaleout_algo algo = tune_attr.algo();

    // when the proxy service is enabled for allreduce, the selected algorithm runs its
    // scale-out exchange on the proxy thread instead of in a host task; a service that
    // failed to start (already warned about) leaves the non-proxy implementation
    const bool use_proxy = ccl::global_data::env().sycl_enable_allreduce_scaleout_proxy &&
                           ccl::global_data::get().proxy_service->is_started();

    sycl::event ev;
    done = false;

    switch (algo) {
        case ccl::sycl_policy::algorithm::direct:
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allreduce_scaleout_sycl_simple",
                                          "send_size",
                                          count * ccl_dtype.size(),
                                          comm->unique_id());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allreduce scaleout selects simple (direct) kernel, count:",
                      count,
                      " datatype: ",
                      dtype,
                      " use_proxy: ",
                      use_proxy);
            ev = allreduce_scaleout_sycl_simple(q,
                                                send_buf,
                                                recv_buf,
                                                count,
                                                dtype,
                                                reduction,
                                                comm,
                                                deps,
                                                done,
                                                copy_to_host,
                                                is_cpu_buffers,
                                                use_proxy);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            // no need to goto fallback, just return
            return ccl::event::create_from_native(ev);
        case ccl::sycl_policy::algorithm::rabenseifner:
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin("allreduce_scaleout_sycl_rabenseifner",
                                          "send_size",
                                          count * ccl_dtype.size(),
                                          comm->unique_id());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allreduce scaleout selects rabenseifner kernel, count:",
                      count,
                      " datatype: ",
                      dtype);
            ev = allreduce_scaleout_sycl_rabenseifner(q,
                                                      send_buf,
                                                      recv_buf,
                                                      count,
                                                      dtype,
                                                      reduction,
                                                      comm,
                                                      deps,
                                                      original_deps,
                                                      tune_attr,
                                                      done);
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
            if (!done) {
                goto fallback;
            }
            return ccl::event::create_from_native(ev);
        case ccl::sycl_policy::algorithm::ring:
#ifdef CCL_ENABLE_ITT
            ccl::profile::itt::task_begin(
                use_proxy ? "allreduce_scaleout_sycl_proxy_ring" : "allreduce_scaleout_sycl_ring",
                "send_size",
                count * ccl_dtype.size(),
                comm->unique_id());
#endif // CCL_ENABLE_ITT
            LOG_DEBUG("|CCL_SYCL| allreduce scaleout selects ring kernel, count:",
                      count,
                      " datatype: ",
                      dtype,
                      " use_proxy: ",
                      use_proxy);
            if (use_proxy) {
                ev = allreduce_scaleout_sycl_proxy_ring(
                    q, send_buf, recv_buf, count, dtype, reduction, comm, deps, tune_attr, done);
            }
            else {
                ev = allreduce_scaleout_sycl_ring(q,
                                                  send_buf,
                                                  recv_buf,
                                                  count,
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
            if (!done) {
                goto fallback;
            }
            return ccl::event::create_from_native(ev);
        default: goto fallback;
    }

fallback:
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin(
        "allreduce_scaleout_sycl_simple", "send_size", count * ccl_dtype.size(), comm->unique_id());
#endif // CCL_ENABLE_ITT
    LOG_DEBUG("|CCL_SYCL| allreduce scaleout selects default simple (direct) kernel, count:",
              count,
              " datatype: ",
              dtype);
    ev = allreduce_scaleout_sycl_simple(q,
                                        send_buf,
                                        recv_buf,
                                        count,
                                        dtype,
                                        reduction,
                                        comm,
                                        deps,
                                        done,
                                        copy_to_host,
                                        is_cpu_buffers);
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT
    return ccl::event::create_from_native(ev);
}

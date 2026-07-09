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
#include "common/log/log.hpp"

void log_debug(std::string msg) {
    if (ccl::global_data::env().use_root_print_wa) {
        if (ccl_logger::get_instance().is_root()) {
            LOG_DEBUG(msg);
        }
    }
    else {
        LOG_DEBUG(msg);
    }
}

ccl::event invoke_mpi_bcast(void* buf,
                            size_t count,
                            ccl::datatype dtype,
                            int root,
                            ccl_comm* comm,
                            const ccl::stream& op_stream,
                            const ccl::broadcast_attr& attr,
                            const ccl::vector_class<ccl::event>& deps) {
#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    log_debug("Entering invoke_mpi_bcast");
    for (auto& dep : deps) {
        sycl::event e = dep.get_native();
        e.wait();
    }
    sycl::queue& q = const_cast<sycl::queue&>(op_stream.get_native());
    q.wait();
    log_debug("Finishing dependencies in invoke_mpi_bcast");

    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const int dsize = ccl_dtype.size();
    int ep_idx = 0;
    atl_req_t req;
    std::shared_ptr<atl_base_comm> atl_comm = comm->get_atl_comm();

    ATL_CALL_THROW_IF_ERROR(atl_comm->bcast(ep_idx, buf, count * dsize, root, req));
    ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
    if (!req.is_completed) {
        // We do not want to call check() in a loop (because we would call MPI_Test repeatedly). Call MPI_Wait() instead.
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
    }
    else {
        // The operation was probably blocking, since it finished really quickly
    }
    log_debug("Exiting invoke_mpi_bcast");
#else
    CCL_THROW("invoke_mpi directly not supported without sycl");
#endif //#if defined(CCL_ENABLE_ZE) || defined(CCL_ENABLE_SYCL)
    return ccl::event();
}

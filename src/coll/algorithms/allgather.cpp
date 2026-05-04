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
#include "coll/algorithms/algorithms.hpp"
#include "coll/coll_util.hpp"
#include "comm/comm.hpp"
#include "sched/entry/factory/chunked_entry_factory.hpp"
#include "sched/entry/factory/entry_factory.hpp"

ccl::status ccl_coll_build_direct_allgather(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            ccl_comm* comm) {
    LOG_DEBUG("build direct allgather");

    std::vector<size_t> recv_counts(comm->size(), count);
    entry_factory::create<allgatherv_entry>(
        sched, send_buf, count, recv_buf, recv_counts.data(), dtype, comm);
    return ccl::status::success;
}
ccl::status ccl_coll_build_naive_allgather(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t count,
                                           const ccl_datatype& dtype,
                                           ccl_comm* comm) {
    LOG_DEBUG("build naive allgather");

    ccl::status status = ccl::status::success;

    int comm_size = comm->size();
    int comm_rank = comm->rank();
    size_t dtype_size = dtype.size();
    std::vector<size_t> offsets(comm_size);

    offsets[0] = 0;
    for (int rank = 1; rank < comm_size; rank++) {
        offsets[rank] = offsets[rank - 1] + count * dtype_size;
    }
    std::vector<size_t> recv_counts(comm_size, count);
    bool is_inplace = ccl::is_allgatherv_inplace(send_buf.get_ptr(),
                                                 count,
                                                 recv_buf.get_ptr(),
                                                 recv_counts.data(),
                                                 offsets.data(),
                                                 dtype.size(),
                                                 comm_rank,
                                                 comm_size);

    if ((!is_inplace) && (count > 0)) {
        // out-of-place case
        entry_factory::create<copy_entry>(
            sched, send_buf, recv_buf + offsets[comm_rank], count, dtype);
    }

    for (int idx = 1; idx < comm_size; idx++) {
        int dst = (comm_rank + idx) % comm_size;
        int src = (comm_rank - idx + comm_size) % comm_size;

        if (count > 0) {
            // send own buffer to other ranks
            entry_factory::create<send_entry>(
                sched, recv_buf + offsets[comm_rank], count, dtype, dst, comm);
            // recv other's rank buffer
            entry_factory::create<recv_entry>(
                sched, recv_buf + offsets[src], count, dtype, src, comm);
        }
    }

    return status;
}

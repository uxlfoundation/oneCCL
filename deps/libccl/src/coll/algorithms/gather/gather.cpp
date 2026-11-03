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

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll_util.hpp"
#include "sched/entry/factory/entry_factory.hpp"

using namespace ccl::utils;

ccl::status ccl_coll_build_direct_gather(ccl_sched* sched,
                                         ccl_buffer send_buf,
                                         ccl_buffer recv_buf,
                                         size_t count,
                                         const ccl_datatype& dtype,
                                         int root,
                                         ccl_comm* comm) {
    ccl::status status = ccl::status::success;

    int rank = comm->rank();
    int comm_size = comm->size();
    size_t block_bytes = count * dtype.size();

    const bool inplace = ccl::is_gather_inplace(
        send_buf.get_ptr(), recv_buf.get_ptr(), count, dtype.size(), rank, root, comm_size);
    LOG_DEBUG("build direct gather: ", inplace ? "in-place" : "out-of-place");

    if (comm_size == 1) {
        if (!inplace) {
            entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        }
        return status;
    }

    if (rank == root) {
        /* root: receive one block from each non-root rank */
        for (int i = 0; i < comm_size; i++) {
            if (i == rank) {
                continue;
            }
            ccl_buffer slot = recv_buf + static_cast<size_t>(i) * block_bytes;
            entry_factory::create<recv_entry>(sched, slot, count, dtype, i, comm);
        }

        /* root copies its own block if not in-place */
        if (!inplace) {
            ccl_buffer own_slot = recv_buf + static_cast<size_t>(root) * block_bytes;
            entry_factory::create<copy_entry>(sched, send_buf, own_slot, count, dtype);
        }
    }
    else {
        /* non-root: send its block to root */
        entry_factory::create<send_entry>(sched, send_buf, count, dtype, root, comm);
    }

    return status;
}

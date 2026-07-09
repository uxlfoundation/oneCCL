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

ccl::status ccl_coll_build_naive_scatter(ccl_sched* sched,
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

    const bool inplace = ccl::is_scatter_inplace(
        send_buf.get_ptr(), recv_buf.get_ptr(), count, dtype.size(), rank, root, comm_size);
    LOG_DEBUG("build naive scatter: ", inplace ? "in-place" : "out-of-place");

    /* single rank: out-of-place copies own block to recv_buf;
     * in-place is a no-op and we return immediately. */
    if (comm_size == 1) {
        if (!inplace) {
            entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        }
        return status;
    }

    if (rank == root) {
        /* root: send block i to each non-root rank, then handle own block */
        for (int i = 0; i < comm_size; i++) {
            if (i == rank) {
                continue;
            }
            ccl_buffer block = send_buf + static_cast<size_t>(i) * block_bytes;
            entry_factory::create<send_entry>(sched, block, count, dtype, i, comm);
        }

        /* root handles its own block: in-place means recv_buf aliases send_buf[root] */
        if (!inplace) {
            ccl_buffer own_block = send_buf + static_cast<size_t>(root) * block_bytes;
            entry_factory::create<copy_entry>(sched, own_block, recv_buf, count, dtype);
        }
    }
    else {
        /* non-root: receive one block from root */
        entry_factory::create<recv_entry>(sched, recv_buf, count, dtype, root, comm);
    }

    return status;
}

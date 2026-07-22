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


/*
*
*  (C) 2001 by Argonne National Laboratory.
*      See COPYRIGHT in top-level directory.
*/

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll_util.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#define MIN(a, b) std::min(a, b)

using namespace ccl::utils;

ccl::status ccl_coll_build_direct_broadcast(ccl_sched* sched,
                                            ccl_buffer send_buf,
                                            ccl_buffer recv_buf,
                                            size_t count,
                                            const ccl_datatype& dtype,
                                            int root,
                                            ccl_comm* comm) {
    LOG_DEBUG("build direct broadcast");

    if (comm->rank() == root && send_buf != recv_buf) {
        entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        sched->add_barrier();
    }

    entry_factory::create<broadcast_entry>(sched, send_buf, recv_buf, count, dtype, root, comm);

    return ccl::status::success;
}

ccl::status ccl_coll_build_naive_broadcast(ccl_sched* sched,
                                           ccl_buffer send_buf,
                                           ccl_buffer recv_buf,
                                           size_t count,
                                           const ccl_datatype& dtype,
                                           int root,
                                           ccl_comm* comm) {
    LOG_DEBUG("build naive broadcast");

    ccl::status status = ccl::status::success;

    int rank = comm->rank();
    int comm_size = comm->size();

    if (rank == root && send_buf != recv_buf) {
        entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
        sched->add_barrier();
    }

    if (comm_size == 1)
        goto fn_exit;

    if (rank == root) {
        for (int idx = 0; idx < comm_size; idx++) {
            if (idx != rank) {
                entry_factory::create<send_entry>(sched, send_buf, count, dtype, idx, comm);
            }
        }
    }
    else {
        entry_factory::create<recv_entry>(sched, recv_buf, count, dtype, root, comm);
    }

fn_exit:
    return status;
}

ccl::status ccl_coll_build_scatter_for_broadcast(ccl_sched* sched,
                                                 ccl_buffer send_buf,
                                                 ccl_buffer recv_buf,
                                                 int root,
                                                 size_t nbytes,
                                                 ccl_comm* comm) {
    LOG_DEBUG("build scatter_for_broadcast");

    ccl::status status = ccl::status::success;
    int rank, local_root, comm_size, src, dst;
    int relative_rank, mask;
    int scatter_size, curr_size, recv_size, send_size;
    bool is_inplace = (send_buf == recv_buf);

    comm_size = comm->size();
    rank = comm->rank();
    local_root = static_cast<int>(root);
    relative_rank = (rank >= local_root) ? rank - local_root : rank - local_root + comm_size;

    /* The scatter algorithm divides the buffer into nprocs pieces and
     * scatters them among the processes. Root gets the first piece,
     * root+1 gets the second piece, and so forth. Uses the same
     * binomial tree algorithm as above. Ceiling division is used to
     * compute the size of each piece. This means some processes may
     * not get any data. For example if bufsize = 97 and nprocs = 16,
     * ranks 15 and 16 will get 0 data. On each process, the scattered
     * data is stored at the same offset in the buffer as it is on the
     * root process. */

    scatter_size = (nbytes + comm_size - 1) / comm_size; /* ceiling division */
    curr_size = (rank == local_root) ? nbytes : 0; /* root starts with all the data */

    mask = 0x1;
    while (mask < comm_size) {
        if (relative_rank & mask) {
            src = rank - mask;
            if (src < 0)
                src += comm_size;

            /* compute the exact recv_size to avoid writing this NBC
             * in callback style */
            recv_size = nbytes - (relative_rank * scatter_size);
            if (recv_size < 0)
                recv_size = 0;

            curr_size = recv_size;

            if (recv_size > 0) {
                entry_factory::create<recv_entry>(sched,
                                                  recv_buf + relative_rank * scatter_size,
                                                  recv_size,
                                                  ccl_datatype_int8,
                                                  src,
                                                  comm);
                sched->add_barrier();
            }
            break;
        }
        mask <<= 1;
    }

    /* This process is responsible for all processes that have bits
     * set from the LSB upto (but not including) mask.  Because of the
     * "not including", we start by shifting mask back down one. */

    mask >>= 1;
    while (mask > 0) {
        if (relative_rank + mask < comm_size) {
            send_size = curr_size - scatter_size * mask;

            /* mask is also the size of this process's subtree */

            if (send_size > 0) {
                dst = rank + mask;
                if (dst >= comm_size)
                    dst -= comm_size;

                // Skip sending to self if in-place and root.
                if (!(is_inplace && dst == local_root)) {
                    entry_factory::create<send_entry>(
                        sched,
                        send_buf + scatter_size * (relative_rank + mask),
                        send_size,
                        ccl_datatype_int8,
                        dst,
                        comm);
                }
                sched->add_barrier();
                curr_size -= send_size;
            }
        }
        mask >>= 1;
    }

    return status;
}

ccl::status ccl_coll_build_scatter_ring_allgather_broadcast(ccl_sched* sched,
                                                            ccl_buffer send_buf,
                                                            ccl_buffer recv_buf,
                                                            size_t count,
                                                            const ccl_datatype& dtype,
                                                            int root,
                                                            ccl_comm* comm) {
    LOG_DEBUG("build scatter_ring_allgather broadcast");

    ccl::status status = ccl::status::success;

    int comm_size, rank, nbytes;
    int scatter_size, curr_size;
    int i, j, jnext, left, right;
    size_t dtype_size = dtype.size();

    comm_size = comm->size();
    rank = comm->rank();

    ccl_buffer send_tmp_buf(send_buf);
    ccl_buffer recv_tmp_buf(recv_buf);

    /* If there is only one process, return */
    if (comm_size == 1)
        goto fn_exit;

    nbytes = dtype_size * count;

    CCL_CALL(ccl_coll_build_scatter_for_broadcast(
        sched, send_tmp_buf, recv_tmp_buf, root, nbytes, comm));

    /* this is the block size used for the scatter operation */
    scatter_size = (nbytes + comm_size - 1) / comm_size; /* ceiling division */

    /* curr_size is the amount of data that this process now has stored in
     * buffer at byte offset (rank*scatter_size) */
    curr_size = MIN(scatter_size, (nbytes - (rank * scatter_size)));
    if (curr_size < 0)
        curr_size = 0;

    /* long-message allgather or medium-size but non-power-of-two. use ring algorithm. */

    left = (comm_size + rank - 1) % comm_size;
    right = (rank + 1) % comm_size;

    j = rank;
    jnext = left;
    for (i = 1; i < comm_size; i++) {
        int left_count, right_count, left_disp, right_disp, rel_j, rel_jnext;

        rel_j = (j - root + comm_size) % comm_size;
        rel_jnext = (jnext - root + comm_size) % comm_size;
        left_count = MIN(scatter_size, (nbytes - rel_jnext * scatter_size));
        if (left_count < 0)
            left_count = 0;
        left_disp = rel_jnext * scatter_size;
        right_count = MIN(scatter_size, (nbytes - rel_j * scatter_size));
        if (right_count < 0)
            right_count = 0;
        right_disp = rel_j * scatter_size;
        entry_factory::create<send_entry>(
            sched, send_tmp_buf + right_disp, right_count, ccl_datatype_int8, right, comm);
        /* sendrecv, no barrier here */
        entry_factory::create<recv_entry>(
            sched, recv_tmp_buf + left_disp, left_count, ccl_datatype_int8, left, comm);
        sched->add_barrier();

        j = jnext;
        jnext = (comm_size + jnext - 1) % comm_size;
    }

    if (rank == root && send_buf != recv_buf) {
        entry_factory::create<copy_entry>(sched, send_tmp_buf, recv_tmp_buf, count, dtype);
        sched->add_barrier();
    }

fn_exit:
    return status;
}

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

ccl::status ccl_coll_build_topo_broadcast(ccl_sched* sched,
                                          ccl_buffer send_buf,
                                          ccl_buffer recv_buf,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          int root,
                                          ccl_comm* comm) {
    LOG_DEBUG("build topo broadcast");

    if (count == 0) {
        return ccl::status::success;
    }

    ccl_comm* node_comm = comm->get_node_comm().get();

    const std::vector<ze_handle_exchange_entry::mem_desc_t> buffers{
        { send_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 0
        { recv_buf.get_ptr(), ccl::ze::ipc_mem_type::memory }, // 1
    };
    LOG_DEBUG("broadcast send_buf = ", send_buf.get_ptr(), " and root = ", root);

    std::vector<ze_event_handle_t> wait_events;
    ze_event_handle_t out_event;

    sched->try_enable_ze_single_list();

    ccl::add_handle_exchange(sched, node_comm, wait_events, out_event, buffers);
    clear_and_push_back(wait_events, out_event);

    // inplace bcast
    if (send_buf == recv_buf) {
        // copy for remote ranks, don't need to copy
        // for root because send_buf == recv_buf
        if (comm->rank() != root) {
            auto entry =
                entry_factory::create<ze_copy_entry>(sched,
                                                     ccl_buffer(),
                                                     recv_buf,
                                                     count,
                                                     dtype,
                                                     copy_attr(root, 0, copy_direction::d2d),
                                                     wait_events);
            clear_and_push_back(wait_events, entry->entry_event);
        }
    }
    else { // out plcace bcast
        auto entry =
            entry_factory::create<ze_copy_entry>(sched,
                                                 comm->rank() != root ? ccl_buffer() : send_buf,
                                                 recv_buf,
                                                 count,
                                                 dtype,
                                                 copy_attr(root, 0, copy_direction::d2d),
                                                 wait_events);
        clear_and_push_back(wait_events, entry->entry_event);
    }

    ccl::add_comm_barrier(sched, node_comm, wait_events, out_event);

    entry_factory::create<ze_execute_cmdlists_on_init_entry>(sched);

    return ccl::status::success;
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

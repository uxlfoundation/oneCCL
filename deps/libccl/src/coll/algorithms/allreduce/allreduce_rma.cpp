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
#include "coll/algorithms/allreduce/allreduce_rma.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "exec/exec.hpp"

ccl::status rma_ring_allreduce_reset_sync_flag(const void* ctx) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    ar_handler->sync_flag = 0;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_reset_dst_ready_flag(const void* ctx) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    ar_handler->dst_ready_flag = 0;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_remote_sync_flag_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = &(ar_handler->remote_sync_flag_mr);
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_sync_flag_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = ar_handler->sync_flag_mr;
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_sync_flags_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = ar_handler->sync_flags_mr;
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_send_buf_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = ar_handler->send_buf_mr;
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_recv_buf_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = ar_handler->recv_buf_mr;
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_tmp_buf_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = ar_handler->tmp_buf_mr;
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_dst_ready_flag_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = ar_handler->dst_ready_flag_mr;
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_dst_ready_value_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = ar_handler->dst_ready_value_mr;
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_remote_dst_ready_flag_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = &(ar_handler->remote_dst_ready_flag_mr);
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_remote_rs_dst_buf_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = &(ar_handler->remote_rs_dst_buf_mr);
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status rma_ring_allreduce_get_remote_recv_buf_mr(const void* ctx, void* field_ptr) {
    ccl_rma_ring_allreduce_handler* ar_handler = (ccl_rma_ring_allreduce_handler*)ctx;
    atl_mr_t* mr = &(ar_handler->remote_recv_buf_mr);
    atl_mr_t** mr_ptr = (atl_mr_t**)field_ptr;
    *mr_ptr = mr;
    return ccl::status::success;
}

ccl::status ccl_coll_build_ring_rma_allreduce(ccl_sched* sched,
                                              ccl_buffer send_buf,
                                              ccl_buffer recv_buf,
                                              size_t count,
                                              const ccl_datatype& dtype,
                                              ccl::reduction op,
                                              ccl_comm* comm) {
    int inplace = (send_buf == recv_buf) ? 1 : 0;
    LOG_DEBUG("build ring rma allreduce (", (inplace) ? "in-place" : "out-of-place", ")");

    CCL_THROW_IF_NOT(sched && send_buf && recv_buf,
                     "incorrect values, sched ",
                     sched,
                     ", send ",
                     send_buf,
                     ", recv ",
                     recv_buf);

    ccl::status status = ccl::status::success;
    int comm_size, rank;
    size_t dtype_size = dtype.size();
    int idx = 0;
    ccl_buffer tmp_buf;
    comm_size = comm->size();
    rank = comm->rank();

    if (comm_size == 1) {
        if (!inplace) {
            entry_factory::create<copy_entry>(sched, send_buf, recv_buf, count, dtype);
            sched->add_barrier();
        }
        return ccl::status::success;
    }

    ccl_rma_ring_allreduce_handler* ar_handler =
        (ccl_rma_ring_allreduce_handler*)sched->alloc_buffer(sizeof(ccl_rma_ring_allreduce_handler))
            .get_ptr();
    ar_handler->sync_flags =
        (uint64_t*)sched->alloc_buffer(2 * comm_size * sizeof(uint64_t)).get_ptr();

    sched->set_entry_exec_mode(ccl_sched_entry_exec_once);

    entry_factory::create<register_entry>(
        sched,
        2 * comm_size * sizeof(uint64_t),
        ccl_buffer(ar_handler->sync_flags, 2 * comm_size * sizeof(uint64_t)),
        &ar_handler->sync_flags_mr,
        comm);
    entry_factory::create<register_entry>(
        sched,
        sizeof(uint64_t),
        ccl_buffer((void*)&ar_handler->sync_flag, sizeof(uint64_t)),
        &ar_handler->sync_flag_mr,
        comm);
    entry_factory::create<register_entry>(
        sched,
        sizeof(uint64_t),
        ccl_buffer((void*)&ar_handler->dst_ready_flag, sizeof(uint64_t)),
        &ar_handler->dst_ready_flag_mr,
        comm);
    entry_factory::create<register_entry>(
        sched,
        sizeof(uint64_t),
        ccl_buffer(&ar_handler->dst_ready_value, sizeof(uint64_t)),
        &ar_handler->dst_ready_value_mr,
        comm);

    if (inplace) {
        tmp_buf = sched->alloc_buffer(count * dtype_size);
        entry_factory::create<register_entry>(
            sched, count * dtype_size, tmp_buf, &ar_handler->tmp_buf_mr, comm);
    }
    else
        entry_factory::create<register_entry>(
            sched, count * dtype_size, send_buf, &ar_handler->send_buf_mr, comm);
    entry_factory::create<register_entry>(
        sched, count * dtype_size, recv_buf, &ar_handler->recv_buf_mr, comm);

    sched->set_entry_exec_mode(ccl_sched_entry_exec_regular);

    ar_handler->wait_dst = 1;
    for (idx = 0; idx < 2 * comm_size; idx++)
        ar_handler->sync_flags[idx] = idx + 1;
    ar_handler->dst_ready_flag = 0;
    ar_handler->src_peer = (comm_size + rank - 1) % comm_size;
    ar_handler->dst_peer = (comm_size + rank + 1) % comm_size;

    entry_factory::create<function_entry>(sched, rma_ring_allreduce_reset_sync_flag, ar_handler);
    sched->add_barrier();

    sched->set_entry_exec_mode(ccl_sched_entry_exec_once);

    send_entry* e{};
    if (inplace) {
        e = entry_factory::create<send_entry>(sched,
                                              ccl_buffer(&ar_handler->tmp_buf_mr, sizeof(atl_mr_t)),
                                              sizeof(atl_mr_t),
                                              ccl_datatype_int8,
                                              ar_handler->src_peer,
                                              comm);
        e->set_field_fn<ccl_sched_entry_field_buf>(rma_ring_allreduce_get_tmp_buf_mr, ar_handler);
    }
    else {
        e = entry_factory::create<send_entry>(
            sched,
            ccl_buffer(&ar_handler->recv_buf_mr, sizeof(atl_mr_t)),
            sizeof(atl_mr_t),
            ccl_datatype_int8,
            ar_handler->src_peer,
            comm);
        e->set_field_fn<ccl_sched_entry_field_buf>(rma_ring_allreduce_get_recv_buf_mr, ar_handler);
    }
    e = entry_factory::create<send_entry>(sched,
                                          ccl_buffer(&ar_handler->recv_buf_mr, sizeof(atl_mr_t)),
                                          sizeof(atl_mr_t),
                                          ccl_datatype_int8,
                                          ar_handler->src_peer,
                                          comm);
    e->set_field_fn<ccl_sched_entry_field_buf>(rma_ring_allreduce_get_recv_buf_mr, ar_handler);

    e = entry_factory::create<send_entry>(sched,
                                          ccl_buffer(&ar_handler->sync_flag_mr, sizeof(atl_mr_t)),
                                          sizeof(atl_mr_t),
                                          ccl_datatype_int8,
                                          ar_handler->src_peer,
                                          comm);
    e->set_field_fn<ccl_sched_entry_field_buf>(rma_ring_allreduce_get_sync_flag_mr, ar_handler);

    entry_factory::create<recv_entry>(
        sched,
        ccl_buffer(&ar_handler->remote_rs_dst_buf_mr, sizeof(atl_mr_t)),
        sizeof(atl_mr_t),
        ccl_datatype_int8,
        ar_handler->dst_peer,
        comm);
    entry_factory::create<recv_entry>(sched,
                                      ccl_buffer(&ar_handler->remote_recv_buf_mr, sizeof(atl_mr_t)),
                                      sizeof(atl_mr_t),
                                      ccl_datatype_int8,
                                      ar_handler->dst_peer,
                                      comm);
    entry_factory::create<recv_entry>(
        sched,
        ccl_buffer(&ar_handler->remote_sync_flag_mr, sizeof(atl_mr_t)),
        sizeof(atl_mr_t),
        ccl_datatype_int8,
        ar_handler->dst_peer,
        comm);

    if (ar_handler->wait_dst) {
        e = entry_factory::create<send_entry>(
            sched,
            ccl_buffer(ar_handler->dst_ready_flag_mr, sizeof(atl_mr_t)),
            sizeof(atl_mr_t),
            ccl_datatype_int8,
            ar_handler->dst_peer,
            comm);
        e->set_field_fn<ccl_sched_entry_field_buf>(rma_ring_allreduce_get_dst_ready_flag_mr,
                                                   ar_handler);
        entry_factory::create<recv_entry>(
            sched,
            ccl_buffer(&ar_handler->remote_dst_ready_flag_mr, sizeof(atl_mr_t)),
            sizeof(atl_mr_t),
            ccl_datatype_int8,
            ar_handler->src_peer,
            comm);
    }
    sched->add_barrier();

    sched->set_entry_exec_mode(ccl_sched_entry_exec_regular);

    if (ar_handler->wait_dst) {
        /* let src side to know that this rank (i.e. dst for src rank) is ready for write ops */
        ar_handler->dst_ready_value = 1;
        write_entry* entry = entry_factory::create<write_entry>(
            sched,
            ccl_buffer(&ar_handler->dst_ready_value, sizeof(uint64_t)),
            (atl_mr_t*)nullptr, /* src_mr */
            sizeof(uint64_t),
            ccl_datatype_int8,
            ar_handler->src_peer,
            (atl_mr_t*)nullptr, /* dst_mr */
            0 /* dst_buf_offset */,
            comm);
        entry->set_field_fn<ccl_sched_entry_field_src_mr>(rma_ring_allreduce_get_dst_ready_value_mr,
                                                          ar_handler);
        entry->set_field_fn<ccl_sched_entry_field_dst_mr>(
            rma_ring_allreduce_get_remote_dst_ready_flag_mr, ar_handler);

        /* wait when dst side will be ready for write ops */
        entry_factory::create<wait_value_entry>(
            sched, &(ar_handler->dst_ready_flag), 1, ccl_condition_equal);

        /* reset dst_ready_flag for next allreduce call */
        entry_factory::create<function_entry>(
            sched, rma_ring_allreduce_reset_dst_ready_flag, ar_handler);
    }

    int block_idx = rank;
    size_t main_block_count = count / comm_size;
    size_t buf_offset;

    /* reduce-scatter */
    for (idx = 0; idx < (comm_size - 1); idx++) {
        size_t block_count = main_block_count;
        if (block_idx == (comm_size - 1))
            block_count += count % comm_size;
        buf_offset = main_block_count * dtype_size * block_idx;

        ccl_buffer src_buf;
        if (inplace)
            src_buf = recv_buf;
        else
            src_buf = (idx == 0) ? send_buf : recv_buf;

        write_entry* entry = entry_factory::create<write_entry>(sched,
                                                                src_buf + buf_offset,
                                                                (atl_mr_t*)nullptr, /* src_mr */
                                                                block_count,
                                                                dtype,
                                                                ar_handler->dst_peer,
                                                                (atl_mr_t*)nullptr, /* dst_mr */
                                                                buf_offset,
                                                                comm);
        entry->set_field_fn<ccl_sched_entry_field_src_mr>(
            (inplace) ? rma_ring_allreduce_get_recv_buf_mr
                      : ((idx == 0) ? rma_ring_allreduce_get_send_buf_mr
                                    : rma_ring_allreduce_get_recv_buf_mr),
            ar_handler);
        entry->set_field_fn<ccl_sched_entry_field_dst_mr>(
            rma_ring_allreduce_get_remote_rs_dst_buf_mr, ar_handler);

        if (block_count * dtype.size() > atl_base_comm::attr.out.max_order_waw_size)
            sched->add_barrier();

        entry = entry_factory::create<write_entry>(
            sched,
            ccl_buffer(&ar_handler->sync_flags[idx], sizeof(uint64_t)),
            (atl_mr_t*)nullptr, /* src_mr */
            sizeof(uint64_t),
            ccl_datatype_int8,
            ar_handler->dst_peer,
            (atl_mr_t*)nullptr, /* dst_mr */
            0 /* dst_buf_offset */,
            comm);
        entry->set_field_fn<ccl_sched_entry_field_src_mr>(rma_ring_allreduce_get_sync_flags_mr,
                                                          ar_handler);
        entry->set_field_fn<ccl_sched_entry_field_dst_mr>(
            rma_ring_allreduce_get_remote_sync_flag_mr, ar_handler);

        block_idx = (block_idx + comm_size - 1) % comm_size;
        block_count = main_block_count;
        if (block_idx == (comm_size - 1))
            block_count += count % comm_size;
        buf_offset = main_block_count * dtype_size * block_idx;

        entry_factory::create<wait_value_entry>(
            sched, &(ar_handler->sync_flag), (idx + 1), ccl_condition_greater_or_equal);

        ccl_buffer reduce_in_buf = (inplace) ? tmp_buf : send_buf;
        ccl_buffer reduce_inout_buf = recv_buf;
        entry_factory::create<reduce_local_entry>(sched,
                                                  reduce_in_buf + buf_offset,
                                                  block_count,
                                                  reduce_inout_buf + buf_offset,
                                                  nullptr,
                                                  dtype,
                                                  op);
    }

    /* allgather */
    size_t flag_idx_offset = (comm_size - 1);
    for (idx = 0; idx < (comm_size - 1); idx++) {
        size_t block_count = main_block_count;
        if (block_idx == (comm_size - 1))
            block_count += count % comm_size;
        buf_offset = main_block_count * dtype_size * block_idx;

        ccl_buffer src_buf = recv_buf;
        write_entry* entry = entry_factory::create<write_entry>(sched,
                                                                src_buf + buf_offset,
                                                                (atl_mr_t*)nullptr, /* src_mr */
                                                                block_count,
                                                                dtype,
                                                                ar_handler->dst_peer,
                                                                (atl_mr_t*)nullptr, /* dst_mr */
                                                                buf_offset,
                                                                comm);
        entry->set_field_fn<ccl_sched_entry_field_src_mr>(rma_ring_allreduce_get_recv_buf_mr,
                                                          ar_handler);
        entry->set_field_fn<ccl_sched_entry_field_dst_mr>(rma_ring_allreduce_get_remote_recv_buf_mr,
                                                          ar_handler);

        if (block_count * dtype.size() > atl_base_comm::attr.out.max_order_waw_size)
            sched->add_barrier();

        entry = entry_factory::create<write_entry>(
            sched,
            ccl_buffer(&ar_handler->sync_flags[flag_idx_offset + idx], sizeof(uint64_t)),
            (atl_mr_t*)nullptr, /* src_mr */
            sizeof(uint64_t),
            ccl_datatype_int8,
            ar_handler->dst_peer,
            (atl_mr_t*)nullptr, /* dst_mr */
            0 /* dst_buf_offset */,
            comm);
        entry->set_field_fn<ccl_sched_entry_field_src_mr>(rma_ring_allreduce_get_sync_flags_mr,
                                                          ar_handler);
        entry->set_field_fn<ccl_sched_entry_field_dst_mr>(
            rma_ring_allreduce_get_remote_sync_flag_mr, ar_handler);

        block_idx = (block_idx + comm_size - 1) % comm_size;

        entry_factory::create<wait_value_entry>(sched,
                                                &(ar_handler->sync_flag),
                                                (flag_idx_offset + idx + 1),
                                                ccl_condition_greater_or_equal);
    }

    return status;
}

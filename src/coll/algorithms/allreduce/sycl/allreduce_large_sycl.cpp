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
#include "coll/algorithms/allreduce/sycl/allreduce_large_sycl.hpp"

#define MAX_RANK 16

void *allreduce_large_buffer = NULL;
void *allreduce_large_buffers[MAX_RANK];
void *allreduce_large_sync_buffer[MAX_RANK];
size_t allreduce_large_offsets[MAX_RANK];
ze_ipc_mem_handle_t allreduce_large_ipc_handle[MAX_RANK];
int allreduce_large_buffer_index = 0;

#define ALLREDUCE_LARGE_API_DECL(TYPE) \
    void init_allreduce_large_##TYPE(ccl::datatype dtype, \
                                     sycl::queue &queue, \
                                     ccl_comm *comm, \
                                     ccl_stream *stream, \
                                     uint32_t rank_in, \
                                     uint32_t world_in); \
    ccl::event run_allreduce_large_##TYPE(ccl::datatype dtype, \
                                          sycl::queue &queue, \
                                          const void *in_buf, \
                                          void *out_buf, \
                                          size_t count, \
                                          ccl::reduction reduction, \
                                          const ccl::vector_class<ccl::event> &deps, \
                                          bool &done)

ALLREDUCE_LARGE_API_DECL(fp16);
ALLREDUCE_LARGE_API_DECL(bf16);
ALLREDUCE_LARGE_API_DECL(fp32);
ALLREDUCE_LARGE_API_DECL(int32);

#define SWITCH_INIT_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        init_allreduce_large_##TYPE(dtype, queue, comm, stream, rank_in, world_in); \
        break;

void init_allreduce_large(ccl::datatype dtype,
                          sycl::queue &queue,
                          ccl_comm *comm,
                          ccl_stream *stream,
                          uint32_t rank_in,
                          uint32_t world_in) {
    switch (dtype) {
        SWITCH_INIT_TYPE(fp16, ccl::datatype::float16)
        SWITCH_INIT_TYPE(bf16, ccl::datatype::bfloat16)
        SWITCH_INIT_TYPE(fp32, ccl::datatype::float32)
        SWITCH_INIT_TYPE(int32, ccl::datatype::int32)
        default: CCL_THROW("unsupported datatype for allreduce"); assert(0);
    }
}

#define SWITCH_RUN_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        e = run_allreduce_large_##TYPE( \
            dtype, queue, in_buf, out_buf, count, reduction, deps, done); \
        break;

ccl::event run_allreduce_large(ccl::datatype dtype,
                               sycl::queue &queue,
                               const void *in_buf,
                               void *out_buf,
                               size_t count,
                               ccl::reduction reduction,
                               const ccl::vector_class<ccl::event> &deps,
                               bool &done) {
    ccl::event e;
    switch (dtype) {
        SWITCH_RUN_TYPE(fp16, ccl::datatype::float16)
        SWITCH_RUN_TYPE(bf16, ccl::datatype::bfloat16)
        SWITCH_RUN_TYPE(fp32, ccl::datatype::float32)
        SWITCH_RUN_TYPE(int32, ccl::datatype::int32)
        default: CCL_THROW("unsupported datatype for allreduce"); assert(0);
    }
    return e;
}

#include "coll/algorithms/allreduce/sycl/allreduce_large_sycl_impl.hpp"

ccl::event allreduce_large(const void *send_buf,
                           void *recv_buf,
                           size_t count,
                           ccl::datatype dtype,
                           ccl::reduction reduction,
                           ccl_comm *comm,
                           ccl_stream *global_stream,
                           const ccl::vector_class<ccl::event> &deps) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("invoking MT allreduce_large");
        ccl::global_data::env().sycl_allreduce_tmp_buf = 1;
        CCL_THROW_IF_NOT(ccl::global_data::env().sycl_allreduce_tmp_buf == 1,
                         "MT large kernel doesnt support disabled tmp buf");
    }
    else {
        LOG_DEBUG("invoking allreduce_large");
    }

    sycl_ptrs_type sycl_ptrs;
    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();

    // use full vector (>= 8 bytes) if buffers are 4 byte aligned
    // we dont have to take into account of the count while calculating alignment,
    // since we divide count such that all ranks have aligned addresses
    bool use_full_vector = can_use_full_vector(send_buf, recv_buf, 0, 0);

    const bool is_use_tmp = ccl::global_data::env().sycl_allreduce_tmp_buf;

    if (is_use_tmp) {
        // global rank of pair_comm neighbors should be adjacent for using tmp buffer
        // i.e. the ranks can be 2,3 or 3,2 but not 1,3
        CCL_ASSERT(pair_comm->size() <= 2);
        if (pair_comm->size() == 2) {
            const int rank_diff = pair_comm->get_node_rank(0) - pair_comm->get_node_rank(1);
            CCL_THROW_IF_NOT(
                abs(rank_diff) == 1,
                "communicator rank reordering not allowed with tmp buffer, set CCL_SYCL_ALLREDUCE_TMP_BUF=0 and CCL_SYCL_AUTO_USE_TMP_BUF=0");
        }
    }

    if (!is_use_tmp) {
        std::vector<void *> ptrs{ (void *)send_buf, recv_buf }; // index 0 and 1
        // UMF: umf doesn't support user's allocation through the sycl yet.
        //      Once it is supported, we can use  UMF as exchnage mechanism for allreduce_large
        auto [sched, exchange_entry] = do_ipc_exchange(comm, global_stream, ptrs);

        sycl_ptrs.xelink_ptrs_rd =
            get_ipc_ptrs<void, MAX_GPUS>(even_comm, 0, (void *)send_buf, sched);
        sycl_ptrs.xelink_ptrs_wr =
            get_ipc_ptrs<void, MAX_GPUS>(even_comm, 1, (void *)recv_buf, sched);
        // use full vector (>= 8 bytes) if remote buffers are 4 byte aligned
        use_full_vector =
            use_full_vector &&
            all_aligned(sycl_ptrs.xelink_ptrs_rd.data(), even_comm->size(), 0, 0, 4) &&
            all_aligned(sycl_ptrs.xelink_ptrs_wr.data(), even_comm->size(), 0, 0, 4);

        if (pair_comm->size() > 1) {
            assert(pair_comm->size() == MAX_TILES);
            int peer_pair_rank = pair_comm->rank() ? 0 : 1;
            sycl_ptrs.mdfi_ptr_rd = get_ipc_ptrs<void, MAX_TILES>(
                pair_comm, 0, (void *)send_buf, sched)[peer_pair_rank];
            sycl_ptrs.mdfi_ptr_wr = get_ipc_ptrs<void, MAX_TILES>(
                pair_comm, 1, (void *)recv_buf, sched)[peer_pair_rank];
            use_full_vector = use_full_vector && all_aligned(&sycl_ptrs.mdfi_ptr_rd, 1, 0, 0, 4) &&
                              all_aligned(&sycl_ptrs.mdfi_ptr_wr, 1, 0, 0, 4);
        }

        delete exchange_entry;
        delete sched;

        coll_init(comm, global_stream);
    }
    else {
        if (comm->is_multi_thread_instance() == true) {
            coll_initExt(comm, ccl::global_data::get().shared_data->hash_table, global_stream);
        }
        else {
            coll_init(comm, global_stream);
        }

        // 0 index is used for tmp work buffer and
        // 1 index is used to copy input data
        // 2 index is used to copy output data
        sycl_ptrs.xelink_ptrs_rd = get_remote_even_tmp_buf(1, comm);
        sycl_ptrs.xelink_ptrs_wr = get_remote_even_tmp_buf(2, comm);
        if (pair_comm->size() > 1) {
            assert(pair_comm->size() == MAX_TILES);
            int peer_pair_rank = pair_comm->rank() ? 0 : 1;
            sycl_ptrs.mdfi_ptr_rd = get_remote_pair_tmp_buf(1, comm)[peer_pair_rank];
            sycl_ptrs.mdfi_ptr_wr = get_remote_pair_tmp_buf(2, comm)[peer_pair_rank];
        }
    }

    auto lambda = [&]<typename T, int NE, int NP>() {
        if (use_full_vector) {
            return allreduce_large_impl<T, NE, NP, true>(
                send_buf, recv_buf, count, dtype, reduction, comm, global_stream, sycl_ptrs, deps);
        }
        else {
            return allreduce_large_impl<T, NE, NP, false>(
                send_buf, recv_buf, count, dtype, reduction, comm, global_stream, sycl_ptrs, deps);
        }
    };

    return invoke_collective(lambda, comm, dtype);
}

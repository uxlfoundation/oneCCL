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
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl.hpp"

#define REDUCE_SCATTER_LARGE_API_DECL(TYPE) \
    void init_reduce_scatter_large_##TYPE(ccl::datatype dtype, \
                                          sycl::queue &queue, \
                                          ccl_comm *comm, \
                                          ccl_stream *stream, \
                                          uint32_t rank_in, \
                                          uint32_t world_in); \
    ccl::event run_reduce_scatter_large_##TYPE(ccl::datatype dtype, \
                                               sycl::queue &queue, \
                                               const void *send_buf, \
                                               void *recv_buf, \
                                               size_t recv_count, \
                                               ccl::reduction reduction, \
                                               const ccl::vector_class<ccl::event> &deps, \
                                               bool &done);

REDUCE_SCATTER_LARGE_API_DECL(fp16);
REDUCE_SCATTER_LARGE_API_DECL(bf16);
REDUCE_SCATTER_LARGE_API_DECL(fp32);
REDUCE_SCATTER_LARGE_API_DECL(int32);

#define SWITCH_INIT_TYPE(TYPE, ccl_type) \
    case ccl_type: init_reduce_scatter_large_##TYPE(dtype, queue, comm, stream, rank_in, world_in); break;

void init_reduce_scatter_large(ccl::datatype dtype,
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
        default: assert(0);
    }
}

#define SWITCH_RUN_TYPE(TYPE, ccl_type) \
    case ccl_type: \
        e = run_reduce_scatter_large_##TYPE(dtype, queue, send_buf, recv_buf, recv_count, reduction, deps, done); \
        break;

ccl::event run_reduce_scatter_large(ccl::datatype dtype,
                                    sycl::queue &queue,
                                    const void *send_buf,
                                    void *recv_buf,
                                    size_t recv_count,
                                    ccl::reduction reduction,
                                    const ccl::vector_class<ccl::event> &deps,
                                    bool &done) {
    ccl::event e;
    switch (dtype) {
        SWITCH_RUN_TYPE(fp16, ccl::datatype::float16)
        SWITCH_RUN_TYPE(bf16, ccl::datatype::bfloat16)
        SWITCH_RUN_TYPE(fp32, ccl::datatype::float32)
        SWITCH_RUN_TYPE(int32, ccl::datatype::int32)
        default: assert(0);
    }
    return e;
}

#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/algorithms/reduce_scatter/sycl/reduce_scatter_large_sycl_impl.hpp"

ccl::event reduce_scatter_large(const void *send_buf,
                                void *recv_buf,
                                size_t recv_count,
                                ccl::datatype dtype,
                                ccl::reduction reduction,
                                ccl_comm *comm,
                                ccl_stream *global_stream,
                                const ccl::vector_class<ccl::event> &deps,
                                sycl_coll_scaleup_attr coll_attr) {
    if (comm->is_multi_thread_instance() == true) {
        LOG_DEBUG("invoking MT reduce_scatter_large");
        ccl::global_data::env().sycl_reduce_scatter_tmp_buf = 1;
        CCL_THROW_IF_NOT(ccl::global_data::env().sycl_reduce_scatter_tmp_buf == 1,
                         "MT large kernel doesnt support disabled tmp buf");
    }
    else {
        LOG_DEBUG("invoking reduce_scatter_large");
    }

    sycl_ptrs_type sycl_ptrs;
    std::shared_ptr<ccl_comm> pair_comm = comm->get_pair_comm();
    std::shared_ptr<ccl_comm> even_comm = comm->get_even_comm();

    const size_t dsize = ccl::global_data::get().dtypes->get(dtype).size();
    // use full vector (>= 8 bytes) if buffers and data size are 4 byte aligned
    bool use_full_vector = can_use_full_vector(send_buf, recv_buf, recv_count, dsize);
    // TODO : generalize constraints for different hardware.
    // kernels with remote access is best performant at 64 bytes alignment (sycl_kernels_line_size/2) on PVC
    const size_t align_size = ccl::global_data::env().sycl_kernels_line_size / 2;
    const bool is_aligned = (recv_count * dsize) % align_size == 0;
    // use tmp buf for types < 4 byte size with odd count or non 4 byte aligned data
    // use tmp buf when data count bytes is not 64 byte aligned
    // since tmp buf version performs better in that case
    bool is_tmp_used = ccl::global_data::env().sycl_reduce_scatter_tmp_buf ||
                       ((!use_full_vector || !is_aligned) && ccl::global_data::env().sycl_auto_use_tmp_buf);
    if (even_comm->size() == 1) {
        is_tmp_used = ccl::global_data::env().sycl_reduce_scatter_tmp_buf;
    }
    // scale-out step can force to use "tmp buffer version" for async support purposes
    is_tmp_used |= (coll_attr.force_use_tmp && ccl::global_data::env().sycl_force_use_tmp_buf_scaleout);

    if (is_tmp_used) {
        // global rank of pair_comm neighbors should be adjacent for using tmp buffer
        // i.e. the ranks can be 2,3 or 3,2 but not 1,3
        CCL_ASSERT(pair_comm->size() <= 2);
        if (pair_comm->size() == 2) {
            const int rank_diff = pair_comm->get_node_rank(0) - pair_comm->get_node_rank(1);
            CCL_THROW_IF_NOT(
                abs(rank_diff) == 1,
                "communicator rank reordering not allowed with tmp buffer, set CCL_SYCL_REDUCE_SCATTER_TMP_BUF=0 and CCL_SYCL_AUTO_USE_TMP_BUF=0");
        }
    }

    if (!is_tmp_used) {
        std::vector<void *> ptrs{ (void *)send_buf, recv_buf }; // index 0 and 1
        auto [sched, exchange_entry] = do_ipc_exchange(comm, global_stream, ptrs);

        sycl_ptrs.xelink_ptrs_rd = get_ipc_ptrs<void, MAX_GPUS>(even_comm, 0, (void *)send_buf, sched);
        // use full vector (>= 8 bytes) if remote buffers and data size are 4 byte aligned
        use_full_vector = use_full_vector &&
                          all_aligned(sycl_ptrs.xelink_ptrs_rd.data(), even_comm->size(), recv_count, dsize, 4);

        if (pair_comm->size() > 1) {
            assert(pair_comm->size() == MAX_TILES);
            int peer_pair_rank = pair_comm->rank() ? 0 : 1;
            sycl_ptrs.mdfi_ptr_rd =
                get_ipc_ptrs<void, MAX_TILES>(pair_comm, 0, (void *)send_buf, sched)[peer_pair_rank];
            use_full_vector = use_full_vector && all_aligned(&sycl_ptrs.mdfi_ptr_rd, 1, recv_count, dsize, 4);
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
        sycl_ptrs.xelink_ptrs_rd = get_remote_even_tmp_buf(1, comm);
        if (pair_comm->size() > 1) {
            assert(pair_comm->size() == MAX_TILES);
            int peer_pair_rank = pair_comm->rank() ? 0 : 1;
            sycl_ptrs.mdfi_ptr_rd = get_remote_pair_tmp_buf(1, comm)[peer_pair_rank];
        }
    }

    auto lambda = [&]<typename T, int NE, int NP>() {
        if (use_full_vector) {
            return reduce_scatter_large_impl<T, NE, NP, true>(send_buf,
                                                              recv_buf,
                                                              recv_count,
                                                              dtype,
                                                              reduction,
                                                              comm,
                                                              global_stream,
                                                              sycl_ptrs,
                                                              deps,
                                                              is_tmp_used);
        }
        else {
            return reduce_scatter_large_impl<T, NE, NP, false>(send_buf,
                                                               recv_buf,
                                                               recv_count,
                                                               dtype,
                                                               reduction,
                                                               comm,
                                                               global_stream,
                                                               sycl_ptrs,
                                                               deps,
                                                               is_tmp_used);
        }
    };

    return invoke_collective(lambda, comm, dtype);
}

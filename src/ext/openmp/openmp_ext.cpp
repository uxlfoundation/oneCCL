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
#if defined(CCL_ENABLE_MPI) && defined(CCL_ENABLE_OMP)

#include "coll/coll_param.hpp"
#include "oneapi/ccl.hpp"
#include "oneapi/ccl/types.hpp"
#include "atl/mpi/atl_mpi.hpp"
#include "common/api_wrapper/mpi_api_wrapper.hpp"
#include "comm/comm.hpp"
#include "coll/selection/selector.hpp"
#include "coll/selection/selector_impl.hpp"
#include "coll/coll_util.hpp"
#include <omp.h>

#define CCL_MAX_OPENMP_THREADS 64

static MPI_Datatype ccl2mpi_dtype(ccl::datatype dtype) {
    switch (dtype) {
        case ccl::datatype::int8: return MPI_CHAR;
        case ccl::datatype::uint8: return MPI_UNSIGNED_CHAR;
        case ccl::datatype::int16: return MPI_INT16_T;
        case ccl::datatype::uint16: return MPI_UINT16_T;
        case ccl::datatype::int32: return MPI_INT;
        case ccl::datatype::uint32: return MPI_UINT32_T;
        case ccl::datatype::int64: return MPI_LONG_LONG;
        case ccl::datatype::uint64: return MPI_UNSIGNED_LONG_LONG;
        case ccl::datatype::float16: return MPIX_C_FLOAT16;
        case ccl::datatype::float32: return MPI_FLOAT;
        case ccl::datatype::float64: return MPI_DOUBLE;
        case ccl::datatype::bfloat16: return MPIX_C_BF16;
        default: CCL_THROW("unsupported datatype: ", dtype);
    }
}

static MPI_Op ccl2mpi_op(ccl::reduction rtype) {
    switch (rtype) {
        case ccl::reduction::sum: return MPI_SUM;
        case ccl::reduction::prod: return MPI_PROD;
        case ccl::reduction::min: return MPI_MIN;
        case ccl::reduction::max: return MPI_MAX;
        default: CCL_THROW("unknown reduction type: ", static_cast<int>(rtype));
    }
}

static MPI_Comm comm_array[CCL_MAX_OPENMP_THREADS];
static MPI_Comm static_comm = MPI_COMM_NULL;
static int static_num_segs = 0;
static ccl_selection_table_t<int> allreduce_selection_table;

void init(const std::string& omp_allreduce_num_threads) {
    fill_table_from_str(omp_allreduce_num_threads, allreduce_selection_table);
}

void allreduce_impl(const void* send_buf,
                    void* recv_buf,
                    size_t count,
                    ccl::datatype dtype,
                    ccl::reduction reduction,
                    const ccl_coll_attr& attr,
                    ccl_comm* comm,
                    const ccl_stream* stream,
                    const std::vector<ccl::event>& deps) {
    if (count == 0) {
        return;
    }
    MPI_Datatype mpi_dtype = ccl2mpi_dtype(dtype);
    MPI_Op mpi_op = ccl2mpi_op(reduction);
    int type_size = 0;
    ccl::mpi_lib_ops_t local_mpi_lib_ops = ccl::get_mpi_lib_ops();
    local_mpi_lib_ops.MPI_Type_size_ptr(mpi_dtype, &type_size);
    size_t message_size = count * type_size;
    // get number of threads from the table based on message size
    // set num_segs to min(num_threads from the table, omp_get_max_threads, CCL_MAX_OPENMP_THREADS, count)
    int num_segs = ccl_algorithm_selector_base<int>::get_value_from_table(
        message_size, allreduce_selection_table);
    if (num_segs > omp_get_max_threads()) {
        num_segs = omp_get_max_threads();
    }
    if (num_segs > CCL_MAX_OPENMP_THREADS) {
        num_segs = CCL_MAX_OPENMP_THREADS;
    }
    if (static_cast<size_t>(num_segs) > count) {
        num_segs = count;
    }
    int seg_count = count / num_segs;
    int seg_count_max = seg_count + count % num_segs;
    MPI_Comm mpi_comm = comm->get_atl_comm()->get_mpi_comm();
    if (static_comm != mpi_comm || static_num_segs < num_segs) {
        // cannot reuse comm_array, reset it
        // only need to reset it if communicator is different or more num_segs are requried
        for (int i = 0; i < static_num_segs; i++) {
            local_mpi_lib_ops.MPI_Comm_free_ptr(&comm_array[i]);
        }
        for (int i = 0; i < num_segs; i++) {
            local_mpi_lib_ops.MPI_Comm_dup_ptr(mpi_comm, &comm_array[i]);
        }
        static_comm = mpi_comm;
        static_num_segs = num_segs;
    }
#pragma omp parallel num_threads(num_segs)
    {
        int tid = omp_get_thread_num();
        int local_seg_count = (tid == num_segs - 1) ? seg_count_max : seg_count;
        void* local_send_buf = (void*)((char*)send_buf + tid * seg_count * type_size);
        void* local_recv_buf = (void*)((char*)recv_buf + tid * seg_count * type_size);
        local_mpi_lib_ops.MPI_Allreduce_ptr(
            (local_send_buf && (local_send_buf == local_recv_buf)) ? MPI_IN_PLACE : local_send_buf,
            local_recv_buf,
            local_seg_count,
            mpi_dtype,
            mpi_op,
            comm_array[tid]);
    }
    return;
}

void allgatherv_impl(const void* send_buf,
                     size_t send_len,
                     void* recv_buf,
                     const size_t* recv_lens,
                     const size_t* offsets,
                     const ccl_coll_attr& attr,
                     ccl_comm* comm,
                     const ccl_stream* stream,
                     const std::vector<ccl::event>& deps) {
    ccl::mpi_lib_ops_t local_mpi_lib_ops = ccl::get_mpi_lib_ops();
    MPI_Comm mpi_comm = comm->get_atl_comm()->get_mpi_comm();
    int comm_size, rank;
    local_mpi_lib_ops.MPI_Comm_size_ptr(mpi_comm, &comm_size);
    local_mpi_lib_ops.MPI_Comm_rank_ptr(mpi_comm, &rank);
    // set up recv_lens_size_t, recv_conv_lens and recv_conv_offsets
    std::vector<size_t> recv_lens_size_t(comm_size, 0);
    std::vector<Compat_MPI_Count_t> recv_conv_lens(comm_size);
    std::vector<Compat_MPI_Aint_t> recv_conv_offsets(comm_size);
    for (int i = 0; i < comm_size; ++i) {
        recv_lens_size_t[i] = recv_lens[i];
        recv_conv_lens[i] = static_cast<Compat_MPI_Count_t>(recv_lens[i]);
        recv_conv_offsets[i] = static_cast<Compat_MPI_Aint_t>(offsets[i]);
    }
    bool inplace = ccl::is_allgatherv_inplace(send_buf,
                                              send_len,
                                              recv_buf,
                                              recv_lens_size_t.data(),
                                              1 /*dtype_size*/, // size of MPI_CHAR dtype is 1
                                              rank,
                                              comm_size);

    // currently, only use one thread
    local_mpi_lib_ops.MPI_Allgatherv_c_ptr(inplace ? MPI_IN_PLACE : send_buf,
                                           send_len,
                                           MPI_CHAR,
                                           recv_buf,
                                           recv_conv_lens.data(),
                                           recv_conv_offsets.data(),
                                           MPI_CHAR,
                                           mpi_comm);
}

extern "C" CCL_API void* ccl_openmp_allreduce() {
    return reinterpret_cast<void*>(&allreduce_impl);
}

extern "C" CCL_API void* ccl_openmp_allgatherv() {
    return reinterpret_cast<void*>(&allgatherv_impl);
}

extern "C" CCL_API void* ccl_openmp_thread_num() {
    return reinterpret_cast<void*>(&omp_get_max_threads);
}

extern "C" CCL_API void* ccl_openmp_init() {
    return reinterpret_cast<void*>(&init);
}

#endif // CCL_ENABLE_MPI && CCL_ENABLE_OMP

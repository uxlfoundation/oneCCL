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

#include "oneapi/ccl/config.h"

#if defined(CCL_ENABLE_MPI)

#include <mpi.h>

namespace ccl {

typedef struct mpi_lib_ops {
    decltype(MPI_Allgather) *MPI_Allgather_ptr;
    decltype(MPI_Allgather_c) *MPI_Allgather_c_ptr;
    decltype(MPI_Allgatherv) *MPI_Allgatherv_ptr;
    decltype(MPI_Allgatherv_c) *MPI_Allgatherv_c_ptr;
    decltype(MPI_Allreduce) *MPI_Allreduce_ptr;
    decltype(MPI_Allreduce_c) *MPI_Allreduce_c_ptr;
    decltype(MPI_Alltoall) *MPI_Alltoall_ptr;
    decltype(MPI_Alltoall_c) *MPI_Alltoall_c_ptr;
    decltype(MPI_Alltoallv) *MPI_Alltoallv_ptr;
    decltype(MPI_Alltoallv_c) *MPI_Alltoallv_c_ptr;
    decltype(MPI_Barrier) *MPI_Barrier_ptr;
    decltype(MPI_Bcast) *MPI_Bcast_ptr;
    decltype(MPI_Bcast_c) *MPI_Bcast_c_ptr;
    decltype(MPI_Cancel) *MPI_Cancel_ptr;
    decltype(MPI_Comm_create_group) *MPI_Comm_create_group_ptr;
    decltype(MPI_Comm_free) *MPI_Comm_free_ptr;
    decltype(MPI_Comm_get_attr) *MPI_Comm_get_attr_ptr;
    decltype(MPI_Comm_get_info) *MPI_Comm_get_info_ptr;
    decltype(MPI_Comm_group) *MPI_Comm_group_ptr;
    decltype(MPI_Comm_rank) *MPI_Comm_rank_ptr;
    decltype(MPI_Comm_set_info) *MPI_Comm_set_info_ptr;
    decltype(MPI_Comm_size) *MPI_Comm_size_ptr;
    decltype(MPI_Comm_split) *MPI_Comm_split_ptr;
    decltype(MPI_Comm_split_type) *MPI_Comm_split_type_ptr;
    decltype(MPI_Error_string) *MPI_Error_string_ptr;
    decltype(MPI_Finalize) *MPI_Finalize_ptr;
    decltype(MPI_Finalized) *MPI_Finalized_ptr;
    decltype(MPI_Get_count) *MPI_Get_count_ptr;
    decltype(MPI_Get_library_version) *MPI_Get_library_version_ptr;
    decltype(MPI_Group_incl) *MPI_Group_incl_ptr;
    decltype(MPI_Iallgather) *MPI_Iallgather_ptr;
    decltype(MPI_Iallgather_c) *MPI_Iallgather_c_ptr;
    decltype(MPI_Iallgatherv) *MPI_Iallgatherv_ptr;
    decltype(MPI_Iallgatherv_c) *MPI_Iallgatherv_c_ptr;
    decltype(MPI_Iallreduce) *MPI_Iallreduce_ptr;
    decltype(MPI_Iallreduce_c) *MPI_Iallreduce_c_ptr;
    decltype(MPI_Ialltoall) *MPI_Ialltoall_ptr;
    decltype(MPI_Ialltoall_c) *MPI_Ialltoall_c_ptr;
    decltype(MPI_Ialltoallv) *MPI_Ialltoallv_ptr;
    decltype(MPI_Ialltoallv_c) *MPI_Ialltoallv_c_ptr;
    decltype(MPI_Ibarrier) *MPI_Ibarrier_ptr;
    decltype(MPI_Ibcast) *MPI_Ibcast_ptr;
    decltype(MPI_Ibcast_c) *MPI_Ibcast_c_ptr;
    decltype(MPI_Info_create) *MPI_Info_create_ptr;
    decltype(MPI_Info_free) *MPI_Info_free_ptr;
    decltype(MPI_Info_get) *MPI_Info_get_ptr;
    decltype(MPI_Info_set) *MPI_Info_set_ptr;
    decltype(MPI_Initialized) *MPI_Initialized_ptr;
    decltype(MPI_Init) *MPI_Init_ptr;
    decltype(MPI_Init_thread) *MPI_Init_thread_ptr;
    decltype(MPI_Iprobe) *MPI_Iprobe_ptr;
    decltype(MPI_Irecv) *MPI_Irecv_ptr;
    decltype(MPI_Irecv_c) *MPI_Irecv_c_ptr;
    decltype(MPI_Ireduce) *MPI_Ireduce_ptr;
    decltype(MPI_Ireduce_c) *MPI_Ireduce_c_ptr;
    decltype(MPI_Ireduce_scatter_block) *MPI_Ireduce_scatter_block_ptr;
    decltype(MPI_Ireduce_scatter_block_c) *MPI_Ireduce_scatter_block_c_ptr;
    decltype(MPI_Isend) *MPI_Isend_ptr;
    decltype(MPI_Isend_c) *MPI_Isend_c_ptr;
    decltype(MPI_Op_create) *MPI_Op_create_ptr;
    decltype(MPI_Op_free) *MPI_Op_free_ptr;
    decltype(MPI_Query_thread) *MPI_Query_thread_ptr;
    decltype(MPI_Reduce) *MPI_Reduce_ptr;
    decltype(MPI_Reduce_c) *MPI_Reduce_c_ptr;
    decltype(MPI_Reduce_local) *MPI_Reduce_local_ptr;
    decltype(MPI_Reduce_scatter) *MPI_Reduce_scatter_ptr;
    decltype(MPI_Reduce_scatter_block) *MPI_Reduce_scatter_block_ptr;
    decltype(MPI_Reduce_scatter_block_c) *MPI_Reduce_scatter_block_c_ptr;
    decltype(MPI_Test) *MPI_Test_ptr;
    decltype(MPI_Type_commit) *MPI_Type_commit_ptr;
    decltype(MPI_Type_contiguous) *MPI_Type_contiguous_ptr;
    decltype(MPI_Type_free) *MPI_Type_free_ptr;
    decltype(MPI_Wait) *MPI_Wait_ptr;
    decltype(MPI_Waitall) *MPI_Waitall_ptr;
    decltype(MPI_Type_size) *MPI_Type_size_ptr;
    decltype(MPI_Comm_dup) *MPI_Comm_dup_ptr;
} mpi_lib_ops_t;

static std::vector<std::string> mpi_fn_names = {
    "MPI_Allgather",
    "MPI_Allgather_c",
    "MPI_Allgatherv",
    "MPI_Allgatherv_c",
    "MPI_Allreduce",
    "MPI_Allreduce_c",
    "MPI_Alltoall",
    "MPI_Alltoall_c",
    "MPI_Alltoallv",
    "MPI_Alltoallv_c",
    "MPI_Barrier",
    "MPI_Bcast",
    "MPI_Bcast_c",
    "MPI_Cancel",
    "MPI_Comm_create_group",
    "MPI_Comm_free",
    "MPI_Comm_get_attr",
    "MPI_Comm_get_info",
    "MPI_Comm_group",
    "MPI_Comm_rank",
    "MPI_Comm_set_info",
    "MPI_Comm_size",
    "MPI_Comm_split",
    "MPI_Comm_split_type",
    "MPI_Error_string",
    "MPI_Finalize",
    "MPI_Finalized",
    "MPI_Get_count",
    "MPI_Get_library_version",
    "MPI_Group_incl",
    "MPI_Iallgather",
    "MPI_Iallgather_c",
    "MPI_Iallgatherv",
    "MPI_Iallgatherv_c",
    "MPI_Iallreduce",
    "MPI_Iallreduce_c",
    "MPI_Ialltoall",
    "MPI_Ialltoall_c",
    "MPI_Ialltoallv",
    "MPI_Ialltoallv_c",
    "MPI_Ibarrier",
    "MPI_Ibcast",
    "MPI_Ibcast_c",
    "MPI_Info_create",
    "MPI_Info_free",
    "MPI_Info_get",
    "MPI_Info_set",
    "MPI_Initialized",
    "MPI_Init",
    "MPI_Init_thread",
    "MPI_Iprobe",
    "MPI_Irecv",
    "MPI_Irecv_c",
    "MPI_Ireduce",
    "MPI_Ireduce_c",
    "MPI_Ireduce_scatter_block",
    "MPI_Ireduce_scatter_block_c",
    "MPI_Isend",
    "MPI_Isend_c",
    "MPI_Op_create",
    "MPI_Op_free",
    "MPI_Query_thread",
    "MPI_Reduce",
    "MPI_Reduce_c",
    "MPI_Reduce_local",
    "MPI_Reduce_scatter",
    "MPI_Reduce_scatter_block",
    "MPI_Reduce_scatter_block_c",
    "MPI_Test",
    "MPI_Type_commit",
    "MPI_Type_contiguous",
    "MPI_Type_free",
    "MPI_Wait",
    "MPI_Waitall",
    "MPI_Type_size",
    "MPI_Comm_dup",
};

extern ccl::mpi_lib_ops_t mpi_lib_ops;

#if (MPI_VERSION >= 4) || (defined(I_MPI_NUMVERSION) && (I_MPI_NUMVERSION >= 20211200300))
#define ENABLE_LARGE_COUNT_SUPPORT 1
#define Compat_MPI_Count_t         MPI_Count
#define Compat_MPI_Aint_t          MPI_Aint
#else // large count support
#define ENABLE_LARGE_COUNT_SUPPORT 0
#define Compat_MPI_Count_t         int
#define Compat_MPI_Aint_t          int
#endif // large count support

// MPI collectives

#define MPI_Allgather ccl::mpi_lib_ops.MPI_Allgather_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Allgather_c ccl::mpi_lib_ops.MPI_Allgather_c_ptr
#else // MPI_Allgather_c
#define MPI_Allgather_c ccl::mpi_lib_ops.MPI_Allgather_ptr
#endif // MPI_Allgather_c
#define MPI_Allgatherv ccl::mpi_lib_ops.MPI_Allgatherv_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Allgatherv_c ccl::mpi_lib_ops.MPI_Allgatherv_c_ptr
#else // MPI_Allgatherv_c
#define MPI_Allgatherv_c ccl::mpi_lib_ops.MPI_Allgatherv_ptr
#endif // MPI_Allgatherv_c
#define MPI_Allreduce ccl::mpi_lib_ops.MPI_Allreduce_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Allreduce_c ccl::mpi_lib_ops.MPI_Allreduce_c_ptr
#else // MPI_Allreduce_c
#define MPI_Allreduce_c ccl::mpi_lib_ops.MPI_Allreduce_ptr
#endif // MPI_Allreduce_c
#define MPI_Alltoall ccl::mpi_lib_ops.MPI_Alltoall_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Alltoall_c ccl::mpi_lib_ops.MPI_Alltoall_c_ptr
#else // MPI_Alltoall_c
#define MPI_Alltoall_c ccl::mpi_lib_ops.MPI_Alltoall_ptr
#endif // MPI_Alltoall_c
#define MPI_Alltoallv ccl::mpi_lib_ops.MPI_Alltoallv_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Alltoallv_c ccl::mpi_lib_ops.MPI_Alltoallv_c_ptr
#else // MPI_Alltoallv_c
#define MPI_Alltoallv_c ccl::mpi_lib_ops.MPI_Alltoallv_ptr
#endif // MPI_Alltoallv_c
#define MPI_Barrier ccl::mpi_lib_ops.MPI_Barrier_ptr
#define MPI_Bcast   ccl::mpi_lib_ops.MPI_Bcast_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Bcast_c ccl::mpi_lib_ops.MPI_Bcast_c_ptr
#else // MPI_Bcast_c
#define MPI_Bcast_c ccl::mpi_lib_ops.MPI_Bcast_ptr
#endif // MPI_Bcast_c
#define MPI_Cancel              ccl::mpi_lib_ops.MPI_Cancel_ptr
#define MPI_Comm_create_group   ccl::mpi_lib_ops.MPI_Comm_create_group_ptr
#define MPI_Comm_free           ccl::mpi_lib_ops.MPI_Comm_free_ptr
#define MPI_Comm_get_attr       ccl::mpi_lib_ops.MPI_Comm_get_attr_ptr
#define MPI_Comm_get_info       ccl::mpi_lib_ops.MPI_Comm_get_info_ptr
#define MPI_Comm_group          ccl::mpi_lib_ops.MPI_Comm_group_ptr
#define MPI_Comm_rank           ccl::mpi_lib_ops.MPI_Comm_rank_ptr
#define MPI_Comm_set_info       ccl::mpi_lib_ops.MPI_Comm_set_info_ptr
#define MPI_Comm_size           ccl::mpi_lib_ops.MPI_Comm_size_ptr
#define MPI_Comm_split          ccl::mpi_lib_ops.MPI_Comm_split_ptr
#define MPI_Comm_split_type     ccl::mpi_lib_ops.MPI_Comm_split_type_ptr
#define MPI_Error_string        ccl::mpi_lib_ops.MPI_Error_string_ptr
#define MPI_Finalize            ccl::mpi_lib_ops.MPI_Finalize_ptr
#define MPI_Finalized           ccl::mpi_lib_ops.MPI_Finalized_ptr
#define MPI_Get_count           ccl::mpi_lib_ops.MPI_Get_count_ptr
#define MPI_Get_library_version ccl::mpi_lib_ops.MPI_Get_library_version_ptr
#define MPI_Group_incl          ccl::mpi_lib_ops.MPI_Group_incl_ptr
#define MPI_Iallgather          ccl::mpi_lib_ops.MPI_Iallgather_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Iallgather_c ccl::mpi_lib_ops.MPI_Iallgather_c_ptr
#else // MPI_Iallgather_c
#define MPI_Iallgather_c ccl::mpi_lib_ops.MPI_Iallgather_ptr
#endif // MPI_Iallgather_c
#define MPI_Iallgatherv ccl::mpi_lib_ops.MPI_Iallgatherv_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Iallgatherv_c ccl::mpi_lib_ops.MPI_Iallgatherv_c_ptr
#else // MPI_Iallgatherv_c
#define MPI_Iallgatherv_c ccl::mpi_lib_ops.MPI_Iallgatherv_ptr
#endif // MPI_Iallgatherv_c
#define MPI_Iallreduce ccl::mpi_lib_ops.MPI_Iallreduce_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Iallreduce_c ccl::mpi_lib_ops.MPI_Iallreduce_c_ptr
#else // MPI_Iallreduce_c
#define MPI_Iallreduce_c ccl::mpi_lib_ops.MPI_Iallreduce_ptr
#endif // MPI_Iallreduce_c
#define MPI_Ialltoall ccl::mpi_lib_ops.MPI_Ialltoall_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Ialltoall_c ccl::mpi_lib_ops.MPI_Ialltoall_c_ptr
#else // MPI_Ialltoall_c
#define MPI_Ialltoall_c ccl::mpi_lib_ops.MPI_Ialltoall_ptr
#endif // MPI_Ialltoall_c
#define MPI_Ialltoallv ccl::mpi_lib_ops.MPI_Ialltoallv_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Ialltoallv_c ccl::mpi_lib_ops.MPI_Ialltoallv_c_ptr
#else // MPI_Ialltoallv_c
#define MPI_Ialltoallv_c ccl::mpi_lib_ops.MPI_Ialltoallv_ptr
#endif // MPI_Ialltoallv_c
#define MPI_Ibarrier ccl::mpi_lib_ops.MPI_Ibarrier_ptr
#define MPI_Ibcast   ccl::mpi_lib_ops.MPI_Ibcast_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Ibcast_c ccl::mpi_lib_ops.MPI_Ibcast_c_ptr
#else // MPI_Ibcast_c
#define MPI_Ibcast_c ccl::mpi_lib_ops.MPI_Ibcast_ptr
#endif // MPI_Ibcast_c
#define MPI_Info_create ccl::mpi_lib_ops.MPI_Info_create_ptr
#define MPI_Info_free   ccl::mpi_lib_ops.MPI_Info_free_ptr
#define MPI_Info_get    ccl::mpi_lib_ops.MPI_Info_get_ptr
#define MPI_Info_set    ccl::mpi_lib_ops.MPI_Info_set_ptr
#define MPI_Initialized ccl::mpi_lib_ops.MPI_Initialized_ptr
#define MPI_Init        ccl::mpi_lib_ops.MPI_Init_ptr
#define MPI_Init_thread ccl::mpi_lib_ops.MPI_Init_thread_ptr
#define MPI_Iprobe      ccl::mpi_lib_ops.MPI_Iprobe_ptr
#define MPI_Irecv       ccl::mpi_lib_ops.MPI_Irecv_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Irecv_c ccl::mpi_lib_ops.MPI_Irecv_c_ptr
#else // MPI_Irecv_c
#define MPI_Irecv_c ccl::mpi_lib_ops.MPI_Irecv_ptr
#endif // MPI_Irecv_c
#define MPI_Ireduce ccl::mpi_lib_ops.MPI_Ireduce_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Ireduce_c ccl::mpi_lib_ops.MPI_Ireduce_c_ptr
#else // MPI_Ireduce_c
#define MPI_Ireduce_c ccl::mpi_lib_ops.MPI_Ireduce_ptr
#endif // MPI_Ireduce_c
#define MPI_Ireduce_scatter_block ccl::mpi_lib_ops.MPI_Ireduce_scatter_block_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Ireduce_scatter_block_c ccl::mpi_lib_ops.MPI_Ireduce_scatter_block_c_ptr
#else // MPI_Ireduce_scatter_block_c
#define MPI_Ireduce_scatter_block_c ccl::mpi_lib_ops.MPI_Ireduce_scatter_block_ptr
#endif // MPI_Ireduce_scatter_block_c
#define MPI_Isend ccl::mpi_lib_ops.MPI_Isend_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Isend_c ccl::mpi_lib_ops.MPI_Isend_c_ptr
#else // MPI_Isend_c
#define MPI_Isend_c ccl::mpi_lib_ops.MPI_Isend_ptr
#endif // MPI_Isend_c
#define MPI_Op_create    ccl::mpi_lib_ops.MPI_Op_create_ptr
#define MPI_Op_free      ccl::mpi_lib_ops.MPI_Op_free_ptr
#define MPI_Query_thread ccl::mpi_lib_ops.MPI_Query_thread_ptr
#define MPI_Reduce       ccl::mpi_lib_ops.MPI_Reduce_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Reduce_c ccl::mpi_lib_ops.MPI_Reduce_c_ptr
#else // MPI_Reduce_c
#define MPI_Reduce_c ccl::mpi_lib_ops.MPI_Reduce_ptr
#endif // MPI_Reduce_c
#define MPI_Reduce_local         ccl::mpi_lib_ops.MPI_Reduce_local_ptr
#define MPI_Reduce_scatter       ccl::mpi_lib_ops.MPI_Reduce_scatter_ptr
#define MPI_Reduce_scatter_block ccl::mpi_lib_ops.MPI_Reduce_scatter_block_ptr
#if ENABLE_LARGE_COUNT_SUPPORT
#define MPI_Reduce_scatter_block_c ccl::mpi_lib_ops.MPI_Reduce_scatter_block_c_ptr
#else // MPI_Reduce_scatter_block_c
#define MPI_Reduce_scatter_block_c ccl::mpi_lib_ops.MPI_Reduce_scatter_block_ptr
#endif // MPI_Reduce_scatter_block_c
#define MPI_Test            ccl::mpi_lib_ops.MPI_Test_ptr
#define MPI_Type_commit     ccl::mpi_lib_ops.MPI_Type_commit_ptr
#define MPI_Type_contiguous ccl::mpi_lib_ops.MPI_Type_contiguous_ptr
#define MPI_Type_free       ccl::mpi_lib_ops.MPI_Type_free_ptr
#define MPI_Wait            ccl::mpi_lib_ops.MPI_Wait_ptr
#define MPI_Waitall         ccl::mpi_lib_ops.MPI_Waitall_ptr
#define MPI_Type_size       ccl::mpi_lib_ops.MPI_Type_size_ptr
#define MPI_Comm_dup        ccl::mpi_lib_ops.MPI_Comm_dup_ptr

bool mpi_api_init();
void mpi_api_fini();
const CCL_API mpi_lib_ops_t &get_mpi_lib_ops();

} //namespace ccl

#endif //CCL_ENABLE_MPI

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

#ifdef CCL_ENABLE_MPI

#include "atl/atl_def.h"
#include "common/api_wrapper/mpi_api_wrapper.hpp"
#include "comp/bf16/bf16.hpp"
#include "comp/bf16/bf16_intrisics.hpp"
#include "comp/fp16/fp16_intrisics.hpp"

#define ATL_MPI_BF16

#ifdef CCL_FP16_COMPILER
#define ATL_MPI_FP16
#endif // CCL_FP16_COMPILER

class atl_mpi_ctx {
public:
    typedef enum { ATL_MPI_LIB_IMPI, ATL_MPI_LIB_MPICH, ATL_MPI_LIB_NONE } atl_mpi_lib_type_t;

private:
    typedef struct {
        atl_mpi_lib_type_t type;

        /* current version of library */
        int version_value;
        /* current sub-version of library */
        int sub_version_value;

        int hmem;
    } atl_mpi_lib_attr_t;
    typedef struct {
        // custom MPI operations for BF16
        MPI_Op sum_op;
        MPI_Op prod_op;
        MPI_Op min_op;
        MPI_Op max_op;
        // custom MPI dtype for BF16
        MPI_Datatype dtype;
    } atl_mpi_bf16_data_t;

    typedef struct {
        // custom MPI operations for FP16
        MPI_Op sum_op;
        MPI_Op prod_op;
        MPI_Op min_op;
        MPI_Op max_op;
        // custom MPI dtype for FP16
        MPI_Datatype dtype;
    } atl_mpi_fp16_data_t;

    typedef struct {
        atl_mpi_lib_type_t type;
        const char* name;

        /* string prefix before numerical version of library, mandatory */
        const char* version_prefix_1;

        /* string prefix before numerical version of library, following prefix_1, optional */
        const char* version_prefix_2;

        /* minimal expected version of library, mandatory */
        int min_version_value;

        /* minimal expected version of library with hmem support, mandatory */
        int min_hmem_version_value;

        /* string prefix before library kind, optional */
        const char* kind_prefix;

        /* library kind, optional */
        const char* kind_value;
    } atl_mpi_lib_info_t;

#define MPI_LIB_INFO_MAX_COUNT 3

    static const atl_mpi_lib_info_t mpi_lib_infos[MPI_LIB_INFO_MAX_COUNT];

    size_t get_nic_count(const char* nic_count_key);

public:
    static const char* EP_IDX_KEY;

    const char* NIC_IDX_KEY = "multi_nic_pref_nic";
    const char* GLOBAL_NIC_COUNT_KEY = "num_nics";
    const char* LOCAL_NIC_COUNT_KEY = "num_close_nics";

    int is_external_init;
    int extra_ep;
    atl_mnic_t mnic_type;
    size_t mnic_count;
    atl_mnic_offset_t mnic_offset;
    atl_mpi_bf16_data_t bf16;
    bool is_bf16_native;
    atl_mpi_fp16_data_t fp16;
    bool is_fp16_native;
    atl_progress_mode_t progress_mode;
    bool sync_coll;
    size_t ep_count;

    static atl_mpi_lib_attr_t mpi_lib_attr;

    atl_mpi_ctx()
            : is_external_init(0),
              extra_ep(0),
              mnic_type(ATL_MNIC_NONE),
              mnic_count(1),
              mnic_offset(ATL_MNIC_OFFSET_NONE),
              is_bf16_native(0),
              is_fp16_native(0),
              progress_mode(ATL_PROGRESS_CHECK),
              sync_coll(0),
              ep_count(0) {
        bf16.dtype = MPI_DATATYPE_NULL;
        bf16.sum_op = MPI_OP_NULL;
        bf16.prod_op = MPI_OP_NULL;
        bf16.min_op = MPI_OP_NULL;
        bf16.max_op = MPI_OP_NULL;

        fp16.dtype = MPI_DATATYPE_NULL;
        fp16.sum_op = MPI_OP_NULL;
        fp16.prod_op = MPI_OP_NULL;
        fp16.min_op = MPI_OP_NULL;
        fp16.max_op = MPI_OP_NULL;
    }

    static atl_mpi_lib_attr_t get_lib_attr();
    static size_t get_ep_count(const atl_attr_t& attr);

    int bf16_init();
    void bf16_finalize();
    int fp16_init();
    void fp16_finalize();

    MPI_Op atl2mpi_op_bf16(atl_reduction_t rtype);
    MPI_Op atl2mpi_op_fp16(atl_reduction_t rtype);

    static atl_status_t set_env(const atl_attr_t& attr);
    static atl_status_t set_base_env(const atl_attr_t& attr);
    static atl_status_t set_impi_env(const atl_attr_t& attr, const atl_mpi_lib_attr_t& lib_attr);
    static atl_status_t set_mpich_env(const atl_attr_t& attr);
    static atl_status_t check_impi_env(const atl_attr_t& attr);
    static atl_status_t check_mpich_env(const atl_attr_t& attr);

    atl_status_t update_global_data(const atl_attr_t& attr);

    void print_mpi_error(int error);
    std::string to_string();
};

#endif

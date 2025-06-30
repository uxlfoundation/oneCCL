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
#ifdef CCL_ENABLE_MPI

#include "atl/mpi/atl_mpi.hpp"
#include "atl/mpi/atl_mpi_ctx.hpp"
#include "common/global/global.hpp"
#include "common/log/log.hpp"

atl_mpi_ctx::atl_mpi_lib_attr_t atl_mpi_ctx::mpi_lib_attr{ ATL_MPI_LIB_NONE, 0, 0, 0 };

const atl_mpi_ctx::atl_mpi_lib_info_t atl_mpi_ctx::mpi_lib_infos[MPI_LIB_INFO_MAX_COUNT] = {
    { ATL_MPI_LIB_IMPI,
      "impi",
      "Intel(R) MPI Library",
      NULL,
      2019,
      2021,
      "library kind:",
      "release" },
    { ATL_MPI_LIB_MPICH, "mpich", "MPICH Custom Information:", "drop", 34, 40, NULL, NULL },
    { ATL_MPI_LIB_NONE, "none", "", NULL, 0, -1, NULL, NULL },
};

const char* atl_mpi_ctx::EP_IDX_KEY = "vci";

void check_op_params(void* in_buf,
                     void* inout_buf,
                     int* length,
                     MPI_Datatype* datatype,
                     const char* caller_func_name) {
    (void)datatype;
    CCL_THROW_IF_NOT(in_buf && inout_buf && length,
                     caller_func_name,
                     " requested, bad arguments: ",
                     in_buf,
                     " ",
                     inout_buf,
                     " ",
                     length);
}

#ifdef ATL_MPI_FP16

void fp16_base_op(void* in, void* inout, int* length, ccl::reduction op) {
    unsigned short* in_buf = (unsigned short*)in;
    unsigned short* inout_buf = (unsigned short*)inout;

    size_t len = *length;
    ccl_fp16_reduce_impl(in_buf, inout_buf, len, op);
}

void fp16_sum_op(void* in, void* inout, int* length, MPI_Datatype* datatype) {
    check_op_params(in, inout, length, datatype, __FUNCTION__);
    fp16_base_op(in, inout, length, ccl::reduction::sum);
}

void fp16_prod_op(void* in, void* inout, int* length, MPI_Datatype* datatype) {
    check_op_params(in, inout, length, datatype, __FUNCTION__);
    fp16_base_op(in, inout, length, ccl::reduction::prod);
}

void fp16_min_op(void* in, void* inout, int* length, MPI_Datatype* datatype) {
    check_op_params(in, inout, length, datatype, __FUNCTION__);
    fp16_base_op(in, inout, length, ccl::reduction::min);
}

void fp16_max_op(void* in, void* inout, int* length, MPI_Datatype* datatype) {
    check_op_params(in, inout, length, datatype, __FUNCTION__);
    fp16_base_op(in, inout, length, ccl::reduction::max);
}
#endif // ATL_MPI_FP16

void BF16_INLINE_TARGET_ATTRIBUTE_ALL bf16_base_op(void* in,
                                                   void* inout,
                                                   int* length,
                                                   ccl::reduction op) {
    ccl_bf16_reduce(in, *length, inout, nullptr, op);
}

void BF16_TARGET_ATTRIBUTE_ALL bf16_sum_op(void* in,
                                           void* inout,
                                           int* length,
                                           MPI_Datatype* datatype) {
    check_op_params(in, inout, length, datatype, __FUNCTION__);
    bf16_base_op(in, inout, length, ccl::reduction::sum);
}

void BF16_TARGET_ATTRIBUTE_ALL bf16_prod_op(void* in,
                                            void* inout,
                                            int* length,
                                            MPI_Datatype* datatype) {
    check_op_params(in, inout, length, datatype, __FUNCTION__);
    bf16_base_op(in, inout, length, ccl::reduction::prod);
}

void BF16_TARGET_ATTRIBUTE_ALL bf16_min_op(void* in,
                                           void* inout,
                                           int* length,
                                           MPI_Datatype* datatype) {
    check_op_params(in, inout, length, datatype, __FUNCTION__);
    bf16_base_op(in, inout, length, ccl::reduction::min);
}

void BF16_TARGET_ATTRIBUTE_ALL bf16_max_op(void* in,
                                           void* inout,
                                           int* length,
                                           MPI_Datatype* datatype) {
    check_op_params(in, inout, length, datatype, __FUNCTION__);
    bf16_base_op(in, inout, length, ccl::reduction::max);
}

size_t atl_mpi_ctx::get_nic_count(const char* nic_count_key) {
    size_t count = 1;
    atl_mpi_env_info_t info = atl_mpi::get_env_info(nic_count_key);
    CCL_THROW_IF_NOT(info.found, "MPI env key ", nic_count_key, " was not set");

    count = atoi(info.value);
    if (count <= 0) {
        count = 1;
    }

    return count;
}

atl_mpi_ctx::atl_mpi_lib_attr_t atl_mpi_ctx::get_lib_attr() {
    atl_mpi_lib_attr_t lib_attr = { ATL_MPI_LIB_NONE, 0, 0, 0 };

    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING] = { 0 };
    int mpi_version_len = -1, i;
    const atl_mpi_lib_info_t* final_info = NULL;

    /* can be called before MPI_Init */
    int ret = MPI_Get_library_version(mpi_version, &mpi_version_len);

    if ((ret != MPI_SUCCESS) || (mpi_version_len < 0) ||
        (mpi_version_len > MPI_MAX_LIBRARY_VERSION_STRING)) {
        LOG_WARN("can not retrieve MPI version, mpi_version_len ", mpi_version_len, ", ret", ret);
        return lib_attr;
    }

    /* remove trailing spaces at the end for more compact log */
    while (strlen(mpi_version) && isspace(mpi_version[strlen(mpi_version) - 1]))
        mpi_version[strlen(mpi_version) - 1] = '\0';

    LOG_DEBUG("MPI version: ", mpi_version);

    /* for filtering */
    char* lib_type_env = getenv("CCL_ATL_MPI");

    for (i = 0; i < MPI_LIB_INFO_MAX_COUNT; i++) {
        const atl_mpi_lib_info_t* info = &(mpi_lib_infos[i]);

        if (info->type == ATL_MPI_LIB_NONE)
            continue;

        if (lib_type_env) {
            if (strcmp(lib_type_env, info->name)) {
                LOG_DEBUG("library ", info->name, " is filtered out by user input ", lib_type_env);
                continue;
            }
            else {
                LOG_DEBUG("use lib_type = ", lib_type_env, " because it is requested explicitly");
            }
        }

        CCL_THROW_IF_NOT(info->version_prefix_1, "empty version_prefix_1");
        CCL_THROW_IF_NOT(info->min_version_value >= 0, "unexpected minimal version");

        const char* version_substr = NULL;
        if ((version_substr = strstr(mpi_version, info->version_prefix_1))) {
            version_substr += strlen(info->version_prefix_1);
            LOG_DEBUG("version_substr: ", version_substr);

            if (info->version_prefix_2) {
                version_substr = strstr(version_substr, info->version_prefix_2);
                if (!version_substr) {
                    LOG_DEBUG("can't find version_prefix_2 ", info->version_prefix_2);
                    continue;
                }
                version_substr += strlen(info->version_prefix_2);
                LOG_DEBUG("version_substr: ", version_substr);
            }

            int version_value = (version_substr) ? atoi(version_substr) : -1;
            LOG_DEBUG("MPI numerical version: ", version_value);
            int sub_version_value = -1;
            /* find sub version value for impi */
            if (i == 0) {
                version_substr = strstr(version_substr, ".");
                if (version_substr) {
                    version_substr++;
                    if (version_substr && version_substr[0] != '\0') {
                        sub_version_value = atoi(version_substr);
                    }
                }
                if (sub_version_value == -1) {
                    LOG_DEBUG("can't find sub_version_value");
                    sub_version_value = 0;
                }
            }

            if (version_value < info->min_version_value) {
                LOG_WARN("loaded MPI doesn't match with expected version, "
                         "consider to switch to ",
                         info->version_prefix_1,
                         " ",
                         (info->version_prefix_2 ? info->version_prefix_2 : ""),
                         info->min_version_value,
                         " (min) ",
                         (info->kind_value ? info->kind_value : ""),
                         "\n");
                continue;
            }

            if (info->kind_prefix && info->kind_value) {
                const char* kind_substr = mpi_version;

                if ((kind_substr = strstr(kind_substr, info->kind_prefix))) {
                    kind_substr += strlen(info->kind_prefix);
                    while ((isspace(*kind_substr)) &&
                           (kind_substr < (mpi_version + mpi_version_len)))
                        kind_substr++;

                    LOG_DEBUG("kind_substr: ", kind_substr);

                    if (strncmp(kind_substr, info->kind_value, strlen(info->kind_value))) {
                        LOG_WARN("loaded MPI version (",
                                 version_value,
                                 ") ",
                                 "is higher or equal to minimal expected version (",
                                 info->min_version_value,
                                 ") ",
                                 "but kind (",
                                 kind_substr,
                                 ") doesn't match with expected kind (",
                                 info->kind_value,
                                 "), "
                                 "consider to switch to ",
                                 info->version_prefix_1,
                                 " ",
                                 (info->version_prefix_2 ? info->version_prefix_2 : ""),
                                 info->min_version_value,
                                 " (min version) ",
                                 (info->kind_value ? info->kind_value : ""),
                                 "\n");
                    }
                }
                else {
                    LOG_DEBUG("MPI version is high enough, but kind_prefix (",
                              info->kind_prefix,
                              ") can not be found",
                              " treat this like expected kind (",
                              info->kind_value,
                              ") was found");
                }
            }

            final_info = info;

            lib_attr.type = info->type;
            lib_attr.version_value = version_value;
            lib_attr.sub_version_value = sub_version_value;
            LOG_DEBUG("set lib_attr.type = ",
                      info->name,
                      ", version ",
                      version_value,
                      ", minimal expected version ",
                      info->min_version_value);

            lib_attr.hmem = (version_value >= info->min_hmem_version_value) ? 1 : 0;
            LOG_DEBUG("set lib_attr.hmem = ",
                      lib_attr.hmem,
                      ", version ",
                      version_value,
                      ", minimal expected hmem version ",
                      info->min_hmem_version_value);

            break;
        }
    }

    if (final_info) {
        LOG_DEBUG("MPI library type: ", final_info->name);
    }
    else {
        LOG_DEBUG("MPI library type: none");
    }

    return lib_attr;
}

size_t atl_mpi_ctx::get_ep_count(const atl_attr_t& attr) {
    size_t mpi_ep_count = attr.in.ep_count;
    if (attr.in.enable_extra_ep)
        mpi_ep_count += attr.in.enable_extra_ep;
    return mpi_ep_count;
}

int atl_mpi_ctx::bf16_init() {
    int ret = MPI_SUCCESS;

    bool env_mpi_bf16_native = (bool)(ccl::global_data::env().mpi_bf16_native);
#ifdef MPIX_C_BF16
    if (mpi_lib_attr.type == atl_mpi_lib_type_t::ATL_MPI_LIB_IMPI &&
        mpi_lib_attr.version_value >= 2021 && mpi_lib_attr.sub_version_value >= 14 &&
        env_mpi_bf16_native) {
        is_bf16_native = true;
        bf16.dtype = MPIX_C_BF16;
        bf16.sum_op = MPI_SUM;
        bf16.prod_op = MPI_PROD;
        bf16.min_op = MPI_MIN;
        bf16.max_op = MPI_MAX;
    }
    else {
        is_bf16_native = false;
    }
#else
    is_bf16_native = false;
#endif /* MPIX_C_BF16 */
    if (env_mpi_bf16_native && !is_bf16_native) {
        LOG_INFO_ROOT("native Intel MPI BF16 is not available");
    }
    if (is_bf16_native) {
        LOG_INFO_ROOT("native Intel MPI BF16 is enabled");
    }
    else {
        // create custom MPI BF16 dtype
        ret = MPI_Type_contiguous(2, MPI_BYTE, &bf16.dtype);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI BF16 dtype");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        ret = MPI_Type_commit(&bf16.dtype);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot commit MPI BF16 type");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        // create custom MPI BF16 summation op
        ret = MPI_Op_create(&bf16_sum_op, 1, &bf16.sum_op);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI BF16 sum op");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        // create custom MPI BF16 production op
        ret = MPI_Op_create(&bf16_prod_op, 1, &bf16.prod_op);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI BF16 prod op");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        // create custom MPI BF16 min op
        ret = MPI_Op_create(&bf16_min_op, 1, &bf16.min_op);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI BF16 min op");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        // create custom MPI BF16 max op
        ret = MPI_Op_create(&bf16_max_op, 1, &bf16.max_op);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI BF16 max op");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }
    }

    return ATL_STATUS_SUCCESS;
}

void atl_mpi_ctx::bf16_finalize() {
    if (!is_bf16_native) {
        if (bf16.dtype != MPI_DATATYPE_NULL) {
            MPI_Type_free(&bf16.dtype);
        }

        if (bf16.sum_op != MPI_OP_NULL) {
            MPI_Op_free(&bf16.sum_op);
        }

        if (bf16.prod_op != MPI_OP_NULL) {
            MPI_Op_free(&bf16.prod_op);
        }

        if (bf16.min_op != MPI_OP_NULL) {
            MPI_Op_free(&bf16.min_op);
        }

        if (bf16.max_op != MPI_OP_NULL) {
            MPI_Op_free(&bf16.max_op);
        }
    }
}

int atl_mpi_ctx::fp16_init() {
    if (ccl::global_data::env().fp16_impl_type <= ccl_fp16_no_hardware_support) {
        return ATL_STATUS_SUCCESS;
    }

#ifdef ATL_MPI_FP16

    int ret = MPI_SUCCESS;

    bool env_mpi_fp16_native = (bool)(ccl::global_data::env().mpi_fp16_native);
#ifdef MPIX_C_FLOAT16
    if (mpi_lib_attr.type == atl_mpi_lib_type_t::ATL_MPI_LIB_IMPI &&
        mpi_lib_attr.version_value >= 2021 && mpi_lib_attr.sub_version_value >= 14 &&
        env_mpi_fp16_native) {
        is_fp16_native = true;
        fp16.dtype = MPIX_C_FLOAT16;
        fp16.sum_op = MPI_SUM;
        fp16.prod_op = MPI_PROD;
        fp16.min_op = MPI_MIN;
        fp16.max_op = MPI_MAX;
    }
    else {
        is_fp16_native = false;
    }
#else
    is_fp16_native = false;
#endif /* MPIX_C_FLOAT16 */
    if (env_mpi_fp16_native && !is_fp16_native) {
        LOG_INFO_ROOT("native Intel MPI FP16 is not available");
    }
    if (is_fp16_native) {
        LOG_INFO_ROOT("native Intel MPI FP16 is enabled");
    }
    else {
        // create custom MPI FP16 dtype
        ret = MPI_Type_contiguous(2, MPI_BYTE, &fp16.dtype);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI FP16 dtype");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        ret = MPI_Type_commit(&fp16.dtype);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot commit MPI FP16 type");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        // create custom MPI FP16 summation op
        ret = MPI_Op_create(&fp16_sum_op, 1, &fp16.sum_op);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI FP16 sum op");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        // create custom MPI FP16 production op
        ret = MPI_Op_create(&fp16_prod_op, 1, &fp16.prod_op);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI FP16 prod op");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        // create custom MPI FP16 min op
        ret = MPI_Op_create(&fp16_min_op, 1, &fp16.min_op);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI FP16 min op");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }

        // create custom MPI FP16 max op
        ret = MPI_Op_create(&fp16_max_op, 1, &fp16.max_op);
        if (ret != MPI_SUCCESS) {
            LOG_ERROR("cannot create MPI FP16 max op");
            print_mpi_error(ret);
            return ATL_STATUS_FAILURE;
        }
    }

#endif // ATL_MPI_FP16

    return ATL_STATUS_SUCCESS;
}

void atl_mpi_ctx::fp16_finalize() {
    if (!is_fp16_native) {
        if (fp16.dtype != MPI_DATATYPE_NULL) {
            MPI_Type_free(&fp16.dtype);
        }

        if (fp16.sum_op != MPI_OP_NULL) {
            MPI_Op_free(&fp16.sum_op);
        }

        if (fp16.prod_op != MPI_OP_NULL) {
            MPI_Op_free(&fp16.prod_op);
        }

        if (fp16.min_op != MPI_OP_NULL) {
            MPI_Op_free(&fp16.min_op);
        }

        if (fp16.max_op != MPI_OP_NULL) {
            MPI_Op_free(&fp16.max_op);
        }
    }
}

MPI_Op atl_mpi_ctx::atl2mpi_op_bf16(atl_reduction_t rtype) {
    switch (rtype) {
        case ATL_REDUCTION_SUM: return bf16.sum_op;
        case ATL_REDUCTION_PROD: return bf16.prod_op;
        case ATL_REDUCTION_MIN: return bf16.min_op;
        case ATL_REDUCTION_MAX: return bf16.max_op;
        default: printf("unknown reduction type: %d\n", rtype); exit(1);
    }
}

#ifdef ATL_MPI_FP16
MPI_Op atl_mpi_ctx::atl2mpi_op_fp16(atl_reduction_t rtype) {
    switch (rtype) {
        case ATL_REDUCTION_SUM: return fp16.sum_op;
        case ATL_REDUCTION_PROD: return fp16.prod_op;
        case ATL_REDUCTION_MIN: return fp16.min_op;
        case ATL_REDUCTION_MAX: return fp16.max_op;
        default: printf("unknown reduction type: %d\n", rtype); exit(1);
    }
}
#endif // ATL_MPI_FP16

atl_status_t atl_mpi_ctx::set_env(const atl_attr_t& attr) {
    if (mpi_lib_attr.type != ATL_MPI_LIB_NONE) {
        /* library type was already detected and env was set, make sanity check */
        if (mpi_lib_attr.type == ATL_MPI_LIB_IMPI) {
            return check_impi_env(attr);
        }
        else if (mpi_lib_attr.type == ATL_MPI_LIB_MPICH) {
            return check_mpich_env(attr);
        }
        return ATL_STATUS_SUCCESS;
    }

    set_base_env(attr);

    mpi_lib_attr = get_lib_attr();

    if (mpi_lib_attr.type == ATL_MPI_LIB_NONE) {
        return ATL_STATUS_SUCCESS;
    }

    if (mpi_lib_attr.type == ATL_MPI_LIB_IMPI) {
        set_impi_env(attr, mpi_lib_attr);
        check_impi_env(attr);
    }
    else if (mpi_lib_attr.type == ATL_MPI_LIB_MPICH) {
        set_mpich_env(attr);
        check_mpich_env(attr);
    }

    int is_mpi_inited = 0;
    MPI_Initialized(&is_mpi_inited);
    if (is_mpi_inited) {
        LOG_WARN("MPI was initialized externally, CCL-MPI specific environment is ignored");
    }
    else {
        LOG_DEBUG("set CCL-MPI specific environment");
    }

    return ATL_STATUS_SUCCESS;
}

/* set these knobs without detection of MPI library type */
atl_status_t atl_mpi_ctx::set_base_env(const atl_attr_t& attr) {
    setenv("PSM2_MULTI_EP", "1", 0);
    setenv("FI_OFI_RXM_USE_HASH", "0", 0);

#ifdef CCL_ENABLE_SYCL
    setenv("FI_SHM_DISABLE_CMA", "1", 0);
#endif // CCL_ENABLE_SYCL

    setenv("MPIR_CVAR_DEFAULT_THREAD_LEVEL", "MPI_THREAD_MULTIPLE", 0);

    /* request IMPI level append library kind into MPI_Get_library_version output */
    setenv("I_MPI_INFO_LIBRARY_KIND", "1", 0);

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_ctx::set_impi_env(const atl_attr_t& attr, const atl_mpi_lib_attr_t& lib_attr) {
    char ep_count_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(ep_count_str, MPI_MAX_INFO_VAL, "%zu", get_ep_count(attr));

    if (attr.in.ep_count)
        setenv("I_MPI_OFI_ISEND_INJECT_THRESHOLD", "0", 0);

#ifdef CCL_ENABLE_SYCL
    setenv("I_MPI_SHM_CMA", "0", 0);
    if (attr.in.enable_hmem && lib_attr.hmem) {
        setenv("I_MPI_OFFLOAD", "2", 0);
        if (attr.in.ep_count > 1) {
            /* try to set global lock level before vci level
               because setenv is invoked with overwrite=0 */
            setenv("I_MPI_THREAD_LOCK_LEVEL", "global", 0);
        }
        LOG_DEBUG("IMPI case: gpu support is enabled");
    }
#endif // CCL_ENABLE_SYCL

    setenv("I_MPI_THREAD_SPLIT", "1", 0);
#ifdef CCL_ENABLE_OMP
    if (getenv("OMP_NUM_THREADS") != NULL) {
        setenv("I_MPI_THREAD_RUNTIME", "openmp", 0);
        if (strcmp(getenv("I_MPI_THREAD_RUNTIME"), "openmp") != 0) {
            LOG_WARN_ROOT(
                "I_MPI_THREAD_RUNTIME must be unset or set to 'openmp' to achieve optimal performance");
        }
        // set I_MPI_THREAD_LOCK_LEVEL to vci for any number of ccl workers
        setenv("I_MPI_THREAD_LOCK_LEVEL", "vci", 0);
        if (strcmp(getenv("I_MPI_THREAD_LOCK_LEVEL"), "vci") != 0) {
            LOG_WARN_ROOT(
                "I_MPI_THREAD_LOCK_LEVEL must be unset or set to 'vci' to achieve optimal performance");
        }
    }
    // disable omp collectives when OMP_NUM_THREADS is not specified
    else {
        setenv("I_MPI_THREAD_RUNTIME", "generic", 0);
        ccl::global_data::env().enable_omp_coll = false;
    }
#else
    setenv("I_MPI_THREAD_RUNTIME", "generic", 0);
#endif // CCL_ENABLE_OMP
    setenv("I_MPI_THREAD_MAX", ep_count_str, 0);
    setenv("I_MPI_THREAD_ID_KEY", EP_IDX_KEY, 0);
    // this setenv will not overwrite I_MPI_THREAD_LOCK_LEVEL if it has value
    setenv("I_MPI_THREAD_LOCK_LEVEL", (attr.in.ep_count == 1) ? "global" : "vci", 0);

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_ctx::set_mpich_env(const atl_attr_t& attr) {
    char ep_count_str[MPI_MAX_INFO_VAL] = { 0 };
    snprintf(ep_count_str, MPI_MAX_INFO_VAL, "%zu", get_ep_count(attr));

    setenv("MPIR_CVAR_CH4_MT_MODEL", "direct", 0);
    setenv("MPIR_CVAR_CH4_NUM_VCIS", ep_count_str, 0);
    setenv("MPIR_CVAR_CH4_OFI_MAX_VCIS", ep_count_str, 0);
    setenv("MPIR_COMM_HINT_VCI", EP_IDX_KEY, 0);

    auto& env = ccl::global_data::env();
    if (env.log_level >= ccl_log_level::debug) {
        setenv("MPIR_CVAR_CH4_RUNTIME_CONF_DEBUG", "1", 0);
        setenv("MPIR_CVAR_CH4_OFI_CAPABILITY_SETS_DEBUG", "1", 0);
        setenv("MPIR_CVAR_DEBUG_SUMMARY", "1", 0);
    }

    setenv("FI_PSM2_DELAY", "0", 0);
    setenv("FI_PSM2_TIMEOUT", "0", 0);
    setenv("FI_PSM2_NAME_SERVER", "0", 0);
    setenv("HFI_NO_CPUAFFINITY", "1", 0);

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_ctx::check_impi_env(const atl_attr_t& attr) {
    char* ep_count_env = getenv("I_MPI_THREAD_MAX");
    if (!ep_count_env)
        return ATL_STATUS_FAILURE;
    if (atoi(ep_count_env) != (int)(get_ep_count(attr)))
        return ATL_STATUS_FAILURE;

    if (!getenv("ONEAPI_ROOT") && !getenv("I_MPI_ROOT")) {
        atl_mpi_lib_type_t type = ATL_MPI_LIB_IMPI;
        LOG_ERROR("CCL/MPI uses ",
                  mpi_lib_infos[type].version_prefix_1,
                  " but neither I_MPI_ROOT nor ONEAPI_ROOT is set. ",
                  "Please source ",
                  mpi_lib_infos[type].kind_value,
                  " version of ",
                  mpi_lib_infos[type].version_prefix_1,
                  " (",
                  mpi_lib_infos[type].min_version_value,
                  " or higher version).");
        return ATL_STATUS_FAILURE;
    }

    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_ctx::check_mpich_env(const atl_attr_t& attr) {
    char* ep_count_env = getenv("MPIR_CVAR_CH4_OFI_MAX_VCIS");
    if (!ep_count_env)
        return ATL_STATUS_FAILURE;
    if (atoi(ep_count_env) != (int)(get_ep_count(attr)))
        return ATL_STATUS_FAILURE;
    return ATL_STATUS_SUCCESS;
}

atl_status_t atl_mpi_ctx::update_global_data(const atl_attr_t& attr) {
    if (mpi_lib_attr.type == ATL_MPI_LIB_NONE)
        mpi_lib_attr = get_lib_attr();

    extra_ep = attr.in.enable_extra_ep;

    mnic_type = attr.in.mnic_type;
    if (mpi_lib_attr.type != ATL_MPI_LIB_MPICH) {
        /* only MPICH supports multi-NIC */
        mnic_type = ATL_MNIC_NONE;
    }

    if (mnic_type == ATL_MNIC_LOCAL) {
        mnic_count = get_nic_count(LOCAL_NIC_COUNT_KEY);
    }
    else if (mnic_type == ATL_MNIC_GLOBAL) {
        mnic_count = get_nic_count(GLOBAL_NIC_COUNT_KEY);
    }
    else if (mnic_type == ATL_MNIC_NONE) {
        mnic_count = 1;
    }
    mnic_count = std::min(mnic_count, attr.in.mnic_count);
    mnic_count = std::max(mnic_count, (size_t)(1));
    mnic_offset = attr.in.mnic_offset;

    if (bf16_init() == ATL_STATUS_FAILURE) {
        bf16_finalize();
        return ATL_STATUS_FAILURE;
    }

    if (fp16_init() == ATL_STATUS_FAILURE) {
        fp16_finalize();
        return ATL_STATUS_FAILURE;
    }
    return ATL_STATUS_SUCCESS;
}

void atl_mpi_ctx::print_mpi_error(int error) {
    char str_error[MPI_MAX_ERROR_STRING];
    int result_len = MPI_MAX_ERROR_STRING;

    MPI_Error_string(error, str_error, &result_len);

    if (result_len > MPI_MAX_ERROR_STRING) {
        result_len = MPI_MAX_ERROR_STRING;
    }
    str_error[result_len - 1] = '\0';

    ccl_logger::format(std::cout, "MPI error: %s (%d)", str_error, error);
}

std::string atl_mpi_ctx::to_string() {
    std::stringstream ss;
    ss << "{\n"
       << "  is_external_init: " << is_external_init << "\n"
       << "  mpi_lib_attr.type: " << mpi_lib_infos[mpi_lib_attr.type].name << "\n"
       << "  mpi_lib_attr.hmem: " << mpi_lib_attr.hmem << "\n"
       << "  extra_ep: " << extra_ep << "\n"
       << "  mnic_type: " << ::to_string(mnic_type) << "\n";
    if (mnic_type != ATL_MNIC_NONE) {
        ss << "  mnic_count: " << mnic_count << "\n"
           << "  mnic_offset: " << ::to_string(mnic_offset) << "\n";
    }
    ss << "  progress_mode: " << progress_mode << "\n"
       << "  sync_coll: " << sync_coll << "\n"
       << "}";
    return ss.str();
}

#endif // CCL_ENABLE_MPI

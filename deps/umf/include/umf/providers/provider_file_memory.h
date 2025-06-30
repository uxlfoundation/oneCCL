/*
 * Copyright (C) 2024 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef UMF_FILE_MEMORY_PROVIDER_H
#define UMF_FILE_MEMORY_PROVIDER_H

#include <umf/providers/provider_os_memory.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @cond
#define UMF_FILE_RESULTS_START_FROM 3000
/// @endcond

struct umf_file_memory_provider_params_t;

typedef struct umf_file_memory_provider_params_t
    *umf_file_memory_provider_params_handle_t;

/// @brief  Create a struct to store parameters of the File Memory Provider.
/// @param  hParams [out] handle to the newly created parameters struct.
/// @param  path path to the file.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfFileMemoryProviderParamsCreate(
    umf_file_memory_provider_params_handle_t *hParams, const char *path);

/// @brief  Destroy parameters struct.
/// @param  hParams handle to the parameters of the File Memory Provider.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfFileMemoryProviderParamsDestroy(
    umf_file_memory_provider_params_handle_t hParams);

/// @brief  Set the path in the parameters struct.
/// @param  hParams handle to the parameters of the File Memory Provider.
/// @param  path path to the file.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfFileMemoryProviderParamsSetPath(
    umf_file_memory_provider_params_handle_t hParams, const char *path);

/// @brief  Set the protection in the parameters struct.
/// @param  hParams handle to the parameters of the File Memory Provider.
/// @param  protection protection. Combination of \p umf_mem_protection_flags_t flags
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfFileMemoryProviderParamsSetProtection(
    umf_file_memory_provider_params_handle_t hParams, unsigned protection);

/// @brief  Set the visibility in the parameters struct.
/// @param  hParams handle to the parameters of the File Memory Provider.
/// @param  visibility memory visibility mode.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfFileMemoryProviderParamsSetVisibility(
    umf_file_memory_provider_params_handle_t hParams,
    umf_memory_visibility_t visibility);

/// @brief File Memory Provider operation results
typedef enum umf_file_memory_provider_native_error {
    UMF_FILE_RESULT_SUCCESS = UMF_FILE_RESULTS_START_FROM, ///< Success
    UMF_FILE_RESULT_ERROR_ALLOC_FAILED,       ///< Memory allocation failed
    UMF_FILE_RESULT_ERROR_FREE_FAILED,        ///< Memory deallocation failed
    UMF_FILE_RESULT_ERROR_PURGE_FORCE_FAILED, ///< Force purging failed
} umf_file_memory_provider_native_error_t;

umf_memory_provider_ops_t *umfFileMemoryProviderOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_FILE_MEMORY_PROVIDER_H */

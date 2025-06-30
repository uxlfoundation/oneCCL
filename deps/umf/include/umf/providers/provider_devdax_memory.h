/*
 * Copyright (C) 2024 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef UMF_DEVDAX_MEMORY_PROVIDER_H
#define UMF_DEVDAX_MEMORY_PROVIDER_H

#include <umf/providers/provider_os_memory.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @cond
#define UMF_DEVDAX_RESULTS_START_FROM 2000
/// @endcond

struct umf_devdax_memory_provider_params_t;

typedef struct umf_devdax_memory_provider_params_t
    *umf_devdax_memory_provider_params_handle_t;

/// @brief  Create a struct to store parameters of the Devdax Memory Provider.
/// @param  hParams [out] handle to the newly created parameters struct.
/// @param  path [in] path of the device DAX.
/// @param  size [in] size of the device DAX in bytes.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfDevDaxMemoryProviderParamsCreate(
    umf_devdax_memory_provider_params_handle_t *hParams, const char *path,
    size_t size);

/// @brief  Destroy parameters struct.
/// @param  hParams [in] handle to the parameters of the Devdax Memory Provider.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfDevDaxMemoryProviderParamsDestroy(
    umf_devdax_memory_provider_params_handle_t hParams);

/// @brief  Set a device DAX in the parameters struct. Overwrites the previous value.
///         It provides an ability to use the same instance of params to create multiple
///         instances of the provider for different DAX devices.
/// @param  hParams [in] handle to the parameters of the Devdax Memory Provider.
/// @param  path [in] path of the device DAX.
/// @param  size [in] size of the device DAX in bytes.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfDevDaxMemoryProviderParamsSetDeviceDax(
    umf_devdax_memory_provider_params_handle_t hParams, const char *path,
    size_t size);

/// @brief  Set the protection flags in the parameters struct.
/// @param  hParams [in] handle to the parameters of the Devdax Memory Provider.
/// @param  protection [in] combination of 'umf_mem_protection_flags_t' flags.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfDevDaxMemoryProviderParamsSetProtection(
    umf_devdax_memory_provider_params_handle_t hParams, unsigned protection);

/// @brief Devdax Memory Provider operation results
typedef enum umf_devdax_memory_provider_native_error {
    UMF_DEVDAX_RESULT_SUCCESS = UMF_DEVDAX_RESULTS_START_FROM, ///< Success
    UMF_DEVDAX_RESULT_ERROR_ALLOC_FAILED,        ///< Memory allocation failed
    UMF_DEVDAX_RESULT_ERROR_ADDRESS_NOT_ALIGNED, ///< Allocated address is not aligned
    UMF_DEVDAX_RESULT_ERROR_FREE_FAILED,         ///< Memory deallocation failed
    UMF_DEVDAX_RESULT_ERROR_PURGE_FORCE_FAILED, ///< Force purging failed
} umf_devdax_memory_provider_native_error_t;

umf_memory_provider_ops_t *umfDevDaxMemoryProviderOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_DEVDAX_MEMORY_PROVIDER_H */

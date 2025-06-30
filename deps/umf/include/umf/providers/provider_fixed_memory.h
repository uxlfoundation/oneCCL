/*
 * Copyright (C) 2024 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef UMF_FIXED_MEMORY_PROVIDER_H
#define UMF_FIXED_MEMORY_PROVIDER_H

#include <umf/providers/provider_os_memory.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @cond
#define UMF_FIXED_RESULTS_START_FROM 4000
/// @endcond

struct umf_fixed_memory_provider_params_t;

typedef struct umf_fixed_memory_provider_params_t
    *umf_fixed_memory_provider_params_handle_t;

/// @brief  Create a struct to store parameters of the Fixed Memory Provider.
/// @param  hParams [out] handle to the newly created parameters struct.
/// @param  ptr [in] pointer to the memory region.
/// @param  size [in] size of the memory region in bytes.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfFixedMemoryProviderParamsCreate(
    umf_fixed_memory_provider_params_handle_t *hParams, void *ptr, size_t size);

/// @brief  Set the memory region in params struct. Overwrites the previous value.
///         It provides an ability to use the same instance of params to create multiple
///         instances of the provider for different memory regions.
/// @param  hParams [in] handle to the parameters of the Fixed Memory Provider.
/// @param  ptr [in] pointer to the memory region.
/// @param  size [in] size of the memory region in bytes.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfFixedMemoryProviderParamsSetMemory(
    umf_fixed_memory_provider_params_handle_t hParams, void *ptr, size_t size);

/// @brief  Destroy parameters struct.
/// @param  hParams [in] handle to the parameters of the Fixed Memory Provider.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfFixedMemoryProviderParamsDestroy(
    umf_fixed_memory_provider_params_handle_t hParams);

/// @brief Retrieve the operations structure for the Fixed Memory Provider.
/// @return Pointer to the umf_memory_provider_ops_t structure.
umf_memory_provider_ops_t *umfFixedMemoryProviderOps(void);

/// @brief Fixed Memory Provider operation results
typedef enum umf_fixed_memory_provider_native_error {
    UMF_FIXED_RESULT_SUCCESS = UMF_FIXED_RESULTS_START_FROM, ///< Success
    UMF_FIXED_RESULT_ERROR_PURGE_FORCE_FAILED, ///< Force purging failed
} umf_fixed_memory_provider_native_error_t;

#ifdef __cplusplus
}
#endif

#endif /* UMF_FIXED_MEMORY_PROVIDER_H */

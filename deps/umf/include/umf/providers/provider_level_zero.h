/*
 * Copyright (C) 2024-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef UMF_PROVIDER_LEVEL_ZERO_H
#define UMF_PROVIDER_LEVEL_ZERO_H

#include <umf/memory_provider_gpu.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _ze_device_handle_t *ze_device_handle_t;
typedef struct _ze_context_handle_t *ze_context_handle_t;

struct umf_level_zero_memory_provider_params_t;

/// @brief handle to the parameters of the Level Zero Memory Provider.
typedef struct umf_level_zero_memory_provider_params_t
    *umf_level_zero_memory_provider_params_handle_t;

/// @brief  Create a struct to store parameters of the Level Zero Memory Provider.
/// @param  hParams [out] handle to the newly created parameters struct.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfLevelZeroMemoryProviderParamsCreate(
    umf_level_zero_memory_provider_params_handle_t *hParams);

/// @brief  Destroy parameters struct.
/// @param  hParams handle to the parameters of the Level Zero Memory Provider.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfLevelZeroMemoryProviderParamsDestroy(
    umf_level_zero_memory_provider_params_handle_t hParams);

/// @brief  Set the Level Zero context handle in the parameters struct.
/// @param  hParams handle to the parameters of the Level Zero Memory Provider.
/// @param  hContext handle to the Level Zero context. Cannot be \p NULL.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfLevelZeroMemoryProviderParamsSetContext(
    umf_level_zero_memory_provider_params_handle_t hParams,
    ze_context_handle_t hContext);

/// @brief  Set the Level Zero device handle in the parameters struct.
/// @param  hParams handle to the parameters of the Level Zero Memory Provider.
/// @param  hDevice handle to the Level Zero device. Can be \p NULL if memory type is \p UMF_MEMORY_TYPE_HOST.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfLevelZeroMemoryProviderParamsSetDevice(
    umf_level_zero_memory_provider_params_handle_t hParams,
    ze_device_handle_t hDevice);

/// @brief  Set the memory type in the parameters struct.
/// @param  hParams handle to the parameters of the Level Zero Memory Provider.
/// @param  memoryType memory type.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfLevelZeroMemoryProviderParamsSetMemoryType(
    umf_level_zero_memory_provider_params_handle_t hParams,
    umf_usm_memory_type_t memoryType);

/// @brief  Set the resident devices in the parameters struct.
/// @param  hParams handle to the parameters of the Level Zero Memory Provider.
/// @param  hDevices array of devices for which the memory should be made resident.
/// @param  deviceCount number of devices for which the memory should be made resident.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfLevelZeroMemoryProviderParamsSetResidentDevices(
    umf_level_zero_memory_provider_params_handle_t hParams,
    ze_device_handle_t *hDevices, uint32_t deviceCount);

typedef enum umf_level_zero_memory_provider_free_policy_t {
    UMF_LEVEL_ZERO_MEMORY_PROVIDER_FREE_POLICY_DEFAULT =
        0, ///< Free memory immediately. Default.
    UMF_LEVEL_ZERO_MEMORY_PROVIDER_FREE_POLICY_BLOCKING_FREE, ///< Blocks until all commands using the memory are complete before freeing.
    UMF_LEVEL_ZERO_MEMORY_PROVIDER_FREE_POLICY_DEFER_FREE, ///< Schedules the memory to be freed but does not free immediately.
} umf_level_zero_memory_provider_free_policy_t;

/// @brief  Set the memory free policy.
/// @param  hParams handle to the parameters of the Level Zero Memory Provider.
/// @param  policy memory free policy.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfLevelZeroMemoryProviderParamsSetFreePolicy(
    umf_level_zero_memory_provider_params_handle_t hParams,
    umf_level_zero_memory_provider_free_policy_t policy);

/// @brief  Set the device ordinal in the parameters struct.
/// @param  hParams handle to the parameters of the Level Zero Memory Provider.
/// @param  deviceOrdinal device ordinal.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfLevelZeroMemoryProviderParamsSetDeviceOrdinal(
    umf_level_zero_memory_provider_params_handle_t hParams,
    uint32_t deviceOrdinal);

umf_memory_provider_ops_t *umfLevelZeroMemoryProviderOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_PROVIDER_LEVEL_ZERO_H */

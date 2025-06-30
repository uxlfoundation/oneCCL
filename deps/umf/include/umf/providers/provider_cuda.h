/*
 * Copyright (C) 2024-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef UMF_PROVIDER_CUDA_H
#define UMF_PROVIDER_CUDA_H

#include <umf/memory_provider_gpu.h>

#ifdef __cplusplus
extern "C" {
#endif

struct umf_cuda_memory_provider_params_t;

typedef struct umf_cuda_memory_provider_params_t
    *umf_cuda_memory_provider_params_handle_t;

/// @brief  Create a struct to store parameters of the CUDA Memory Provider.
/// @param  hParams [out] handle to the newly created parameters structure,
///         initialized with the default (current) context and device ID.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfCUDAMemoryProviderParamsCreate(
    umf_cuda_memory_provider_params_handle_t *hParams);

/// @brief  Destroy parameters struct.
/// @param  hParams handle to the parameters of the CUDA Memory Provider.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfCUDAMemoryProviderParamsDestroy(
    umf_cuda_memory_provider_params_handle_t hParams);

/// @brief  Set the CUDA context handle in the parameters struct.
/// @param  hParams handle to the parameters of the CUDA Memory Provider.
/// @param  hContext handle to the CUDA context.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfCUDAMemoryProviderParamsSetContext(
    umf_cuda_memory_provider_params_handle_t hParams, void *hContext);

/// @brief  Set the CUDA device handle in the parameters struct.
/// @param  hParams handle to the parameters of the CUDA Memory Provider.
/// @param  hDevice handle to the CUDA device.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfCUDAMemoryProviderParamsSetDevice(
    umf_cuda_memory_provider_params_handle_t hParams, int hDevice);

/// @brief  Set the memory type in the parameters struct.
/// @param  hParams handle to the parameters of the CUDA Memory Provider.
/// @param  memoryType memory type.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfCUDAMemoryProviderParamsSetMemoryType(
    umf_cuda_memory_provider_params_handle_t hParams,
    umf_usm_memory_type_t memoryType);

/// @brief  Set the allocation flags in the parameters struct.
/// @param  hParams handle to the parameters of the CUDA Memory Provider.
/// @param  flags valid combination of CUDA allocation flags.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfCUDAMemoryProviderParamsSetAllocFlags(
    umf_cuda_memory_provider_params_handle_t hParams, unsigned int flags);

umf_memory_provider_ops_t *umfCUDAMemoryProviderOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_PROVIDER_CUDA_H */

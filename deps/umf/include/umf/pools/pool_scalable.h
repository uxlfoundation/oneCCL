/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_SCALABLE_MEMORY_POOL_H
#define UMF_SCALABLE_MEMORY_POOL_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>

#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

struct umf_scalable_pool_params_t;

/// @brief handle to the parameters of the scalable pool.
typedef struct umf_scalable_pool_params_t *umf_scalable_pool_params_handle_t;

/// @brief  Create a struct to store parameters of scalable pool.
/// @param  hParams [out] handle to the newly created parameters struct.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfScalablePoolParamsCreate(umf_scalable_pool_params_handle_t *hParams);

/// @brief  Destroy parameters struct.
/// @param  hParams handle to the parameters of the scalable pool.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfScalablePoolParamsDestroy(umf_scalable_pool_params_handle_t hParams);

/// @brief  Set granularity of allocations that scalable pool requests from a memory provider.
/// @param  hParams handle to the parameters of the scalable pool.
/// @param  granularity granularity in bytes.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfScalablePoolParamsSetGranularity(umf_scalable_pool_params_handle_t hParams,
                                    size_t granularity);

/// @brief  Set if scalable pool should keep all memory allocated from memory provider till destruction.
/// @param  hParams handle to the parameters of the scalable pool.
/// @param  keepAllMemory \p true if the scalable pool should not call
/// \p umfMemoryProviderFree until it is destroyed, \p false otherwise.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfScalablePoolParamsSetKeepAllMemory(umf_scalable_pool_params_handle_t hParams,
                                      bool keepAllMemory);

/// @brief  Return \p ops structure containing pointers to the scalable pool implementation.
/// @return pointer to the \p umf_memory_pool_ops_t struct.
umf_memory_pool_ops_t *umfScalablePoolOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_SCALABLE_MEMORY_POOL_H */

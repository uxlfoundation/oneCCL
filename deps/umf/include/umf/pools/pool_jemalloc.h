/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_JEMALLOC_MEMORY_POOL_H
#define UMF_JEMALLOC_MEMORY_POOL_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <umf/memory_pool_ops.h>

struct umf_jemalloc_pool_params_t;

/// @brief handle to the optional parameters of the jemalloc pool.
typedef struct umf_jemalloc_pool_params_t *umf_jemalloc_pool_params_handle_t;

/// @brief Create an optional struct to store parameters of jemalloc pool.
/// @param hParams [out] handle to the newly created parameters struct.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfJemallocPoolParamsCreate(umf_jemalloc_pool_params_handle_t *hParams);

/// @brief Destroy parameters struct.
/// @param hParams handle to the parameters of the jemalloc pool.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfJemallocPoolParamsDestroy(umf_jemalloc_pool_params_handle_t hParams);

/// @brief Customize number of arenas created for this pool. Default is the number of CPU cores * 4.
/// \details
/// The number of arenas is limited by jemalloc; setting this value too high may reduce the number of pools available for creation.
/// @param hParams handle to the parameters of the jemalloc pool.
/// @param numArenas number of arenas.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfJemallocPoolParamsSetNumArenas(umf_jemalloc_pool_params_handle_t hParams,
                                  size_t numArenas);

/// @brief Set custom name of the jemalloc pool used in traces.
/// @param hParams handle to the parameters of the jemalloc pool.
/// @param name custom name. Must not be NULL. Name longer than 63 characters
///        will be truncated.
/// \details Name should contain only [a-zA-Z0-9_-] characters.
/// Other names are deprecated and may limit CTL functionality.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfJemallocPoolParamsSetName(umf_jemalloc_pool_params_handle_t hParams,
                             const char *name);

const umf_memory_pool_ops_t *umfJemallocPoolOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_JEMALLOC_MEMORY_POOL_H */

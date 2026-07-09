/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_DISJOINT_MEMORY_POOL_H
#define UMF_DISJOINT_MEMORY_POOL_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

#define UMF_DISJOINT_POOL_MIN_BUCKET_DEFAULT_SIZE ((size_t)8)

/// @brief Memory limits that can be shared between multiple pool instances,
///        i.e. if multiple pools use the same shared limits, sum of those pools'
///        sizes cannot exceed MaxSize.
typedef struct umf_disjoint_pool_shared_limits_t
    *umf_disjoint_pool_shared_limits_handle_t;

struct umf_disjoint_pool_params_t;
/// @brief handle to the parameters of the disjoint pool.
typedef struct umf_disjoint_pool_params_t *umf_disjoint_pool_params_handle_t;

/// @brief Create a pool limits struct.
/// @param MaxSize specifies hard limit for memory allocated from a provider.
/// @param hSharedLimits [out] handle to the newly created shared limits struct.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfDisjointPoolSharedLimitsCreate(
    size_t MaxSize, umf_disjoint_pool_shared_limits_handle_t *hSharedLimits);

/// @brief Destroy previously created pool limits struct.
/// @param hSharedLimits handle to the shared limits struct.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfDisjointPoolSharedLimitsDestroy(
    umf_disjoint_pool_shared_limits_handle_t hSharedLimits);

/// @brief  Create a struct to store parameters of disjoint pool.
/// @param  hParams [out] handle to the newly created parameters struct.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfDisjointPoolParamsCreate(umf_disjoint_pool_params_handle_t *hParams);

/// @brief  Destroy parameters struct.
/// @param  hParams handle to the parameters of the disjoint pool.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfDisjointPoolParamsDestroy(umf_disjoint_pool_params_handle_t hParams);

/// @brief Set minimum allocation size that will be requested from the memory provider.
/// @details Default value for minimum size of slab's is 64KB.
/// @param hParams handle to the parameters of the disjoint pool.
/// @param slabMinSize minimum allocation size.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfDisjointPoolParamsSetSlabMinSize(umf_disjoint_pool_params_handle_t hParams,
                                    size_t slabMinSize);

/// @brief Set size limit for allocations that are subject to pooling.
/// @details Default value for maximum poolable size is 2MB.
/// @param hParams handle to the parameters of the disjoint pool.
/// @param maxPoolableSize maximum poolable size.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfDisjointPoolParamsSetMaxPoolableSize(
    umf_disjoint_pool_params_handle_t hParams, size_t maxPoolableSize);

/// @brief Set maximum capacity of each bucket. Each bucket will hold a
///        max of \p maxCapacity unfreed slabs.
/// @details Default value for capacity is 4.
/// @param hParams handle to the parameters of the disjoint pool.
/// @param maxCapacity maximum capacity of each bucket.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfDisjointPoolParamsSetCapacity(umf_disjoint_pool_params_handle_t hParams,
                                 size_t maxCapacity);

/// @brief Set minimum bucket allocation size.
/// @details Default value for minimum bucket size is 8.
/// @param hParams handle to the parameters of the disjoint pool.
/// @param minBucketSize minimum bucket size. Must be power of 2.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfDisjointPoolParamsSetMinBucketSize(umf_disjoint_pool_params_handle_t hParams,
                                      size_t minBucketSize);

/// @brief Set trace level for pool usage statistics.
/// @details Default value for pool trace is 0 (no traces).
/// @param hParams handle to the parameters of the disjoint pool.
/// @param poolTrace trace level.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfDisjointPoolParamsSetTrace(umf_disjoint_pool_params_handle_t hParams,
                              int poolTrace);

/// @brief Set shared limits for disjoint pool.
/// @param hParams handle to the parameters of the disjoint pool.
/// @param hSharedLimits handle to the shared limits.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfDisjointPoolParamsSetSharedLimits(
    umf_disjoint_pool_params_handle_t hParams,
    umf_disjoint_pool_shared_limits_handle_t hSharedLimits);

/// @brief Set custom name of the disjoint pool to be used in the traces.
/// @param hParams handle to the parameters of the disjoint pool.
/// @param name custom name of the pool. Must not be NULL. Name longer than 63
///        characters will be truncated.
/// \details Name should contain only [a-zA-Z0-9_-] characters.
/// Other names are deprecated and may limit CTL functionality.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t
umfDisjointPoolParamsSetName(umf_disjoint_pool_params_handle_t hParams,
                             const char *name);

const umf_memory_pool_ops_t *umfDisjointPoolOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_DISJOINT_MEMORY_POOL_H */

/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMPOLICY_H
#define UMF_MEMPOLICY_H 1

#include <umf/base.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct umf_mempolicy_t *umf_mempolicy_handle_t;
typedef const struct umf_mempolicy_t *umf_const_mempolicy_handle_t;

typedef enum umf_mempolicy_membind_t {
    /// Interleave memory from all memory in memspace
    UMF_MEMPOLICY_INTERLEAVE,
    /// Bind memory to memspace
    UMF_MEMPOLICY_BIND,
    /// Prefer memory from memspace but fallback to other memory if not available
    UMF_MEMPOLICY_PREFERRED,
    /// Allocation will be split evenly across nodes specified in nodemask.
    /// umf_mempolicy_split_partition_t can be used to specify different distribution.
    UMF_MEMPOLICY_SPLIT
} umf_mempolicy_membind_t;

/// user defined partition for UMF_MEMPOLICY_SPLIT mode
typedef struct umf_mempolicy_split_partition_t {
    /// The weight of the partition, representing the proportion of
    /// the allocation that should be assigned to this NUMA node.
    unsigned weight;
    /// The NUMA node where the pages assigned to this partition will be bound.
    unsigned target;
} umf_mempolicy_split_partition_t;

///
/// @brief Creates a new memory policy
/// @param bind memory binding policy
/// @param hPolicy [out] handle to the newly created memory policy
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMempolicyCreate(umf_mempolicy_membind_t bind,
                                umf_mempolicy_handle_t *hPolicy);

///
/// @brief Destroys memory policy
/// @param hPolicy handle to memory policy
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMempolicyDestroy(umf_mempolicy_handle_t hPolicy);

///
/// @brief Sets custom part size for interleaved memory policy - by default it's interleaved by pages
/// @param hPolicy handle to memory policy
/// @param partSize size of the part or zero to use default part size (page size)
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMempolicySetInterleavePartSize(umf_mempolicy_handle_t hPolicy,
                                               size_t partSize);

///
/// @brief Sets custom split partitions
/// @param hPolicy handle to memory policy
/// @param partList ordered array of partitions
/// @param partListLen length of the partList array
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t
umfMempolicySetCustomSplitPartitions(umf_mempolicy_handle_t hPolicy,
                                     umf_mempolicy_split_partition_t *partList,
                                     size_t partListLen);
#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMPOLICY_H */

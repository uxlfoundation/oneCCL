/*
 * Copyright (C) 2023-2024 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef UMF_COARSE_PROVIDER_H
#define UMF_COARSE_PROVIDER_H

#include <stdbool.h>
#include <string.h>

#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Coarse Memory Provider allocation strategy
typedef enum coarse_memory_provider_strategy_t {
    /// Always allocate a free block of the (size + alignment) size
    /// and cut out the properly aligned part leaving two remaining parts.
    /// It is the fastest strategy but causes memory fragmentation
    /// when alignment is greater than 0.
    /// It is the best strategy when alignment always equals 0.
    UMF_COARSE_MEMORY_STRATEGY_FASTEST = 0,

    /// Check if the first free block of the 'size' size has the correct alignment.
    /// If not, use the `UMF_COARSE_MEMORY_STRATEGY_FASTEST` strategy.
    UMF_COARSE_MEMORY_STRATEGY_FASTEST_BUT_ONE,

    /// Look through all free blocks of the 'size' size
    /// and choose the first one with the correct alignment.
    /// If none of them had the correct alignment,
    /// use the `UMF_COARSE_MEMORY_STRATEGY_FASTEST` strategy.
    UMF_COARSE_MEMORY_STRATEGY_CHECK_ALL_SIZE,

    /// The maximum value (it has to be the last one).
    UMF_COARSE_MEMORY_STRATEGY_MAX
} coarse_memory_provider_strategy_t;

/// @brief Coarse Memory Provider settings struct.
typedef struct coarse_memory_provider_params_t {
    /// Handle to the upstream memory provider.
    /// It has to be NULL if init_buffer is set
    /// (exactly one of them has to be non-NULL).
    umf_memory_provider_handle_t upstream_memory_provider;

    /// Memory allocation strategy.
    /// See coarse_memory_provider_strategy_t for details.
    coarse_memory_provider_strategy_t allocation_strategy;

    /// A pre-allocated buffer that will be the only memory that
    /// the coarse provider can provide (the fixed-size memory provider option).
    /// If it is non-NULL, `init_buffer_size ` has to contain its size.
    /// It has to be NULL if upstream_memory_provider is set
    /// (exactly one of them has to be non-NULL).
    void *init_buffer;

    /// Size of the initial buffer:
    /// 1) `init_buffer` if it is non-NULL xor
    /// 2) that will be allocated from the upstream_memory_provider
    ///    (if it is non-NULL) in the `.initialize` operation.
    size_t init_buffer_size;

    /// When it is true and the upstream_memory_provider is given,
    /// the init buffer (of `init_buffer_size` bytes) would be pre-allocated
    /// during creation time using the `upstream_memory_provider`.
    /// If upstream_memory_provider is not given,
    /// the init_buffer is always used instead
    /// (regardless of the value of this parameter).
    bool immediate_init_from_upstream;

    /// Destroy upstream_memory_provider in finalize().
    bool destroy_upstream_memory_provider;
} coarse_memory_provider_params_t;

/// @brief Coarse Memory Provider stats (TODO move to CTL)
typedef struct coarse_memory_provider_stats_t {
    /// Total allocation size.
    size_t alloc_size;

    /// Size of used memory.
    size_t used_size;

    /// Number of memory blocks allocated from the upstream provider.
    size_t num_upstream_blocks;

    /// Total number of allocated memory blocks.
    size_t num_all_blocks;

    /// Number of free memory blocks.
    size_t num_free_blocks;
} coarse_memory_provider_stats_t;

umf_memory_provider_ops_t *umfCoarseMemoryProviderOps(void);

// TODO use CTL
coarse_memory_provider_stats_t umfCoarseMemoryProviderGetStats(
    umf_memory_provider_handle_t provider);

/// @brief Create default params for the coarse memory provider
static inline coarse_memory_provider_params_t umfCoarseMemoryProviderParamsDefault(void) {
    coarse_memory_provider_params_t coarse_memory_provider_params;
    memset(&coarse_memory_provider_params, 0, sizeof(coarse_memory_provider_params));
    return coarse_memory_provider_params;
}

#ifdef __cplusplus
}
#endif

#endif // UMF_COARSE_PROVIDER_H

/*
 *
 * Copyright (C) 2022-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_OS_MEMORY_PROVIDER_H
#define UMF_OS_MEMORY_PROVIDER_H

#include "umf/memory_provider.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @cond
#define UMF_OS_RESULTS_START_FROM 1000
/// @endcond

/// @brief Memory binding mode
/// Specifies how memory is bound to NUMA nodes on systems that support NUMA.
/// Not every mode is supported on every system.
typedef enum umf_numa_mode_t {
    /// Default binding mode. Actual binding policy is system-specific. On
    /// Linux this corresponds to MPOL_DEFAULT. If this mode is specified,
    /// nodemask must be NULL and maxnode must be 0.
    UMF_NUMA_MODE_DEFAULT,

    /// Restricts memory allocation to nodes specified in nodemask. Allocations
    /// might come from any of the allowed nodes. Nodemask must specify at
    // least one node.
    UMF_NUMA_MODE_BIND,

    /// Interleaves memory allocations across the set of nodes specified in
    /// nodemask. Nodemask must specify at least one node.
    UMF_NUMA_MODE_INTERLEAVE,

    /// Specifies preferred node for allocation. If allocation cannot be
    /// fulfilled, memory will be allocated from other nodes.
    UMF_NUMA_MODE_PREFERRED,

    /// Allocation will be split evenly across nodes specified in nodemask.
    /// umf_numa_split_partition_t can be passed in umf_os_memory_provider_params_t structure
    /// to specify other distribution.
    UMF_NUMA_MODE_SPLIT,
    /// The memory is allocated on the node of the CPU that triggered the
    /// allocation. If this mode is specified, nodemask must be NULL and
    /// maxnode must be 0.
    UMF_NUMA_MODE_LOCAL, // TODO: should this be a hint or strict policy?
} umf_numa_mode_t;

/// @brief This structure specifies a user-defined page distribution
/// within a single allocation in UMF_NUMA_MODE_SPLIT mode.
typedef struct umf_numa_split_partition_t {
    /// The weight of the partition, representing the proportion of
    /// the allocation that should be assigned to this NUMA node.
    unsigned weight;
    /// The NUMA node where the pages assigned to this partition will be bound.
    unsigned target;
} umf_numa_split_partition_t;

struct umf_os_memory_provider_params_t;

typedef struct umf_os_memory_provider_params_t
    *umf_os_memory_provider_params_handle_t;

/// @brief  Create a struct to store parameters of the OS memory provider.
/// @param  hParams [out] handle to the newly created parameters struct.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsCreate(
    umf_os_memory_provider_params_handle_t *hParams);

/// @brief  Destroy parameters struct.
/// @param  hParams handle to the parameters of the OS memory provider.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsDestroy(
    umf_os_memory_provider_params_handle_t hParams);

/// @brief  Set protection flags for the OS memory provider.
/// @param  hParams handle to the parameters of the OS memory provider.
/// @param  protection combination of \p umf_mem_protection_flags_t flags.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsSetProtection(
    umf_os_memory_provider_params_handle_t hParams, unsigned protection);

/// @brief  Set visibility mode for the OS memory provider.
/// @param  hParams handle to the parameters of the OS memory provider.
/// @param  visibility memory visibility mode.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsSetVisibility(
    umf_os_memory_provider_params_handle_t hParams,
    umf_memory_visibility_t visibility);

/// @brief  Set a name of a shared memory file for the OS memory provider.
/// @param  hParams handle to the parameters of the OS memory provider.
/// @param  shm_name a name of a shared memory file.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsSetShmName(
    umf_os_memory_provider_params_handle_t hParams, const char *shm_name);

/// @brief  Set NUMA nodes for the OS memory provider.
/// @param  hParams handle to the parameters of the OS memory provider.
/// @param  numa_list ordered list of NUMA nodes.
/// @param  numa_list_len length of the numa_list.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsSetNumaList(
    umf_os_memory_provider_params_handle_t hParams, unsigned *numa_list,
    unsigned numa_list_len);

/// @brief  Set NUMA mode for the OS memory provider.
/// @param  hParams handle to the parameters of the OS memory provider.
/// @param  numa_mode NUMA mode. Describes how node list is interpreted.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsSetNumaMode(
    umf_os_memory_provider_params_handle_t hParams, umf_numa_mode_t numa_mode);

/// @brief  Set part size for the interleave mode. 0 means default (system specific)
///         It might be rounded up because of HW constraints.
/// @param  hParams handle to the parameters of the OS memory provider.
/// @param  part_size part size for interleave mode.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsSetPartSize(
    umf_os_memory_provider_params_handle_t hParams, size_t part_size);

/// @brief  Set partitions for the split mode.
/// @param  hParams handle to the parameters of the OS memory provider.
/// @param  partitions ordered list of the partitions for the split mode.
/// @param  partitions_len length of the partitions array.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOsMemoryProviderParamsSetPartitions(
    umf_os_memory_provider_params_handle_t hParams,
    umf_numa_split_partition_t *partitions, unsigned partitions_len);

/// @brief OS Memory Provider operation results
typedef enum umf_os_memory_provider_native_error {
    UMF_OS_RESULT_SUCCESS = UMF_OS_RESULTS_START_FROM, ///< Success
    UMF_OS_RESULT_ERROR_ALLOC_FAILED,        ///< Memory allocation failed
    UMF_OS_RESULT_ERROR_ADDRESS_NOT_ALIGNED, ///< Allocated address is not aligned
    UMF_OS_RESULT_ERROR_BIND_FAILED, ///< Binding memory to NUMA node failed
    UMF_OS_RESULT_ERROR_FREE_FAILED, ///< Memory deallocation failed
    UMF_OS_RESULT_ERROR_PURGE_LAZY_FAILED,     ///< Lazy purging failed
    UMF_OS_RESULT_ERROR_PURGE_FORCE_FAILED,    ///< Force purging failed
    UMF_OS_RESULT_ERROR_TOPO_DISCOVERY_FAILED, ///< HWLOC topology discovery failed
} umf_os_memory_provider_native_error_t;

umf_memory_provider_ops_t *umfOsMemoryProviderOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_OS_MEMORY_PROVIDER_H */

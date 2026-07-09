/*
 *
 * Copyright (C) 2023-2024 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMSPACE_H
#define UMF_MEMSPACE_H 1

#include <umf/base.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>
#include <umf/mempolicy.h>
#include <umf/memtarget.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct umf_memspace_t *umf_memspace_handle_t;
typedef const struct umf_memspace_t *umf_const_memspace_handle_t;

///
/// \brief Creates new memory pool from memspace and policy.
/// \param hMemspace handle to memspace
/// \param hPolicy handle to policy
/// \param hPool [out] handle to the newly created memory pool
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfPoolCreateFromMemspace(umf_const_memspace_handle_t hMemspace,
                                       umf_const_mempolicy_handle_t hPolicy,
                                       umf_memory_pool_handle_t *hPool);

///
/// \brief Creates new memory provider from memspace and policy.
/// \param hMemspace handle to memspace
/// \param hPolicy handle to policy
/// \param hProvider [out] handle to the newly created memory provider
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t
umfMemoryProviderCreateFromMemspace(umf_const_memspace_handle_t hMemspace,
                                    umf_const_mempolicy_handle_t hPolicy,
                                    umf_memory_provider_handle_t *hProvider);
///
/// \brief Creates new memspace from array of NUMA node ids.
/// \param nodeIds array of NUMA node ids
/// \param numIds size of the array
/// \param hMemspace [out] handle to the newly created memspace
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMemspaceCreateFromNumaArray(unsigned *nodeIds, size_t numIds,
                                            umf_memspace_handle_t *hMemspace);

///
/// \brief Destroys memspace
/// \param hMemspace handle to memspace
///
void umfMemspaceDestroy(umf_memspace_handle_t hMemspace);

///
/// \brief Retrieves predefined host all memspace.
/// \return host all memspace handle on success or NULL on failure.
///
umf_const_memspace_handle_t umfMemspaceHostAllGet(void);

///
/// \brief Retrieves predefined highest capacity memspace.
/// \return highest capacity memspace handle on success or NULL on failure.
///
umf_const_memspace_handle_t umfMemspaceHighestCapacityGet(void);

/// \brief Retrieves predefined highest bandwidth memspace.
/// \return highest bandwidth memspace handle on success or NULL on
///         failure (no HMAT support).
///
umf_const_memspace_handle_t umfMemspaceHighestBandwidthGet(void);

/// \brief Retrieves predefined lowest latency memspace.
/// \return lowest latency memspace handle on success or NULL on
///         failure (no HMAT support).
///
umf_const_memspace_handle_t umfMemspaceLowestLatencyGet(void);

/// \brief Creates new empty memspace, which can be populated with umfMemspaceMemtargetAdd()
/// \param hMemspace [out] handle to the newly created memspace
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMemspaceNew(umf_memspace_handle_t *hMemspace);

/// \brief Returns number of memory targets in memspace.
/// \param hMemspace handle to memspace
/// \return number of memory targets in memspace
///
size_t umfMemspaceMemtargetNum(umf_const_memspace_handle_t hMemspace);

/// \brief Returns memory target by index.
/// \param hMemspace handle to memspace
/// \param targetNum index of the memory target
/// \return memory target handle on success or NULL on invalid input.
///
umf_const_memtarget_handle_t
umfMemspaceMemtargetGet(umf_const_memspace_handle_t hMemspace,
                        unsigned targetNum);

/// \brief Adds memory target to memspace.
///
/// \details
/// This function duplicates the memory target and then adds it to the memspace.
/// This means that original memtarget handle and the handle of the duplicated memtarget are different
/// and you cannot use it interchangeably.
/// You can use `umfMemspaceMemtargetGet()` to retrieve new handle.
///
/// \param hMemspace handle to memspace
/// \param hMemtarget handle to memory target
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMemspaceMemtargetAdd(umf_memspace_handle_t hMemspace,
                                     umf_const_memtarget_handle_t hMemtarget);

/// \brief Removes memory target from memspace.
///
/// \param hMemspace handle to memspace
/// \param hMemtarget handle to memory target
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t
umfMemspaceMemtargetRemove(umf_memspace_handle_t hMemspace,
                           umf_const_memtarget_handle_t hMemtarget);

/// \brief Clones memspace.
///
/// \param hMemspace handle to memspace
/// \param hNewMemspace [out] handle to the newly created memspace
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMemspaceClone(umf_const_memspace_handle_t hMemspace,
                              umf_memspace_handle_t *hNewMemspace);

/// \brief Custom filter function for umfMemspaceUserFilter
///
/// \param hMemspace handle to memspace
/// \param hMemtarget handle to memory target
/// \param args user provided arguments
/// \return zero if hMemtarget should be removed from memspace, positive otherwise, and negative on error
///
typedef int (*umf_memspace_filter_func_t)(
    umf_const_memspace_handle_t hMemspace,
    umf_const_memtarget_handle_t hMemtarget, void *args);

/// \brief Removes all memory targets with non-matching numa node ids.
///
/// \param hMemspace handle to memspace
/// \param ids array of numa node ids
/// \param size size of the array
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
/// If the error code is UMF_RESULT_UNKNOWN the memspace is corrupted, otherwise the memspace is not modified.
///
umf_result_t umfMemspaceFilterById(umf_memspace_handle_t hMemspace,
                                   unsigned *ids, size_t size);

/// \brief Filters out memory targets that capacity is less than specified size.
///
/// \param hMemspace handle to memspace
/// \param size minimum capacity of memory target
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
/// If the error code is UMF_RESULT_UNKNOWN the memspace is corrupted, otherwise the memspace is not modified.
/// \details Negative values of size parameters are reserved for future
/// extension of functionality of this function.
///
umf_result_t umfMemspaceFilterByCapacity(umf_memspace_handle_t hMemspace,
                                         int64_t size);

/// \brief Filters out memory targets based on user provided function
///
/// \param hMemspace handle to memspace
/// \param filter user provided function
/// \param args user provided arguments
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
/// If the error code is UMF_RESULT_UNKNOWN the memspace is corrupted, otherwise the memspace is not modified.
///
umf_result_t umfMemspaceUserFilter(umf_memspace_handle_t hMemspace,
                                   umf_memspace_filter_func_t filter,
                                   void *args);
#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMSPACE_H */

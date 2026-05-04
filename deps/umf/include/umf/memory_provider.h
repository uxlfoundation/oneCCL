/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMORY_PROVIDER_H
#define UMF_MEMORY_PROVIDER_H 1

#include <umf/base.h>
#include <umf/memory_provider_ops.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Memory visibility mode
typedef enum umf_memory_visibility_t {
    UMF_MEM_MAP_PRIVATE = 1, ///< private memory mapping
    UMF_MEM_MAP_SHARED,      ///< shared memory mapping (Linux only)
} umf_memory_visibility_t;

/// @brief Protection of the memory allocations
typedef enum umf_mem_protection_flag_t {
    UMF_PROTECTION_NONE = (1 << 0),  ///< Memory allocations can not be accessed
    UMF_PROTECTION_READ = (1 << 1),  ///< Memory allocations can be read.
    UMF_PROTECTION_WRITE = (1 << 2), ///< Memory allocations can be written.
    UMF_PROTECTION_EXEC = (1 << 3),  ///< Memory allocations can be executed.
    /// @cond
    UMF_PROTECTION_MAX // must be the last one
    /// @endcond
} umf_mem_protection_flag_t;

/// @brief A struct containing memory provider specific set of functions
typedef struct umf_memory_provider_t *umf_memory_provider_handle_t;

///
/// @brief Creates new memory provider.
/// @param ops instance of umf_memory_provider_ops_t
/// @param params pointer to provider-specific parameters
/// @param hProvider [out] pointer to the newly created memory provider
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMemoryProviderCreate(const umf_memory_provider_ops_t *ops,
                                     const void *params,
                                     umf_memory_provider_handle_t *hProvider);

///
/// @brief Destroys memory provider.
/// @param hProvider handle to the memory provider
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t umfMemoryProviderDestroy(umf_memory_provider_handle_t hProvider);

///
/// @brief Allocates \p size bytes of uninitialized storage from memory \p hProvider
///        with the specified \p alignment
/// @param hProvider handle to the memory provider
/// @param size number of bytes to allocate
/// @param alignment alignment of the allocation in bytes, it has to be a multiple or a divider of the minimum page size
/// @param ptr [out] pointer to the allocated memory
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure
///
umf_result_t umfMemoryProviderAlloc(umf_memory_provider_handle_t hProvider,
                                    size_t size, size_t alignment, void **ptr);

///
/// @brief Frees the memory space pointed by \p ptr from the memory \p hProvider
/// @param hProvider handle to the memory provider
/// @param ptr pointer to the allocated memory to free
/// @param size size of the allocation
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure
///
umf_result_t umfMemoryProviderFree(umf_memory_provider_handle_t hProvider,
                                   void *ptr, size_t size);

///
/// @brief Retrieve string representation of the underlying provider specific
///        result reported by the last API that returned
///        UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC. Allows for a provider
///        independent way to return a provider specific result.
///
/// \details
/// * Implementations *must* store the message and error code in thread-local
///   storage prior to returning UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC.
///
/// * The message and error code will only be valid if a previously
///   called entry-point returned UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC.
///
/// * The memory pointed to by the C string returned in \p ppMessage is owned by
///   the adapter and *must* be null terminated.
///
/// * The application *may* call this function from simultaneous threads.
///
/// * The implementation of this function *should* be lock-free.
/// @param hProvider handle to the memory provider
/// @param ppMessage [out] pointer to a string containing provider specific
///        result in string representation
/// @param pError [out] pointer to an integer where the adapter specific error code will be stored
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t
umfMemoryProviderGetLastNativeError(umf_memory_provider_handle_t hProvider,
                                    const char **ppMessage, int32_t *pError);

///
/// @brief Retrieve recommended page size for a given allocation size.
/// @param hProvider handle to the memory provider
/// @param size allocation size
/// @param pageSize [out] pointer to the recommended page size
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t
umfMemoryProviderGetRecommendedPageSize(umf_memory_provider_handle_t hProvider,
                                        size_t size, size_t *pageSize);

///
/// @brief Retrieve minimum possible page size used by memory region referenced by a given \p ptr
///        or minimum possible page size that can be used by the \p hProvider if \p ptr is NULL.
/// @param hProvider handle to the memory provider
/// @param ptr [optional] pointer to memory allocated by the memory \p hProvider
/// @param pageSize [out] pointer to the minimum possible page size
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t
umfMemoryProviderGetMinPageSize(umf_memory_provider_handle_t hProvider,
                                const void *ptr, size_t *pageSize);

///
/// @brief Discard physical pages within the virtual memory mapping associated at the given addr
///        and \p size. This call is asynchronous and may delay purging the pages indefinitely.
/// @param hProvider handle to the memory provider
/// @param ptr beginning of the virtual memory range
/// @param size size of the virtual memory range
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///         UMF_RESULT_ERROR_INVALID_ALIGNMENT if ptr or size is not page-aligned.
///         UMF_RESULT_ERROR_NOT_SUPPORTED if operation is not supported by this provider.
///
umf_result_t umfMemoryProviderPurgeLazy(umf_memory_provider_handle_t hProvider,
                                        void *ptr, size_t size);

///
/// @brief Discard physical pages within the virtual memory mapping associated at the given addr and \p size.
///        This call is synchronous and if it succeeds, pages are guaranteed to be zero-filled on the next access.
/// @param hProvider handle to the memory provider
/// @param ptr beginning of the virtual memory range
/// @param size size of the virtual memory range
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure
///         UMF_RESULT_ERROR_INVALID_ALIGNMENT if ptr or size is not page-aligned.
///         UMF_RESULT_ERROR_NOT_SUPPORTED if operation is not supported by this provider.
///
umf_result_t umfMemoryProviderPurgeForce(umf_memory_provider_handle_t hProvider,
                                         void *ptr, size_t size);

///
/// @brief Retrieve the size of opaque data structure required to store IPC data.
/// \param hProvider [in] handle to the memory provider.
/// \param size [out] pointer to the size.
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///         UMF_RESULT_ERROR_NOT_SUPPORTED if IPC functionality is not supported by this provider.
umf_result_t
umfMemoryProviderGetIPCHandleSize(umf_memory_provider_handle_t hProvider,
                                  size_t *size);

///
/// @brief Retrieve an IPC memory handle for the specified allocation.
/// \param hProvider [in] handle to the memory provider.
/// \param ptr [in] beginning of the virtual memory range returned by
///                 umfMemoryProviderAlloc function.
/// \param size [in] size of the memory address range.
/// \param providerIpcData [out] pointer to the preallocated opaque data structure to store IPC handle.
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///         UMF_RESULT_ERROR_INVALID_ARGUMENT if ptr was not allocated by this provider.
///         UMF_RESULT_ERROR_NOT_SUPPORTED if IPC functionality is not supported by this provider.
umf_result_t
umfMemoryProviderGetIPCHandle(umf_memory_provider_handle_t hProvider,
                              const void *ptr, size_t size,
                              void *providerIpcData);

///
/// @brief Release IPC handle retrieved with get_ipc_handle function.
/// @param hProvider [in] handle to the memory provider.
/// @param providerIpcData [in] pointer to the IPC opaque data structure.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///         UMF_RESULT_ERROR_INVALID_ARGUMENT if providerIpcData was not created by this provider.
///         UMF_RESULT_ERROR_NOT_SUPPORTED if IPC functionality is not supported by this provider.
umf_result_t
umfMemoryProviderPutIPCHandle(umf_memory_provider_handle_t hProvider,
                              void *providerIpcData);

///
/// @brief Open IPC handle.
/// @param hProvider [in] handle to the memory provider.
/// @param providerIpcData [in] pointer to the IPC opaque data structure.
/// @param ptr [out] pointer to the memory to be used in the current process.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///         UMF_RESULT_ERROR_INVALID_ARGUMENT if providerIpcData cannot be handled by the provider.
///         UMF_RESULT_ERROR_NOT_SUPPORTED if IPC functionality is not supported by this provider.
umf_result_t
umfMemoryProviderOpenIPCHandle(umf_memory_provider_handle_t hProvider,
                               void *providerIpcData, void **ptr);

///
/// @brief Close an IPC memory handle.
/// @param hProvider [in] handle to the memory provider.
/// @param ptr [in] pointer returned by umfMemoryProviderOpenIPCHandle function.
/// @param size [in] size of the memory address range.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///         UMF_RESULT_ERROR_INVALID_ARGUMENT if invalid \p hProvider or \p ptr are passed.
///         UMF_RESULT_ERROR_NOT_SUPPORTED if IPC functionality is not supported by this provider.
umf_result_t
umfMemoryProviderCloseIPCHandle(umf_memory_provider_handle_t hProvider,
                                void *ptr, size_t size);

///
/// @brief Retrieve name of a given memory \p hProvider.
/// @param hProvider handle to the memory provider
/// @param name [out] pointer to the provider name string
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure
umf_result_t umfMemoryProviderGetName(umf_memory_provider_handle_t hProvider,
                                      const char **name);

///
/// @brief Retrieve handle to the last memory provider that returned status other
///        than UMF_RESULT_SUCCESS on the calling thread.
/// @param provider [out] pointer to the handle to the last failed memory provider
///
/// \details Handle to the memory provider is stored in  the thread local
///          storage. The handle is updated every time a memory provider
///          returns status other than UMF_RESULT_SUCCESS.
///
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
///
umf_result_t
umfGetLastFailedMemoryProvider(umf_memory_provider_handle_t *provider);

///
/// @brief Splits a coarse grain allocation into 2 adjacent allocations that
///        can be managed (freed) separately.
/// @param hProvider handle to the memory provider
/// @param ptr pointer to the beginning of the allocation
/// @param totalSize total size of the allocation to be split
/// @param firstSize size of the first new allocation, second allocation
//         has a size equal to totalSize - firstSize
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure
///
umf_result_t
umfMemoryProviderAllocationSplit(umf_memory_provider_handle_t hProvider,
                                 void *ptr, size_t totalSize, size_t firstSize);

///
/// @brief Merges two coarse grain allocations into a single allocation that
///        can be managed (freed) as a whole.
/// @param hProvider handle to the memory provider
/// @param lowPtr pointer to the first allocation
/// @param highPtr pointer to the second allocation (should be > lowPtr)
/// @param totalSize size of a new merged allocation. Should be equal
///        to the sum of sizes of allocations beginning at lowPtr and highPtr
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure
///
umf_result_t
umfMemoryProviderAllocationMerge(umf_memory_provider_handle_t hProvider,
                                 void *lowPtr, void *highPtr, size_t totalSize);

#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMORY_PROVIDER_H */

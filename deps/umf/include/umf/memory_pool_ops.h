/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMORY_POOL_OPS_H
#define UMF_MEMORY_POOL_OPS_H 1

#include <umf/base.h>
#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Version of the Memory Pool ops structure.
/// NOTE: This is equal to the latest UMF version, in which the ops structure
/// has been modified.
#define UMF_POOL_OPS_VERSION_CURRENT UMF_MAKE_VERSION(0, 11)

///
/// @brief This structure comprises function pointers used by corresponding umfPool*
/// calls. Each memory pool implementation should initialize all function
/// pointers.
///
typedef struct umf_memory_pool_ops_t {
    /// Version of the ops structure.
    /// Should be initialized using UMF_POOL_OPS_VERSION_CURRENT.
    uint32_t version;

    ///
    /// @brief Initializes memory pool.
    /// @param providers array of memory providers that will be used for coarse-grain allocations.
    ///        Should contain at least one memory provider.
    /// @param numProvider number of elements in the providers array
    /// @param params pool-specific params
    /// @param pool [out] returns pointer to the pool
    /// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
    ///
    umf_result_t (*initialize)(umf_memory_provider_handle_t provider,
                               void *params, void **pool);

    ///
    /// @brief Finalizes memory pool
    /// @param pool pool to finalize
    ///
    void (*finalize)(void *pool);

    ///
    /// @brief Allocates \p size bytes of uninitialized storage from \p pool
    /// @param pool pointer to the memory pool
    /// @param size number of bytes to allocate
    /// @return Pointer to the allocated memory
    ///
    void *(*malloc)(void *pool, size_t size);

    ///
    /// @brief Allocates memory from \p pool for an array of \p num elements
    ///        of \p size bytes each and initializes all bytes in the allocated storage
    ///        to zero
    /// @param pool pointer to the memory pool
    /// @param num number of objects
    /// @param size number of bytes to allocate for each object
    /// @return Pointer to the allocated memory
    ///
    void *(*calloc)(void *pool, size_t num, size_t size);

    ///
    /// @brief Reallocates memory from \p pool
    /// @param pool pointer to the memory pool
    /// @param ptr pointer to the memory block to be reallocated
    /// @param size new size for the memory block in bytes
    /// @return Pointer to the allocated memory
    ///
    void *(*realloc)(void *pool, void *ptr, size_t size);

    ///
    /// @brief Allocates \p size bytes of uninitialized storage from the specified \p pool
    /// with the specified \p alignment
    /// @param pool pointer to the memory pool
    /// @param size number of bytes to allocate
    /// @param alignment alignment of the allocation in bytes
    /// @return Pointer to the allocated memory
    ///
    void *(*aligned_malloc)(void *pool, size_t size, size_t alignment);

    ///
    /// @brief Obtains size of block of memory allocated from the \p pool for a given \p ptr
    /// @param pool pointer to the memory pool
    /// @param ptr pointer to the allocated memory
    /// @return size of the memory block allocated from the \p pool
    ///
    size_t (*malloc_usable_size)(void *pool, void *ptr);

    ///
    /// @brief Frees the memory space of the specified \p pool pointed by \p ptr
    /// @param pool pointer to the memory pool
    /// @param ptr pointer to the allocated memory to free
    /// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
    ///         Whether any status other than UMF_RESULT_SUCCESS can be returned
    ///         depends on the memory provider used by the \p pool.
    ///
    umf_result_t (*free)(void *pool, void *);

    ///
    /// @brief Retrieve \p umf_result_t representing the error of the last failed allocation
    ///        operation in this thread (malloc, calloc, realloc, aligned_malloc).
    ///
    /// \details
    /// * Implementations *must* store the error code in thread-local
    ///   storage prior to returning NULL from the allocation functions.
    ///
    /// * If the last allocation/de-allocation operation succeeded, the value returned by
    ///   this function is unspecified.
    ///
    /// * The application *may* call this function from simultaneous threads.
    ///
    /// * The implementation of this function *should* be lock-free.
    /// @param pool pointer to the memory pool for which the last allocation error is returned
    /// @return Error code describing the failure of the last failed allocation operation.
    ///         The value is undefined if the previous allocation was successful.
    ///
    umf_result_t (*get_last_allocation_error)(void *pool);
} umf_memory_pool_ops_t;

#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMORY_POOL_OPS_H */

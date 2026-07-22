/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_BASE_H
#define UMF_BASE_H 1

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Generates generic 'UMF' API versions
#define UMF_MAKE_VERSION(_major, _minor)                                       \
    ((_major << 16) | (_minor & 0x0000ffff))

/// @brief Extracts 'UMF' API major version
#define UMF_MAJOR_VERSION(_ver) (_ver >> 16)

/// @brief Extracts 'UMF' API minor version
#define UMF_MINOR_VERSION(_ver) (_ver & 0x0000ffff)

/// @brief Current version of the UMF headers
#define UMF_VERSION_CURRENT UMF_MAKE_VERSION(1, 0)

/// @brief Operation results
typedef enum umf_result_t {
    UMF_RESULT_SUCCESS = 0, ///< Success
    UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY =
        1, ///< Insufficient host memory to satisfy call
    UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC =
        2, /*!< A provider specific warning/error has been reported and can be
              Retrieved via the umfMemoryProviderGetLastNativeError entry point. */
    UMF_RESULT_ERROR_INVALID_ARGUMENT =
        3, ///< Generic error code for invalid arguments
    UMF_RESULT_ERROR_INVALID_ALIGNMENT =
        4,                              ///< Invalid alignment of an argument
    UMF_RESULT_ERROR_NOT_SUPPORTED = 5, ///< Operation not supported
    UMF_RESULT_ERROR_USER_SPECIFIC =
        6, ///< Failure in user provider code (i.e in user provided callback)
    UMF_RESULT_ERROR_DEPENDENCY_UNAVAILABLE =
        7, ///< External required dependency is unavailable or missing
    UMF_RESULT_ERROR_OUT_OF_RESOURCES = 8, ///< Out of internal resources
    UMF_RESULT_ERROR_INVALID_CTL_PATH =
        9, ///< CTL path is not supported or not found
    UMF_RESULT_ERROR_UNKNOWN = 0x7ffffffe ///< Unknown error
} umf_result_t;

/// @brief Handle to the memory properties structure
typedef struct umf_memory_properties_t *umf_memory_properties_handle_t;

/// @brief ID of the memory property
typedef enum umf_memory_property_id_t {
    UMF_MEMORY_PROPERTY_INVALID = -1, ///< Invalid property

    // UMF specific
    UMF_MEMORY_PROPERTY_PROVIDER_HANDLE = 0, ///< Handle to the memory provider
    UMF_MEMORY_PROPERTY_POOL_HANDLE = 1,     ///< Handle to the memory pool

    // generic pointer properties
    UMF_MEMORY_PROPERTY_BASE_ADDRESS = 10, ///< Base address of the allocation
    UMF_MEMORY_PROPERTY_BASE_SIZE = 11,    ///< Base size of the allocation
    UMF_MEMORY_PROPERTY_BUFFER_ID = 12,    ///< Unique identifier for the buffer

    // GPU specific
    UMF_MEMORY_PROPERTY_POINTER_TYPE = 20, ///< Type of the pointer
    UMF_MEMORY_PROPERTY_CONTEXT = 21,      ///< GPU context of the allocation
    UMF_MEMORY_PROPERTY_DEVICE =
        22, ///< GPU device where the allocation resides

    /// @cond
    UMF_MEMORY_PROPERTY_MAX_RESERVED = 0x1000, ///< Maximum reserved value
    /// @endcond
} umf_memory_property_id_t;

/// @brief Type of the CTL query
typedef enum umf_ctl_query_type {
    CTL_QUERY_READ,
    CTL_QUERY_WRITE,
    CTL_QUERY_RUNNABLE,
} umf_ctl_query_type_t;

typedef enum ctl_query_source {
    CTL_UNKNOWN_QUERY_SOURCE,
    /* query executed directly from the program */
    CTL_QUERY_PROGRAMMATIC,
    /* query executed from the config file */
    CTL_QUERY_CONFIG_INPUT
} umf_ctl_query_source_t;

#ifdef __cplusplus
}
#endif

#endif /* UMF_BASE_H */

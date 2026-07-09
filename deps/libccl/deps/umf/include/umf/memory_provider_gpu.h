/*
 *
 * Copyright (C) 2023-2024 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMORY_PROVIDER_GPU_H
#define UMF_MEMORY_PROVIDER_GPU_H 1

#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @brief USM memory allocation type
typedef enum umf_usm_memory_type_t {
    UMF_MEMORY_TYPE_UNKNOWN = 0, ///< The memory pointed to is of unknown type
    UMF_MEMORY_TYPE_HOST,        ///< The memory pointed to is a host allocation
    UMF_MEMORY_TYPE_DEVICE, ///< The memory pointed to is a device allocation
    UMF_MEMORY_TYPE_SHARED, ///< The memory pointed to is a shared ownership allocation
} umf_usm_memory_type_t;

#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMORY_PROVIDER_GPU_H */

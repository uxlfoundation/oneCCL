/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_UNIFIED_MEMORY_FRAMEWORK_H
#define UMF_UNIFIED_MEMORY_FRAMEWORK_H 1

#include <umf/base.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

///
/// @brief  Increment the usage reference counter and initialize the global state of libumf
///         if the usage reference counter was equal to 0.
///         It must be called just after dlopen() and it is not required in other scenarios.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfInit(void);

///
/// @brief Decrement the usage reference counter and destroy the global state of libumf
///        if the usage reference counter is equal to 0.
///        It must be called just before dlclose() and it is not required in other scenarios.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfTearDown(void);

///
/// @brief Get the current version of the UMF headers defined by UMF_VERSION_CURRENT.
/// @return The current version of the UMF headers.
int umfGetCurrentVersion(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_UNIFIED_MEMORY_FRAMEWORK_H */

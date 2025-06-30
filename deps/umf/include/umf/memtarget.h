/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_MEMTARGET_H
#define UMF_MEMTARGET_H 1

#include <stddef.h>
#include <umf/base.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct umf_memtarget_t *umf_memtarget_handle_t;
typedef const struct umf_memtarget_t *umf_const_memtarget_handle_t;

typedef enum umf_memtarget_type_t {
    UMF_MEMTARGET_TYPE_UNKNOWN = 0,
    UMF_MEMTARGET_TYPE_NUMA = 1,
} umf_memtarget_type_t;

/// \brief Gets the type of the memory target.
/// \param hMemtarget handle to the memory target
/// \param type [out] type of the memory target
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfMemtargetGetType(umf_const_memtarget_handle_t hMemtarget,
                                 umf_memtarget_type_t *type);

/// \brief Get size of the memory target in bytes.
/// \param hMemtarget handle to the memory target
/// \param capacity [out] capacity of the memory target
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfMemtargetGetCapacity(umf_const_memtarget_handle_t hMemtarget,
                                     size_t *capacity);

/// \brief Get physical ID of the memory target.
/// \param hMemtarget handle to the memory target
/// \param id [out] id of the memory target
/// \return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfMemtargetGetId(umf_const_memtarget_handle_t hMemtarget,
                               unsigned *id);

#ifdef __cplusplus
}
#endif

#endif /* UMF_MEMTARGET_H */

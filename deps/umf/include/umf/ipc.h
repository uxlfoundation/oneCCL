/*
 *
 * Copyright (C) 2023-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_IPC_H
#define UMF_IPC_H 1

#include <umf/base.h>
#include <umf/memory_pool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct umf_ipc_data_t *umf_ipc_handle_t;

typedef void *umf_ipc_handler_handle_t;

///
/// @brief Returns the size of IPC handles for the specified pool.
/// @param hPool [in] Pool handle
/// @param size [out] size of IPC handle in bytes.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfPoolGetIPCHandleSize(umf_memory_pool_handle_t hPool,
                                     size_t *size);

///
/// @brief Creates an IPC handle for the specified UMF allocation.
/// @param ptr pointer to the allocated memory.
/// @param ipcHandle [out] returned IPC handle.
/// @param size [out] size of IPC handle in bytes.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfGetIPCHandle(const void *ptr, umf_ipc_handle_t *ipcHandle,
                             size_t *size);

///
/// @brief Release IPC handle retrieved by umfGetIPCHandle.
/// @param ipcHandle IPC handle.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfPutIPCHandle(umf_ipc_handle_t ipcHandle);

///
/// @brief Open IPC handle retrieved by umfGetIPCHandle.
/// @param hIPCHandler [in] IPC Handler handle used to open the IPC handle.
/// @param ipcHandle [in] IPC handle.
/// @param ptr [out] pointer to the memory in the current process.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfOpenIPCHandle(umf_ipc_handler_handle_t hIPCHandler,
                              umf_ipc_handle_t ipcHandle, void **ptr);

///
/// @brief Close IPC handle.
/// @param ptr [in] pointer to the memory.
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfCloseIPCHandle(void *ptr);

/// @brief Get handle to the IPC handler from existing pool.
/// @param hPool [in] Pool handle
/// @param hIPCHandler [out] handle to the IPC handler
/// @return UMF_RESULT_SUCCESS on success or appropriate error code on failure.
umf_result_t umfPoolGetIPCHandler(umf_memory_pool_handle_t hPool,
                                  umf_ipc_handler_handle_t *hIPCHandler);

#ifdef __cplusplus
}
#endif

#endif /* UMF_IPC_H */

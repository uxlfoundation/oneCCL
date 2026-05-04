/*
 *
 * Copyright (C) 2024-2025 Intel Corporation
 *
 * Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef UMF_PROXY_MEMORY_POOL_H
#define UMF_PROXY_MEMORY_POOL_H 1

#include <umf/base.h>
#include <umf/memory_pool.h>
#include <umf/memory_provider.h>

#ifdef __cplusplus
extern "C" {
#endif

const umf_memory_pool_ops_t *umfProxyPoolOps(void);

#ifdef __cplusplus
}
#endif

#endif /* UMF_PROXY_MEMORY_POOL_H */

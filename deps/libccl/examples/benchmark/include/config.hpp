/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#pragma once

#define REG_MSG_ALIGNMENT   (4096)
#define LARGE_MSG_ALIGNMENT (2 * 1024 * 1024)
#define LARGE_MSG_THRESHOLD (1 * 1024 * 1024)

#define ALL_COLLS_LIST \
    "allgather,allgatherv,allreduce,alltoall,alltoallv,bcast,broadcast,reduce,reduce_scatter"

#define ALL_DTYPES_LIST "int8,int32,int64,uint64,float16,float32,float64,bfloat16"

#define ALL_REDUCTIONS_LIST            "sum,prod,min,max"
#define ALL_REDUCTIONS_LIST_WITH_CHECK "sum"

#define COL_WIDTH     (14)
#define COL_PRECISION (2)

#ifdef CCL_ENABLE_SYCL
#define DEFAULT_BACKEND BACKEND_SYCL
#else // CCL_ENABLE_SYCL
#define DEFAULT_BACKEND BACKEND_HOST
#endif // CCL_ENABLE_SYCL
#define DEFAULT_ITERS           (16)
#define DEFAULT_WARMUP_ITERS    (16)
#define DEFAULT_ITER_POLICY     ITER_POLICY_AUTO
#define DEFAULT_BUF_COUNT       (1)
#define DEFAULT_MIN_ELEM_COUNT  (0)
#define DEFAULT_MAX_ELEM_COUNT  (128)
#define DEFAULT_ELEM_OFFSET     (0)
#define DEFAULT_CHECK_VALUES    CHECK_LAST_ITER
#define DEFAULT_EXT_VALUES      EXT_AUTO
#define DEFAULT_CACHE_OPS       (1)
#define DEFAULT_INPLACE         (0)
#define DEFAULT_RANKS_PER_PROC  (1)
#define DEFAULT_NUMA_NODE       (-1)
#define DEFAULT_NUMA_NODE_STR   "<default>"
#define DEFAULT_SYCL_DEV_TYPE   SYCL_DEV_GPU
#define DEFAULT_SYCL_ROOT_DEV   (0)
#define DEFAULT_SYCL_MEM_TYPE   SYCL_MEM_USM
#define DEFAULT_SYCL_USM_TYPE   SYCL_USM_DEVICE
#define DEFAULT_SYCL_QUEUE_TYPE SYCL_QUEUE_IN_ORDER

#define DEFAULT_COLL_LIST       "allreduce"
#define DEFAULT_DTYPES_LIST     "float32"
#define DEFAULT_REDUCTIONS_LIST "sum"
#define DEFAULT_VERBOSITY       (0)
#define DEFAULT_CSV_FILEPATH    ""

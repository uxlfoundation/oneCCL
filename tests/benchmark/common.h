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

#ifndef ONECCL_BENCHMARK_COMMON_H
#define ONECCL_BENCHMARK_COMMON_H

#define MPI_SUPPORT 1

#include "oneapi/ccl.h"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <unistd.h>
#include <cstdint>
#include <algorithm>
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <pthread.h>
#include "timer.h"

#define ONECCLCHECK(cmd) do {                                               \
  onecclResult_t const result = cmd;                                        \
  if (result != onecclSuccess) {                                            \
    char hostname[253];                                                    \
    get_host_name(hostname, 253);                                            \
    printf("%s: Test oneCCL failure %s:%d '%s'\n",                          \
           hostname,__FILE__,__LINE__, onecclGetErrorString(result));       \
    return testOneccleError;                                                \
  }                                                                         \
} while(0)

#define SYCLCHECK(cmd) do { \
    try { \
        cmd; \
    } catch(sycl::exception const& e) { \
        char hostname[253]; \
        get_host_name(hostname, 253); \
        printf("%s: Test SYCL failure %s:%d '%s'\n", \
            hostname, __FILE__, __LINE__, e.what()); \
        return testSyclError; \
    } \
} while(0)


// Test result enumeration
typedef enum {
    testSuccess = 0,
    testInternalError = 1,
    testSyclError = 2,
    testOneccleError = 3,
    testTimeout = 4,
    testNumResults = 5
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do { \
    testResult_t const result = cmd; \
    if (result != testSuccess) { \
        char hostname[253]; \
        get_host_name(hostname, 253); \
        printf(" .. %s pid %d: Test failure %s:%d\n", \
            hostname, getpid(), \
            __FILE__,__LINE__); \
        return result; \
    } \
} while(0)

// Forward declaration of thread arguments struct
struct threadArgs;

// Per-collective test descriptor
struct testColl {
    const char name[20];
    void (*getCollByteCount)(
        size_t *sendcount, size_t *recvcount, size_t *paramcount,
        size_t *sendInplaceOffset, size_t *recvInplaceOffset,
        size_t count, size_t eltSize, int nranks);
    testResult_t (*initData)(struct threadArgs* args, onecclDataType_t type,
        onecclRedOp_t op, int root, int rep, int in_place);
    void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
    testResult_t (*runColl)(void* sendbuff, void* recvbuff, size_t count, onecclDataType_t type,
        onecclRedOp_t op, int root, int rank, onecclComm_t comm, sycl::queue* stream);
};

// Test engine structure
struct testEngine {
    void (*getBuffSize)(size_t *sendcount, size_t *recvcount, size_t count, int nranks);
    testResult_t (*runTest)(struct threadArgs* args, int root, onecclDataType_t type,
        const char* typeName, onecclRedOp_t op, const char* opName);
};

extern struct testEngine* currentTestEngine;

// Arguments for each thread
struct threadArgs {
    size_t nbytes;
    size_t minbytes;
    size_t maxbytes;
    size_t stepbytes;
    size_t stepfactor;

    int totalProcs;
    int nProcs;
    int proc;
    int nThreads;
    int thread;
    int nGpus;
    int* gpus;
    int localRank;
    void** sendbuffs;
    size_t sendBytes;
    size_t sendInplaceOffset;
    void** recvbuffs;
    size_t recvInplaceOffset;
    onecclUniqueId cclId;
    onecclComm_t* comms;
    sycl::queue** queues;

    void** expected;
    size_t expectedBytes;
    int* errors;
    double* bw;
    int* bw_count;

    int reportErrors;
    int runInplace;

    struct testColl* collTest;
};

// Thread function pointer
typedef testResult_t (*threadFunc_t)(struct threadArgs* args);

// Thread structure
struct testThread {
    pthread_t thread;
    threadFunc_t func;
    struct threadArgs args;
    testResult_t ret;
};

// Common utility functions
extern void barrier(struct threadArgs* args);
extern testResult_t timeTest(struct threadArgs* args, onecclDataType_t type, const char* typeName, onecclRedOp_t op, const char* opName, int root);
extern testResult_t initData(sycl::queue* queue, void* data, const size_t count, onecclDataType_t type, int rank, int nranks);
extern testResult_t initDataReduce(sycl::queue* queue, void* data, const size_t count, onecclDataType_t type, onecclRedOp_t operation, int nranks);
extern testResult_t checkData(sycl::queue* queue, void* recvbuff, void* expected, const size_t count, onecclDataType_t type, int64_t* wrongElts);
extern testResult_t allocateBuffs(sycl::queue* queue, void **sendbuff, size_t sendBytes, void **recvbuff, size_t recvBytes, void **expected, size_t maxbytes);

// Utility to get a simplified hostname
static void get_host_name(char* hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i=0; i< maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

static size_t word_size(onecclDataType_t type) {
    switch(type) {
        case onecclInt8:
        case onecclUint8:
            return 1;
        case onecclFloat16:
        case onecclBfloat16:
            return 2;
        case onecclInt32:
        case onecclUint32:
        case onecclFloat32:
            return 4;
        case onecclInt64:
        case onecclUint64:
        case onecclFloat64:
            return 8;
        default: return 0;
    }
}

// Globals for supported types and operations
extern int test_opnum;
extern int test_typenum;
extern onecclDataType_t test_types[];
extern const char *test_typenames[];
extern onecclRedOp_t test_ops[];
extern const char *test_opnames[];

// Convert string to oneCCL data type index
static int oneccl_string_to_type(char *str) {
    for (int t=0; t<test_typenum; t++) {
        if (strcmp(str, test_typenames[t]) == 0) return t;
    }
    if (strcmp(str, "all") == 0) return -1;
    printf("invalid type %s, defaulting to %s .. \n", str, test_typenames[onecclFloat32]);
    return onecclFloat32;
}

// Convert string to oneCCL reduction operation index
static int oneccl_string_to_op(char *str) {
    for (int o=0; o<test_opnum; o++) {
        if (strcmp(str, test_opnames[o]) == 0) return o;
    }
    if (strcmp(str, "all") == 0) return -1;
    printf("invalid op %s, defaulting to %s .. \n", str, test_opnames[onecclSum]);
    return onecclSum;
}

extern int is_main_proc;
extern thread_local int is_main_thread;
#define PRINT if (is_main_thread) printf

#endif // ONECCL_BENCHMARK_COMMON_H

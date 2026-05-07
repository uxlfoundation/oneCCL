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

#include "common.h"
#include "oneapi/ccl.h"

namespace {

// Byte count calculations for broadcast
void BcastGetCollByteCount(size_t *sendcount, size_t *recvcount,
                           size_t *paramcount, size_t *sendInplaceOffset,
                           size_t *recvInplaceOffset, size_t count,
                           size_t eltSize, int nranks) {
    *sendcount = count;
    *recvcount = count;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
}

// Initialize buffers for broadcast
testResult_t BcastInitData(struct threadArgs *args, onecclDataType_t type,
                           onecclRedOp_t op, int root, int rep, int in_place) {
    size_t sendcount = args->sendBytes / word_size(type);
    size_t recvcount = args->expectedBytes / word_size(type);

    for (int i = 0; i < args->nGpus; i++) {
        int rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        SYCLCHECK(args->queues[i]
                      ->memset(args->recvbuffs[i], 0, args->expectedBytes)
                      .wait());
        void *data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
        if (rank == root) {
            TESTCHECK(initData(args->queues[i], data, sendcount, type, rank,
                               args->nProcs * args->nThreads * args->nGpus));
        }
        TESTCHECK(initData(args->queues[i], args->expected[i], recvcount, type,
                           root, args->nProcs * args->nThreads * args->nGpus));
        SYCLCHECK(args->queues[i]->wait());
    }
    return testSuccess;
}

// Bandwidth calculation for broadcast
void BcastGetBw(size_t count, int typesize, double sec, double *algBw,
                double *busBw, int nranks) {
    double baseBw = (double)(count * typesize) / 1.0E9 / sec;
    *algBw = baseBw;
    double factor = 1;
    *busBw = baseBw * factor;
}

// Run the actual broadcast collective
testResult_t BcastRunColl(void *sendbuff, void *recvbuff, size_t count,
                          onecclDataType_t type, onecclRedOp_t op, int root,
                          int rank, onecclComm_t comm, sycl::queue *stream) {
    ONECCLCHECK(
        onecclBroadcast(sendbuff, recvbuff, count, type, root, comm, stream));
    return testSuccess;
}

// Struct for the broadcast test
struct testColl bcastTest = {"Broadcast", BcastGetCollByteCount, BcastInitData,
                             BcastGetBw, BcastRunColl};

void BcastGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count,
                      int nranks) {
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    BcastGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset,
                          &recvInplaceOffset, count, /*eltSize=*/16, nranks);
}

// Main test loop for broadcast
testResult_t BcastRunTest(struct threadArgs *args, int root,
                          onecclDataType_t type, const char *typeName,
                          onecclRedOp_t op, const char *opName) {
    args->collTest = &bcastTest;
    onecclDataType_t *run_types;
    const char **run_typenames;
    int type_count;
    int begin_root, end_root;

    if ((int)type != -1) {
        type_count = 1;
        run_types = &type;
        run_typenames = &typeName;
    } else {
        type_count = test_typenum;
        run_types = test_types;
        run_typenames = test_typenames;
    }

    if (root != -1) {
        begin_root = end_root = root;
    } else {
        begin_root = 0;
        end_root = args->nProcs * args->nThreads * args->nGpus - 1;
    }

    for (int i = 0; i < type_count; i++) {
        for (int j = begin_root; j <= end_root; j++) {
            TESTCHECK(timeTest(args, run_types[i], run_typenames[i],
                               (onecclRedOp_t)0, "none", j));
        }
    }
    return testSuccess;
}

} // anonymous namespace

struct testEngine bcastEngine = {BcastGetBuffSize, BcastRunTest};

// Set this as the current engine for the broadcast executable
__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &bcastEngine;
}

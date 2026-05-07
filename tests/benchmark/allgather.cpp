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

namespace {

void AllGatherGetCollByteCount(size_t *sendcount, size_t *recvcount,
                               size_t *paramcount, size_t *sendInplaceOffset,
                               size_t *recvInplaceOffset, size_t count,
                               size_t /*eltSize*/, int nranks) {
    // count = per-rank element count
    *sendcount = count;          // elements each rank sends
    *recvcount = count * nranks; // elements each rank receives
    *paramcount = count;         // API count argument
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
}

testResult_t AllGatherInitData(struct threadArgs *args, onecclDataType_t type,
                               onecclRedOp_t op, int root, int rep,
                               int in_place) {
    size_t sendcount = args->sendBytes / word_size(type);     // base
    size_t recvcount = args->expectedBytes / word_size(type); // base * nranks
    int nranks = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++) {
        int rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);

        // Clear receive buffer
        SYCLCHECK(args->queues[i]
                      ->memset(args->recvbuffs[i], 0, args->expectedBytes)
                      .wait());

        if (in_place) {
            // For in-place, initialize this rank's data in its correct position
            // within recvbuff Each rank's data starts at rank * sendcount
            // elements
            void *rank_data_pos =
                (char *)args->recvbuffs[i] + rank * sendcount * word_size(type);
            TESTCHECK(initData(args->queues[i], rank_data_pos, sendcount, type,
                               33 * rep + rank, nranks));
        } else {
            // For out-of-place, initialize send buffer normally
            TESTCHECK(initData(args->queues[i], args->sendbuffs[i], sendcount,
                               type, 33 * rep + rank, nranks));
        }

        // Initialize expected data - each rank contributes its data to all
        // positions
        for (int j = 0; j < nranks; j++) {
            void *expected_pos =
                (char *)args->expected[i] + j * sendcount * word_size(type);
            TESTCHECK(initData(args->queues[i], expected_pos, sendcount, type,
                               33 * rep + j, nranks));
        }

        SYCLCHECK(args->queues[i]->wait());
    }
    return testSuccess;
}

void AllGatherGetBw(size_t count, int typesize, double sec, double *algBw,
                    double *busBw, int nranks) {
    double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)nranks);
    *busBw = baseBw * factor;
}

testResult_t AllGatherRunColl(void *sendbuff, void *recvbuff, size_t count,
                              onecclDataType_t type, onecclRedOp_t op, int root,
                              int rank, onecclComm_t comm, sycl::queue *queue) {
    void *actualSendBuff = sendbuff;

    // Handle in-place mode: if sendbuff == recvbuff, we need to use the rank's
    // portion as send buffer
    if (sendbuff == recvbuff) {
        // For in-place AllGather, sendbuff should point to this rank's data
        // within recvbuff
        actualSendBuff = (char *)recvbuff + rank * count * word_size(type);
    }

    ONECCLCHECK(
        onecclAllGather(actualSendBuff, recvbuff, count, type, comm, queue));
    return testSuccess;
}

struct testColl allGatherTest = {"AllGather", AllGatherGetCollByteCount,
                                 AllGatherInitData, AllGatherGetBw,
                                 AllGatherRunColl};

void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count,
                          int nranks) {
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    AllGatherGetCollByteCount(sendcount, recvcount, &paramcount,
                              &sendInplaceOffset, &recvInplaceOffset, count,
                              /*eltSize=*/16, nranks);
}

testResult_t AllGatherRunTest(struct threadArgs *args, int root,
                              onecclDataType_t type, const char *typeName,
                              onecclRedOp_t op, const char *opName) {
    args->collTest = &allGatherTest;
    onecclDataType_t *run_types;
    const char **run_typenames;
    int type_count;

    if ((int)type != -1) {
        type_count = 1;
        run_types = &type;
        run_typenames = &typeName;
    } else {
        type_count = test_typenum;
        run_types = test_types;
        run_typenames = test_typenames;
    }

    for (int i = 0; i < type_count; i++) {
        TESTCHECK(timeTest(args, run_types[i], run_typenames[i],
                           (onecclRedOp_t)0, "none", -1));
    }
    return testSuccess;
}

} // anonymous namespace

struct testEngine allGatherEngine = {AllGatherGetBuffSize, AllGatherRunTest};

// Set this as the current engine for the allgather executable
__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &allGatherEngine;
}

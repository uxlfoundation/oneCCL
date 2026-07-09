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

void AllToAllGetCollByteCount(size_t *sendcount, size_t *recvcount,
                              size_t *paramcount, size_t *sendInplaceOffset,
                              size_t *recvInplaceOffset, size_t count,
                              size_t /*eltSize*/, int nranks) {
    // count = per-peer element count
    *sendcount = count * nranks;
    *recvcount = count * nranks;
    *paramcount = count;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
}

testResult_t AllToAllInitData(struct threadArgs *args, onecclDataType_t type,
                              onecclRedOp_t /*op*/, int /*root*/, int rep,
                              int in_place) {
    size_t const count = args->nbytes / word_size(type);
    int const nranks = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++) {
        int const rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);

        SYCLCHECK(args->queues[i]
                      ->memset(args->recvbuffs[i], 0, args->expectedBytes)
                      .wait());

        void *data_to_init =
            (in_place != 0) ? args->recvbuffs[i] : args->sendbuffs[i];
        TESTCHECK(initData(args->queues[i], data_to_init, count * nranks, type,
                           33 * rep + rank, nranks));

        for (int peer = 0; peer < nranks; peer++) {
            void *expected_pos =
                static_cast<char *>(args->expected[i]) +
                static_cast<size_t>(peer) * count * word_size(type);
            TESTCHECK(initData(args->queues[i], expected_pos, count, type,
                               33 * rep + peer, nranks));
        }

        SYCLCHECK(args->queues[i]->wait());
    }
    return testSuccess;
}

void AllToAllGetBw(size_t count, int typesize, double sec, double *algBw,
                   double *busBw, int nranks) {
    double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;
    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)nranks);
    *busBw = baseBw * factor;
}

testResult_t AllToAllRunColl(void *sendbuff, void *recvbuff, size_t count,
                             onecclDataType_t type, onecclRedOp_t /*op*/,
                             int /*root*/, int /*rank*/, onecclComm_t comm,
                             sycl::queue *queue) {
    ONECCLCHECK(onecclAllToAll(sendbuff, recvbuff, count, type, comm, queue));
    return testSuccess;
}

struct testColl allToAllTest = {"AllToAll", AllToAllGetCollByteCount,
                                AllToAllInitData, AllToAllGetBw,
                                AllToAllRunColl};

void AllToAllGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count,
                         int nranks) {
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    AllToAllGetCollByteCount(sendcount, recvcount, &paramcount,
                             &sendInplaceOffset, &recvInplaceOffset, count,
                             /*eltSize=*/16, nranks);
}

testResult_t AllToAllRunTest(struct threadArgs *args, int /*root*/,
                             onecclDataType_t type, const char *typeName,
                             onecclRedOp_t /*op*/, const char * /*opName*/) {
    args->collTest = &allToAllTest;
    onecclDataType_t *run_types = nullptr;
    const char **run_typenames = nullptr;
    int type_count = 0;

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

struct testEngine allToAllEngine = {AllToAllGetBuffSize, AllToAllRunTest};

__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &allToAllEngine;
}

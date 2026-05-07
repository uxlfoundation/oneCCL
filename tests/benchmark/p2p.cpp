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

void P2PGetCollByteCount(size_t *sendcount, size_t *recvcount,
                         size_t *paramcount, size_t *sendInplaceOffset,
                         size_t *recvInplaceOffset, size_t count,
                         size_t /*eltSize*/, int /*nranks*/) {
    *sendcount = count;
    *recvcount = count;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
}

// initialize: each rank’s send buffer gets its own pattern; expected is pattern
// from recvPeer=(rank-1+nranks)%nranks
testResult_t P2PInitData(struct threadArgs *args, onecclDataType_t type,
                         onecclRedOp_t /*op*/, int /*root*/, int /*rep*/,
                         int in_place) {
    size_t sendcount = args->sendBytes / word_size(type);
    size_t recvcount = args->expectedBytes / word_size(type);
    int nranks = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++) {
        int rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);

        // clear recv buffer
        SYCLCHECK(args->queues[i]
                      ->memset(args->recvbuffs[i], 0, args->expectedBytes)
                      .wait());

        // in-place is not meaningful for p2p ring; keep for interface
        // compatibility but disable error checking if set.
        void *data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

        // fill send buffer with this rank’s pattern
        TESTCHECK(
            initData(args->queues[i], data, sendcount, type, rank, nranks));

        // expected data equals what we receive from recvPeer
        int recvPeer = (rank - 1 + nranks) % nranks;
        TESTCHECK(initData(args->queues[i], args->expected[i], recvcount, type,
                           recvPeer, nranks));

        SYCLCHECK(args->queues[i]->wait());
    }

    // Do not report errors for in-place (unsupported/ignored)
    args->reportErrors = in_place ? 0 : 1;
    return testSuccess;
}

void P2PGetBw(size_t count, int typesize, double sec, double *algBw,
              double *busBw, int /*nranks*/) {
    double baseBw = (double)(count * typesize) / 1.0E9 / sec;
    *algBw = baseBw;
    *busBw = baseBw;
}

// bidirectional ring using group semantics: send to (rank+1)%nranks, recv from
// (rank-1+nranks)%nranks
testResult_t P2PRunColl(void *sendbuff, void *recvbuff, size_t count,
                        onecclDataType_t type, onecclRedOp_t /*op*/,
                        int /*root*/, int rank, onecclComm_t comm,
                        sycl::queue *stream) {
    int nRanks = 0;
    ONECCLCHECK(onecclCommCount(comm, &nRanks));

    int recvPeer = (rank - 1 + nRanks) % nRanks;
    int sendPeer = (rank + 1) % nRanks;

    ONECCLCHECK(onecclGroupStart());
    ONECCLCHECK(onecclSend(sendbuff, count, type, sendPeer, comm, stream));
    ONECCLCHECK(onecclRecv(recvbuff, count, type, recvPeer, comm, stream));
    ONECCLCHECK(onecclGroupEnd());

    return testSuccess;
}

struct testColl sendRecvTest = {"SendRecv", P2PGetCollByteCount, P2PInitData,
                                P2PGetBw, P2PRunColl};

void P2PGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count,
                    int nranks) {
    size_t paramcount, sOff, rOff;
    P2PGetCollByteCount(sendcount, recvcount, &paramcount, &sOff, &rOff, count,
                        /*eltSize=*/16, nranks);
}

testResult_t P2PRunTest(struct threadArgs *args, int /*root*/,
                        onecclDataType_t type, const char *typeName,
                        onecclRedOp_t op, const char *opName) {
    args->collTest = &sendRecvTest;

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

    // run across all types; op/opName are ignored for p2p but kept for
    // interface compatibility
    for (int i = 0; i < type_count; i++) {
        TESTCHECK(timeTest(args, run_types[i], run_typenames[i], op, opName,
                           /*root*/ -1));
    }
    return testSuccess;
}

} // namespace

struct testEngine p2pEngine = {P2PGetBuffSize, P2PRunTest};

__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &p2pEngine;
}

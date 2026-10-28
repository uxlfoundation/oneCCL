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

// Byte count calculation for point-to-point (sender -> receiver)
void P2PGetCollByteCount(size_t *sendcount, size_t *recvcount,
                         size_t *paramcount, size_t *sendInplaceOffset,
                         size_t *recvInplaceOffset, size_t count,
                         size_t /*eltSize*/, int /*nranks*/) {
    *sendcount = count;
    *recvcount = count;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount; // measure one transfer
}

// initialize buffers: root fills sendbuff, peer zeros recvbuff.
testResult_t P2PInitData(struct threadArgs *args, onecclDataType_t type,
                         onecclRedOp_t /*op*/, int root, int /*rep*/,
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

        // note: "in-place" handling here exists only for interface
        // compatibility with common.cpp. semantics do not apply and provide no
        // benefit. the flag is ignored for p2p.
        void *sendptr = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

        if (rank == root) {
            TESTCHECK(initData(args->queues[i], sendptr, sendcount, type, rank,
                               nranks));
        }

        TESTCHECK(initData(args->queues[i], args->expected[i], recvcount, type,
                           root, nranks));
        SYCLCHECK(args->queues[i]->wait());
    }
    return testSuccess;
}

void P2PGetBw(size_t count, int typesize, double sec, double *algBw,
              double *busBw, int /*nranks*/) {
    double base = (double)count * typesize / 1.0E9 / sec;
    *algBw = base;
    *busBw = base;
}

// Run one send/recv pair for given root (sender)
testResult_t P2PRunColl(void *sendbuff, void *recvbuff, size_t count,
                        onecclDataType_t type, onecclRedOp_t /*op*/, int root,
                        int rank, onecclComm_t comm, sycl::queue *stream) {
    // expect exactly 2 ranks for this benchmark
    // Root sends, peer receives.
    if (rank == root) {
        // find peer rank (0->1 or 1->0)
        int peer = (root == 0) ? 1 : 0;
        ONECCLCHECK(onecclSend(sendbuff, count, type, peer, comm, stream));
    } else {
        ONECCLCHECK(onecclRecv(recvbuff, count, type, root, comm, stream));
    }
    return testSuccess;
}

// testColl descriptor
struct testColl p2pTest = {"PointToPoint", P2PGetCollByteCount, P2PInitData,
                           P2PGetBw, P2PRunColl};

// buffer size helper
void P2PGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count,
                    int nranks) {
    size_t paramcount, sOff, rOff;
    P2PGetCollByteCount(sendcount, recvcount, &paramcount, &sOff, &rOff, count,
                        1, nranks);
}

testResult_t P2PRunTest(struct threadArgs *args, int root,
                        onecclDataType_t type, const char *typeName,
                        onecclRedOp_t /*op*/, const char * /*opName*/) {
    args->collTest = &p2pTest;
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

    int begin_root, end_root;
    if (root != -1) {
        begin_root = end_root = root;
    } else {
        begin_root = 0;
        end_root = 1; // only 2 ranks supported
    }

    for (int i = 0; i < type_count; i++) {
        for (int r = begin_root; r <= end_root; r++) {
            TESTCHECK(timeTest(args, run_types[i], run_typenames[i],
                               (onecclRedOp_t)0, "none", r));
        }
    }
    return testSuccess;
}

} // namespace

struct testEngine p2pEngine = {P2PGetBuffSize, P2PRunTest};

__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &p2pEngine;
}

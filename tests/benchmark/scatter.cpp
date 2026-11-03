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

// Scatter performance benchmark: root-only-send collective.
// Root distributes count elements to each of nranks receivers.
#include "common.h"

namespace {

void ScatterGetCollByteCount(size_t *sendcount, size_t *recvcount,
                             size_t *paramcount, size_t *sendInplaceOffset,
                             size_t *recvInplaceOffset, size_t count,
                             size_t eltSize, int nranks) {
    // count = whole-buffer element count, as in nccl-tests
    size_t base = (count / nranks) & -(16 / eltSize);
    *sendcount = base * nranks;
    *recvcount = base;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = base;
}

testResult_t ScatterInitData(struct threadArgs *args, onecclDataType_t type,
                             onecclRedOp_t op, int root, int rep,
                             int in_place) {
    size_t recvcount = args->expectedBytes / word_size(type);
    int nranks = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++) {
        int rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);

        SYCLCHECK(args->queues[i]
                      ->memset(args->recvbuffs[i], 0, args->expectedBytes)
                      .wait());

        // Only root supplies the send buffer; it holds count*nranks elements.
        // In-place aliases the result into this buffer at root*count, so it is
        // initialized for both placements.
        if (rank == root) {
            TESTCHECK(initData(args->queues[i], args->sendbuffs[i],
                               recvcount * nranks, type, rank, nranks));
        }

        // Expected: root fills uniformly with (root+1); every rank receives
        // that value.
        TESTCHECK(initData(args->queues[i], args->expected[i], recvcount, type,
                           root, nranks));

        SYCLCHECK(args->queues[i]->wait());
    }
    return testSuccess;
}

void ScatterGetBw(size_t count, int typesize, double sec, double *algBw,
                  double *busBw, int nranks) {
    double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)nranks);
    *busBw = baseBw * factor;
}

testResult_t ScatterRunColl(void *sendbuff, void *recvbuff, size_t count,
                            onecclDataType_t type, onecclRedOp_t op, int root,
                            int rank, onecclComm_t comm, sycl::queue *stream) {
    // Non-root ranks supply no send buffer. In in-place mode the root's recv
    // aliases into its own block of the send buffer (recv == send +
    // root*count); oneccl_start_coll passes recvbuff == sendbuff for scatter so
    // we recover the slice here.
    const void *send = (rank == root) ? sendbuff : nullptr;
    void *recv = recvbuff;
    if (rank == root && sendbuff == recvbuff) {
        recv = static_cast<char *>(recvbuff) +
               static_cast<size_t>(root) * count * word_size(type);
    }
    ONECCLCHECK(onecclScatter(send, recv, count, type, root, comm, stream));
    return testSuccess;
}

struct testColl scatterTest = {"Scatter", ScatterGetCollByteCount,
                               ScatterInitData, ScatterGetBw, ScatterRunColl};

void ScatterGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count,
                        int nranks) {
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    ScatterGetCollByteCount(sendcount, recvcount, &paramcount,
                            &sendInplaceOffset, &recvInplaceOffset, count, 16,
                            nranks);
}

testResult_t ScatterRunTest(struct threadArgs *args, int root,
                            onecclDataType_t type, const char *typeName,
                            onecclRedOp_t op, const char *opName) {
    args->collTest = &scatterTest;
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
        begin_root = root;
        end_root = root;
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

} // namespace

struct testEngine scatterEngine = {ScatterGetBuffSize, ScatterRunTest};

__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &scatterEngine;
}

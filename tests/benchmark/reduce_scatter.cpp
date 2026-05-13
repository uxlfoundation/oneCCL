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

void ReduceScatterGetCollByteCount(size_t *sendcount, size_t *recvcount,
                                   size_t *paramcount,
                                   size_t *sendInplaceOffset,
                                   size_t *recvInplaceOffset, size_t count,
                                   size_t eltSize, int nranks) {

    size_t base = (count / nranks) & -(16 / eltSize);
    *sendcount = base * nranks;
    *recvcount = base;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = base;
}

testResult_t ReduceScatterInitData(struct threadArgs *args,
                                   onecclDataType_t type, onecclRedOp_t op,
                                   int root, int rep, int in_place) {
    size_t sendcount =
        args->sendBytes / word_size(type); // == recvcount * nranks
    size_t recvcount = args->expectedBytes / word_size(type);
    int nranks = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++) {
        int rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);

        // Clear receive segment (recvcount elements worth of bytes)
        SYCLCHECK(args->queues[i]
                      ->memset(args->recvbuffs[i], 0, args->expectedBytes)
                      .wait());

        void *data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];

        // Initialize full send buffer region (sendcount elements) with rank+1
        // pattern
        TESTCHECK(
            initData(args->queues[i], data, sendcount, type, rank, nranks));

        // Prepare expected (recvcount elements) = reduction of rank+1 over all
        // ranks
        SYCLCHECK(args->queues[i]
                      ->memcpy(args->expected[i], args->recvbuffs[i],
                               args->expectedBytes)
                      .wait());
        TESTCHECK(initDataReduce(args->queues[i], args->expected[i], recvcount,
                                 type, op, nranks));

        SYCLCHECK(args->queues[i]->wait());
    }
    return testSuccess;
}

void ReduceScatterGetBw(size_t count, int typesize, double sec, double *algBw,
                        double *busBw, int nranks) {
    double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)nranks);
    *busBw = baseBw * factor;
}

testResult_t ReduceScatterRunColl(void *sendbuff, void *recvbuff, size_t count,
                                  onecclDataType_t type, onecclRedOp_t op,
                                  int root, int rank, onecclComm_t comm,
                                  sycl::queue *queue) {
    /* In-place semantics adaptation:
       The same pointer is passed for sendbuff and recvbuff in in-place mode.
       oneCCL reduce_scatter expects:
         - sendbuff: base of concatenated input (nranks * recvcount elements)
         - recvbuff: start of this rank's recvcount slice (offset
       rank*recvcount)
       If we detect sendbuff == recvbuff, adjust recv pointer by
       rank*count*typesize. */
    if (sendbuff == recvbuff) {
        size_t typesize = word_size(type);
        void *recv_slice = static_cast<char *>(recvbuff) +
                           static_cast<size_t>(rank) * count * typesize;
        ONECCLCHECK(onecclReduceScatter(sendbuff, recv_slice, count, type, op,
                                        comm, queue));
    } else {
        ONECCLCHECK(onecclReduceScatter(sendbuff, recvbuff, count, type, op,
                                        comm, queue));
    }
    return testSuccess;
}

struct testColl reduceScatterTest = {
    "ReduceScatter", ReduceScatterGetCollByteCount, ReduceScatterInitData,
    ReduceScatterGetBw, ReduceScatterRunColl};

void ReduceScatterGetBuffSize(size_t *sendcount, size_t *recvcount,
                              size_t count, int nranks) {
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    ReduceScatterGetCollByteCount(sendcount, recvcount, &paramcount,
                                  &sendInplaceOffset, &recvInplaceOffset, count,
                                  16, nranks);
}

testResult_t ReduceScatterRunTest(struct threadArgs *args, int root,
                                  onecclDataType_t type, const char *typeName,
                                  onecclRedOp_t op, const char *opName) {
    args->collTest = &reduceScatterTest;
    onecclDataType_t *run_types;
    onecclRedOp_t *run_ops;
    const char **run_typenames, **run_opnames;
    int type_count, op_count;

    if ((int)type != -1) {
        type_count = 1;
        run_types = &type;
        run_typenames = &typeName;
    } else {
        type_count = test_typenum;
        run_types = test_types;
        run_typenames = test_typenames;
    }

    if ((int)op != -1) {
        run_ops = &op;
        run_opnames = &opName;
        op_count = 1;
    } else {
        op_count = test_opnum;
        run_ops = test_ops;
        run_opnames = test_opnames;
    }

    for (int i = 0; i < type_count; i++) {
        for (int j = 0; j < op_count; j++) {
            TESTCHECK(timeTest(args, run_types[i], run_typenames[i], run_ops[j],
                               run_opnames[j], -1));
        }
    }
    return testSuccess;
}

} // namespace

struct testEngine reduceScatterEngine = {ReduceScatterGetBuffSize,
                                         ReduceScatterRunTest};

// Set this as the current engine for the reduce_scatter executable
__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &reduceScatterEngine;
}

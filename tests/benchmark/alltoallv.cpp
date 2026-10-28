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

#include <vector>

// seed for initData() function, which is used to fill the send buffer
// and expected buffer
#define PRIME_SEED 33

namespace {

/* We initialize per-peer send_counts and recv_counts by evenly splitting
 * total_count across all rank-pairs, then distributing the remainder so
 * that it would not break the logic when rank A sends message to rank B,
 * reporting this intention in send_counts[B], and rank B awaits a message
 * from rank A, setting the proper element amount in recv_counts[A]. Thus
 * each rank sends and receives exactly total_count overall making the
 * case look balanced but still non-uniform.
 */
void AllToAllVConfig(size_t total_count, int nranks, int rank,
                     std::vector<size_t> &send_counts,
                     std::vector<size_t> &recv_counts) {
    send_counts.assign(nranks, 0);
    recv_counts.assign(nranks, 0);

    size_t const base = total_count / static_cast<size_t>(nranks);
    size_t const rem = total_count % static_cast<size_t>(nranks);

    auto count_for_pair = [base, rem, nranks](int src, int dst) {
        // mapping: count sent from src to dst
        int const idx = (src - dst + nranks) % nranks;
        return base + ((idx < static_cast<int>(rem)) ? 1 : 0);
    };

    for (int peer = 0; peer < nranks; peer++) {
        send_counts[peer] = count_for_pair(rank, peer);
        recv_counts[peer] = count_for_pair(peer, rank);
    }
}

void AllToAllVGetCollByteCount(size_t *sendcount, size_t *recvcount,
                               size_t *paramcount, size_t *sendInplaceOffset,
                               size_t *recvInplaceOffset, size_t count,
                               size_t eltSize, int /*nranks*/) {
    // count is the full buffer element count
    size_t const total = count & -(16 / eltSize);
    *sendcount = total;
    *recvcount = total;
    *paramcount = total;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
}

testResult_t AllToAllVInitData(struct threadArgs *args, onecclDataType_t type,
                               onecclRedOp_t /*op*/, int /*root*/, int rep,
                               int in_place) {
    size_t const total_count = args->nbytes / word_size(type);
    int const nranks = args->nProcs * args->nThreads * args->nGpus;
    size_t const type_size = word_size(type);

    for (int i = 0; i < args->nGpus; i++) {
        int const rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);

        std::vector<size_t> send_counts;
        std::vector<size_t> recv_counts;
        AllToAllVConfig(total_count, nranks, rank, send_counts, recv_counts);

        SYCLCHECK(args->queues[i]
                      ->memset(args->recvbuffs[i], 0, args->expectedBytes)
                      .wait());

        void *data_to_init = args->sendbuffs[i];
        TESTCHECK(initData(args->queues[i], data_to_init, total_count, type,
                           PRIME_SEED * rep + rank, nranks));

        size_t recv_prefix = 0;
        for (int peer = 0; peer < nranks; peer++) {
            void *expected_pos = static_cast<char *>(args->expected[i]) +
                                 recv_prefix * type_size;
            TESTCHECK(initData(args->queues[i], expected_pos, recv_counts[peer],
                               type, PRIME_SEED * rep + peer, nranks));
            recv_prefix += recv_counts[peer];
        }

        SYCLCHECK(args->queues[i]->wait());
    }

    args->reportErrors = 1;
    return testSuccess;
}

void AllToAllVGetBw(size_t count, int typesize, double sec, double *algBw,
                    double *busBw, int nranks) {
    double baseBw = (double)(count * typesize) / 1.0E9 / sec;
    *algBw = baseBw;
    double factor = ((double)(nranks - 1)) / ((double)nranks);
    *busBw = baseBw * factor;
}

testResult_t AllToAllVRunColl(void *sendbuff, void *recvbuff, size_t count,
                              onecclDataType_t type, onecclRedOp_t /*op*/,
                              int /*root*/, int rank, onecclComm_t comm,
                              sycl::queue *queue) {
    int nranks = 0;
    ONECCLCHECK(onecclCommCount(comm, &nranks));

    std::vector<size_t> send_counts;
    std::vector<size_t> recv_counts;
    AllToAllVConfig(count, nranks, rank, send_counts, recv_counts);

    ONECCLCHECK(onecclAllToAllV(sendbuff, send_counts.data(), recvbuff,
                                recv_counts.data(), type, comm, queue));

    return testSuccess;
}

struct testColl allToAllVTest = {"AllToAllV", AllToAllVGetCollByteCount,
                                 AllToAllVInitData, AllToAllVGetBw,
                                 AllToAllVRunColl};

void AllToAllVGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count,
                          int nranks) {
    size_t paramcount, sendInplaceOffset, recvInplaceOffset;
    AllToAllVGetCollByteCount(sendcount, recvcount, &paramcount,
                              &sendInplaceOffset, &recvInplaceOffset, count,
                              /*eltSize=*/16, nranks);
}

testResult_t AllToAllVRunTest(struct threadArgs *args, int /*root*/,
                              onecclDataType_t type, const char *typeName,
                              onecclRedOp_t /*op*/, const char * /*opName*/) {
    args->collTest = &allToAllVTest;
    // alltoallv does not support in-place in this benchmark implementation
    args->runInplace = 0;
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

struct testEngine allToAllVEngine = {AllToAllVGetBuffSize, AllToAllVRunTest};

__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &allToAllVEngine;
}

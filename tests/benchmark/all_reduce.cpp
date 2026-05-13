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

// Calculate buffer sizes for AllReduce
void oneccl_all_reduce_get_coll_byte_count(size_t *sendcount, size_t *recvcount,
                                           size_t *paramcount,
                                           size_t *sendInplaceOffset,
                                           size_t *recvInplaceOffset,
                                           size_t count, size_t /*eltSize*/,
                                           int /*nranks*/) {
    *sendcount = count;
    *recvcount = count;
    *sendInplaceOffset = 0;
    *recvInplaceOffset = 0;
    *paramcount = *sendcount;
}

// Initialize data for AllReduce operation
testResult_t oneccl_all_reduce_init_data(struct threadArgs *args,
                                         onecclDataType_t type,
                                         onecclRedOp_t red_op, int /*root*/,
                                         int /*rep*/, int in_place) {
    size_t const sendcount = args->sendBytes / word_size(type);
    size_t const recvcount = args->expectedBytes / word_size(type);
    int const nranks = args->nProcs * args->nThreads * args->nGpus;

    for (int i = 0; i < args->nGpus; i++) {
        int const rank =
            (((args->proc * args->nThreads + args->thread) * args->nGpus) + i);
        SYCLCHECK(args->queues[i]
                      ->memset(args->recvbuffs[i], 0, args->expectedBytes)
                      .wait());

        void *data_to_init =
            (in_place != 0) ? args->recvbuffs[i] : args->sendbuffs[i];
        TESTCHECK(initData(args->queues[i], data_to_init, sendcount, type, rank,
                           nranks));
        TESTCHECK(initDataReduce(args->queues[i], args->expected[i], recvcount,
                                 type, red_op, nranks));
        SYCLCHECK(args->queues[i]->wait());
    }
    return testSuccess;
}

// Calculate bandwidth for AllReduce
void oneccl_all_reduce_get_bw(size_t count, int typesize, double sec,
                              double *algBw, double *busBw, int nranks) {
    double const base_bw = static_cast<double>(count * typesize) / 1.0E9 / sec;
    *algBw = base_bw;
    double const factor = (nranks > 1)
                              ? (static_cast<double>(2 * (nranks - 1))) /
                                    (static_cast<double>(nranks))
                              : 0.0;
    *busBw = base_bw * factor;
}

// Run the AllReduce collective
testResult_t oneccl_all_reduce_run_coll(void *sendbuff, void *recvbuff,
                                        size_t count, onecclDataType_t type,
                                        onecclRedOp_t red_op, int /*root*/,
                                        int /*rank*/, onecclComm_t comm,
                                        sycl::queue *stream) {
    ONECCLCHECK(
        onecclAllReduce(sendbuff, recvbuff, count, type, red_op, comm, stream));
    return testSuccess;
}

// Define the testColl struct for AllReduce
struct testColl all_reduce_test = {
    "AllReduce", oneccl_all_reduce_get_coll_byte_count,
    oneccl_all_reduce_init_data, oneccl_all_reduce_get_bw,
    oneccl_all_reduce_run_coll};

// Get buffer size needed for the test
void oneccl_all_reduce_get_buff_size(size_t *sendcount, size_t *recvcount,
                                     size_t count, int nranks) {
    size_t paramcount = 0;
    size_t send_inplace_offset = 0;
    size_t recv_inplace_offset = 0;
    oneccl_all_reduce_get_coll_byte_count(
        sendcount, recvcount, &paramcount, &send_inplace_offset,
        &recv_inplace_offset, count, 1, nranks);
}

// Main test runner for AllReduce
testResult_t oneccl_all_reduce_run_test(struct threadArgs *args, int /*root*/,
                                        onecclDataType_t type,
                                        const char *typeName,
                                        onecclRedOp_t red_op,
                                        const char *opName) {
    args->collTest = &all_reduce_test;
    onecclDataType_t const *run_types = nullptr;
    onecclRedOp_t const *run_ops = nullptr;
    const char *const *run_typenames = nullptr;
    const char *const *run_opnames = nullptr;
    int type_count = 0;
    int op_count = 0;

    if (static_cast<int>(type) != -1) {
        type_count = 1;
        run_types = &type;
        run_typenames = &typeName;
    } else {
        type_count = test_typenum;
        run_types = test_types;
        run_typenames = test_typenames;
    }

    if (static_cast<int>(red_op) != -1) {
        op_count = 1;
        run_ops = &red_op;
        run_opnames = &opName;
    } else {
        op_count = test_opnum;
        run_ops = test_ops;
        run_opnames = test_opnames;
    }

    for (int i = 0; i < type_count; i++) {
        for (int j = 0; j < op_count; j++) {
            // AllReduce root is ignored, passing -1
            TESTCHECK(timeTest(args, run_types[i], run_typenames[i], run_ops[j],
                               run_opnames[j], -1));
        }
    }
    return testSuccess;
}

} // anonymous namespace

// Define the test engine for AllReduce
struct testEngine allReduceEngine = {oneccl_all_reduce_get_buff_size,
                                     oneccl_all_reduce_run_test};

__attribute__((constructor)) static void setCurrentEngine() {
    currentTestEngine = &allReduceEngine;
}

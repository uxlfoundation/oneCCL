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

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <libgen.h>
#include <pthread.h>
#include <type_traits>

#include "common.h"

// oneCCL-supported data types and their names
onecclDataType_t test_types[] = {
    onecclInt8,   onecclUint8,   onecclInt32,   onecclUint32,  onecclInt64,
    onecclUint64, onecclFloat16, onecclFloat32, onecclFloat64, onecclBfloat16};
const char *test_typenames[] = {"int8",   "uint8", "int32", "uint32", "int64",
                                "uint64", "fp16",  "fp32",  "fp64",   "bf16"};
int test_typenum = sizeof(test_types) / sizeof(onecclDataType_t);

// oneCCL-supported reduction operations and their names
onecclRedOp_t test_ops[] = {onecclSum, onecclProd, onecclMax, onecclMin};
const char *test_opnames[] = {"sum", "prod", "max", "min"};
int test_opnum = sizeof(test_ops) / sizeof(onecclRedOp_t);

int is_main_proc = 0;
thread_local int is_main_thread = 0;

// Command line parameter defaults
static int n_threads = 1;
static int n_gpus = 1; // Represents devices per thread
static size_t min_bytes = static_cast<size_t>(32) * 1024 * 1024;
static size_t max_bytes = static_cast<size_t>(32) * 1024 * 1024;
static size_t step_bytes = static_cast<size_t>(1) * 1024 * 1024;
static size_t step_factor = 1;
static int datacheck = 1;
static int warmup_iters = 5;
static int iters = 20;
static int cclop = onecclSum;
static int ccltype = onecclFloat32;
static int cclroot = 0;
static int blocking_coll = 0;
static int timeout = 0;
static int report_cputime = 0;
static int average = 1;
static int run_inplace = 1;

// Forward declaration for header printing used in timeTest
static void print_benchmark_headers(int run_inplace);

// Global current engine (will be set by each executable)
struct testEngine *currentTestEngine = nullptr;

// Helper to parse size arguments (e.g., 1K, 2M, 1G)
static bool parse_size(const char *value, size_t *out) {
    int units = 0;
    double size = NAN;
    char size_lit[2] = {0};

    int const count = sscanf(value, "%lf%1s", &size, size_lit);

    switch (count) {
    case 2:
        switch (size_lit[0]) {
        case 'G':
        case 'g':
            units = 1024 * 1024 * 1024;
            break;
        case 'M':
        case 'm':
            units = 1024 * 1024;
            break;
        case 'K':
        case 'k':
            units = 1024;
            break;
        default:
            return false;
        };
        break;
    case 1:
        units = 1;
        break;
    default:
        return false;
    }

    *out = static_cast<size_t>(size * units);
    return true;
}

// SYCL kernel to initialize input data
template <typename T>
static testResult_t oneccl_init_data_kernel(sycl::queue *queue, T *data,
                                            const size_t count, int rank) {
    // NOLINTBEGIN
    SYCLCHECK(queue
                  ->submit([&](sycl::handler &cgh) {
                      cgh.parallel_for(
                          sycl::range<1>(count),
                          [=](sycl::id<1> i) { data[i] = (T)(rank + 1); });
                  })
                  .wait());
    // NOLINTEND
    return testSuccess;
}

// SYCL kernel to initialize data for reduction (expected values)
template <typename T>
static testResult_t
oneccl_init_data_reduce_kernel(sycl::queue *queue, T *data, const size_t count,
                               onecclRedOp_t operation, int nranks) {
    T expected = 0;
    if (operation == onecclSum) {
        for (int i = 0; i < nranks; ++i) {
            expected += static_cast<T>(i + 1);
        }
    } else if (operation == onecclProd) {
        expected = 1;
        for (int i = 0; i < nranks; ++i) {
            expected *= static_cast<T>(i + 1);
        }
    } else if (operation == onecclMax) {
        expected = static_cast<T>(nranks);
    } else if (operation == onecclMin) {
        expected = static_cast<T>(1);
    } else {
        // Unsupported op for data check
        expected = static_cast<T>(0);
    }

    // NOLINTBEGIN
    SYCLCHECK(queue
                  ->submit([&](sycl::handler &cgh) {
                      cgh.parallel_for(
                          sycl::range<1>(count),
                          [=](sycl::id<1> i) { data[i] = expected; });
                  })
                  .wait());
    // NOLINTEND
    return testSuccess;
}

// SYCL kernel to check data correctness
template <typename T>
static testResult_t oneccl_check_data_kernel(sycl::queue *queue, T *recvbuff,
                                             T *expected, const size_t count,
                                             int64_t *wrongElts) {
    auto *d_wrong_elts = sycl::malloc_device<int64_t>(1, *queue);
    SYCLCHECK(queue->memset(d_wrong_elts, 0, sizeof(int64_t)).wait());

    // NOLINTBEGIN
    SYCLCHECK(
        queue
            ->submit([&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
                    if (recvbuff[i] != expected[i]) {
                        // Create atomic_ref inside the kernel from the raw
                        // pointer
                        sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>
                            atomic_wrong(*d_wrong_elts);
                        atomic_wrong.fetch_add(1);
                    }
                });
            })
            .wait());
    // NOLINTEND

    SYCLCHECK(queue->memcpy(wrongElts, d_wrong_elts, sizeof(int64_t)).wait());
    sycl::free(d_wrong_elts, *queue);
    return testSuccess;
}

// Dispatcher for InitData based on data type
testResult_t initData(sycl::queue *queue, void *data, const size_t count,
                      onecclDataType_t type, int rank, int /*nranks*/) {
    switch (type) {
    case onecclInt8:
        return oneccl_init_data_kernel<int8_t>(
            queue, static_cast<int8_t *>(data), count, rank);
    case onecclUint8:
        return oneccl_init_data_kernel<uint8_t>(
            queue, static_cast<uint8_t *>(data), count, rank);
    case onecclInt32:
        return oneccl_init_data_kernel<int32_t>(
            queue, static_cast<int32_t *>(data), count, rank);
    case onecclUint32:
        return oneccl_init_data_kernel<uint32_t>(
            queue, static_cast<uint32_t *>(data), count, rank);
    case onecclInt64:
        return oneccl_init_data_kernel<int64_t>(
            queue, static_cast<int64_t *>(data), count, rank);
    case onecclUint64:
        return oneccl_init_data_kernel<uint64_t>(
            queue, static_cast<uint64_t *>(data), count, rank);
    case onecclFloat32:
        return oneccl_init_data_kernel<float>(queue, static_cast<float *>(data),
                                              count, rank);
    case onecclFloat64:
        return oneccl_init_data_kernel<double>(
            queue, static_cast<double *>(data), count, rank);
    case onecclFloat16:
        return oneccl_init_data_kernel<sycl::half>(
            queue, static_cast<sycl::half *>(data), count, rank);
    case onecclBfloat16:
        return oneccl_init_data_kernel<sycl::ext::oneapi::bfloat16>(
            queue, static_cast<sycl::ext::oneapi::bfloat16 *>(data), count,
            rank);
    default:
        return testInternalError;
    }
}

// Dispatcher for InitDataReduce based on data type
testResult_t initDataReduce(sycl::queue *queue, void *data, const size_t count,
                            onecclDataType_t type, onecclRedOp_t operation,
                            int nranks) {
    switch (type) {
    case onecclInt8:
        return oneccl_init_data_reduce_kernel<int8_t>(
            queue, static_cast<int8_t *>(data), count, operation, nranks);
    case onecclUint8:
        return oneccl_init_data_reduce_kernel<uint8_t>(
            queue, static_cast<uint8_t *>(data), count, operation, nranks);
    case onecclInt32:
        return oneccl_init_data_reduce_kernel<int32_t>(
            queue, static_cast<int32_t *>(data), count, operation, nranks);
    case onecclUint32:
        return oneccl_init_data_reduce_kernel<uint32_t>(
            queue, static_cast<uint32_t *>(data), count, operation, nranks);
    case onecclInt64:
        return oneccl_init_data_reduce_kernel<int64_t>(
            queue, static_cast<int64_t *>(data), count, operation, nranks);
    case onecclUint64:
        return oneccl_init_data_reduce_kernel<uint64_t>(
            queue, static_cast<uint64_t *>(data), count, operation, nranks);
    case onecclFloat32:
        return oneccl_init_data_reduce_kernel<float>(
            queue, static_cast<float *>(data), count, operation, nranks);
    case onecclFloat64:
        return oneccl_init_data_reduce_kernel<double>(
            queue, static_cast<double *>(data), count, operation, nranks);
    case onecclFloat16:
        return oneccl_init_data_reduce_kernel<sycl::half>(
            queue, static_cast<sycl::half *>(data), count, operation, nranks);
    case onecclBfloat16:
        return oneccl_init_data_reduce_kernel<sycl::ext::oneapi::bfloat16>(
            queue, static_cast<sycl::ext::oneapi::bfloat16 *>(data), count,
            operation, nranks);
    default:
        return testInternalError;
    }
}

// Dispatcher for CheckData based on data type
testResult_t checkData(sycl::queue *queue, void *recvbuff, void *expected,
                       const size_t count, onecclDataType_t type,
                       int64_t *wrongElts) {
    switch (type) {
    case onecclInt8:
        return oneccl_check_data_kernel<int8_t>(
            queue, static_cast<int8_t *>(recvbuff),
            static_cast<int8_t *>(expected), count, wrongElts);
    case onecclUint8:
        return oneccl_check_data_kernel<uint8_t>(
            queue, static_cast<uint8_t *>(recvbuff),
            static_cast<uint8_t *>(expected), count, wrongElts);
    case onecclInt32:
        return oneccl_check_data_kernel<int32_t>(
            queue, static_cast<int32_t *>(recvbuff),
            static_cast<int32_t *>(expected), count, wrongElts);
    case onecclUint32:
        return oneccl_check_data_kernel<uint32_t>(
            queue, static_cast<uint32_t *>(recvbuff),
            static_cast<uint32_t *>(expected), count, wrongElts);
    case onecclInt64:
        return oneccl_check_data_kernel<int64_t>(
            queue, static_cast<int64_t *>(recvbuff),
            static_cast<int64_t *>(expected), count, wrongElts);
    case onecclUint64:
        return oneccl_check_data_kernel<uint64_t>(
            queue, static_cast<uint64_t *>(recvbuff),
            static_cast<uint64_t *>(expected), count, wrongElts);
    case onecclFloat32:
        return oneccl_check_data_kernel<float>(
            queue, static_cast<float *>(recvbuff),
            static_cast<float *>(expected), count, wrongElts);
    case onecclFloat64:
        return oneccl_check_data_kernel<double>(
            queue, static_cast<double *>(recvbuff),
            static_cast<double *>(expected), count, wrongElts);
    case onecclFloat16:
        return oneccl_check_data_kernel<sycl::half>(
            queue, static_cast<sycl::half *>(recvbuff),
            static_cast<sycl::half *>(expected), count, wrongElts);
    case onecclBfloat16:
        return oneccl_check_data_kernel<sycl::ext::oneapi::bfloat16>(
            queue, static_cast<sycl::ext::oneapi::bfloat16 *>(recvbuff),
            static_cast<sycl::ext::oneapi::bfloat16 *>(expected), count,
            wrongElts);
    default:
        return testInternalError;
    }
}

// Inter-thread barrier
void barrier(struct threadArgs *args) {
    // Skip barrier overhead when running single-threaded
    if (args->nThreads == 1) {
        return;
    }
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    static int volatile barrier_count = 0;
    static int volatile barrier_generation = 0;

    int const gen = barrier_generation;

    pthread_mutex_lock(&mutex);
    if (++barrier_count == args->nThreads) {
        barrier_count = 0;
        barrier_generation++;
        pthread_cond_broadcast(&cond);
    } else {
        while (gen == barrier_generation) {
            pthread_cond_wait(&cond, &mutex);
        }
    }
    pthread_mutex_unlock(&mutex);
}

// Helper function to apply reduction operation
template <typename T>
static void apply_reduction_op(T *accumulator, T value, int average) {
    switch (average) {
    case /*r0*/ 0:
        // No-op for case 0, handled by caller
        break;
    case /*avg*/ 1:
    case /*sum*/ 4:
        *accumulator += value;
        break;
    case /*min*/ 2:
        *accumulator = std::min<T>(*accumulator, value);
        break;
    case /*max*/ 3:
        *accumulator = std::max<T>(*accumulator, value);
        break;
    default:
        // Unknown average mode
        break;
    }
}

#ifdef MPI_SUPPORT
// Helper function to get MPI datatype
template <typename T> static MPI_Datatype get_mpi_datatype() {
    static_assert(std::is_same_v<T, long long> || std::is_same_v<T, double>,
                  "Allreduce<T> only for T in {long long, double}");
    if constexpr (std::is_same_v<T, long long>) {
        return MPI_LONG_LONG;
    } else {
        return MPI_DOUBLE;
    }
}

// Helper function to get MPI operation
static MPI_Op get_mpi_op(int average) {
    switch (average) {
    case 1:
    case 4:
        return MPI_SUM;
    case 2:
        return MPI_MIN;
    case 3:
        return MPI_MAX;
    default:
        return MPI_Op();
    }
}
#endif

// Inter-thread/process barrier + allreduce
template <typename T>
static void oneccl_allreduce(struct threadArgs *args, T *value, int average) {
    static thread_local int epoch = 0;
    static pthread_mutex_t lock[2] = {PTHREAD_MUTEX_INITIALIZER,
                                      PTHREAD_MUTEX_INITIALIZER};
    static pthread_cond_t cond[2] = {PTHREAD_COND_INITIALIZER,
                                     PTHREAD_COND_INITIALIZER};
    static T accumulator[2];
    static int counter[2] = {0, 0};

    pthread_mutex_lock(&lock[epoch]);

    // Initialize or accumulate value
    if (counter[epoch] == 0) {
        if (average != 0 || args->thread == 0) {
            accumulator[epoch] = *value;
        }
    } else {
        if (average == 0 && args->thread == 0) {
            accumulator[epoch] = *value;
        } else {
            apply_reduction_op(&accumulator[epoch], *value, average);
        }
    }

    // Signal when all threads have contributed
    if (++counter[epoch] == args->nThreads) {
        pthread_cond_broadcast(&cond[epoch]);
    }

    // Last thread performs cross-process reduction
    if (args->thread + 1 == args->nThreads) {
        while (counter[epoch] != args->nThreads) {
            pthread_cond_wait(&cond[epoch], &lock[epoch]);
        }

#ifdef MPI_SUPPORT
        if (average != 0) {
            MPI_Datatype const mpi_type = get_mpi_datatype<T>();
            MPI_Op const mpi_operation = get_mpi_op(average);
            MPI_Allreduce(MPI_IN_PLACE,
                          static_cast<void *>(&accumulator[epoch]), 1, mpi_type,
                          mpi_operation, MPI_COMM_WORLD);
        }
#endif

        if (average == 1) {
            accumulator[epoch] /=
                static_cast<long long>(args->totalProcs) * args->nThreads;
        }
        counter[epoch] = 0;
        pthread_cond_broadcast(&cond[epoch]);
    } else {
        while (counter[epoch] != 0) {
            pthread_cond_wait(&cond[epoch], &lock[epoch]);
        }
    }
    pthread_mutex_unlock(&lock[epoch]);

    *value = accumulator[epoch];
    epoch ^= 1;
}

// Check data correctness across all devices in a thread
static testResult_t oneccl_check_correctness(struct threadArgs *args,
                                             onecclDataType_t type,
                                             onecclRedOp_t /*operation*/,
                                             int /*root*/, int in_place,
                                             int64_t *wrongElts) {
    size_t const count = args->expectedBytes / word_size(type);

    auto *wrong_per_gpu = new int64_t[args->nGpus];
    bool isReduceScatter =
        (args->collTest && strcmp(args->collTest->name, "ReduceScatter") == 0);
    size_t recvcount = args->expectedBytes / word_size(type);

    for (int i = 0; i < args->nGpus; i++) {
        int rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);

        void *data;
        if (isReduceScatter) {
            size_t offBytes = (size_t)rank * recvcount * word_size(type);
            if (in_place) {
                // in-place reduce_scatter result slice stored in recv buffer
                // rank segment
                data = (char *)args->recvbuffs[i] + offBytes;
            } else {
                data = args->recvbuffs[i];
            }
        } else {
            // other collectives: existing semantics
            data = in_place ? (void *)args->recvbuffs[i] : args->recvbuffs[i];
        }

        TESTCHECK(checkData(args->queues[i], data, args->expected[i], count,
                            type, &wrong_per_gpu[i]));
    }

    *wrongElts = 0;
    for (int i = 0; i < args->nGpus; i++) {
        *wrongElts += wrong_per_gpu[i];
    }
    delete[] wrong_per_gpu;

    if ((args->reportErrors != 0) && (*wrongElts != 0)) {
        args->errors[0]++;
    }
    return testSuccess;
}

// Synchronize SYCL queues and check for async errors
static testResult_t oneccl_test_queue_synchronize(int ngpus,
                                                  sycl::queue **queues,
                                                  onecclComm_t *comms) {
    for (int i = 0; i < ngpus; ++i) {
        SYCLCHECK(queues[i]->wait_and_throw());
        if (comms != nullptr) {
            // TODO:
            // onecclResult_t asyncErr;
            // ONECCLCHECK(onecclCommGetAsyncError(comms[i], &asyncErr));
            // if (asyncErr != onecclSuccess) {
            //     for (int j=0; j<ngpus; j++)
            //     ONECCLCHECK(onecclCommAbort(comms[j]));
            //     ONECCLCHECK(asyncErr);
            // }
        }
    }
    return testSuccess;
}

// Start a collective operation
static testResult_t oneccl_start_coll(struct threadArgs *args,
                                      onecclDataType_t type,
                                      onecclRedOp_t operation, int root,
                                      int in_place, int /*iter*/) {
    size_t const count = args->nbytes / word_size(type);

    if (args->nGpus > 1) {
        fprintf(stderr, "ERROR: Multiple GPUs per process are not currently "
                        "supported. Please use nGpus=1.\n");
        return testInternalError;
    }

    if (args->nGpus > 1) {
        ONECCLCHECK(onecclGroupStart());
    }
    for (int i = 0; i < args->nGpus; i++) {
        int rank =
            ((args->proc * args->nThreads + args->thread) * args->nGpus + i);
        void *recv_buff = args->recvbuffs[i];
        void *send_buff = args->sendbuffs[i];
        TESTCHECK(args->collTest->runColl(
            (in_place ? recv_buff : send_buff), recv_buff, count, type,
            operation, root, rank, args->comms[i], args->queues[i]));
    }
    if (args->nGpus > 1) {
        ONECCLCHECK(onecclGroupEnd());
    }

    if (blocking_coll != 0) {
        TESTCHECK(oneccl_test_queue_synchronize(args->nGpus, args->queues,
                                                args->comms));
        barrier(args);
    }
    return testSuccess;
}

// Complete a collective operation
// Note: If blocking_coll is true, synchronization already happened in
// start_coll, so no need to wait again. If false, we must synchronize here to
// measure actual execution time.
static testResult_t oneccl_complete_coll(struct threadArgs *args) {
    if (blocking_coll != 0) {
        return testSuccess;
    }
    TESTCHECK(
        oneccl_test_queue_synchronize(args->nGpus, args->queues, args->comms));
    return testSuccess;
}

// Main benchmark timing function
static testResult_t oneccl_bench_time(struct threadArgs *args,
                                      onecclDataType_t type,
                                      onecclRedOp_t operation, int root,
                                      int in_place) {
    size_t const count = args->nbytes / word_size(type);
    if (datacheck != 0) {
        TESTCHECK(args->collTest->initData(args, type, operation, root, 99,
                                           in_place));
    }

    TESTCHECK(oneccl_start_coll(args, type, operation, root, in_place, 0));
    TESTCHECK(oneccl_complete_coll(args));
    barrier(args);

    timer const tim;
    for (int iter = 0; iter < iters; iter++) {
        TESTCHECK(
            oneccl_start_coll(args, type, operation, root, in_place, iter));
    }
    TESTCHECK(oneccl_complete_coll(args));

    double delta_sec = tim.elapsed() / iters;
    oneccl_allreduce(args, &delta_sec, average);

    double alg_bw = NAN;
    double bus_bw = NAN;
    args->collTest->getBw(count, static_cast<int>(word_size(type)), delta_sec,
                          &alg_bw, &bus_bw,
                          args->nProcs * args->nThreads * args->nGpus);
    barrier(args);

    int64_t wrong_elts = 0;
    if (datacheck != 0) {
        TESTCHECK(args->collTest->initData(args, type, operation, root, 100,
                                           in_place));
        TESTCHECK(oneccl_start_coll(args, type, operation, root, in_place, 0));
        TESTCHECK(oneccl_complete_coll(args));
        TESTCHECK(oneccl_check_correctness(args, type, operation, root,
                                           in_place, &wrong_elts));
        long long wrong_elts1 = wrong_elts;
        oneccl_allreduce(args, &wrong_elts1, 4); // Sum of wrong elements
        wrong_elts = wrong_elts1;
    }

    double const time_usec = delta_sec * 1.0E6;
    char time_str[100];
    if (time_usec >= 10000.0) {
        sprintf(time_str, "%7.0f", time_usec);
    } else if (time_usec >= 100.0) {
        sprintf(time_str, "%7.1f", time_usec);
    } else {
        sprintf(time_str, "%7.2f", time_usec);
    }

    if (args->reportErrors != 0) {
        PRINT("  %7s  %6.2f  %6.2f  %5.0f", time_str, alg_bw, bus_bw,
              static_cast<double>(wrong_elts));
    } else {
        PRINT("  %7s  %6.2f  %6.2f  %5s", time_str, alg_bw, bus_bw, "N/A");
    }

    args->bw[0] += bus_bw;
    args->bw_count[0]++;
    return testSuccess;
}

static void oneccl_setup_args(size_t size, onecclDataType_t type,
                              struct threadArgs *args) {
    int const nranks = args->nProcs * args->nGpus * args->nThreads;
    size_t count = 0;
    size_t send_count = 0;
    size_t recv_count = 0;
    size_t param_count = 0;
    size_t send_inplace_offset = 0;
    size_t recv_inplace_offset = 0;

    count = size / word_size(type);
    args->collTest->getCollByteCount(&send_count, &recv_count, &param_count,
                                     &send_inplace_offset, &recv_inplace_offset,
                                     count, word_size(type), nranks);

    args->nbytes = param_count * word_size(type);
    args->sendBytes = send_count * word_size(type);
    args->expectedBytes = recv_count * word_size(type);
}

// Run tests for a given collective
testResult_t timeTest(struct threadArgs *args, onecclDataType_t type,
                      const char *typeName, onecclRedOp_t operation,
                      const char *opName, int root) {
    static bool header_printed = false;

    barrier(args);

    // Warm-up
    oneccl_setup_args(args->maxbytes, type, args);
    for (int iter = 0; iter < warmup_iters; iter++) {
        TESTCHECK(oneccl_start_coll(args, type, operation, root, 0, iter));
    }
    TESTCHECK(oneccl_complete_coll(args));

    if (!header_printed) {
        print_benchmark_headers(run_inplace);
        header_printed = true;
    }

    // Benchmark
    for (size_t size = args->minbytes; size <= args->maxbytes;
         size = ((args->stepfactor > 1) ? size * args->stepfactor
                                        : size + args->stepbytes)) {
        oneccl_setup_args(size, type, args);
        PRINT("%12li  %12li  %8s  %6s  %6i",
              std::max(args->sendBytes, args->expectedBytes),
              args->nbytes / word_size(type), typeName, opName, root);
        TESTCHECK(
            oneccl_bench_time(args, type, operation, root, 0)); // out-of-place
        if (args->runInplace != 0) {
            TESTCHECK(
                oneccl_bench_time(args, type, operation, root, 1)); // in-place
        }
        PRINT("\n");
    }

    return testSuccess;
}

static testResult_t oneccl_thread_run_tests(struct threadArgs *args) {
    SYCLCHECK(args->queues[0]
                  ->get_device()
                  .get_platform()
                  .get_devices()[args->gpus[0]]);
    TESTCHECK(currentTestEngine->runTest(
        args, cclroot, (onecclDataType_t)ccltype, test_typenames[ccltype],
        (onecclRedOp_t)cclop, test_opnames[cclop]));
    return testSuccess;
}

static testResult_t oneccl_thread_init(struct threadArgs *args) {
    int const nranks = args->nProcs * args->nThreads * args->nGpus;
    is_main_thread = ((is_main_proc != 0) && args->thread == 0) ? 1 : 0;

    // ONECCLCHECK(onecclGroupStart());
    for (int i = 0; i < args->nGpus; i++) {
        // Calculate the local device ordinal for this thread/gpu
        // This assumes gpus[] was filled as in run():
        // gpus[i] = (localRank * nGpus * nThreads) + thread * nGpus + i;
        int const local_device_ordinal = args->gpus[i];
        // Set the device for this rank/thread/gpu before comm creation
        ONECCLCHECK(onecclSetDevice(local_device_ordinal));
        int const rank = (args->proc * args->nThreads * args->nGpus) +
                         (args->thread * args->nGpus) + i;
        ONECCLCHECK(
            onecclCommInitRank(args->comms + i, nranks, args->cclId, rank));
    }
    // ONECCLCHECK(onecclGroupEnd());

    TESTCHECK(oneccl_thread_run_tests(args));

    for (int i = 0; i < args->nGpus; i++) {
        ONECCLCHECK(onecclCommDestroy(args->comms[i]));
    }
    return testSuccess;
}

static void *oneccl_thread_launcher(void *thread_) {
    auto *thread = static_cast<struct testThread *>(thread_);
    thread->ret = thread->func(&thread->args);
    return nullptr;
}

static testResult_t oneccl_thread_launch(struct testThread *thread) {
    pthread_create(&thread->thread, nullptr, oneccl_thread_launcher, thread);
    return testSuccess;
}

testResult_t allocateBuffs(sycl::queue *queue, void **sendbuff,
                           size_t sendBytes, void **recvbuff, size_t recvBytes,
                           void **expected, size_t maxbytes) {
    // ensure enough space for collectives with expanded recv(allgather) or
    // expanded send(reduce_scatter).
    size_t sendAlloc = sendBytes > maxbytes ? sendBytes : maxbytes;
    size_t recvAlloc = recvBytes > maxbytes ? recvBytes : maxbytes;
    SYCLCHECK(*sendbuff = sycl::malloc_device(sendAlloc, *queue));
    SYCLCHECK(*recvbuff = sycl::malloc_device(recvAlloc, *queue));
    if (datacheck != 0) {
        SYCLCHECK(*expected = sycl::malloc_device(recvBytes, *queue));
    }
    return testSuccess;
}

// Helper function to clean up allocated buffers and queues
static void cleanup_buffers_and_queues(sycl::queue **queues, void **sendbuffs,
                                       void **recvbuffs, void **expected,
                                       int num_items) {
    for (int i = 0; i < num_items; i++) {
        if (queues[i] != nullptr) {
            if (sendbuffs[i] != nullptr) {
                sycl::free(sendbuffs[i], *queues[i]);
            }
            if (recvbuffs[i] != nullptr) {
                sycl::free(recvbuffs[i], *queues[i]);
            }
            if (datacheck != 0 && expected[i] != nullptr) {
                sycl::free(expected[i], *queues[i]);
            }
            delete queues[i];
        }
    }
}

// Helper function to allocate all buffers and queues
static testResult_t allocate_all_buffers(
    sycl::queue **queues, void **sendbuffs, void **recvbuffs, void **expected,
    int *gpus, std::vector<sycl::device> &devices, int local_rank,
    int num_items, size_t send_bytes, size_t recv_bytes, size_t max_bytes) {
    for (int i = 0; i < num_items; i++) {
        gpus[i] = (local_rank * num_items) + i;
        sycl::property_list const props = {sycl::property::queue::in_order{}};
        queues[i] = new sycl::queue(devices[gpus[i]], props);

        testResult_t const alloc_result =
            allocateBuffs(queues[i], sendbuffs + i, send_bytes, recvbuffs + i,
                          recv_bytes, expected + i, max_bytes);
        if (alloc_result != testSuccess) {
            cleanup_buffers_and_queues(queues, sendbuffs, recvbuffs, expected,
                                       i + 1);
            char hostname[253];
            get_host_name(hostname, 253);
            printf(" .. %s pid %d: Test failure %s:%d\n", hostname, getpid(),
                   __FILE__, __LINE__);
            return alloc_result;
        }
    }
    return testSuccess;
}

// Helper function to discover and select SYCL platform
static testResult_t discover_sycl_platform(sycl::platform *backend_platform) {
    auto platforms = sycl::platform::get_platforms();
    bool platform_found = false;

    for (const auto &platform : platforms) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            *backend_platform = platform;
            platform_found = true;
            break;
        }
    }

    if (!platform_found) {
        *backend_platform = platforms[0];
    }

    return testSuccess;
}

// Helper function to print benchmark headers
static void print_benchmark_headers(int run_inplace) {
    PRINT("#\n");
    if (run_inplace != 0) {
        PRINT("# %10s  %12s  %8s  %6s  %6s                 out-of-place        "
              "                         in-place         \n",
              "", "", "", "", "");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s %6s  %7s  %6s  %6s "
              "%6s\n",
              "size", "count", "type", "redop", "root", "time", "algbw",
              "busbw", "#wrong", "time", "algbw", "busbw", "#wrong");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s  %7s  %6s  %6s  "
              "%5s\n",
              "(B)", "(elements)", "", "", "", "(us)", "(GB/s)", "(GB/s)", "",
              "(us)", "(GB/s)", "(GB/s)", "");
    } else {
        PRINT("# %10s  %12s  %8s  %6s  %6s                 out-of-place        "
              " \n",
              "", "", "", "", "");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s %6s\n", "size",
              "count", "type", "redop", "root", "time", "algbw", "busbw",
              "#wrong");
        PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %5s\n", "(B)",
              "(elements)", "", "", "", "(us)", "(GB/s)", "(GB/s)", "");
    }
}

// Helper function to initialize thread arguments
static void initialize_thread_args(struct testThread *threads, int n_threads,
                                   int *gpus, void **sendbuffs,
                                   void **recvbuffs, void **expected,
                                   onecclUniqueId ccl_id, onecclComm_t *comms,
                                   sycl::queue **queues, int *errors,
                                   double *bandwidth, int *bw_count,
                                   int local_rank, int total_procs, int proc) {
    for (int thread_idx = 0; thread_idx < n_threads; thread_idx++) {
        threads[thread_idx].args.minbytes = min_bytes;
        threads[thread_idx].args.maxbytes = max_bytes;
        threads[thread_idx].args.stepbytes = step_bytes;
        threads[thread_idx].args.stepfactor = step_factor;
        threads[thread_idx].args.localRank = local_rank;
        threads[thread_idx].args.totalProcs = total_procs;
        threads[thread_idx].args.nProcs = total_procs;
        threads[thread_idx].args.proc = proc;
        threads[thread_idx].args.nThreads = n_threads;
        threads[thread_idx].args.thread = thread_idx;
        threads[thread_idx].args.nGpus = n_gpus;
        ptrdiff_t const offset = static_cast<ptrdiff_t>(thread_idx) * n_gpus;
        threads[thread_idx].args.gpus = gpus + offset;
        threads[thread_idx].args.sendbuffs = sendbuffs + offset;
        threads[thread_idx].args.recvbuffs = recvbuffs + offset;
        threads[thread_idx].args.expected = expected + offset;
        threads[thread_idx].args.cclId = ccl_id;
        threads[thread_idx].args.comms = comms + offset;
        threads[thread_idx].args.queues = queues + offset;
        threads[thread_idx].args.errors = errors + thread_idx;
        threads[thread_idx].args.bw = bandwidth + thread_idx;
        threads[thread_idx].args.bw_count = bw_count + thread_idx;
        threads[thread_idx].args.reportErrors = datacheck;
        threads[thread_idx].args.runInplace = run_inplace;
        threads[thread_idx].func = oneccl_thread_init;
    }
}

// Helper function to launch threads and wait for completion
static void launch_and_wait_threads(struct testThread *threads, int n_threads) {
    for (int thread_idx = 0; thread_idx < n_threads; thread_idx++) {
        if (thread_idx > 0) {
            oneccl_thread_launch(threads + thread_idx);
        } else {
            threads[thread_idx].func(&threads[thread_idx].args);
        }
    }

    for (int thread_idx = 1; thread_idx < n_threads; thread_idx++) {
        pthread_join(threads[thread_idx].thread, nullptr);
    }
}

// Helper function to aggregate results across threads
static void aggregate_thread_results(double *bandwidth, int *bw_count,
                                     int *errors, int n_threads) {
    for (int thread_idx = 1; thread_idx < n_threads; thread_idx++) {
        bandwidth[0] += bandwidth[thread_idx];
        bw_count[0] += bw_count[thread_idx];
        errors[0] += errors[thread_idx];
    }
}

// Helper function to initialize buffer pointers to nullptr
static void initialize_buffer_pointers(sycl::queue **queues, void **sendbuffs,
                                       void **recvbuffs, void **expected,
                                       int num_devices) {
    for (int i = 0; i < num_devices; i++) {
        queues[i] = nullptr;
        sendbuffs[i] = nullptr;
        recvbuffs[i] = nullptr;
        expected[i] = nullptr;
    }
}

static testResult_t oneccl_run() {
    int total_procs = 1;
    int proc = 0;
    int local_rank = 0;
    char hostname[253];
    get_host_name(hostname, 253);

#ifdef MPI_SUPPORT
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    // Simple local rank calculation
    MPI_Comm local_comm = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, proc,
                        MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
#endif
    is_main_proc = static_cast<int>(proc == 0);
    is_main_thread = is_main_proc;

    PRINT("# oneCCL test starting\n");
    PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) "
          "warmup iters: %d iters: %d validation: %d inplace: %d\n",
          n_threads, n_gpus, min_bytes, max_bytes,
          (step_factor > 1) ? step_factor : step_bytes,
          (step_factor > 1) ? "factor" : "bytes", warmup_iters, iters,
          datacheck, run_inplace);
    PRINT("# procs: %d x threads: %d x gpus: %d\n", total_procs, n_threads,
          n_gpus);

    // Discover SYCL devices
    sycl::platform backend_platform;
    TESTCHECK(discover_sycl_platform(&backend_platform));

    auto devices = backend_platform.get_devices();
    size_t const required_devices = static_cast<size_t>(local_rank + 1) *
                                    static_cast<size_t>(n_gpus) *
                                    static_cast<size_t>(n_threads);
    if (devices.size() < required_devices) {
        printf("Not enough SYCL devices available on this node.\n");
        return testInternalError;
    }

    PRINT("# Using devices from platform: %s\n",
          backend_platform.get_info<sycl::info::platform::name>().c_str());

    onecclUniqueId ccl_id;
    if (proc == 0) {
        ONECCLCHECK(onecclGetUniqueId(&ccl_id));
    }
#ifdef MPI_SUPPORT
    MPI_Bcast(&ccl_id, sizeof(ccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    int const num_devices = n_gpus * n_threads;
    int *gpus = static_cast<int *>(malloc(sizeof(int) * num_devices));
    auto **queues = static_cast<sycl::queue **>(
        malloc(sizeof(sycl::queue *) * num_devices));
    void **sendbuffs =
        static_cast<void **>(malloc(sizeof(void *) * num_devices));
    void **recvbuffs =
        static_cast<void **>(malloc(sizeof(void *) * num_devices));
    void **expected =
        static_cast<void **>(malloc(sizeof(void *) * num_devices));

    // Initialize pointers to nullptr for safe cleanup
    initialize_buffer_pointers(queues, sendbuffs, recvbuffs, expected,
                               num_devices);

    size_t send_bytes = 0;
    size_t recv_bytes = 0;
    currentTestEngine->getBuffSize(&send_bytes, &recv_bytes, max_bytes,
                                   total_procs * num_devices);

    testResult_t const alloc_result = allocate_all_buffers(
        queues, sendbuffs, recvbuffs, expected, gpus, devices, local_rank,
        num_devices, send_bytes, recv_bytes, max_bytes);
    if (alloc_result != testSuccess) {
        free(static_cast<void *>(gpus));
        free(static_cast<void *>(queues));
        free(static_cast<void *>(sendbuffs));
        free(static_cast<void *>(recvbuffs));
        free(static_cast<void *>(expected));
        return alloc_result;
    }

    auto *comms = static_cast<onecclComm_t *>(
        malloc(sizeof(onecclComm_t) * n_threads * n_gpus));

    int *errors = new int[n_threads];
    auto *bandwidth = new double[n_threads];
    int *bw_count = new int[n_threads];
    memset(errors, 0, sizeof(int) * n_threads);
    memset(bandwidth, 0, sizeof(double) * n_threads);
    memset(bw_count, 0, sizeof(int) * n_threads);

    struct testThread threads[n_threads];
    memset(threads, 0, sizeof(struct testThread) * n_threads);

    initialize_thread_args(threads, n_threads, gpus, sendbuffs, recvbuffs,
                           expected, ccl_id, comms, queues, errors, bandwidth,
                           bw_count, local_rank, total_procs, proc);

    launch_and_wait_threads(threads, n_threads);

    aggregate_thread_results(bandwidth, bw_count, errors, n_threads);
#ifdef MPI_SUPPORT
    MPI_Allreduce(MPI_IN_PLACE, &bandwidth[0], 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bw_count[0], 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &errors[0], 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
    if (bw_count[0] > 0) {
        bandwidth[0] /= bw_count[0];
    }

    double const check_avg_bw = -1;

    // Print error status
    const char *error_status = "OK";
    if (errors[0] != 0) {
        error_status = "FAILED";
    }
    PRINT("# Out of bounds values : %d %s\n", errors[0], error_status);

    // Print bandwidth status
    const char *bw_status = "";
    if (check_avg_bw != -1) {
        if (bandwidth[0] < check_avg_bw * 0.9) {
            bw_status = "FAILED";
        } else {
            bw_status = "OK";
        }
    }
    PRINT("# Avg bus bandwidth    : %g %s\n", bandwidth[0], bw_status);
    PRINT("# oneCCL test concluded.\n");

    // Save values before cleanup for error checking
    bool has_errors = (errors[0] != 0);
    if (check_avg_bw != -1 && bandwidth[0] < check_avg_bw * 0.9) {
        has_errors = true;
    }
    double const final_bw = bandwidth[0];
    int const final_errors = errors[0];

    // Cleanup
    cleanup_buffers_and_queues(queues, sendbuffs, recvbuffs, expected,
                               num_devices);
    free(static_cast<void *>(comms));
    free(static_cast<void *>(gpus));
    free(static_cast<void *>(queues));
    free(static_cast<void *>(sendbuffs));
    free(static_cast<void *>(recvbuffs));
    free(static_cast<void *>(expected));
    delete[] errors;
    delete[] bandwidth;
    delete[] bw_count;

    if (has_errors) {
        fprintf(stderr, "Error: bw[0]=%f, check_avg_bw=%f, errors[0]=%d\n",
                final_bw, check_avg_bw, final_errors);
        exit(EXIT_FAILURE);
    }

#ifdef MPI_SUPPORT
    MPI_Finalize();
#endif
    return testSuccess;
}

int main(int argc, char *argv[]) {
    setlinebuf(stdout);

    bool parsed = false;
    int longindex = 0;
    static struct option longopts[] = {
        {"nthreads", required_argument, nullptr, 't'},
        {"ngpus", required_argument, nullptr, 'g'},
        {"minbytes", required_argument, nullptr, 'b'},
        {"maxbytes", required_argument, nullptr, 'e'},
        {"stepbytes", required_argument, nullptr, 'i'},
        {"stepfactor", required_argument, nullptr, 'f'},
        {"iters", required_argument, nullptr, 'n'},
        {"warmup_iters", required_argument, nullptr, 'w'},
        {"check", required_argument, nullptr, 'c'},
        {"op", required_argument, nullptr, 'o'},
        {"datatype", required_argument, nullptr, 'd'},
        {"root", required_argument, nullptr, 'r'},
        {"average", required_argument, nullptr, 'a'},
        {"blocking", required_argument, nullptr, 'z'},
        {"timeout", required_argument, nullptr, 'T'},
        {"inplace", required_argument, nullptr, 'p'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    // Reset getopt global state in case of multiple parses (for safety)
    optind = 1;

    // Parse all options before doing anything else
    while (true) {
        int const getopt_char =
            getopt_long(argc, argv, "t:g:b:e:i:f:n:w:c:o:d:r:a:z:T:p:h",
                        longopts, &longindex);
        if (getopt_char == -1) {
            break;
        }

        switch (getopt_char) {
        case 't':
            n_threads = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'g':
            n_gpus = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'b':
            parsed = parse_size(optarg, &min_bytes);
            if (!parsed) {
                fprintf(stderr, "invalid size specified for 'minbytes'\n");
                return -1;
            }
            break;
        case 'e':
            parsed = parse_size(optarg, &max_bytes);
            if (!parsed) {
                fprintf(stderr, "invalid size specified for 'maxbytes'\n");
                return -1;
            }
            break;
        case 'i':
            parsed = parse_size(optarg, &step_bytes);
            if (!parsed) {
                fprintf(stderr, "invalid size specified for 'stepBytes'\n");
                return -1;
            }
            break;
        case 'f':
            step_factor = strtol(optarg, nullptr, 0);
            break;
        case 'n':
            iters = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'w':
            warmup_iters = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'c':
            datacheck = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'o':
            cclop = oneccl_string_to_op(optarg);
            break;
        case 'd':
            ccltype = oneccl_string_to_type(optarg);
            break;
        case 'r':
            cclroot = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'a':
            average = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'z':
            blocking_coll = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'T':
            timeout = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'p':
            run_inplace = static_cast<int>(strtol(optarg, nullptr, 0));
            break;
        case 'h':
        default:
            printf(
                "USAGE: %s [-t nthreads] [-g ngpus] [-b minbytes] [-e "
                "maxbytes] [-i stepbytes] [-f stepfactor] [-p inplace] ...\n",
                argv[0]);
            return 0;
        }
    }

    if (min_bytes > max_bytes) {
        fprintf(stderr,
                "invalid sizes for 'minbytes' and 'maxbytes': %llu > %llu\n",
                static_cast<unsigned long long>(min_bytes),
                static_cast<unsigned long long>(max_bytes));
        return -1;
    }

    if (n_threads > 1) {
        fprintf(stderr, "Error: Multi-threading is not supported\n");
        fprintf(stderr, "       Please use nThreads=1 (current value: %d)\n",
                n_threads);
        fprintf(stderr, "       To scale across multiple ranks, use MPI "
                        "processes with multiple GPUs instead.\n");
        return -1;
    }

#ifdef MPI_SUPPORT
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
#endif
    TESTCHECK(oneccl_run());
    return 0;
}

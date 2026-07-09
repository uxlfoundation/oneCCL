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

#include "base.hpp"

#include <unistd.h>

/* msg sizes in bytes in backprop order */

size_t msg_sizes_vgg16[] = { 16384000, 4000, 67108864, 16384, 411041792, 16384, 9437184, 2048,
                             9437184,  2048, 9437184,  2048,  9437184,   2048,  9437184, 2048,
                             4718592,  2048, 2359296,  1024,  2359296,   1024,  1179648, 1024,
                             589824,   512,  294912,   512,   147456,    256,   6912,    256 };

//size_t msg_sizes_test[] = { 9437184, 2048, 4718592, 2048, 2359296, 1024, 589824, 512, 147456, 256, 6912, 256 };

size_t msg_sizes_test[] = { 589824, 512, 147456, 256, 6912, 256, 2048, 2048, 2048, 2048, 2048, 1024,
                            2048,   512, 2048,   256, 2048, 256, 2048, 2048, 2048, 2048, 2048, 1024,
                            2048,   512, 2048,   256, 2048, 256, 2048, 2048, 2048, 2048, 2048, 1024,
                            2048,   512, 2048,   256, 2048, 256, 2048, 2048, 2048, 2048, 2048, 1024,
                            2048,   512, 2048,   256, 2048, 256, 2048, 2048, 2048, 2048, 2048, 1024,
                            2048,   512, 2048,   256, 2048, 256, 2048, 2048, 2048, 2048, 2048, 1024,
                            2048,   512, 2048,   256, 2048, 256 };

#define DELIMETER \
    "---------------------------------------------------------------------------------------------------------------------------------------------\n"

#define PRINT_HEADER() \
    do { \
        printf(DELIMETER); \
        printf("msg_idx | size (bytes) | " \
               "msg_time (usec) | msg_start_time (usec) | " \
               "msg_wait_time (usec) | msg_iso_time (usec) |" \
               "avg_msg_time(usec)|stddev (%%)|\n"); \
        printf(DELIMETER); \
    } while (0)

#define msg_sizes msg_sizes_test

int comp_iter_time_ms = 0;

#define sizeofa(arr)   (sizeof(arr) / sizeof(*arr))
#define DTYPE          float
#define CACHELINE_SIZE 64

#define MSG_COUNT         sizeofa(msg_sizes)
#define ITER_COUNT        20
#define WARMUP_ITER_COUNT 1

int collect_iso = 1;

void* msg_buffers[MSG_COUNT];
int msg_priorities[MSG_COUNT];
int msg_completions[MSG_COUNT];
ccl::event msg_requests[MSG_COUNT];

double tmp_start_timer, tmp_stop_timer;
double iter_start, iter_stop, iter_timer, iter_iso_timer;

double iter_timer_avg, iter_timer_stddev;

double msg_starts[MSG_COUNT];
double msg_stops[MSG_COUNT];
double msg_timers[MSG_COUNT];
double msg_iso_timers[MSG_COUNT];

double msg_pure_start_timers[MSG_COUNT];
double msg_pure_wait_timers[MSG_COUNT];

double msg_timers_avg[MSG_COUNT];
double msg_timers_stddev[MSG_COUNT];

size_t comp_delay_ms;

void do_iter(size_t iter_idx, ccl::communicator& comm) {
    if (comm.rank() == 0) {
        printf("started iter %zu\n", iter_idx);
        fflush(stdout);
    }

    size_t idx, msg_idx;
    char match_id[16];

    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
    attr.set<ccl::operation_attr_id::to_cache>(true);

    if (collect_iso) {
        ccl::barrier(comm);

        iter_start = when();
        for (idx = 0; idx < MSG_COUNT; idx++) {
            sprintf(match_id, "%zu", idx);

            attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(match_id));

            tmp_start_timer = when();
            ccl::allreduce(msg_buffers[idx],
                           msg_buffers[idx],
                           msg_sizes[idx] / ccl::get_datatype_size(ccl::datatype::float32),
                           ccl::datatype::float32,
                           ccl::reduction::sum,
                           comm,
                           attr)
                .wait();
            tmp_stop_timer = when();
            msg_iso_timers[idx] += (tmp_stop_timer - tmp_start_timer);
        }
        iter_stop = when();
        iter_iso_timer += (iter_stop - iter_start);
        collect_iso = 0;
    }

    ccl::barrier(comm);

    memset(msg_completions, 0, MSG_COUNT * sizeof(int));
    size_t completions = 0;

    iter_start = when();
    for (idx = 0; idx < MSG_COUNT; idx++) {
        if (idx % 2 == 0)
            usleep(comp_delay_ms * 1000);

        /* sequentially increase priority over iterations and messages */
        attr.set<ccl::operation_attr_id::priority>(iter_idx * MSG_COUNT + idx);

        sprintf(match_id, "%zu", idx);

        attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(match_id));

        msg_starts[idx] = when();
        tmp_start_timer = when();
        msg_requests[idx] =
            ccl::allreduce(msg_buffers[idx],
                           msg_buffers[idx],
                           msg_sizes[idx] / ccl::get_datatype_size(ccl::datatype::float32),
                           ccl::datatype::float32,
                           ccl::reduction::sum,
                           comm,
                           attr);
        tmp_stop_timer = when();
        msg_pure_start_timers[idx] += (tmp_stop_timer - tmp_start_timer);
    }

    while (completions < MSG_COUNT) {
        for (idx = 0; idx < MSG_COUNT; idx++) {
            /* complete in reverse list order */
            msg_idx = (MSG_COUNT - idx - 1);

            /* complete in direct list order */
            //msg_idx = idx;

            if (msg_completions[msg_idx])
                continue;

            int is_completed = 0;

            tmp_start_timer = when();

            // msg_requests[msg_idx].wait(); is_completed = 1;
            is_completed = msg_requests[msg_idx].test();

            tmp_stop_timer = when();
            msg_pure_wait_timers[msg_idx] += (tmp_stop_timer - tmp_start_timer);

            if (is_completed) {
                msg_stops[msg_idx] = when();
                msg_timers[msg_idx] += (msg_stops[msg_idx] - msg_starts[msg_idx]);
                msg_completions[msg_idx] = 1;
                completions++;
            }
        }
    }
    iter_stop = when();
    iter_timer += (iter_stop - iter_start);

    if (comm.rank() == 0) {
        printf("completed iter %zu\n", iter_idx);
        fflush(stdout);
    }
}

int main() {
    setenv("CCL_PRIORITY", "direct", 0);

    ccl::init();

    size_t size = 0, rank = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, (int*)&size);
    MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);

    atexit(mpi_finalize);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(size, rank, kvs);

    char* comp_iter_time_ms_env = getenv("COMP_ITER_TIME_MS");
    if (comp_iter_time_ms_env) {
        comp_iter_time_ms = atoi(comp_iter_time_ms_env);
    }

    if (comp_iter_time_ms < 0)
        comp_iter_time_ms = 0;

    if (comp_iter_time_ms)
        comp_delay_ms = 2 * comp_iter_time_ms / MSG_COUNT;

    size_t total_msg_size = 0;
    size_t idx;
    for (idx = 0; idx < MSG_COUNT; idx++)
        total_msg_size += msg_sizes[idx];

    if (rank == 0) {
        printf("iter_count: %d, warmup_iter_count: %d\n", ITER_COUNT, WARMUP_ITER_COUNT);
        printf("msg_count: %zu, total_msg_size: %zu bytes\n", MSG_COUNT, total_msg_size);
        printf("comp_iter_time_ms: %d, comp_delay_ms: %zu (between each pair of messages)\n",
               comp_iter_time_ms,
               comp_delay_ms);
        printf("messages are started in direct order and completed in reverse order\n");
        fflush(stdout);
    }

    for (idx = 0; idx < MSG_COUNT; idx++) {
        size_t msg_size = msg_sizes[idx];
        int pm_ret = posix_memalign(&(msg_buffers[idx]), CACHELINE_SIZE, msg_size);
        assert((pm_ret == 0) && msg_buffers[idx]);
        (void)pm_ret;
        memset(msg_buffers[idx], 'a' + idx, msg_size);
        msg_priorities[idx] = idx;
    }

    /* warmup */
    collect_iso = 0;
    for (idx = 0; idx < WARMUP_ITER_COUNT; idx++) {
        do_iter(idx, comm);
    }

    /* reset timers */
    iter_start = iter_stop = iter_timer = iter_iso_timer = 0;
    for (idx = 0; idx < MSG_COUNT; idx++) {
        msg_starts[idx] = msg_stops[idx] = msg_timers[idx] = msg_iso_timers[idx] =
            msg_pure_start_timers[idx] = msg_pure_wait_timers[idx] = 0;
    }

    /* main loop */
    collect_iso = 1;
    for (idx = 0; idx < ITER_COUNT; idx++) {
        do_iter(idx, comm);
    }

    iter_timer /= ITER_COUNT;
    for (idx = 0; idx < MSG_COUNT; idx++) {
        msg_timers[idx] /= ITER_COUNT;
        msg_pure_start_timers[idx] /= ITER_COUNT;
        msg_pure_wait_timers[idx] /= ITER_COUNT;
    }

    auto attr = ccl::create_operation_attr<ccl::barrier_attr>();
    ccl::barrier(comm, attr);

    std::vector<double> recv_msg_timers(size * MSG_COUNT);
    std::vector<size_t> recv_msg_timers_counts(size, MSG_COUNT);

    std::vector<double> recv_iter_timers(size);
    std::vector<size_t> recv_iter_timers_counts(size, 1);

    ccl::allgatherv(msg_timers, MSG_COUNT, recv_msg_timers.data(), recv_msg_timers_counts, comm)
        .wait();

    ccl::allgatherv(&iter_timer, 1, recv_iter_timers.data(), recv_iter_timers_counts, comm).wait();

    if (rank == 0) {
        size_t rank_idx;
        for (idx = 0; idx < MSG_COUNT; idx++) {
            msg_timers_avg[idx] = 0;
            for (rank_idx = 0; rank_idx < size; rank_idx++) {
                double val = recv_msg_timers[rank_idx * MSG_COUNT + idx];
                msg_timers_avg[idx] += val;
            }
            msg_timers_avg[idx] /= size;

            msg_timers_stddev[idx] = 0;
            double sum = 0;
            for (rank_idx = 0; rank_idx < size; rank_idx++) {
                double avg = msg_timers_avg[idx];
                double val = recv_msg_timers[rank_idx * MSG_COUNT + idx];
                sum += (val - avg) * (val - avg);
            }
            msg_timers_stddev[idx] = std::sqrt(sum / size) / msg_timers_avg[idx] * 100;
            printf("size %10zu bytes, avg %10.2lf us, stddev %5.1lf %%\n",
                   msg_sizes[idx],
                   msg_timers_avg[idx],
                   msg_timers_stddev[idx]);
        }

        iter_timer_avg = 0;
        for (rank_idx = 0; rank_idx < size; rank_idx++) {
            double val = recv_iter_timers[rank_idx];
            printf("rank %zu, time %f\n", rank_idx, val);
            iter_timer_avg += val;
        }
        iter_timer_avg /= size;

        iter_timer_stddev = 0;
        double sum = 0;
        for (rank_idx = 0; rank_idx < size; rank_idx++) {
            double avg = iter_timer_avg;
            double val = recv_iter_timers[rank_idx];
            sum += (val - avg) * (val - avg);
        }
        iter_timer_stddev = std::sqrt(sum / size) / iter_timer_avg * 100;
    }
    ccl::barrier(comm);

    if (rank == 0)
        PRINT_HEADER();

    for (idx = 0; idx < MSG_COUNT; idx++) {
        free(msg_buffers[idx]);
        if (rank == 0) {
            printf("%7zu | %12zu | %15.2lf | %21.2lf | %20.2lf | %19.2lf | %16.1lf | %10.1lf",
                   idx,
                   msg_sizes[idx],
                   msg_timers[idx],
                   msg_pure_start_timers[idx],
                   msg_pure_wait_timers[idx],
                   msg_iso_timers[idx],
                   msg_timers_avg[idx],
                   msg_timers_stddev[idx]);
            printf("\n");
        }
    }

    if (rank == 0)
        PRINT_HEADER();

    if (rank == 0) {
        printf("iter_time        (usec): %12.2lf\n", iter_timer);
        printf("iter_time_avg    (usec): %12.2lf\n", iter_timer_avg);
        printf("iter_time_stddev (%%): %12.2lf\n", iter_timer_stddev);
        printf("iter_iso_time    (usec): %12.2lf\n", iter_iso_timer);
    }

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}

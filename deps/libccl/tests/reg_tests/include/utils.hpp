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

#pragma once

#define CHECK_BF16_ERROR(send_buf, recv_buf, comm) \
    { \
        /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */ \
        int comm_size = comm.size(); \
        double log_base2 = log(comm_size) / log(2); \
        double precision = 2 * BF16_PRECISION; \
        double g = (log_base2 * precision) / (1 - (log_base2 * precision)); \
        for (size_t i = 0; i < COUNT; i++) { \
            double expected = ((comm_size * (comm_size - 1) / 2) + ((float)(i)*comm_size)); \
            double max_error = g * expected; \
            if (std::fabs(max_error) < std::fabs(expected - recv_buf[i])) { \
                printf( \
                    "[%d] got recv_buf[%zu] = %0.7f, but expected = %0.7f, max_error = %0.16f\n", \
                    comm.rank(), \
                    i, \
                    recv_buf[i], \
                    (float)expected, \
                    (double)max_error); \
                return -1; \
            } \
        } \
    }

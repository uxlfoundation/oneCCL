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

#include "common/global/global.hpp"
#include "sched/entry/factory/entry_factory.hpp"

#define CCL_CHUNKED_ENTRY_FUNCTION(entry_name, dtype, cnt, create_entry_expr, get_sched_expr) \
    do { \
        LOG_DEBUG("creating chunked ", entry_name, " entry"); \
        size_t dtype_size = dtype.size(); \
        size_t bytes = cnt * dtype_size; \
        size_t chunk_count = (bytes >= ccl::global_data::env().min_chunk_size && \
                              bytes >= ccl::global_data::env().chunk_count) \
                                 ? ccl::global_data::env().chunk_count \
                                 : 1; \
        while ((chunk_count > 1) && \
               (bytes / chunk_count < ccl::global_data::env().min_chunk_size)) { \
            chunk_count--; \
        } \
        if (chunk_count == 0) { \
            LOG_ERROR("unexpected chunk_count"); \
            chunk_count = 1; \
        } \
        LOG_DEBUG("cnt ", cnt, ", chunk_count ", chunk_count); \
        size_t main_chunk_size = cnt / chunk_count; \
        size_t last_chunk_size = main_chunk_size + cnt % chunk_count; \
        size_t chunk_size, chunk_offset; \
        ccl_sched* chunk_sched = nullptr; \
        for (size_t chunk_idx = 0; chunk_idx < chunk_count; chunk_idx++) { \
            chunk_size = (chunk_idx == (chunk_count - 1)) ? last_chunk_size : main_chunk_size; \
            chunk_offset = chunk_idx * main_chunk_size * dtype_size; \
            get_sched_expr; \
            create_entry_expr; \
        } \
    } while (0)

namespace entry_factory {
/* chunking w/o spreading between schedules */
void make_chunked_send_entry(ccl_sched* sched,
                             const ccl_buffer buf,
                             size_t cnt,
                             const ccl_datatype& dtype,
                             int dst,
                             ccl_comm* comm);

void make_chunked_recv_entry(ccl_sched* sched,
                             const ccl_buffer buf,
                             size_t cnt,
                             const ccl_datatype& dtype,
                             int src,
                             ccl_comm* comm);

void make_chunked_recv_reduce_entry(
    ccl_sched* sched,
    ccl_buffer inout_buf,
    size_t cnt,
    size_t* out_cnt,
    const ccl_datatype& dtype,
    ccl::reduction reduction_op,
    int src,
    ccl_buffer comm_buf,
    ccl_comm* comm,
    ccl_recv_reduce_result_buf_type result_buf_type = ccl_recv_reduce_local_buf);

/* chunking with spreading between schedules */
void make_chunked_send_entry(std::vector<ccl_sched*>& scheds,
                             size_t first_sched_idx,
                             const ccl_buffer buf,
                             size_t cnt,
                             const ccl_datatype& dtype,
                             int dst,
                             ccl_comm* comm);

void make_chunked_recv_entry(std::vector<ccl_sched*>& scheds,
                             size_t first_sched_idx,
                             const ccl_buffer buf,
                             size_t cnt,
                             const ccl_datatype& dtype,
                             int src,
                             ccl_comm* comm);

void make_chunked_copy_entry(std::vector<ccl_sched*>& scheds,
                             size_t first_sched_idx,
                             const ccl_buffer in_buf,
                             ccl_buffer out_buf,
                             size_t cnt,
                             const ccl_datatype& dtype);
} // namespace entry_factory

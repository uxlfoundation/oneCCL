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

#include "sched/entry/factory/chunked_entry_factory.hpp"

namespace entry_factory {
void make_chunked_send_entry(ccl_sched* sched,
                             const ccl_buffer buf,
                             size_t cnt,
                             const ccl_datatype& dtype,
                             int dst,
                             ccl_comm* comm) {
    CCL_CHUNKED_ENTRY_FUNCTION(
        "send",
        dtype,
        cnt,
        create<send_entry>(chunk_sched, buf + chunk_offset, chunk_size, dtype, dst, comm),
        { chunk_sched = sched; });
}

void make_chunked_recv_entry(ccl_sched* sched,
                             const ccl_buffer buf,
                             size_t cnt,
                             const ccl_datatype& dtype,
                             int src,
                             ccl_comm* comm) {
    CCL_CHUNKED_ENTRY_FUNCTION(
        "recv",
        dtype,
        cnt,
        create<recv_entry>(chunk_sched, buf + chunk_offset, chunk_size, dtype, src, comm),
        { chunk_sched = sched; });
}

void make_chunked_recv_reduce_entry(ccl_sched* sched,
                                    ccl_buffer inout_buf,
                                    size_t cnt,
                                    const ccl_datatype& dtype,
                                    ccl::reduction reduction_op,
                                    int src,
                                    ccl_comm* comm,
                                    ccl_buffer comm_buf,
                                    ccl_recv_reduce_result_buf_type result_buf_type) {
    CCL_CHUNKED_ENTRY_FUNCTION("recv_reduce",
                               dtype,
                               cnt,
                               create<recv_reduce_entry>(chunk_sched,
                                                         inout_buf + chunk_offset,
                                                         chunk_size,
                                                         dtype,
                                                         reduction_op,
                                                         src,
                                                         comm,
                                                         comm_buf + chunk_offset,
                                                         result_buf_type),
                               { chunk_sched = sched; });
}

void make_chunked_send_entry(std::vector<ccl_sched*>& scheds,
                             size_t first_sched_idx,
                             const ccl_buffer buf,
                             size_t cnt,
                             const ccl_datatype& dtype,
                             int dst,
                             ccl_comm* comm) {
    CCL_CHUNKED_ENTRY_FUNCTION(
        "send",
        dtype,
        cnt,
        create<send_entry>(chunk_sched, buf + chunk_offset, chunk_size, dtype, dst, comm),
        { chunk_sched = scheds[(first_sched_idx + chunk_idx) % scheds.size()]; });
}

void make_chunked_recv_entry(std::vector<ccl_sched*>& scheds,
                             size_t first_sched_idx,
                             const ccl_buffer buf,
                             size_t cnt,
                             const ccl_datatype& dtype,
                             int src,
                             ccl_comm* comm) {
    CCL_CHUNKED_ENTRY_FUNCTION(
        "recv",
        dtype,
        cnt,
        create<recv_entry>(chunk_sched, buf + chunk_offset, chunk_size, dtype, src, comm),
        { chunk_sched = scheds[(first_sched_idx + chunk_idx) % scheds.size()]; });
}

void make_chunked_copy_entry(std::vector<ccl_sched*>& scheds,
                             size_t first_sched_idx,
                             const ccl_buffer in_buf,
                             ccl_buffer out_buf,
                             size_t cnt,
                             const ccl_datatype& dtype) {
    CCL_CHUNKED_ENTRY_FUNCTION(
        "copy",
        dtype,
        cnt,
        create<copy_entry>(
            chunk_sched, in_buf + chunk_offset, out_buf + chunk_offset, chunk_size, dtype),
        { chunk_sched = scheds[(first_sched_idx + chunk_idx) % scheds.size()]; });
}
} // namespace entry_factory

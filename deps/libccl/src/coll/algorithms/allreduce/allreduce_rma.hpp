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

typedef struct {
    int wait_dst;

    int src_peer;
    int dst_peer;

    volatile uint64_t sync_flag; // src side will write here the index of iteration it completed
    atl_mr_t* sync_flag_mr;

    uint64_t* sync_flags; // pre-computed values (iteration indexes) to be written to dst side
    atl_mr_t* sync_flags_mr;

    volatile uint64_t
        dst_ready_flag; // dst side will write here '1' to indicate it is ready for write ops
    atl_mr_t* dst_ready_flag_mr;

    uint64_t dst_ready_value;
    atl_mr_t* dst_ready_value_mr;

    atl_mr_t remote_sync_flag_mr;
    atl_mr_t remote_dst_ready_flag_mr;

    atl_mr_t* send_buf_mr;
    atl_mr_t* recv_buf_mr;
    atl_mr_t* tmp_buf_mr;

    atl_mr_t remote_rs_dst_buf_mr;
    atl_mr_t remote_recv_buf_mr;

} ccl_rma_ring_allreduce_handler;

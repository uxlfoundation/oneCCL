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

#include "common/utils/buffer.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"

// ze_a2a_allgatherv_entry
class ze_a2a_allgatherv_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_A2A_ALLGATHERV";
    }

    const char* name() const override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    explicit ze_a2a_allgatherv_entry(
        ccl_sched* sched,
        ccl_buffer send_buf,
        size_t send_count,
        std::vector<ccl_buffer> recv_bufs,
        std::vector<size_t> recv_counts,
        const ccl_datatype& dtype,
        ccl_comm* comm,
        const std::vector<ze_event_handle_t>& wait_events = {},
        size_t peer_buf_idx = 0,
        size_t peer_buf_offset = 0,
        bool is_monolithic_pipeline = false,
        ccl_comm* pipeline_comm = nullptr,
        // whether ipc handle exchange done only for start of the buffer
        // or separate handles for the buffer partition of each rank
        bool is_separate_block_handles = false,
        bool is_scaleout = false,
        size_t scaleout_offset = 0);

    void init_ze_hook() override;
    void update() override;

protected:
    void dump_detail(std::stringstream& str) const override;

private:
    static constexpr size_t event_group_count{ 1 }; // copy phase

    const ccl_buffer send_buf;
    const size_t send_count;
    const std::vector<ccl_buffer> recv_bufs;
    const std::vector<size_t> recv_counts;
    const ccl_datatype dtype;
    const size_t peer_buf_idx;
    const size_t peer_buf_offset;
    const int peer_count;
    const bool is_monolithic_pipeline;
    ccl_comm* pipeline_comm;
    const bool is_separate_block_handles;
    const bool is_scaleout;
    const size_t scaleout_offset;

    std::vector<ze_event_handle_t> copy_events;
    std::vector<ze_kernel> kernels;
    std::vector<ze_event_handle_t> kernel_events;
};

// ze_a2a_allgatherv_op
class ze_a2a_allgatherv_op {
public:
    ze_a2a_allgatherv_op() = delete;
    ze_a2a_allgatherv_op(const ze_a2a_allgatherv_op&) = delete;
    ze_a2a_allgatherv_op(ccl_sched* sched,
                         const ze_base_entry* entry,
                         ccl_comm* comm,
                         ccl_comm* pipeline_comm,
                         const ccl_datatype& dtype,
                         ccl_buffer send_buf,
                         const std::vector<ccl_buffer>& recv_bufs,
                         const std::vector<ccl_buffer>& peer_bufs,
                         const std::vector<ccl_buffer>& pair_peer_bufs,
                         const std::vector<size_t>& copy_bytes,
                         const std::vector<size_t>& recv_counts,
                         int peer_count,
                         const std::vector<size_t>& rank_buf_offsets,
                         size_t peer_buf_offset,
                         std::vector<ze_event_handle_t>& copy_events,
                         std::vector<ze_event_handle_t>& wait_events,
                         ze_event_handle_t out_event,
                         bool is_monolithic,
                         bool is_monolithic_pipeline,
                         bool is_inplace = false,
                         bool is_separate_block_handles = true,
                         bool is_scaleout = false,
                         size_t scaleout_offset = 0);
    // methods
    static void select(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels);
    // common
    ccl_sched* sched;
    const ze_base_entry* entry;
    ccl_comm* comm;
    // pipeline_comm is used for pipeline across MDFI
    ccl_comm* pipeline_comm;
    const ccl_datatype& dtype;
    // bufs
    ccl_buffer send_buf;
    const std::vector<ccl_buffer>& recv_bufs;
    const std::vector<ccl_buffer>& peer_bufs;
    const std::vector<ccl_buffer>& pair_peer_bufs;
    // other
    const std::vector<size_t>& copy_bytes;
    const std::vector<size_t>& recv_counts;
    int peer_count;
    // offsets
    const std::vector<size_t>& rank_buf_offsets;
    size_t peer_buf_offset;
    // events
    std::vector<ze_event_handle_t>& copy_events;
    std::vector<ze_event_handle_t>& wait_events;
    ze_event_handle_t out_event;
    // flags
    bool is_monolithic;
    bool is_monolithic_pipeline;
    bool is_inplace;
    bool is_separate_block_handles;
    bool is_scaleout;
    size_t scaleout_offset;

private:
    static void read_write(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels);
    static void read(ze_a2a_allgatherv_op& args);
    static void write(ze_a2a_allgatherv_op& args, std::vector<ze_kernel>& kernels);
};

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

typedef struct {
    ccl_comm* comm;
    const std::vector<size_t> recv_counts;
    const ccl_datatype dtype;
    ccl::reduction op;
} reduce_scatter_args;

typedef struct {
    ccl_buffer send_buf;
    ccl_buffer recv_buf;
    ccl_buffer tmp_write_buf;
    size_t peer_write_buf_idx;
    size_t send_buf_offset;
} reduce_scatter_bufs;

// Code path for copy_read + kernel call
class ze_a2a_reduce_scatter_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_A2A_REDUCE_SCATTER";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    ze_a2a_reduce_scatter_entry() = delete;
    explicit ze_a2a_reduce_scatter_entry(ccl_sched* sched,
                                         ccl_buffer send_buf,
                                         ccl_buffer recv_buf,
                                         const size_t* recv_counts,
                                         const ccl_datatype& dtype,
                                         ccl::reduction op,
                                         ccl_comm* comm,
                                         const std::vector<ze_event_handle_t>& wait_events = {},
                                         size_t peer_buf_idx = 0,
                                         size_t peer_buf_offset = 0);

    void init_ze_hook() override;
    void update() override;

    static void fill_list(const ze_base_entry* entry,
                          void* send_buf,
                          void* recv_buf,
                          void* tmp_buf,
                          const std::vector<ccl_buffer>& peer_send_bufs,
                          int peer_count,
                          int comm_rank,
                          size_t block_count,
                          size_t rank_buf_offset,
                          std::vector<ze_event_handle_t>& wait_events,
                          std::vector<ze_event_handle_t>& copy_events,
                          std::vector<ze_kernel>& kernels,
                          std::vector<ze_event_handle_t>& kernel_events,
                          ze_event_handle_t& barrier_event,
                          const ccl_datatype& dtype,
                          ze_module_handle_t module,
                          ze_device_handle_t device,
                          ze_context_handle_t context,
                          ccl::reduction op,
                          size_t worker_idx,
                          size_t peer_buf_offset,
                          bool is_monolithic);

private:
    static constexpr size_t event_group_count{ 3 }; // copy + kernel + copy

    const ccl_buffer send_buf;
    const ccl_buffer recv_buf;
    const ccl_datatype dtype;
    const ccl::reduction op;
    const std::vector<size_t> recv_counts;
    const size_t peer_buf_idx;
    const size_t peer_buf_offset;
    const int peer_count;

    std::vector<ze_event_handle_t> pre_copy_events;
    ze_event_handle_t barrier_event{};

    std::vector<ze_kernel> kernels;
    std::vector<ze_event_handle_t> kernel_events;

    static void kernel_init(size_t rank_buf_offset,
                            size_t block_count,
                            void* send_buf,
                            void* recv_buf,
                            void* base_ptr,
                            const std::vector<ccl_buffer>& peer_send_bufs,
                            int peer_count,
                            const ccl_datatype& dtype,
                            int comm_rank,
                            std::vector<ze_kernel>& kernels,
                            ze_module_handle_t module,
                            ze_device_handle_t device,
                            ze_context_handle_t context,
                            ccl::reduction op,
                            size_t worker_idx,
                            size_t peer_buf_offset,
                            bool is_monolithic);
};

class ze_a2a_reduce_scatter_write_copy_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_A2A_REDUCE_SCATTER_WRITE_COPY_ENTRY";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    ze_a2a_reduce_scatter_write_copy_entry() = delete;
    explicit ze_a2a_reduce_scatter_write_copy_entry(
        ccl_sched* sched,
        reduce_scatter_args rs_args,
        reduce_scatter_bufs rs_bufs,
        const std::vector<ze_event_handle_t>& wait_events = {});
    void init_ze_hook() override;

    void update() override;
    static void fill_list_copy(const ze_base_entry* entry,
                               reduce_scatter_args rs_args,
                               reduce_scatter_bufs rs_bufs,
                               const std::vector<ccl_buffer>& peer_recv_bufs,
                               int peer_count,
                               int comm_rank,
                               int comm_size,
                               std::vector<ze_event_handle_t>& copy_events,
                               ze_event_handle_t& barrier_event,
                               ze_module_handle_t module,
                               ze_device_handle_t device,
                               ze_context_handle_t context,
                               size_t worker_idx,
                               const std::vector<ze_event_handle_t>& wait_events);

private:
    static constexpr size_t event_group_count{ 3 };

    const reduce_scatter_args rs_args;
    const reduce_scatter_bufs rs_bufs;
    const int peer_count;

    std::vector<ze_event_handle_t> pre_copy_events;
    ze_event_handle_t barrier_event{};
};

class ze_a2a_reduce_scatter_write_kernel_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_A2A_REDUCE_SCATTER_WRITE_KERNEL_ENTRY";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    ze_a2a_reduce_scatter_write_kernel_entry() = delete;
    explicit ze_a2a_reduce_scatter_write_kernel_entry(
        ccl_sched* sched,
        reduce_scatter_args rs_args,
        reduce_scatter_bufs rs_bufs,
        const std::vector<ze_event_handle_t>& wait_events = {});
    void init_ze_hook() override;

    void update() override;
    static void fill_list_kernel(const ze_base_entry* entry,
                                 reduce_scatter_args rs_args,
                                 reduce_scatter_bufs rs_bufs,
                                 int peer_count,
                                 int comm_rank,
                                 int comm_size,
                                 size_t rank_buf_offset,
                                 std::vector<ze_kernel>& kernels,
                                 std::vector<ze_event_handle_t>& kernel_events,
                                 ze_module_handle_t module,
                                 ze_device_handle_t device,
                                 ze_context_handle_t context,
                                 size_t worker_idx,
                                 const std::vector<ze_event_handle_t>& wait_events);

private:
    static constexpr size_t event_group_count{ 3 };

    const reduce_scatter_args rs_args;
    const reduce_scatter_bufs rs_bufs;
    const int peer_count;

    ze_event_handle_t barrier_event{};

    std::vector<ze_kernel> kernels;
    std::vector<ze_event_handle_t> kernel_events;

    static void kernel_init(size_t rank_buf_offset,
                            reduce_scatter_args rs_args,
                            reduce_scatter_bufs rs_bufs,
                            int peer_count,
                            int comm_rank,
                            std::vector<ze_kernel>& kernels,
                            ze_module_handle_t module,
                            ze_device_handle_t device,
                            ze_context_handle_t context,
                            size_t worker_idx);
};

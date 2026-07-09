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
#include "comp/comp.hpp"
#include "sched/entry/ze/ze_base_entry.hpp"
#include "sched/entry/ze/ze_handle_exchange_entry.hpp"

#include <atomic>
#include <sstream>

class ze_a2a_pipeline_read_write_entry : public ze_base_entry {
public:
    struct attr {
        // whether to divide the data into continous chunks
        // for example, in reduce and allreduce where the buffer
        // is divided into two chunks, one for each tile in the GPU
        // or to divide the data in a strided manner, for example
        // in reduce_scatter we divide the buffer into node_comm size
        // number of partitions and tile 0 uses even chunks and
        // tile 1 uses odd chunks.
        bool use_continous_data;

        // whether the target buffer is remote in which case use
        // the handle manager to get the remote pointer or
        // directly use the local pointer
        bool use_remote_target;
    };

    static constexpr const char* class_name() noexcept {
        return "ZE_READ_WRITE_REDUCE";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    ze_a2a_pipeline_read_write_entry() = delete;
    explicit ze_a2a_pipeline_read_write_entry(ccl_sched* sched,
                                              ccl_comm* comm,
                                              ccl_buffer send_buf,
                                              std::vector<ccl_buffer> tmp_bufs,
                                              size_t tmp_buf_idx_start,
                                              size_t count,
                                              size_t count_per_rank,
                                              size_t read_block_inner_offset,
                                              const ccl_datatype& dtype,
                                              ccl::reduction op,
                                              std::vector<ze_event_handle_t>& wait_events,
                                              const attr& attrs);

    void init_ze_hook() override;

protected:
    void dump_detail(std::stringstream& str) const override;

private:
    ccl_buffer send_buf;
    std::vector<ccl_buffer> tmp_bufs;

    size_t tmp_buf_idx_start;
    size_t count;
    // Count of elements in send/mdfi read buffer
    // for just one rank. In case of chunking,
    // we need the variable to find start
    // of `send_buf` that is required by a single rank.
    size_t count_per_rank;
    // Without chunking, the offset is 0. In case of chunking,
    // it represents chunk offset inside rank's buffer buffer.
    size_t read_block_inner_offset;

    const ccl_datatype dtype;
    ccl::reduction op;
    const attr attrs;

    std::string kernel_name;
};

class ze_a2a_pipeline_reduce_entry : public ze_base_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "ZE_LOCAL_SCATTER";
    }

    const char* name() const noexcept override {
        return class_name();
    }

    virtual std::string name_ext() const override;

    ze_a2a_pipeline_reduce_entry() = delete;
    explicit ze_a2a_pipeline_reduce_entry(ccl_sched* sched,
                                          ccl_comm* comm,
                                          ccl_buffer recv_buf,
                                          std::vector<ccl_buffer> tmp_bufs,
                                          size_t count,
                                          const ccl_datatype& dtype,
                                          ccl::reduction op,
                                          const std::vector<ze_event_handle_t>& wait_events);

    void init_ze_hook() override;

protected:
    void dump_detail(std::stringstream& str) const override;

private:
    ccl_buffer recv_buf;
    std::vector<ccl_buffer> tmp_bufs;

    size_t count;

    const ccl_datatype dtype;
    ccl::reduction op;
    std::string kernel_name;
};

namespace ze_utils {
void alloc_tmp_bufs(ccl_sched* sched,
                    ccl_comm* comm,
                    std::vector<ccl_buffer>& tmp_bufs,
                    std::vector<ze_handle_exchange_entry::mem_desc_t>& in_buffers,
                    size_t& tmp_buf_idx_start,
                    size_t count,
                    const ccl_datatype& dtype);
} // namespace ze_utils

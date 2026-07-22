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

#include "common/stream/stream.hpp"
#include "comp/comp.hpp"
#include "sched/entry/ze/ze_a2a_pipeline_reduce_scatter_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/queue/queue.hpp"
#include "coll/coll_util.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include <string>
#include <sstream>

using namespace ccl;
using namespace ccl::ze;

// ze_a2a_pipeline_read_write_entry and ze_a2a_pipeline_reduce_entry
// combining emulates monolithic pipeline reduce_scatter operation

ze_a2a_pipeline_read_write_entry::ze_a2a_pipeline_read_write_entry(
    ccl_sched* sched,
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
    const attr& attrs)
        : ze_base_entry(sched, wait_events, comm, 1 /* request additional events */),
          send_buf(send_buf),
          tmp_bufs(std::move(tmp_bufs)),
          tmp_buf_idx_start(tmp_buf_idx_start),
          count(count),
          count_per_rank(count_per_rank),
          read_block_inner_offset(read_block_inner_offset),
          dtype(dtype),
          op(op),
          attrs(attrs) {}

void ze_a2a_pipeline_read_write_entry::init_ze_hook() {
    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();
    ccl_comm* node_comm = comm->get_node_comm().get();

    size_t base_count = count / pair_comm->size();
    // TODO: if the code is used with `ccl_offset_explicit` chunking
    // and `use_continous_data` attribute, the `pair_comm_offset_bytes`
    // might need to be calculated using `send_buf_count_per_rank`.
    size_t pair_comm_offset_bytes = base_count * pair_comm->rank() * dtype.size();
    if (pair_comm->rank() == pair_comm->size() - 1)
        base_count += count % pair_comm->size();

    size_t main_block_count = base_count / even_comm->size();
    size_t block_count = main_block_count;
    size_t block_count_last_rank = block_count;
    block_count_last_rank += base_count % even_comm->size();

    // the block stores can`t be used for unaligned data.
    // Address must be aligned to a 128-bit (16-byte) boundary.
    // When using remote target buffers, each of them is allocated and aligned separately.
    // However, for the local tmp buffer, one buffer cut into several blocks of contiguous data is required.
    // Therefore it is hard to align each block inside the buffer
    // and ensure correctness for the subsequent entries.
    int can_use_block = attrs.use_remote_target;

    if (!attrs.use_continous_data) {
        CCL_THROW_IF_NOT(block_count_last_rank == block_count,
                         "block_count : ",
                         block_count,
                         " should be equal to last block_count : ",
                         block_count_last_rank);
    }

    kernel_name = "reduce_read_write_kernel_" + std::to_string(even_comm->size() - 1) + "_" +
                  to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);

    LOG_DEBUG("reduce_read_write kernel name: ", kernel_name);

    auto fill_vec = [&](std::vector<void*>& vec, void* base_ptr) {
        for (int idx = 0; idx < even_comm->size(); idx++) {
            if (!attrs.use_continous_data) {
                const int global_rank = even_comm->get_global_rank(idx);
                const int node_rank = node_comm->get_rank_from_global(global_rank);

                // The offset must be calculated independently from `count`, because it
                // might change depending on chunks count.
                auto even_comm_offset_bytes = count_per_rank * node_rank * dtype.size();
                vec[idx] =
                    static_cast<char*>(base_ptr) + even_comm_offset_bytes + read_block_inner_offset;
            }
            else {
                auto even_comm_offset_bytes = count_per_rank * idx * dtype.size();
                vec[idx] = static_cast<char*>(base_ptr) + pair_comm_offset_bytes +
                           even_comm_offset_bytes + read_block_inner_offset;
            }
        }
    };

    std::vector<void*> local_send_bufs(even_comm->size());
    fill_vec(local_send_bufs, send_buf.get_ptr());

    // mdfi
    ccl_buffer mdfi_buf_ptr;
    int pair_peer_rank = (pair_comm->rank() + 1) % pair_comm->size();
    if (pair_peer_rank != pair_comm->rank()) {
        ccl_buffer buf;
        sched->get_memory().handle_manager.get(pair_peer_rank, 0, buf, pair_comm);
        CCL_THROW_IF_NOT(buf.get_ptr(), "null IPC buffer is received");
        mdfi_buf_ptr = buf;
    }

    std::vector<void*> mdfi_bufs(even_comm->size());
    fill_vec(mdfi_bufs, mdfi_buf_ptr.get_ptr());

    std::vector<void*> tmp_buf_ptrs(even_comm->size());
    for (int idx = 0; idx < even_comm->size(); idx++) {
        int even_peer_rank = (even_comm->rank() + idx + 1) % even_comm->size();
        if (even_peer_rank != even_comm->rank() && attrs.use_remote_target) {
            ccl_buffer buf;
            sched->get_memory().handle_manager.get(
                even_peer_rank, tmp_buf_idx_start + even_comm->rank(), buf, even_comm);
            CCL_THROW_IF_NOT(buf.get_ptr(), "null IPC buffer is received");
            tmp_buf_ptrs[even_peer_rank] = buf.get_ptr();
        }
        else {
            tmp_buf_ptrs[even_peer_rank] = tmp_bufs[even_peer_rank].get_ptr();
        }
    }

    ze_kernel_args_t kernel_args{ local_send_bufs,        mdfi_bufs,     tmp_buf_ptrs, &block_count,
                                  &block_count_last_rank, &can_use_block };

    ze_kernel kernel(module, kernel_name, kernel_args, block_count, worker_idx);
    ZE_APPEND_CALL(ze_cmd_launch_kernel,
                   ze_base_entry::get_comp_list(),
                   std::move(kernel),
                   ze_base_entry::entry_event,
                   wait_events);
}

std::string ze_a2a_pipeline_read_write_entry::name_ext() const {
    std::stringstream out;
    out << name();
    return out.str();
}

void ze_a2a_pipeline_read_write_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       "dt ",
                       ccl::global_data::get().dtypes->name(dtype),
                       ", send_bufs ",
                       &send_buf,
                       ", comm ",
                       comm->to_string(),
                       ", context ",
                       context,
                       "\n");
}

ze_a2a_pipeline_reduce_entry::ze_a2a_pipeline_reduce_entry(
    ccl_sched* sched,
    ccl_comm* comm,
    ccl_buffer recv_buf,
    std::vector<ccl_buffer> tmp_bufs,
    size_t count,
    const ccl_datatype& dtype,
    ccl::reduction op,
    const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, comm, 1 /* request additional events */),
          recv_buf(recv_buf),
          tmp_bufs(std::move(tmp_bufs)),
          count(count),
          dtype(dtype),
          op(op) {}

void ze_a2a_pipeline_reduce_entry::init_ze_hook() {
    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();

    size_t base_count = count / pair_comm->size();
    if (pair_comm->rank() == pair_comm->size() - 1)
        base_count += count % pair_comm->size();

    size_t main_block_count = base_count / even_comm->size();
    size_t block_count = main_block_count;
    if (even_comm->rank() == even_comm->size() - 1)
        block_count += base_count % even_comm->size();

    kernel_name = "local_reduce_kernel_" + std::to_string(even_comm->size() - 1) + "_" +
                  to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);
    LOG_DEBUG("local_reduce kernel name: ", kernel_name);

    std::vector<void*> tmp_buf_ptrs;
    tmp_buf_ptrs.reserve(even_comm->size());
    for (const auto& buf : tmp_bufs) {
        tmp_buf_ptrs.push_back(buf.get_ptr());
    }

    void* recv_buf_ptr = recv_buf.get_ptr();
    ze_kernel_args_t kernel_args{ &block_count, tmp_buf_ptrs, &recv_buf_ptr };
    ze_kernel kernel(module, kernel_name, kernel_args, block_count, worker_idx);
    ZE_APPEND_CALL(ze_cmd_launch_kernel,
                   ze_base_entry::get_comp_list(),
                   std::move(kernel),
                   ze_base_entry::entry_event,
                   wait_events);
}

std::string ze_a2a_pipeline_reduce_entry::name_ext() const {
    std::stringstream out;
    out << name();
    return out.str();
}

void ze_a2a_pipeline_reduce_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       "dt ",
                       ccl::global_data::get().dtypes->name(dtype),
                       ", recv_bufs ",
                       &recv_buf,
                       ", comm ",
                       comm->to_string(),
                       ", context ",
                       context,
                       "\n");
}

namespace ze_utils {
void alloc_tmp_bufs(ccl_sched* sched,
                    ccl_comm* comm,
                    std::vector<ccl_buffer>& tmp_bufs,
                    std::vector<ze_handle_exchange_entry::mem_desc_t>& in_buffers,
                    size_t& tmp_buf_idx_start,
                    size_t count,
                    const ccl_datatype& dtype) {
    ccl_comm* pair_comm = comm->get_pair_comm().get();
    ccl_comm* even_comm = comm->get_even_comm().get();

    tmp_bufs.resize(even_comm->size());
    size_t base_count = count / pair_comm->size();
    if (pair_comm->rank() == pair_comm->size() - 1) {
        base_count += count % pair_comm->size();
    }
    size_t main_block_count = base_count / even_comm->size();
    size_t block_count = main_block_count;
    if (even_comm->rank() == even_comm->size() - 1) {
        block_count += base_count % even_comm->size();
    }

    for (int idx = 0; idx < even_comm->size(); idx++) {
        ccl::alloc_param alloc_tmp_param(
            block_count * dtype.size(), ccl::buffer_type::ze, ccl::buffer_place::device);
        tmp_bufs[idx] = sched->alloc_buffer(alloc_tmp_param);
    }
    tmp_buf_idx_start = in_buffers.size();
    in_buffers.reserve(tmp_buf_idx_start + tmp_bufs.size());
    for (const auto& buf : tmp_bufs) {
        in_buffers.push_back({ buf.get_ptr(), ccl::ze::ipc_mem_type::memory });
    }
}
} // namespace ze_utils

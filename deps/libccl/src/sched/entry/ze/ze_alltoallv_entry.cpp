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
#include "sched/entry/ze/ze_alltoallv_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"
#include "sched/queue/queue.hpp"

#include <string>
#include <sstream>

using namespace ccl;
using namespace ccl::ze;

ze_alltoallv_entry::ze_alltoallv_entry(ccl_sched* sched,
                                       std::vector<ccl_buffer> send_bufs,
                                       std::vector<ccl_buffer> recv_bufs,
                                       std::vector<size_t> counts,
                                       size_t buf_idx_start,
                                       const ccl_datatype& dtype,
                                       ccl_comm* comm,
                                       const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, comm, 1 /* request additional events */),
          send_bufs(std::move(send_bufs)),
          recv_bufs(std::move(recv_bufs)),
          counts(std::move(counts)),
          buf_idx_start(buf_idx_start),
          dtype(dtype) {}

void ze_alltoallv_entry::init_ze_hook() {
    auto comm_size = comm->size();
    auto comm_rank = comm->rank();

    std::vector<void*> in_bufs(send_bufs.size());
    std::vector<void*> out_bufs(recv_bufs.size());

    ccl::global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);
    kernel_name =
        "alltoallv_kernel_" + std::to_string(comm_size) + "_" + to_string(dtype.idx()) + "_custom";
    LOG_DEBUG("kernel name: ", kernel_name);

    auto init_in_out_bufs = [&](std::vector<void*>& src_bufs1,
                                std::vector<void*>& src_bufs2,
                                std::vector<ccl_buffer>& dest_bufs1,
                                std::vector<ccl_buffer>& dest_bufs2) {
        for (int idx = 0; idx < comm_size; idx++) {
            const auto global_rank = comm->get_global_rank(idx);
            if (counts[global_rank] == 0) {
                continue;
            }

            if (idx == comm_rank) {
                // fill local send buf
                src_bufs1[global_rank] = dest_bufs1[global_rank].get_ptr();
            }
            else {
                ccl_buffer buf{};
                sched->get_memory().handle_manager.get(
                    idx, buf_idx_start + comm->get_global_rank(comm_rank), buf, comm);
                CCL_THROW_IF_NOT(buf.get_ptr(), "null IPC buffer is received");
                src_bufs1[global_rank] = buf.get_ptr();
            }
        }

        // recv bufs are local bufs
        for (int idx = 0; idx < comm_size; idx++) {
            const int peer_rank = (comm_rank + idx + 1) % comm_size;
            const auto global_peer_rank = comm->get_global_rank(peer_rank);
            src_bufs2[global_peer_rank] = dest_bufs2[global_peer_rank].get_ptr();
        }
    };

    if (ccl::global_data::env().alltoallv_monolithic_read_kernel) {
        init_in_out_bufs(in_bufs, out_bufs, send_bufs, recv_bufs);
        LOG_DEBUG("alltoallv monolithic read kernel");
    }
    else {
        init_in_out_bufs(out_bufs, in_bufs, recv_bufs, send_bufs);
        LOG_DEBUG("alltoallv monolithic write kernel");
    }

    // sets in/out bufs for kernel
    std::vector<ze_kernel_arg_t> kernel_args;
    kernel_args.reserve(3 * comm_size + 1);
    for (int idx = 0; idx < comm_size; idx++) {
        const auto global_rank = comm->get_global_rank(idx);
        if (in_bufs[global_rank]) {
            kernel_args.push_back(&(in_bufs[global_rank]));
        }
        else {
            CCL_THROW_IF_NOT(counts[global_rank] == 0,
                             "kernel input buffer is null but counts > 0");
            kernel_args.push_back({});
        }
        kernel_args.push_back(&(out_bufs[global_rank]));
        kernel_args.push_back(&(counts[global_rank]));
    }

    kernel_args.push_back(&comm_size);

    ze_kernel kernel(module, kernel_name, kernel_args, counts[comm_rank], worker_idx);

    ZE_APPEND_CALL(ze_cmd_launch_kernel,
                   ze_base_entry::get_comp_list(),
                   std::move(kernel),
                   ze_base_entry::entry_event,
                   wait_events);
}

void ze_alltoallv_entry::finalize_ze_hook() {
    LOG_DEBUG("finalize");
}

void ze_alltoallv_entry::start() {
    size_t kernel_counter = 0;
    if (global_data::env().enable_kernel_sync) {
        kernel_counter = global_data::get().ze_data->kernel_counter++;
    }

    if (kernel_counter == 0) {
        ze_base_entry::start();
    }
    else {
        global_data::get().ze_data->kernel_counter--;
        status = ccl_sched_entry_status_again;
    }
}

void ze_alltoallv_entry::update() {
    ze_base_entry::update();

    if (global_data::env().enable_kernel_sync && global_data::get().ze_data->kernel_counter > 0) {
        global_data::get().ze_data->kernel_counter--;
    }
}

std::string ze_alltoallv_entry::name_ext() const {
    std::stringstream out;
    out << name();
    return out.str();
}

void ze_alltoallv_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       "dt ",
                       ccl::global_data::get().dtypes->name(dtype),
                       ", send_bufs ",
                       &send_bufs,
                       ", recv_bufs ",
                       &recv_bufs,
                       ", comm ",
                       comm->to_string(),
                       ", context ",
                       context,
                       "\n");
}

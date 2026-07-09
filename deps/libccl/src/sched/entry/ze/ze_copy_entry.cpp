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

#include "sched/entry/copy/copy_helper.hpp"
#include "sched/entry/factory/entry_factory.hpp"
#include "sched/entry/ze/ze_copy_entry.hpp"

using namespace ccl;

ze_copy_entry::ze_copy_entry(ccl_sched* sched,
                             ccl_buffer in_buf,
                             ccl_buffer out_buf,
                             size_t count,
                             const ccl_datatype& dtype,
                             const copy_attr& attr,
                             const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events, nullptr /*comm*/, 1 /*add_event_count*/),
          in_buf(in_buf),
          out_buf(out_buf),
          dtype(dtype),
          attr(attr),
          count(count) {
    CCL_THROW_IF_NOT(sched, "no sched");
}

void ze_copy_entry::init_ze_hook() {
    int peer_rank = attr.peer_rank;
    if (attr.pt2pt_op) {
        peer_rank = ccl::ze::ipc_handle_manager::pt2pt_handles_size - 1;
    }

    if (attr.peer_rank != ccl_comm::invalid_rank) {
        if (!out_buf) {
            sched->get_memory().handle_manager.get(
                peer_rank, attr.peer_buf_idx, out_buf, attr.map_comm, attr.pt2pt_op);
        }

        if (!in_buf) {
            sched->get_memory().handle_manager.get(
                peer_rank, attr.peer_buf_idx, in_buf, attr.map_comm, attr.pt2pt_op);
        }
    }

    void* dst = static_cast<char*>(out_buf.get_ptr()) + attr.out_buf_offset * dtype.size();
    void* src = static_cast<char*>(in_buf.get_ptr()) + attr.in_buf_offset * dtype.size();

    ze_command_list_handle_t list =
        ze_base_entry::get_copy_list(attr.direction, attr.hint_queue_index, attr.force_queue_type);

    ZE_APPEND_CALL(ze_cmd_memory_copy,
                   list,
                   dst,
                   src,
                   dtype.size() * count,
                   ze_base_entry::entry_event,
                   wait_events);
}

std::string ze_copy_entry::name_ext() const {
    std::stringstream out;
    out << name();
    if (attr.direction != copy_direction::undefined) {
        out << ":" << to_string(attr.direction);
    }
    out << ":" << count * dtype.size();
    return out.str();
}

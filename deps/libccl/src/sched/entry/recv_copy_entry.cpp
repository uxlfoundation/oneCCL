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

#include "comm/comm.hpp"
#include "comp/comp.hpp"
#include "sched/entry/entry.hpp"
#include "sched/entry/recv_copy_entry.hpp"
#include "sched/queue/queue.hpp"

void recv_copy_entry::start() {
    atl_tag = comm->get_atl_comm()->tag_creator->create(
        src, comm->get_comm_id(), sched->sched_id, sched->get_op_id());
    LOG_DEBUG("starting RECV in RECV_COPY entry, src ",
              src,
              ", tag ",
              atl_tag,
              ", req ",
              req,
              ", bytes ",
              bytes);

    atl_status_t atl_status = comm->get_atl_comm()->recv(
        sched->bin->get_atl_ep(), recv_buf.get_ptr(bytes), bytes, src, atl_tag, req);

    update_status(atl_status);
}

void recv_copy_entry::update() {
    atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);

    if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
        CCL_THROW("RECV_COPY entry failed. atl_status: ", atl_status_to_str(atl_status));
    }

    if (!req.is_completed) {
        return;
    }

    LOG_DEBUG("completed RECV in RECV_COPY entry, req=", req, ", starting COPY");

    auto comp_status = ccl_comp_copy(
        recv_buf.get_ptr(bytes), copy_buf.get_ptr(bytes), bytes, attr.use_nontemporal);
    CCL_ASSERT(comp_status == ccl::status::success, "bad status ", comp_status);

    status = ccl_sched_entry_status_complete;
    LOG_DEBUG("completed COPY in RECV_COPY entry");
}

void recv_copy_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       ", recv_buf ",
                       recv_buf,
                       ", copy_buf ",
                       copy_buf,
                       ", bytes ",
                       bytes,
                       ", src ",
                       src,
                       ", atl_tag ",
                       atl_tag,
                       ", comm_id ",
                       comm->get_comm_id(),
                       ", req ",
                       req,
                       "\n");
}

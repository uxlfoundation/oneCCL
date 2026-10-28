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


#include "coll/algorithms/algorithm_utils.hpp"
#include "common/env/env.hpp"
#include "common/global/global.hpp"
#include "coll/algorithms/utils/sycl_ptrs.hpp"
#include "common/ze/ze_wrapper.hpp"
#include "common/utils/ce_utils.hpp"

// Allgather for large payload using copy engine
// even this function is in allgatherv path, it implements only allgather, 
// recv_counts and offsets are not used, but kept for compatibility with other allgatherv implementations
ccl::event allgather_large_copy_engine(sycl::queue& q,
                                       const void* send_buf,
                                       size_t send_count,
                                       void* recv_buf,
                                       const ccl::vector_class<size_t>& recv_counts,
                                       const ccl::vector_class<size_t>& offsets,
                                       ccl::datatype dtype,
                                       ccl_comm* comm,
                                       ccl_stream* global_stream,
                                       sycl_ptrs_type& sycl_ptrs,
                                       const ccl::vector_class<ccl::event>& deps)
{
    // 1. run prologue - sync and get cmd list
    LOG_TRACE("start - rank(", comm->rank(), ")");

    ce_utils_t ce_utils(comm, q); 
    ze_cmd_list_ptr cmd_list = ce_utils.prolog(comm, q);
    CCL_THROW_IF_NOT(cmd_list, "failed to create command list for copy engine");

    // 2. copy data to all ranks over L0 list

    // get data type size
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    // comm and data size needed for steps calculations
    int my_node_rank = comm->get_node_comm()->rank();
    size_t comm_size = comm->get_node_comm()->size();
    size_t data_size = send_count * dsize;
    LOG_DEBUG("global rank(", comm->rank(),"), node rank(", my_node_rank, "/", comm_size, "), data size(", data_size, "), send from {", send_buf, "}, recv to {", recv_buf, "}");
    
    // send address is the input buffer
    const void* send_addr = send_buf;

    // copy self data to dest at offset
    void* self_dest_addr = (uint8_t*)recv_buf + my_node_rank * data_size;
    LOG_TRACE("copy self data ", data_size, " bytes from {", send_addr, "} to {", self_dest_addr, "}");
    CCL_THROW_IF_NOT(cmd_list->mem_copy(self_dest_addr, send_addr, data_size) == ZE_RESULT_SUCCESS,
                     "failed to copy self data on rank ", my_node_rank);
    
    // comm size - 1 steps
    uint32_t steps = comm_size - 1;

    // pair - wise send/recv, one loop to send, one loop to wait for recv
    for (size_t step = 0; step < steps; step++)
    {
        // calculate step params - to_rank, recv addr, offsets
        // Step k: Send to (my + k + 1)
        int to_rank = (my_node_rank + step + 1) % comm_size;

        // copy destination to_rank recv address + my_rank offset
        void* dest_addr = (uint8_t*)sycl_ptrs.node_ptrs_wr[to_rank] + my_node_rank * data_size;
        LOG_DEBUG("Rank(", my_node_rank, 
                 ") copy ", data_size, 
                 " bytes from {", send_addr, 
                 "} to {", dest_addr, 
                 ") on rank(", to_rank, ")");

        // get the signal event
        ze_cb_event_ptr remote_ev = comm->get_node_comm()->get_remote_cb_event(to_rank);

        // send to to_rank and signal remote event
        CCL_THROW_IF_NOT(cmd_list->mem_copy(dest_addr, send_addr, data_size) == ZE_RESULT_SUCCESS,
                         "failed to copy to rank ", to_rank);
        LOG_TRACE("signal to rank(", to_rank,
             "), remote event ptr {", remote_ev, "}, handle {", remote_ev->handle(), "}");
        CCL_THROW_IF_NOT(remote_ev->signal(*cmd_list) == ZE_RESULT_SUCCESS,
                         "failed to signal remote cb event for rank ", to_rank);
    }

    for (size_t step = 0; step < steps; step++)
    {
        // Step k: Recv from (my - k - 1)
        int from_rank = (my_node_rank - step - 1 + comm_size) % comm_size;

        // get wait event and wait for recv from from_rank
        ze_cb_event_ptr local_ev = comm->get_node_comm()->get_local_cb_event(from_rank);
        LOG_TRACE("add wait to recv from rank(", from_rank,
             "), local event ptr {", local_ev, "}, handle {", local_ev->handle(), "}");
        CCL_THROW_IF_NOT(cmd_list->wait_event(*local_ev) == ZE_RESULT_SUCCESS,
                         "failed to wait on cb event from rank ", from_rank);

        // Reset the counter after waiting to prepare for next collective
        CCL_THROW_IF_NOT(local_ev->reset_counter(*cmd_list) == ZE_RESULT_SUCCESS,
                         "failed to reset local cb event for rank ", from_rank);
    }

    // 3. run epilogue - sync and signal completion
    std::shared_ptr<sycl::event> sycl_ev = ce_utils.epilog(comm, q);
    LOG_TRACE("all ranks synchronized - rank ", my_node_rank);
    ccl::event ccl_ev = ccl::event::create_from_native(*sycl_ev);
    return ccl_ev;
}


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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "oneapi/ccl/types.hpp"
#include "common/ze/ze_wrapper.hpp"
#include "common/utils/ce_utils.hpp"

ccl::event alltoall_large_copy_engine(sycl::queue &q,
                                      ccl_comm* comm,
                                      const std::array<void *, MAX_NODE_RANKS> &send_bufs,
                                      std::array<void *, MAX_NODE_RANKS> &recv_bufs,
                                      const size_t count,
                                      ccl::datatype dtype,
                                      std::vector<size_t> &offsets,
                                      const size_t rank,
                                      const size_t comm_size,
                                      bool is_multi_thread = false) {
    ce_utils_t ce_utils(comm, q);
    ze_cmd_list_ptr cmd_list = ce_utils.prolog(comm, q);
    CCL_THROW_IF_NOT(cmd_list, "failed to create command list for copy engine");

    // get data type size
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);
    const size_t dsize = ccl_dtype.size();

    size_t data_size = count * dsize;
    LOG_DEBUG("rank(",
              rank,
              "/",
              comm_size,
              "), data size(",
              data_size,
              "), send from {",
              send_bufs[rank],
              "}, recv to {",
              recv_bufs[rank],
              "}");

    // copy self data: in alltoall, rank R sends chunk R of send_buf to itself
    const void *self_src_addr = (uint8_t *)send_bufs[rank] + rank * data_size;
    void *self_dest_addr = (uint8_t *)recv_bufs[rank] + rank * data_size;
    LOG_TRACE("copy self data ",
              data_size,
              " bytes from {",
              self_src_addr,
              "} to {",
              self_dest_addr,
              "}");
    CCL_THROW_IF_NOT(cmd_list->mem_copy(self_dest_addr, self_src_addr, data_size) ==
                         ZE_RESULT_SUCCESS,
                     "failed to copy self data on rank ",
                     rank);

    // calculate number of steps according to comm ize
    uint32_t steps = comm_size - 1;

    // pair - wise send/recv
    for (size_t step = 0; step < steps; step++) {
        // Step k: Send to (my + k + 1), Recv from (my - k - 1)
        int to_rank = (rank + step + 1) % comm_size;
        int from_rank = (rank - step - 1 + comm_size) % comm_size;

        // write protocol: push my send data into remote rank's recv buffer
        const void *send_addr = (uint8_t *)send_bufs[rank] + to_rank * data_size;

        // destination: to_rank's recv buffer at my rank's offset
        void *dest_addr = (uint8_t *)recv_bufs[to_rank] + rank * data_size;
        LOG_DEBUG("Rank(",
                  rank,
                  ") write ",
                  data_size,
                  " bytes from {",
                  send_addr,
                  "} to {",
                  dest_addr,
                  "} on rank(",
                  to_rank,
                  ")");

        ze_cb_event_ptr remote_ev = comm->get_node_comm()->get_remote_cb_event(to_rank);

        CCL_THROW_IF_NOT(cmd_list->mem_copy(dest_addr, send_addr, data_size) == ZE_RESULT_SUCCESS,
                         "failed to copy to rank ",
                         to_rank);
        LOG_TRACE("signal to rank(",
                  to_rank,
                  "), remote event ptr {",
                  remote_ev,
                  "}, handle {",
                  remote_ev->handle(),
                  "}");
        CCL_THROW_IF_NOT(remote_ev->signal(*cmd_list) == ZE_RESULT_SUCCESS,
                         "failed to signal remote cb event for rank ",
                         to_rank);

        // wait for from_rank to finish writing to my recv buffer
        ze_cb_event_ptr local_ev = comm->get_node_comm()->get_local_cb_event(from_rank);
        LOG_TRACE("wait for write from rank(",
                  from_rank,
                  "), local event ptr {",
                  local_ev,
                  "}, handle {",
                  local_ev->handle(),
                  "}");
        CCL_THROW_IF_NOT(cmd_list->wait_event(*local_ev) == ZE_RESULT_SUCCESS,
                         "failed to wait on cb event from rank ",
                         from_rank);

        // Reset the counter after waiting to prepare for next collective
        CCL_THROW_IF_NOT(local_ev->reset_counter(*cmd_list) == ZE_RESULT_SUCCESS,
                         "failed to reset local cb event for rank ",
                         from_rank);
    }

    // run epilogue - sync and signal completion
    std::shared_ptr<sycl::event> sycl_ev = ce_utils.epilog(comm, q);
    LOG_TRACE("all ranks synchronized - rank ", rank);
    ccl::event ccl_ev = ccl::event::create_from_native(*sycl_ev);
    return ccl_ev;
}

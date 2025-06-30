/*
 Copyright 2016-2020 Intel Corporation
 
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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "common/request/request.hpp"
#include "common/event/impls/host_event.hpp"
#include "coll/algorithms/recv/sycl/recv_sycl.hpp"
#include "sched/entry/ze/ze_pt2pt_barrier_entry.hpp"

namespace ccl {
namespace v1 {

static ccl::event recv_sycl_single_node(sycl::queue& q,
                                        void* recv_buf,
                                        size_t count,
                                        ccl::datatype dtype,
                                        int peer_rank,
                                        ccl_comm* comm,
                                        ccl_stream* global_stream,
                                        const pt2pt_attr& attr,
                                        const std::vector<ccl::event>& deps,
                                        bool& done) {
    ccl_comm* node_comm = comm->get_node_comm().get();
    auto node_peer_rank = node_comm->get_rank_from_global(peer_rank);
    auto node_curr_rank = node_comm->rank();
    auto sycl_queue = global_stream->get_native_stream();

    auto* tagc = node_comm->get_atl_comm()->tag_creator.get();
    auto comm_id = comm->get_atl_comm()->get_comm_id();
    auto [sync_ready, sync_done] = tagc->get_pt2pt_sync_tags();
    uint64_t tag_ready = tagc->create(node_curr_rank, comm_id, sync_ready);
    uint64_t tag_done = tagc->create(node_curr_rank, comm_id, sync_done);

    LOG_DEBUG("recv_sycl_single_node: recv_buf=",
              recv_buf,
              ", count=",
              count,
              ", peer_rank=",
              node_peer_rank);
    if (count == 0) {
        auto sycl_deps = get_sycl_events(deps);
        sycl::event barrier = submit_wait_on_events(sycl_queue, sycl_deps);

        sycl::event ack_event = post_host_task_ack(sycl_queue,
                                                   std::vector<sycl::event>{ barrier },
                                                   comm,
                                                   /*do_send_ack*/ true,
                                                   node_peer_rank,
                                                   tag_done);

        done = true;
        return ccl::event::create_from_native(ack_event);
    }

    const std::vector<ze_handle_exchange_entry::mem_desc_t> buffer = {
        { recv_buf, ccl::ze::ipc_mem_type::memory }
    };

    ccl::utils::pt2pt_handle_exchange_info info{ node_peer_rank,
                                                 ccl::utils::pt2pt_handle_exchange_role::receiver };
    if (!ccl::global_data::env().sycl_pt2pt_read) {
        info.role = ccl::utils::pt2pt_handle_exchange_role::sender;
    }

    ccl_coll_param param = ccl_coll_param::create_recv_param(
        recv_buf, count, dtype, peer_rank, attr, comm, global_stream, deps);
    ccl_coll_attr coll_attr{};
    ccl_sched* sched = ccl_sched::create(param, coll_attr);

    auto* exchange_entry =
        new ze_handle_exchange_entry(sched, node_comm, buffer, ccl_comm::invalid_rank, info);

    exchange_entry->start();
    while (!exchange_entry->is_completed()) {
        exchange_entry->update();
    }

    ccl::event ret_evt;
    auto sycl_deps = get_sycl_events(deps);

    // Read mode => remote pointer -> local buffer
    if (ccl::global_data::env().sycl_pt2pt_read) {
        LOG_DEBUG("recv_sycl_single_node: read mode enabled");

        // in read mode we have to wait for confirmation from sender
        // this is needed to avoid situation when we will read before the data is ready
        sycl::event sync_event = pt2pt_pre_sync(sycl_queue,
                                                sycl_deps,
                                                comm,
                                                /*do_send=*/false,
                                                node_peer_rank,
                                                tag_ready);

        ccl_buffer out_buf;
        sched->get_memory().handle_manager.get(0, 0, out_buf, node_comm, /*pt2pt_op=*/true);

        CCL_THROW_IF_NOT(out_buf.get_ptr(), "no pointer from peer in read mode");

        int bytes = ccl::get_datatype_size(dtype) * count;
        sycl::event copy_event = sycl_queue.memcpy(
            recv_buf, out_buf.get_ptr(), bytes, std::vector<sycl::event>{ sync_event });

        sycl::event ack_event = post_host_task_ack(sycl_queue,
                                                   std::vector<sycl::event>{ copy_event },
                                                   comm,
                                                   /*do_send=*/true,
                                                   node_peer_rank,
                                                   tag_done);

        ret_evt = ccl::event::create_from_native(ack_event);
        LOG_DEBUG("recv_sycl_single_node: ack_report done, ack_tag=", tag_done);
    }
    // Write mode => we just do ack-accept from the sender
    else {
        LOG_DEBUG("recv_sycl_single_node: write mode enabled");

        // in write mode we have to send the readiness confirmation to sender
        sycl::event sync_event = pt2pt_pre_sync(sycl_queue,
                                                sycl_deps,
                                                comm,
                                                /*do_send=*/true,
                                                node_peer_rank,
                                                tag_ready);

        sycl::event ack_event = post_host_task_ack(sycl_queue,
                                                   std::vector<sycl::event>{ sync_event },
                                                   comm,
                                                   /*do_send=*/false,
                                                   node_peer_rank,
                                                   tag_done);

        ret_evt = ccl::event::create_from_native(ack_event);

        LOG_DEBUG("recv_sycl_single_node: ack_accept done, ack_tag=", tag_done);
    }

    done = true;
    return ret_evt;
}

static ccl::event recv_mt_sycl_single_node(sycl::queue& q,
                                           void* recv_buf,
                                           size_t recv_count,
                                           ccl::datatype dtype,
                                           int peer_rank,
                                           ccl_comm* comm,
                                           ccl_stream* global_stream,
                                           const pt2pt_attr& /*attr*/,
                                           const std::vector<ccl::event>& deps,
                                           bool& done) {
    done = false;
    auto node_comm = comm->get_node_comm();
    auto& g_shared_res = *ccl::global_data::get().shared_data;

    // get the operation ID from the shared handshake
    int op_id = g_shared_res.get_shared_op_id(comm->global_current_id, false);

    // publish our pointer in hash_table
    g_shared_res.do_ipc_exchangeExt(comm,
                                    g_shared_res.hash_table,
                                    global_stream,
                                    { recv_buf },
                                    comm->global_current_id,
                                    true /* is_pt2pt */
    );

    // produce a device event that signals "my dependencies are done, my buffer is ready"
    auto sycl_deps = get_sycl_events(deps);
    sycl::event recv_ready_event = submit_wait_on_events(q, sycl_deps);

    // store that in shared_resources so the sender can wait on it
    g_shared_res.set_receiver_ready_event(op_id, recv_ready_event);

    // wait for the host handshake (a host-level signal that the copy is done)
    {
        auto& handshake = g_shared_res.handshakes[op_id];
        // We mark that we have published our pointer
        {
            std::lock_guard<std::mutex> lk(handshake.m);
            handshake.recv_published = true;
            handshake.cv.notify_one();
        }

        // Wait until the sender says "copy_done" (a host-level guarantee)
        std::unique_lock<std::mutex> lk(handshake.m);
        while (!handshake.copy_done) {
            handshake.cv.wait(lk);
        }
        handshake.copy_done = false; // reset
    }

    // combine the final copy_event from the sender with our local deps
    sycl::event sender_copy_event = g_shared_res.copy_event;
    sycl_deps = get_sycl_events(deps);
    sycl_deps.push_back(sender_copy_event);
    sycl::event ret_sycl_event = submit_wait_on_events(q, sycl_deps);

    done = true;
    return ccl::event::create_from_native(ret_sycl_event);
}

ccl::event recv_sycl(sycl::queue& q,
                     void* recv_buf,
                     size_t count,
                     ccl::datatype dtype,
                     int peer_rank,
                     ccl_comm* comm,
                     ccl_stream* global_stream,
                     const pt2pt_attr& attr,
                     const vector_class<event>& deps,
                     bool& done) {
    bool is_single_node = false;
    bool is_single_card = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
        is_single_card = topo_manager.is_single_card;
    }

    if (is_single_card) {
        LOG_DEBUG("recv_sycl: read mode enabled: tiles on the same card are detected");
        ccl::global_data::env().sycl_pt2pt_read = 1;
    }

    if (is_single_node) {
        if (comm->is_multi_thread_instance() == true) {
            return recv_mt_sycl_single_node(
                q, recv_buf, count, dtype, peer_rank, comm, global_stream, attr, deps, done);
        }
        else {
            return recv_sycl_single_node(
                q, recv_buf, count, dtype, peer_rank, comm, global_stream, attr, deps, done);
        }
    }
    else {
        done = false;
        CCL_THROW("SYCL recv is not supported for multi-node case");
        return ccl::event();
    }
}
} // namespace v1
} // namespace ccl

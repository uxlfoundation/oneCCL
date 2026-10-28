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

#include "coll/coll_util.hpp"
#include "coll/group/group.hpp"
#include <chrono>
#include <map>
#include <thread>
#include <tuple>
#include "common/global/global.hpp"
#include "comm/comm.hpp"
#include "common/utils/utils.hpp"
#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#include "coll/algorithms/utils/barrier_data.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL
#include "coll/algorithms/utils/host_task.hpp"

thread_local bool group_impl::first_group_op = false;
thread_local std::vector<group_impl::group_operation> group_impl::operation_storage;
thread_local std::vector<std::function<bool(atl_req_t&, bool, bool)>>
    group_impl::post_processing_steps;
thread_local size_t group_impl::group_counter = 0;

#ifdef CCL_ENABLE_SYCL
thread_local sycl::queue group_impl::sycl_queue;

thread_local std::vector<group_impl::GroupSendRequest> group_impl::all_group_send_requests;
thread_local std::vector<group_impl::GroupRecvRequest> group_impl::all_group_recv_requests;
#endif // CCL_ENABLE_SYCL
std::mutex group_impl::group_mutex;

#ifdef CCL_ENABLE_SYCL
static void free_in_graph_callback(sycl::queue &q, void *mem) {
#if defined(__INTEL_LLVM_COMPILER) && (__INTEL_LLVM_COMPILER >= 20260200)
     q.ext_oneapi_get_graph().set_destruction_callback([mem](){
         free(mem);
     });
#else
#if defined(__INTEL_LLVM_COMPILER)
    LOG_WARN_ONCE("using sycl graph with group API with compiler [", __INTEL_LLVM_COMPILER,
                  "], which does not support destruction callback; memory leak may occur");
#else
    LOG_WARN_ONCE("using sycl graph with group API with unknown compiler, ",
                  "destruction callback might be unsupported; memory leak may occur");
#endif
#endif
}
#endif

void group_impl::start() {
    if (group_counter++ > 0) {
        LOG_DEBUG("nested group start, ignoring");
        return;
    }
    LOG_DEBUG("group operation is started");
    operation_storage.clear();
#ifdef CCL_ENABLE_SYCL
    all_group_send_requests.clear();
    all_group_recv_requests.clear();
#endif // CCL_ENABLE_SYCL

    // if shared_data is not initialized, assume non-multi-thread instance
    // in case of multithread instance, pt2pt fallback table change might cause the crash
    // as it is not protected for such scenario. As long as the fallback also does not make
    // any sense for the multithread scenario, we just disable it completely
    if (ccl::global_data::get().shared_data) {
        auto& g = *ccl::global_data::get().shared_data;
        if (!g.is_multi_thread_instance) {
            ccl::enable_direct_fallback_for_pt2pt();
        }
    }
    else {
        ccl::enable_direct_fallback_for_pt2pt();
    }
}

// same on all ranks of a comm and unique per comm, so all ranks sort alike
using comm_order_key = std::tuple<int, uint64_t, std::vector<int>>;

static comm_order_key make_comm_order_key(ccl_comm* comm) {
    std::vector<int> ranks(comm->size());
    for (int r = 0; r < comm->size(); ++r) {
        ranks[r] = comm->get_global_rank(r);
    }
    return { comm->get_comm_id(), comm->unique_id(), std::move(ranks) };
}

static bool is_scale_up_pt2pt(const group_impl::group_operation& op) {
#ifdef CCL_ENABLE_SYCL
    return (op.ctype == ccl_coll_send || op.ctype == ccl_coll_recv) && op.stream && op.comm &&
           group_impl::use_group_scale_up(op.comm, op.peer_rank, op.stream);
#else
    return false;
#endif // CCL_ENABLE_SYCL
}

// collectives keep the user order, only scale-up pt2pt is moved to the tail and sorted
static void sort_group_operations() {
    auto& ops = group_impl::operation_storage;
    if (ops.size() < 2) {
        return;
    }
    auto tail =
        std::stable_partition(ops.begin(), ops.end(), [](const group_impl::group_operation& op) {
            return !is_scale_up_pt2pt(op);
        });
    if (tail == ops.end()) {
        return;
    }
    std::map<ccl_comm*, comm_order_key> keys;
    auto key_of = [&keys](ccl_comm* comm) -> const comm_order_key& {
        auto it = keys.find(comm);
        if (it == keys.end()) {
            it = keys.emplace(comm, make_comm_order_key(comm)).first;
        }
        return it->second;
    };
    auto order_of = [](const group_impl::group_operation& op) {
        int src = (op.ctype == ccl_coll_send) ? op.comm->rank() : op.peer_rank;
        int dst = (op.ctype == ccl_coll_send) ? op.peer_rank : op.comm->rank();
        return (src <= dst) ? 0 : 1;
    };
    std::stable_sort(
        tail,
        ops.end(),
        [&](const group_impl::group_operation& a, const group_impl::group_operation& b) {
            const comm_order_key& ka = key_of(a.comm);
            const comm_order_key& kb = key_of(b.comm);
            if (ka != kb) {
                return ka < kb;
            }
            return order_of(a) < order_of(b);
        });
}

void group_impl::end() {
    if (group_counter == 0) {
        CCL_THROW("group_end without group start or double group end detected");
    }
    if (group_counter > 1) {
        --group_counter;
        LOG_DEBUG("nested group end, ignoring");
        return;
    }
    // group_counter == 1
    LOG_DEBUG("group end is started");
    bool is_multi_thread_instance = true;
    if (ccl::global_data::get().shared_data) {
        auto& g = *ccl::global_data::get().shared_data;
        if (!g.is_multi_thread_instance) {
            is_multi_thread_instance = false;
        }
    }
    else {
        is_multi_thread_instance = false;
    }

#ifdef CCL_ENABLE_SYCL
    auto store_ze_pt2pt_read = ccl::global_data::env().ze_pt2pt_read;
    auto store_sycl_pt2pt_read = ccl::global_data::env().sycl_pt2pt_read;
    ccl::global_data::env().ze_pt2pt_read = 1;
    ccl::global_data::env().sycl_pt2pt_read = 1;
#endif

    first_group_op = true;

    sort_group_operations();

    if (is_multi_thread_instance) {
        // === Phase 1: buffer discovery (all recvs register in multi_thread case) ===
        LOG_DEBUG("group: Phase 1 - Buffer discovery phase");
        for (const auto& op : operation_storage) {
            if (op.ctype == ccl_coll_recv) {
                (void)op.fn();
                first_group_op = false;
            }
        }
    }

    if (is_multi_thread_instance) {
        // === Phase 2: execution (all operations) ===
        LOG_DEBUG("group: Phase 2 - Execution phase");
    }
    ccl::event event;
#ifdef CCL_ENABLE_SYCL
    // one comm can use two queues, keep its pt2pt in the sorted order
    std::map<ccl_comm*, sycl::queue> last_pt2pt_queue;
#endif // CCL_ENABLE_SYCL
    for (const auto& operation : operation_storage) {
#ifdef CCL_ENABLE_SYCL
        if ((operation.ctype == ccl_coll_send || operation.ctype == ccl_coll_recv) &&
            operation.stream && operation.comm &&
            use_group_scale_up(operation.comm, operation.peer_rank, operation.stream)) {
            auto q = operation.stream->get_native_stream();
            auto it = last_pt2pt_queue.find(operation.comm);
            if (it != last_pt2pt_queue.end() && it->second != q) {
                q.ext_oneapi_submit_barrier({ it->second.ext_oneapi_submit_barrier() });
            }
            last_pt2pt_queue.insert_or_assign(operation.comm, q);
        }
#endif // CCL_ENABLE_SYCL
        event = operation.fn();
        first_group_op = false;
    }
    first_group_op = false; // needed in case operation_storage is empty
    // wait() is needed to avoid oneCCL destruction prior to device tasks completion
    // wait() can be remove when finalize() is implemented for oneCCL. At that point
    // we need ensure that group execution is not being overlapped between groups
    if (post_processing_steps.size() > 0) {
        LOG_DEBUG("there are post processing steps");
        event.wait();
    }
    if (is_multi_thread_instance) {
        LOG_DEBUG("group: Phase 2 completed");
    }

#ifdef CCL_ENABLE_SYCL
    // Process batched scale-out send/recv requests
    process_group_pt2pt_scale_out_requests();
#endif // CCL_ENABLE_SYCL

    // === Post-processing steps ===
    auto post_processing_array = post_processing_steps;
    if (post_processing_array.size()) {
        LOG_DEBUG("there are post processing steps");
#ifdef CCL_ENABLE_SYCL
        sycl_queue
            .submit([=](sycl::handler& h) {
                h.host_task([=]() {
#endif
                    std::vector<atl_req_t> reqs(post_processing_array.size());
                    bool init = true;
                    while (true) {
                        bool all_done = true;
                        for (size_t i = 0; i < post_processing_array.size(); i++) {
                            if (!post_processing_array[i](reqs[i], false, init)) {
                                all_done = false;
                            }
                        }
                        init = false;
                        if (all_done)
                            break;
                    }
#ifdef CCL_ENABLE_SYCL
                });
            })
            .wait();
#endif
    }

#ifdef CCL_ENABLE_SYCL
    ccl::global_data::env().ze_pt2pt_read = store_ze_pt2pt_read;
    ccl::global_data::env().sycl_pt2pt_read = store_sycl_pt2pt_read;
#endif

    if (!is_multi_thread_instance) {
        ccl::restore_pt2pt_fallback_table();
    }
    LOG_DEBUG("group operation is ended");
    operation_storage.clear();
    post_processing_steps.clear();
    group_counter = 0;
}

void group_impl::add_operation(ccl_coll_type ctype,
                               ccl_comm* comm,
                               int peer_rank,
                               ccl_stream* stream,
                               std::function<ccl::event()> operation) {
    if (is_group_active()) {
        operation_storage.push_back({ ctype, comm, peer_rank, stream, std::move(operation) });
    }
    else {
        CCL_THROW("group API is not active");
    }
}

const std::vector<ccl::event>& group_impl::safe_deps(const std::vector<ccl::event>& deps) {
    static const std::vector<ccl::event> empty_deps;
    return is_group_active() ? empty_deps : deps;
}

#ifdef CCL_ENABLE_SYCL
bool group_impl::use_group_scale_up(ccl_comm* comm, int peer_rank, ccl_stream* stream) {
    // selected from data that both ranks have, so they select the same path
    return !comm->is_multi_thread_instance() && comm->size() == 2 && peer_rank != comm->rank() &&
           ccl::ze::is_arc_card(ccl::ze::get_device_family(stream->get_ze_device())) &&
           !use_recording_path(stream);
}
#endif // CCL_ENABLE_SYCL

bool group_impl::is_group_active() {
    return group_counter > 0;
}

void group_impl::add_post_processing_step(std::function<bool(atl_req_t&, bool, bool)> step) {
    if (is_group_active()) {
        post_processing_steps.push_back(std::move(step));
    }
    else {
        CCL_THROW("group API is not active");
    }
}

#ifdef CCL_ENABLE_SYCL
void group_impl::set_sycl_queue(sycl::queue q) {
    if (is_group_active()) {
        sycl_queue = q;
    }
    else {
        CCL_THROW("group API is not active");
    }
}

void group_impl::add_group_send_request(std::shared_ptr<ccl_internal_comm::group_send_request> req,
                                        sycl::queue q, int tag) {
    all_group_send_requests.push_back(GroupSendRequest { req, q, tag });
}

void group_impl::add_group_recv_request(std::shared_ptr<ccl_internal_comm::group_recv_request> req,
                                        sycl::queue q, int tag) {
    all_group_recv_requests.push_back(GroupRecvRequest { req, q, tag });
}

void group_impl::process_group_pt2pt_scale_out_requests() {
    if (!all_group_send_requests.empty() || !all_group_recv_requests.empty()) {
        std::vector<sycl::queue> used_queues;
        size_t total_requests = all_group_send_requests.size() + all_group_recv_requests.size();
        used_queues.reserve(total_requests);
        bool all_recording = true;
        bool all_not_recording = true;
        bool is_recording {};

        // the code below has O(n2) complexity
        // sycl::queue does not support the "<" operator
        // so std::set cannot be used
        // TODO implement more efficient way 
        for (auto &op : all_group_send_requests) {
            ccl::utils::vec_insert_uniq(used_queues, op.q);
        }
        for (auto &op : all_group_recv_requests) {
            ccl::utils::vec_insert_uniq(used_queues, op.q);
        }
        used_queues.shrink_to_fit();

        for (auto &used_q : used_queues) {
            bool t_is_recording = use_recording_path(used_q);
            if (t_is_recording) {
                all_not_recording = false;
            } else {
                all_recording = false;
            }
        }
        if (all_recording && all_not_recording) {
            LOG_DEBUG("empty scaleout group API submitted");
            // the whole should default to no-op
            is_recording = true; // this will skip all the waits
        } else if (all_recording && !all_not_recording) {
            is_recording = true;
        } else if (!all_recording && all_not_recording) {
            is_recording = false;
        } else if (!all_recording && !all_not_recording) {
            CCL_THROW("scenario not supported: scaleout group API with some requests recording, others non-recording");
        } else {
            CCL_THROW("internal error, this should be dead code");
        }

        LOG_DEBUG("send_requests: ", all_group_send_requests.size(),
                  ", recv_requests: ", all_group_recv_requests.size(),
                  ", unique queues: ", used_queues.size());

        // Step 0: pre-sync all queues
        // Step 1: start all send requests
        // Step 2: start all recv request
        // Step 3: wait for all send requests
        // Step 4: wait for all recv requests
        // Step 5: if not RDMA, add copy-out
        // Step 6: if not RDMA, free recv staging buffers
        // Step 7: post-sync all queues
        // Step 8: add destruction callbacks

        // Step 0: pre-sync all queues
        auto begin_it = used_queues.begin();
        sycl::queue main_q = *begin_it;
        (void)used_queues.erase(begin_it);

        std::vector<sycl::event> sync_events;
        sync_events.reserve(sync_events.size());
        for (sycl::queue used_queue : used_queues) {
            sync_events.push_back(ccl::utils::submit_barrier(used_queue));
        }
        LOG_TRACE("number of sync events: ", sync_events.size());
        if (sync_events.size() > 0) {
            ccl::utils::submit_barrier(main_q, sync_events);
        }

        auto all_group_send_requests_cpy = all_group_send_requests;
        auto all_group_recv_requests_cpy = all_group_recv_requests;

        ccl_host_task(main_q, {}, [all_group_send_requests_cpy, all_group_recv_requests_cpy, is_recording]() {
            // Step 3: Wait for all send requests
            for (auto &send_request : all_group_send_requests_cpy) {
                auto send_req_ptr = send_request.req;
                int tag = send_request.tag;
                LOG_TRACE("processing send request: ", send_req_ptr, "tag: ", tag);
                 auto atl_comm = send_req_ptr->comm->get_atl_comm();
                 ATL_CALL_THROW_IF_ERROR(atl_comm->wait(0, send_req_ptr->atl_send_req));

                // free send host buffer
                if (!send_req_ptr->use_rdma && !is_recording) {
                    free(send_req_ptr->send_host_buf);
                }
                LOG_TRACE("processed send request: ", send_req_ptr, "tag: ", tag);
            }
 
            // Step 4: Wait for all recv requests and copy data
            for (auto& recv_request : all_group_recv_requests_cpy) {
                auto recv_req_ptr = recv_request.req;
                int tag = recv_request.tag;
                 LOG_TRACE("processing recv request: ", recv_req_ptr, "tag: ", tag);
                 auto atl_comm = recv_req_ptr->comm->get_atl_comm();
                 ATL_CALL_THROW_IF_ERROR(atl_comm->wait(0, recv_req_ptr->atl_recv_req));
                 LOG_TRACE("processed recv request: ", recv_req_ptr, "tag: ", tag);
            }
        });

        // Step 5: if not RDMA, add copy-out
        for (auto& recv_request : all_group_recv_requests) {
            auto recv_req_ptr = recv_request.req;
            int tag = recv_request.tag;
 
            if (!recv_req_ptr->use_rdma) {
                if (recv_req_ptr->recv_size > 0) {
                    // calling this with 0-size caused hangs in graph tests
                    main_q.submit([recv_req_ptr](sycl::handler& h) {
                         h.memcpy(recv_req_ptr->recv_user_buf,
                                  recv_req_ptr->recv_host_buf,
                                  recv_req_ptr->recv_size);
                    });
                }
            }
        }

        // Step 6: if not RDMA, free recv staging buffers
        if (!is_recording) {
            ccl_host_task(main_q, {}, [all_group_recv_requests_cpy]() {
                for (auto& recv_request : all_group_recv_requests_cpy) {
                    auto recv_req_ptr = recv_request.req;
                    int tag = recv_request.tag;

                    if (!recv_req_ptr->use_rdma) {
                         free(recv_req_ptr->recv_host_buf);
                     }
                }
            });
        }

        // Step 7: post-sync all queues
        if (used_queues.size() > 0) {
            sycl::event e = ccl::utils::submit_barrier(main_q);
            for (auto &used_queue : used_queues) {
                ccl::utils::submit_barrier(used_queue, e);
            }
        }

        // Step 8: add destruction callbacks
        if (is_recording) {
            for (auto& recv_request : all_group_recv_requests) {
                auto recv_req_ptr = recv_request.req;
                int tag = recv_request.tag;

                if (!recv_req_ptr->use_rdma) {
                    free_in_graph_callback(recv_request.q, recv_req_ptr->recv_host_buf);
                }
            }
            for (auto &send_request : all_group_send_requests) {
                auto send_req_ptr = send_request.req;
                if (!send_req_ptr->use_rdma) {
                    free_in_graph_callback(send_request.q, send_req_ptr->send_host_buf);
                 }
            }
        }

        // Clear the request vectors after processing
        all_group_send_requests.clear();
        all_group_recv_requests.clear();
    }
}
#endif // CCL_ENABLE_SYCL

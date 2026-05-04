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
#include "coll/coll_util.hpp"
#include "coll/group/group.hpp"
#include "common/global/global.hpp"
#include "comm/comm.hpp"

thread_local bool group_impl::is_group_active = false;
thread_local bool group_impl::first_group_op = false;
thread_local std::vector<std::pair<ccl_coll_type, std::function<ccl::event()>>>
    group_impl::operation_storage;
thread_local std::vector<std::function<bool(atl_req_t&, bool, bool)>>
    group_impl::post_processing_steps;
#ifdef CCL_ENABLE_SYCL
thread_local sycl::queue group_impl::sycl_queue;
thread_local std::vector<std::shared_ptr<ccl_internal_comm::group_send_request>>
    group_impl::all_group_send_requests;
thread_local std::vector<std::shared_ptr<ccl_internal_comm::group_recv_request>>
    group_impl::all_group_recv_requests;
#endif // CCL_ENABLE_SYCL
std::mutex group_impl::group_mutex;

void group_impl::start() {
    LOG_DEBUG("group operation is started");
    operation_storage.clear();
#ifdef CCL_ENABLE_SYCL
    all_group_send_requests.clear();
    all_group_recv_requests.clear();
#endif // CCL_ENABLE_SYCL
    is_group_active = true;

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

void group_impl::end() {
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

    if (is_multi_thread_instance) {
        // === Phase 1: buffer discovery (all recvs register in multi_thread case) ===
        LOG_DEBUG("group: Phase 1 - Buffer discovery phase");
        for (const auto& op : operation_storage) {
            if (op.first == ccl_coll_recv) {
                (void)op.second();
                first_group_op = false;
            }
        }
    }

    if (is_multi_thread_instance) {
        // === Phase 2: execution (all operations) ===
        LOG_DEBUG("group: Phase 2 - Execution phase");
    }
    ccl::event event;
    for (const auto& operation : operation_storage) {
        event = operation.second();
        first_group_op = false;
    }
    first_group_op = false; // needed in case operation_storage is empty
    // wait() is needed to avoid oneCCL destruction prior to device tasks completion
    // wait() can be remove when finalize() is implemented for oneCCL. At that point
    // we need ensure that group execution is not being overlapped between groups
    event.wait();
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
    is_group_active = false;
    operation_storage.clear();
    post_processing_steps.clear();
}

void group_impl::add_operation(ccl_coll_type ctype, std::function<ccl::event()> operation) {
    if (is_group_active) {
        operation_storage.push_back(std::make_pair(ctype, std::move(operation)));
    }
    else {
        CCL_THROW("group API is not active");
    }
}

void group_impl::add_post_processing_step(std::function<bool(atl_req_t&, bool, bool)> step) {
    if (is_group_active) {
        post_processing_steps.push_back(std::move(step));
    }
    else {
        CCL_THROW("group API is not active");
    }
}

#ifdef CCL_ENABLE_SYCL
void group_impl::set_sycl_queue(sycl::queue q) {
    if (is_group_active) {
        sycl_queue = q;
    }
    else {
        CCL_THROW("group API is not active");
    }
}

void group_impl::add_group_send_request(std::shared_ptr<ccl_internal_comm::group_send_request> req,
                                        bool is_group) {
    if (is_group) {
        all_group_send_requests.push_back(req);
    }
}

void group_impl::add_group_recv_request(std::shared_ptr<ccl_internal_comm::group_recv_request> req,
                                        bool is_group) {
    if (is_group) {
        all_group_recv_requests.push_back(req);
    }
}

void group_impl::process_group_pt2pt_scale_out_requests() {
    if (!all_group_send_requests.empty() || !all_group_recv_requests.empty()) {
        sycl::event e;

        // Copy the vectors to local variables so they can be captured
        auto local_send_requests = all_group_send_requests;
        auto local_recv_requests = all_group_recv_requests;

        // Step 1: Wait for all send requests
        e = sycl_queue.submit([local_send_requests](sycl::handler& h) mutable {
            h.host_task([local_send_requests]() mutable {
                for (auto& send_req_ptr : local_send_requests) {
                    int ep_idx = 0;
                    auto atl_comm = send_req_ptr->comm->get_atl_comm();
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, send_req_ptr->atl_send_req));
                }
            });
        });

        // Step 2: Wait for all recv requests and copy data
        for (auto& recv_req_ptr : local_recv_requests) {
            e = sycl_queue.submit([recv_req_ptr, e](sycl::handler& h) mutable {
                h.depends_on(e);
                h.host_task([recv_req_ptr]() mutable {
                    int ep_idx = 0;
                    auto atl_comm = recv_req_ptr->comm->get_atl_comm();
                    ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, recv_req_ptr->atl_recv_req));
                });
            });

            e = sycl_queue.submit([recv_req_ptr, e](sycl::handler& h) {
                h.depends_on(e);
                h.memcpy(recv_req_ptr->recv_user_buf,
                         recv_req_ptr->recv_host_buf,
                         recv_req_ptr->recv_size);
            });
        }

        // Step 3: Free all host buffers
        e = sycl_queue.submit([local_send_requests, local_recv_requests, e](sycl::handler& h) {
            h.depends_on(e);
            h.host_task([local_send_requests, local_recv_requests]() {
                for (const auto& send_req_ptr : local_send_requests) {
                    free(send_req_ptr->send_host_buf);
                }

                for (const auto& recv_req_ptr : local_recv_requests) {
                    free(recv_req_ptr->recv_host_buf);
                }
            });
        });

        e.wait();

        // Clear the request vectors after processing
        all_group_send_requests.clear();
        all_group_recv_requests.clear();
    }
}
#endif // CCL_ENABLE_SYCL

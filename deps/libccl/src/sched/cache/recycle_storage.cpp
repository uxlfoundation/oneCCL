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

#include "common/global/global.hpp"
#include "common/request/request.hpp"
#include "exec/exec.hpp"
#include "sched/cache/recycle_storage.hpp"

namespace ccl {

void recycle_storage::recycle_events(size_t threshold, size_t limit) {
#ifdef CCL_ENABLE_SYCL
    std::lock_guard<std::mutex> lg(lock_events);
    if (output_sycl_events.size() > critical_threshold) {
        limit = 0;
    }
    if (output_sycl_events.size() > threshold) {
        LOG_DEBUG("recycle storage has more than ",
                  threshold,
                  " stored events (",
                  output_sycl_events.size(),
                  "), try to recycle");
        std::list<std::list<ze::dynamic_event_pool*>::iterator> ze_pools_to_remove;
        std::list<std::list<std::shared_ptr<sycl::event>>::iterator> sync_events_to_remove;
        std::list<std::list<std::shared_ptr<sycl::event>>::iterator> sycl_output_events_to_remove;
        auto ze_pool_it = ze_pools.begin();
        auto sync_event_it = sync_sycl_events.begin();
        size_t recycling_counter = 0;
        for (auto sycl_output_event_it = output_sycl_events.begin();
             sycl_output_event_it != output_sycl_events.end();
             sycl_output_event_it++) {
            auto sycl_output_event = *sycl_output_event_it;
            auto sync_event = *sync_event_it;
            auto pool = *ze_pool_it;
            if (sycl_output_event &&
                sycl_output_event->get_info<sycl::info::event::command_execution_status>() ==
                    sycl::info::event_command_status::complete) {
                try {
                    CCL_THROW_IF_NOT(
                        sync_event &&
                        sync_event->get_info<sycl::info::event::command_execution_status>() ==
                            sycl::info::event_command_status::complete);
                    pool->put_event(ccl::utils::get_native_event(*sync_event));
                }
                // runtime_error is __SYCL2020_DEPRECATED, catch generic exception
                catch (sycl::exception& e) {
                    LOG_ERROR(
                        "sycl event not recovered: ", e.what(), " potential resource/memory leak");
                }
                catch (ccl::exception& e) {
                    LOG_ERROR("sycl event not recovered: ", e.what());
                }
                catch (...) {
                    LOG_ERROR("sycl event not recovered: unknown exception");
                }
                sycl_output_events_to_remove.push_back(sycl_output_event_it);
                sync_events_to_remove.push_back(sync_event_it);
                ze_pools_to_remove.push_back(ze_pool_it);
            }
            ze_pool_it++;
            sync_event_it++;
            recycling_counter++;
            if (limit && ++recycling_counter == limit) {
                break;
            }
        }
        for (auto& sycl_output_event : sycl_output_events_to_remove) {
            output_sycl_events.erase(sycl_output_event);
        }
        for (auto& sync_event : sync_events_to_remove) {
            sync_sycl_events.erase(sync_event);
        }
        for (auto& ze_pool : ze_pools_to_remove) {
            ze_pools.erase(ze_pool);
        }
    }
#endif // CCL_ENABLE_SYCL
}

void recycle_storage::recycle_requests(size_t threshold, size_t limit) {
#ifdef CCL_ENABLE_SYCL
    std::lock_guard<std::mutex> lg(lock_requests);
    if (requests.size() > threshold) {
        LOG_DEBUG("recycle storage has more than ",
                  threshold,
                  " stored requests (",
                  requests.size(),
                  "), try to recycle");
        std::list<std::list<ccl_request*>::iterator> requests_to_remove;
        size_t recycling_counter = 0;
        for (auto request_it = requests.begin(); request_it != requests.end(); request_it++) {
            auto request = *request_it;
            if (request->is_completed()) {
                requests_to_remove.push_back(request_it);
                ccl_release_request(request);
            }
            if (limit && ++recycling_counter == limit) {
                break;
            }
        }
        for (auto& request : requests_to_remove) {
            requests.erase(request);
        }
    }
#endif // CCL_ENABLE_SYCL
}

#ifdef CCL_ENABLE_SYCL
// Recycle storage respects storage order for ze_pools & related events
void recycle_storage::store_events(ze::dynamic_event_pool* pool,
                                   const std::shared_ptr<sycl::event>& sync_event,
                                   const std::shared_ptr<sycl::event>& output_event) {
    std::lock_guard<std::mutex> lg(lock_events);
    ze_pools.push_back(pool);
    sync_sycl_events.push_back(sync_event);
    output_sycl_events.push_back(output_event);
}

void recycle_storage::store_request(ccl_request* request) {
    std::lock_guard<std::mutex> lg(lock_requests);
    requests.push_back(request);
}
#endif // CCL_ENABLE_SYCL

} // namespace ccl

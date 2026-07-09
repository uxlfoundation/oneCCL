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

#pragma once

#include <list>

#ifdef CCL_ENABLE_SYCL
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

class ccl_request;

namespace ccl {

class recycle_storage {
public:
    recycle_storage() = default;
    ~recycle_storage() = default;
    recycle_storage(const recycle_storage&) = delete;
    recycle_storage& operator=(const recycle_storage&) = delete;

    void recycle_events(size_t threshold = 0, size_t limit = 0);
    void recycle_requests(size_t threshold = 0, size_t limit = 0);

#ifdef CCL_ENABLE_SYCL
    void store_events(ze::dynamic_event_pool* pool,
                      const std::shared_ptr<sycl::event>& sync_event,
                      const std::shared_ptr<sycl::event>& output_event);
    void store_request(ccl_request* request);
#endif // CCL_ENABLE_SYCL

private:
#ifdef CCL_ENABLE_SYCL
    std::list<ze::dynamic_event_pool*> ze_pools;
    std::list<std::shared_ptr<sycl::event>> sync_sycl_events;
    std::list<std::shared_ptr<sycl::event>> output_sycl_events;
    std::list<ccl_request*> requests;
    const size_t critical_threshold = 1000;
    std::mutex lock_events;
    std::mutex lock_requests;
#endif // CCL_ENABLE_SYCL
};

} // namespace ccl

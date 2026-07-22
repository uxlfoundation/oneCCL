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

#include "coll/algorithms/utils/sycl_custom_preop.hpp"

#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/utils/sycl_reductions.hpp"

ccl::event stage_and_apply_custom_preop(sycl::queue& q,
                                        void* dst,
                                        const void* src,
                                        size_t count,
                                        ccl::datatype dtype,
                                        ccl::reduction reduction,
                                        size_t tmp_buf_size,
                                        const std::vector<sycl::event>& dep_events) {
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

    sycl::event sycl_e;
    if ((src != nullptr) && (src != dst)) {
        sycl_e = q.submit([=](sycl::handler& h) {
            h.depends_on(dep_events);
            h.memcpy(dst, src, count * ccl_dtype.size());
        });
    }
    else {
        sycl_e = submit_wait_on_events(q, dep_events);
    }

    ccl_reduction_data reduction_op = make_reduction_operation(reduction);
    auto lambda = [&]<typename T>() {
        sycl_e = pre_operation_invoke<T, 1, 32>(q,
                                                dst,
                                                count,
                                                false,
                                                nullptr,
                                                reduction_op,
                                                tmp_buf_size,
                                                { sycl_e });
        return ccl::event::create_from_native(sycl_e);
    };
    return invoke_collective(lambda, dtype);
}

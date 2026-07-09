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

#include "comp/comp.hpp"
#include "common/datatype/datatype.hpp"
#include "common/stream/stream.hpp"
#include "sched/entry/reduce_local_entry.hpp"
#include "sched/queue/queue.hpp"

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "common/utils/sycl_utils.hpp"
#include "sched/entry/ze/ze_reduce_local_entry.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

using namespace ccl;

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)

using namespace ccl::ze;

void reduce_local_entry::check_use_device() {
    use_device = false;
    ccl_stream* stream = sched->coll_param.stream;
    if (fn || !stream) {
        return;
    }

    size_t bytes = in_cnt * dtype.size();
    auto sycl_stream = stream->get_native_stream();
    auto in_ptr_type = sycl::get_pointer_type(in_buf.get_ptr(bytes), sycl_stream.get_context());
    auto inout_ptr_type =
        sycl::get_pointer_type(inout_buf.get_ptr(bytes), sycl_stream.get_context());

    LOG_DEBUG("in_ptr_type: ",
              ccl::utils::usm_type_to_str(in_ptr_type),
              ", inout_ptr_type: ",
              ccl::utils::usm_type_to_str(inout_ptr_type),
              ", native_stream: ",
              stream->to_string(),
              ", in_count: ",
              in_cnt)

    if ((in_ptr_type == sycl::usm::alloc::device) && (inout_ptr_type == sycl::usm::alloc::device)) {
        use_device = true;
    }
}

void reduce_local_entry::start_on_device() {
    ze_reduce_local->start();
    status = ze_reduce_local->get_status();
}

#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

reduce_local_entry::reduce_local_entry(ccl_sched* sched,
                                       const ccl_buffer in_buf,
                                       size_t in_cnt,
                                       ccl_buffer inout_buf,
                                       size_t* out_cnt,
                                       const ccl_datatype& dtype,
                                       ccl::reduction op)
        : sched_entry(sched),
          in_buf(in_buf),
          in_cnt(in_cnt),
          inout_buf(inout_buf),
          out_cnt(out_cnt),
          dtype(dtype),
          op(op),
          fn(sched->coll_attr.reduction_fn) {
    CCL_THROW_IF_NOT(op != ccl::reduction::custom || fn,
                     "custom reduction requires user provided callback",
                     ", op ",
                     ccl_reduction_to_str(op),
                     ", fn ",
                     fn);

    check_use_device();
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (use_device) {
        ze_reduce_local = std::make_unique<ze_reduce_local_entry>(
            sched, in_buf, in_cnt, inout_buf, out_cnt, dtype, op);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

void reduce_local_entry::start_on_host() {
    size_t bytes = in_cnt * dtype.size();
    size_t offset = inout_buf.get_offset();
    const fn_context context = { sched->coll_attr.match_id.c_str(), offset };
    ccl::status comp_status = ccl_comp_reduce(sched,
                                              in_buf.get_ptr(bytes),
                                              in_cnt,
                                              inout_buf.get_ptr(bytes),
                                              const_cast<size_t*>(out_cnt),
                                              dtype,
                                              op,
                                              fn,
                                              &context);
    CCL_ASSERT(comp_status == ccl::status::success, "bad status ", comp_status);

    status = ccl_sched_entry_status_complete;
}

void reduce_local_entry::start() {
    if (use_device) {
        LOG_DEBUG("start on device");
        start_on_device();
    }
    else {
        LOG_DEBUG("start on host");
        start_on_host();
    }
}

void reduce_local_entry::update() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (use_device) {
        ze_reduce_local->update();
        status = ze_reduce_local->get_status();
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    // status will be set after start_on_host()
}

void reduce_local_entry::reset(size_t idx) {
    sched_entry::reset(idx);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ze_reduce_local) {
        ze_reduce_local->reset(idx);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

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
#include "sched/entry/copy/copy_entry.hpp"
#include "sched/queue/queue.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#ifdef CCL_ENABLE_ZE
#include "sched/entry/ze/ze_copy_entry.hpp"
#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL

copy_entry::copy_entry(ccl_sched* sched,
                       ccl_buffer in_buf,
                       ccl_buffer out_buf,
                       size_t count,
                       const ccl_datatype& dtype,
                       copy_attr attr)
        : sched_entry(sched),
          in_buf(in_buf),
          out_buf(out_buf),
          count(count),
          dtype(dtype),
          attr(attr) {
    CCL_THROW_IF_NOT(sched, "no sched");

    LOG_DEBUG(class_name(), ": in_buf ", in_buf, ", out_buf ", out_buf, ", count ", count);

#ifdef CCL_ENABLE_SYCL
    is_sycl_buf = sched->coll_attr.is_sycl_buf;
    sycl::usm::alloc in_ptr_type = sycl::usm::alloc::unknown;
    sycl::usm::alloc out_ptr_type = sycl::usm::alloc::unknown;

    if (sched->coll_param.stream) {
        auto context = sched->coll_param.stream->get_native_stream().get_context();
        in_ptr_type = sycl::get_pointer_type(in_buf.get_ptr(), context);
        out_ptr_type = sycl::get_pointer_type(out_buf.get_ptr(), context);

        LOG_DEBUG("in_ptr_type: ",
                  ccl::utils::usm_type_to_str(in_ptr_type),
                  ", out_ptr_type: ",
                  ccl::utils::usm_type_to_str(out_ptr_type));

        if (attr.direction == copy_direction::undefined) {
            if (in_ptr_type == sycl::usm::alloc::device &&
                out_ptr_type == sycl::usm::alloc::device) {
                attr.direction = copy_direction::d2d;
            }

            if ((in_ptr_type != sycl::usm::alloc::device) &&
                (out_ptr_type != sycl::usm::alloc::device)) {
                attr.direction = copy_direction::h2h;
            }

            if ((in_ptr_type == sycl::usm::alloc::device) &&
                (out_ptr_type != sycl::usm::alloc::device)) {
                attr.direction = copy_direction::d2h;
            }

            if ((in_ptr_type != sycl::usm::alloc::device) &&
                (out_ptr_type == sycl::usm::alloc::device)) {
                attr.direction = copy_direction::h2d;
            }

            CCL_THROW_IF_NOT(attr.direction != copy_direction::undefined);
        }
    }
#endif // CCL_ENABLE_SYCL

    LOG_DEBUG("count: ", count, ", direction: ", to_string(attr.direction));

    if (!sched->coll_param.stream || (attr.direction == copy_direction::h2h)) {
#ifdef CCL_ENABLE_SYCL
        CCL_THROW_IF_NOT(in_ptr_type != sycl::usm::alloc::device,
                         "unexpected device usm type for input buffer");
        CCL_THROW_IF_NOT(out_ptr_type != sycl::usm::alloc::device,
                         "unexpected device usm type for output buffer");
#endif // CCL_ENABLE_SYCL
        ctype = copy_type::regular;
    }

#ifdef CCL_ENABLE_SYCL
    else if (sched->coll_param.stream->get_backend() != ccl::utils::get_level_zero_backend() ||
             is_sycl_buf) {
        ctype = copy_type::sycl;
        if (!is_sycl_buf) {
            if ((in_ptr_type != sycl::usm::alloc::device) &&
                (out_ptr_type != sycl::usm::alloc::device)) {
                ctype = copy_type::regular;
            }
        }
    }
#ifdef CCL_ENABLE_ZE
    else {
        ctype = copy_type::ze;
        ze_copier = std::make_unique<ze_copy_entry>(sched, in_buf, out_buf, count, dtype, attr);
    }
#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL
}

void copy_entry::start() {
#ifdef CCL_ENABLE_SYCL
    sycl::queue* sycl_queue{};
#endif // CCL_ENABLE_SYCL

    switch (ctype) {
        case copy_type::regular:
            do_regular_copy(); // will set complete status
            break;

#ifdef CCL_ENABLE_SYCL
        case copy_type::sycl:
            sycl_queue = sched->coll_param.stream->get_native_stream(sched->queue->get_idx());
            CCL_THROW_IF_NOT(sycl_queue, "null sycl queue");
            copier = sycl_copier(attr.direction,
                                 in_buf,
                                 out_buf,
                                 count,
                                 dtype,
                                 is_sycl_buf,
                                 attr.in_buf_offset,
                                 attr.out_buf_offset);
            copier.set_queue(sycl_queue);
            ccl_tuple_for_each_indexed<ccl_sycl_buffer_one_dim_types>(copier);
            status = ccl_sched_entry_status_started;
            break;

#ifdef CCL_ENABLE_ZE
        case copy_type::ze:
            ze_copier->start();
            status = ze_copier->get_status();
            break;
#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL

        default: CCL_THROW("unknown copy type"); break;
    }
}

void copy_entry::update() {
#ifdef CCL_ENABLE_SYCL
    if (ctype == copy_type::sycl) {
        if (copier.is_completed()) {
            status = ccl_sched_entry_status_complete;
        }
    }
#ifdef CCL_ENABLE_ZE
    else if (ctype == copy_type::ze) {
        ze_copier->update();
        status = ze_copier->get_status();
    }
#endif // CCL_ENABLE_ZE
#endif // CCL_ENABLE_SYCL

    // status for regular will be set in do_regular_copy()
}

void copy_entry::reset(size_t idx) {
    sched_entry::reset(idx);
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (ze_copier) {
        ze_copier->reset(idx);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

void copy_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str,
                       "dt ",
                       ccl::global_data::get().dtypes->name(dtype),
                       ", count ",
                       count,
                       ", in_buf ",
                       in_buf,
                       ", out_buf ",
                       out_buf,
                       ", in_buf_offset ",
                       attr.in_buf_offset,
                       ", out_buf_offset ",
                       attr.out_buf_offset,
                       ", direction ",
                       to_string(attr.direction),
                       "\n");
}

void copy_entry::do_regular_copy() {
    size_t bytes = dtype.size() * count;
    auto comp_status =
        ccl_comp_copy(in_buf.get_ptr(bytes), out_buf.get_ptr(bytes), bytes, attr.use_nontemporal);
    CCL_ASSERT(comp_status == ccl::status::success, "bad status ", comp_status);
    status = ccl_sched_entry_status_complete;
}

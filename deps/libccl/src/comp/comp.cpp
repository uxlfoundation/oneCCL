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

#include "comp/bf16/bf16.hpp"
#include "comp/comp.hpp"
#include "comp/fp16/fp16.hpp"
#include "common/log/log.hpp"
#include "common/global/global.hpp"
#include "common/utils/enums.hpp"
#include "common/utils/memcpy.hpp"
#include "oneapi/ccl/types.hpp"
#include "sched/queue/queue.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#include <sycl/sycl.hpp>
#endif // CCL_ENABLE_SYCL

#define CCL_REDUCE(type) \
    do { \
        type* in_buf_##type = (type*)in_buf; \
        type* inout_buf_##type = (type*)inout_buf; \
        switch (reduction) { \
            case ccl::reduction::sum: \
                for (i = 0; i < in_count; i++) { \
                    inout_buf_##type[i] += in_buf_##type[i]; \
                } \
                break; \
            case ccl::reduction::prod: \
                for (i = 0; i < in_count; i++) { \
                    inout_buf_##type[i] *= in_buf_##type[i]; \
                } \
                break; \
            case ccl::reduction::min: \
                for (i = 0; i < in_count; i++) { \
                    inout_buf_##type[i] = std::min(in_buf_##type[i], inout_buf_##type[i]); \
                } \
                break; \
            case ccl::reduction::max: \
                for (i = 0; i < in_count; i++) { \
                    inout_buf_##type[i] = std::max(in_buf_##type[i], inout_buf_##type[i]); \
                } \
                break; \
            default: CCL_FATAL("unexpected value ", ccl::utils::enum_to_underlying(reduction)); \
        } \
    } while (0)

ccl::status ccl_comp_copy(const void* in_buf, void* out_buf, size_t bytes, bool use_nontemporal) {
    if (bytes == 0) {
        return ccl::status::success;
    }

    CCL_ASSERT(in_buf, "in_buf is null");
    CCL_ASSERT(out_buf, "out_buf is null");
    if (use_nontemporal) {
        ccl::memcpy_nontemporal(out_buf, in_buf, bytes);
    }
    else {
        ccl::memcpy(out_buf, in_buf, bytes);
    }
    return ccl::status::success;
}

ccl::status ccl_comp_reduce_regular(const void* in_buf,
                                    size_t in_count,
                                    void* inout_buf,
                                    size_t* out_count,
                                    const ccl_datatype& dtype,
                                    ccl::reduction reduction,
                                    ccl::reduction_fn reduction_fn,
                                    const ccl::fn_context* context) {
    if (reduction == ccl::reduction::custom) {
        CCL_THROW_IF_NOT(reduction_fn, "custom reduction requires user callback");
        reduction_fn(in_buf, in_count, inout_buf, out_count, dtype.idx(), context);
        return ccl::status::success;
    }

#ifdef CCL_ENABLE_ITT
    __itt_event comp_reduce_itt_event = ccl::profile::itt::event_get("comp_reduce_regular");
    ccl::profile::itt::event_start(comp_reduce_itt_event);
#endif // CCL_ENABLE_ITT

    size_t i;
    switch (dtype.idx()) {
        case ccl::datatype::int8: CCL_REDUCE(int8_t); break;
        case ccl::datatype::uint8: CCL_REDUCE(uint8_t); break;
        case ccl::datatype::int16: CCL_REDUCE(int16_t); break;
        case ccl::datatype::uint16: CCL_REDUCE(uint16_t); break;
        case ccl::datatype::int32: CCL_REDUCE(int32_t); break;
        case ccl::datatype::uint32: CCL_REDUCE(uint32_t); break;
        case ccl::datatype::int64: CCL_REDUCE(int64_t); break;
        case ccl::datatype::uint64: CCL_REDUCE(uint64_t); break;
        case ccl::datatype::float16:
            ccl_fp16_reduce(in_buf, in_count, inout_buf, out_count, reduction);
            break;
        case ccl::datatype::float32: CCL_REDUCE(float); break;
        case ccl::datatype::float64: CCL_REDUCE(double); break;
        case ccl::datatype::bfloat16:
            ccl_bf16_reduce(in_buf, in_count, inout_buf, out_count, reduction);
            break;
        default: CCL_FATAL("unexpected value ", dtype.idx()); break;
    }

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::event_end(comp_reduce_itt_event);
#endif // CCL_ENABLE_ITT

    return ccl::status::success;
}

ccl::status ccl_comp_reduce(ccl_sched* sched,
                            const void* in_buf,
                            size_t in_count,
                            void* inout_buf,
                            size_t* out_count,
                            const ccl_datatype& dtype,
                            ccl::reduction reduction,
                            ccl::reduction_fn reduction_fn,
                            const ccl::fn_context* context) {
    if (!in_count) {
        return ccl::status::success;
    }

#ifdef CCL_ENABLE_SYCL
    ccl_stream* stream = (ccl_stream*)sched->coll_param.stream;

    if (!stream) {
        return ccl_comp_reduce_regular(
            in_buf, in_count, inout_buf, out_count, dtype, reduction, reduction_fn, context);
    }

    sycl::queue* q = stream->get_native_stream(sched->queue->get_idx());
    CCL_THROW_IF_NOT(q, "null sycl queue");
    auto in_ptr_type = sycl::get_pointer_type(in_buf, q->get_context());
    auto inout_ptr_type = sycl::get_pointer_type(inout_buf, q->get_context());

    LOG_DEBUG("in_ptr_type: ",
              ccl::utils::usm_type_to_str(in_ptr_type),
              ", inout_ptr_type: ",
              ccl::utils::usm_type_to_str(inout_ptr_type),
              ", native_stream: ",
              stream->to_string(),
              ", in_count: ",
              in_count)

    if ((in_ptr_type != sycl::usm::alloc::device) && (inout_ptr_type != sycl::usm::alloc::device)) {
        return ccl_comp_reduce_regular(
            in_buf, in_count, inout_buf, out_count, dtype, reduction, reduction_fn, context);
    }

    void* host_in_buf = (void*)in_buf;
    void* host_inout_buf = inout_buf;
    size_t bytes = in_count * dtype.size();

    ccl::alloc_param alloc_param(bytes, ccl::buffer_type::sycl, ccl::buffer_place::host, false);
    ccl::dealloc_param dealloc_param(nullptr, bytes, ccl::buffer_type::sycl);

    if (in_ptr_type == sycl::usm::alloc::device) {
        host_in_buf = sched->alloc_buffer(alloc_param).get_ptr();
        q->memcpy(host_in_buf, in_buf, bytes).wait();
    }

    if (inout_ptr_type == sycl::usm::alloc::device) {
        host_inout_buf = sched->alloc_buffer(alloc_param).get_ptr();
        q->memcpy(host_inout_buf, inout_buf, bytes).wait();
    }

    ccl_comp_reduce_regular(
        host_in_buf, in_count, host_inout_buf, out_count, dtype, reduction, reduction_fn, context);

    if (host_in_buf != in_buf) {
        dealloc_param.ptr = host_in_buf;
        sched->dealloc_buffer(dealloc_param);
    }

    if (host_inout_buf != inout_buf) {
        q->memcpy(inout_buf, host_inout_buf, bytes).wait();
        dealloc_param.ptr = host_inout_buf;
        sched->dealloc_buffer(dealloc_param);
    }

    return ccl::status::success;

#else // CCL_ENABLE_SYCL
    return ccl_comp_reduce_regular(
        in_buf, in_count, inout_buf, out_count, dtype, reduction, reduction_fn, context);
#endif // CCL_ENABLE_SYCL
}

ccl::status ccl_comp_batch_reduce(const void* in_buf,
                                  const std::vector<size_t>& offsets,
                                  size_t in_count,
                                  void* inout_buf,
                                  size_t* out_count,
                                  const ccl_datatype& dtype,
                                  ccl::reduction reduction,
                                  ccl::reduction_fn reduction_fn,
                                  const ccl::fn_context* context,
                                  int bf16_keep_precision_mode,
                                  float* tmp,
                                  float* acc) {
    if (bf16_keep_precision_mode) {
        //->acc, tmp fusion_buffer_cache???

        /* inout_buf => inout_buffer + offsets[0] */
        ccl_convert_bf16_to_fp32_arrays(inout_buf, acc, in_count);

        for (size_t i = 1; i < offsets.size(); i++) {
            ccl_convert_bf16_to_fp32_arrays(
                (char*)in_buf + dtype.size() * offsets[i], tmp, in_count);
            ccl_comp_reduce_regular(tmp,
                                    in_count,
                                    acc,
                                    out_count,
                                    ccl::global_data::get().dtypes->get(ccl::datatype::float32),
                                    reduction,
                                    reduction_fn,
                                    context);
        }

        ccl_convert_fp32_to_bf16_arrays(acc, inout_buf, in_count);
    }
    else {
        for (size_t i = 1; i < offsets.size(); i++) {
            ccl_comp_reduce_regular((char*)in_buf + dtype.size() * offsets[i],
                                    in_count,
                                    inout_buf,
                                    out_count,
                                    dtype,
                                    reduction,
                                    reduction_fn,
                                    context);
        }
    }

    return ccl::status::success;
}

const char* ccl_reduction_to_str(ccl::reduction type) {
    switch (type) {
        case ccl::reduction::sum: return "sum";
        case ccl::reduction::prod: return "prod";
        case ccl::reduction::min: return "min";
        case ccl::reduction::max: return "max";
        case ccl::reduction::custom: return "custom";
        default: return "unknown";
    }
}

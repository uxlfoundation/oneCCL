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

#include "common/datatype/datatype.hpp"
#include "common/utils/buffer.hpp"
#include "common/utils/enums.hpp"
#include "common/utils/tuple.hpp"

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#endif // CCL_ENABLE_SYCL

enum class copy_direction { undefined, h2h, d2h, h2d, d2d, t2t, c2c };
std::string to_string(copy_direction val);

class ccl_comm;

struct copy_attr {
    int peer_rank = ccl::utils::invalid_peer_rank;
    size_t peer_buf_idx = 0;
    copy_direction direction;
    bool pt2pt_op = false;
    ccl_comm* map_comm = nullptr;
    size_t in_buf_offset = 0;
    size_t out_buf_offset = 0;
    bool use_nontemporal = false;

#ifdef CCL_ENABLE_ZE
    int hint_queue_index = 0;
    ccl::ze::queue_group_type force_queue_type = ccl::ze::queue_group_type::unknown;
#endif // CCL_ENABLE_ZE

    copy_attr();
    copy_attr(int peer_rank,
              size_t peer_buf_idx,
              copy_direction direction,
              bool pt2pt_op = false,
              ccl_comm* map_comm = nullptr,
              size_t in_buf_offset = 0,
              size_t out_buf_offset = 0,
              bool use_nontemporal = false
#ifdef CCL_ENABLE_ZE
              ,
              int hint_queue_index = 0,
              ccl::ze::queue_group_type force_queue_type = ccl::ze::queue_group_type::unknown
#endif // CCL_ENABLE_ZE
    );
    copy_attr(copy_direction direction, size_t in_buf_offset = 0, size_t out_buf_offset = 0);

#ifdef CCL_ENABLE_ZE
    copy_attr(copy_direction direction, ccl::ze::queue_group_type force_queue_type);
#endif // CCL_ENABLE_ZE
};

#ifdef CCL_ENABLE_SYCL
struct sycl_copier {
    sycl_copier() = default;
    sycl_copier(copy_direction direction,
                ccl_buffer in_buf,
                ccl_buffer out_buf,
                size_t count,
                const ccl_datatype& dtype,
                bool is_sycl_buf = false,
                size_t in_buf_offset = 0,
                size_t out_buf_offset = 0);

    bool is_completed() const;
    void set_queue(const sycl::queue* external_q);

    template <size_t index, class specific_sycl_buffer>
    void invoke() {
        const bool dtype_idx_is_matched = index == (int)(dtype.idx());

        if (dtype_idx_is_matched) {
            LOG_DEBUG("visitor matched index: ",
                      index,
                      ", dt: ",
                      get_dtype_name(dtype),
                      ", in: ",
                      __PRETTY_FUNCTION__);

            size_t bytes = count * dtype.size();

            void* in_buf_ptr = in_buf.get_ptr(bytes);
            void* out_buf_ptr = out_buf.get_ptr(bytes);

            if (direction == copy_direction::d2d) {
                CCL_THROW_IF_NOT(!is_sycl_buf, "D2D + SYCL buffer");
                e = q->submit([&](sycl::handler& h) {
                    h.memcpy(static_cast<typename specific_sycl_buffer::value_type*>(out_buf_ptr) +
                                 out_buf_offset,
                             static_cast<typename specific_sycl_buffer::value_type*>(in_buf_ptr) +
                                 in_buf_offset,
                             bytes);
                });
                return;
            }

            void* void_device_ptr = (direction == copy_direction::h2d) ? out_buf_ptr : in_buf_ptr;

            if (!is_sycl_buf) {
                auto device_ptr_type = sycl::get_pointer_type(void_device_ptr, q->get_context());
                CCL_THROW_IF_NOT(device_ptr_type == sycl::usm::alloc::device,
                                 "unexpected USM type ",
                                 ccl::utils::usm_type_to_str(device_ptr_type),
                                 " for device_ptr ",
                                 void_device_ptr);
            }

            LOG_DEBUG("count: ",
                      count,
                      ", in_buf_offset: ",
                      in_buf_offset,
                      ", out_buf_offset: ",
                      out_buf_offset,
                      ", dtype_size: ",
                      dtype.size(),
                      ", bytes: ",
                      bytes,
                      ", direction: ",
                      to_string(direction),
                      ", in_buf_ptr: ",
                      in_buf_ptr,
                      ", out_buf_ptr: ",
                      out_buf_ptr,
                      ", device_ptr: ",
                      void_device_ptr,
                      ", is_sycl_buf: ",
                      (is_sycl_buf) ? "yes" : "no");

            if (is_sycl_buf) {
                /* cast device pointer into SYCL buffer */
                specific_sycl_buffer* device_buf_ptr =
                    static_cast<specific_sycl_buffer*>(void_device_ptr);

                specific_sycl_buffer host_buf(
                    static_cast<typename specific_sycl_buffer::value_type*>(
                        (direction == copy_direction::h2d) ? in_buf_ptr : out_buf_ptr),
                    count,
                    sycl::property::buffer::use_host_ptr{});

                e = q->submit([&](sycl::handler& h) {
                    auto& src_buf = (direction == copy_direction::h2d) ? host_buf : *device_buf_ptr;
                    auto& dst_buf = (direction == copy_direction::h2d) ? *device_buf_ptr : host_buf;
                    auto src_buf_acc = src_buf.template get_access<sycl::access::mode::read>(
                        h, count, in_buf_offset);
                    auto dst_buf_acc = dst_buf.template get_access<sycl::access::mode::write>(
                        h, count, out_buf_offset);
                    h.copy(std::move(src_buf_acc), std::move(dst_buf_acc));
                });
            }
            else {
                /* don't do special cast, provided USM pointer can be used as is in copy kernel */
                e = q->memcpy(static_cast<typename specific_sycl_buffer::value_type*>(out_buf_ptr) +
                                  out_buf_offset,
                              static_cast<typename specific_sycl_buffer::value_type*>(in_buf_ptr) +
                                  in_buf_offset,
                              bytes);
            }

            /* TODO: fix parallel copies */
            e.wait();
        }
        else {
            LOG_TRACE("visitor skipped index: ",
                      index,
                      ", dt: ",
                      get_dtype_name(dtype),
                      ", in: ",
                      __PRETTY_FUNCTION__);
        }
    }

    std::string get_dtype_name(const ccl_datatype& dt) const;

    copy_direction direction;
    ccl_buffer in_buf{};
    ccl_buffer out_buf{};
    size_t count = 0;
    ccl_datatype dtype{};
    bool is_sycl_buf = false;
    sycl::queue* q{ nullptr };
    size_t in_buf_offset = 0;
    size_t out_buf_offset = 0;
    sycl::event e{};
};
#endif // CCL_ENABLE_SYCL

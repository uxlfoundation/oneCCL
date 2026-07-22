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
#include "sched/entry/copy/copy_helper.hpp"

copy_attr::copy_attr()
        : peer_rank(ccl::utils::invalid_rank),
          peer_buf_idx(0),
          direction(copy_direction::undefined),
          pt2pt_op(false),
          map_comm(nullptr),
          in_buf_offset(0),
          out_buf_offset(0),
          use_nontemporal(false)
#ifdef CCL_ENABLE_ZE
          ,
          hint_queue_index(0),
          force_queue_type(ccl::ze::queue_group_type::unknown)
#endif // CCL_ENABLE_ZE
{
}

copy_attr::copy_attr(int peer_rank,
                     size_t peer_buf_idx,
                     copy_direction direction,
                     bool pt2pt_op,
                     ccl_comm* map_comm,
                     size_t in_buf_offset,
                     size_t out_buf_offset,
                     bool use_nontemporal
#ifdef CCL_ENABLE_ZE
                     ,
                     int hint_queue_index,
                     ccl::ze::queue_group_type force_queue_type
#endif // CCL_ENABLE_ZE
                     )
        : peer_rank(peer_rank),
          peer_buf_idx(peer_buf_idx),
          direction(direction),
          pt2pt_op(pt2pt_op),
          map_comm(map_comm),
          in_buf_offset(in_buf_offset),
          out_buf_offset(out_buf_offset),
          use_nontemporal(use_nontemporal)
#ifdef CCL_ENABLE_ZE
          ,
          hint_queue_index(hint_queue_index),
          force_queue_type(force_queue_type)
#endif // CCL_ENABLE_ZE
{
}

copy_attr::copy_attr(copy_direction direction, size_t in_buf_offset, size_t out_buf_offset)
        : direction(direction),
          in_buf_offset(in_buf_offset),
          out_buf_offset(out_buf_offset) {}

#ifdef CCL_ENABLE_ZE
copy_attr::copy_attr(copy_direction direction, ccl::ze::queue_group_type force_queue_type)
        : direction(direction),
          force_queue_type(force_queue_type) {}
#endif // CCL_ENABLE_ZE

using copy_direction_str_enum =
    ccl::utils::enum_to_str<ccl::utils::enum_to_underlying(copy_direction::c2c) + 1>;
std::string to_string(copy_direction val) {
    return copy_direction_str_enum({ "UNDEFINED", "H2H", "D2H", "H2D", "D2D", "T2T", "C2C" })
        .choose(val, "UNKNOWN");
}

#ifdef CCL_ENABLE_SYCL

sycl_copier::sycl_copier(copy_direction direction,
                         ccl_buffer in_buf,
                         ccl_buffer out_buf,
                         size_t count,
                         const ccl_datatype& dtype,
                         bool is_sycl_buf,
                         size_t in_buf_offset,
                         size_t out_buf_offset)
        : direction(direction),
          in_buf(in_buf),
          out_buf(out_buf),
          count(count),
          dtype(dtype),
          is_sycl_buf(is_sycl_buf),
          in_buf_offset(in_buf_offset),
          out_buf_offset(out_buf_offset) {}

bool sycl_copier::is_completed() const {
    return e.get_info<sycl::info::event::command_execution_status>() ==
           sycl::info::event_command_status::complete;
}

void sycl_copier::set_queue(const sycl::queue* external_q) {
    q = const_cast<sycl::queue*>(external_q);
    CCL_THROW_IF_NOT(q);
}

std::string sycl_copier::get_dtype_name(const ccl_datatype& dt) const {
    return ccl::global_data::get().dtypes->name(dt);
}

#endif // CCL_ENABLE_SYCL

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

#include "sched/entry/ze/ze_reduce_local_entry.hpp"
#include "sched/entry/ze/ze_primitives.hpp"
#include "sched/entry/ze/cache/ze_cache.hpp"

#include <string>

using namespace ccl;
using namespace ccl::ze;

ze_reduce_local_entry::ze_reduce_local_entry(ccl_sched* sched,
                                             const ccl_buffer in_buf,
                                             size_t in_cnt,
                                             ccl_buffer inout_buf,
                                             size_t* out_cnt,
                                             const ccl_datatype& dtype,
                                             ccl::reduction op,
                                             const std::vector<ze_event_handle_t>& wait_events)
        : ze_base_entry(sched, wait_events),
          in_buf(in_buf),
          in_cnt(in_cnt),
          inout_buf(inout_buf),
          dtype(dtype),
          op(op) {}

void ze_reduce_local_entry::init_ze_hook() {
    global_data::get().ze_data->cache->get(context, device, "kernels.spv", &module);

    kernel_name =
        "reduce_local_inplace_kernel_" + to_string(dtype.idx()) + "_" + ccl_reduction_to_str(op);

    size_t bytes = in_cnt * dtype.size();
    void* in_buf_ptr = in_buf.get_ptr(bytes);
    void* inout_buf_ptr = inout_buf.get_ptr(bytes);
    ze_kernel_args_t kernel_args{ &in_cnt, &in_buf_ptr, &inout_buf_ptr };

    ze_kernel kernel(module, kernel_name, kernel_args, in_cnt, worker_idx);

    ZE_APPEND_CALL(ze_cmd_launch_kernel,
                   ze_base_entry::get_comp_list(),
                   std::move(kernel),
                   ze_base_entry::entry_event,
                   wait_events);
}

void ze_reduce_local_entry::finalize_ze_hook() {}

std::string ze_reduce_local_entry::name_ext() const {
    std::stringstream out;
    out << name() << ":" << in_cnt * dtype.size();
    return out.str();
}

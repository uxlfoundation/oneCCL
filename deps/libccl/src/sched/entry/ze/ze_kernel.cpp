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
#include "sched/entry/ze/ze_kernel.hpp"

using namespace ccl;
using namespace ccl::ze;

ze_kernel::ze_kernel(ze_module_handle_t module,
                     const std::string &kernel_name,
                     const ze_kernel_args_t &kernel_args,
                     size_t elem_count,
                     size_t worker_idx)
        : module(module),
          kernel_name(kernel_name),
          kernel_args(kernel_args),
          worker_idx(worker_idx) {
    global_data::get().ze_data->cache->get(worker_idx, module, kernel_name, &kernel);
    CCL_THROW_IF_NOT(kernel, "no kernel");
    get_suggested_group_size(kernel, elem_count, &group_size);
    get_suggested_group_count(group_size, elem_count, &group_count);
}

ze_kernel::ze_kernel(ze_module_handle_t module,
                     const std::string &kernel_name,
                     const ze_kernel_args_t &kernel_args,
                     size_t elem_count,
                     const ze_group_count_t &group_count,
                     size_t worker_idx)
        : module(module),
          kernel_name(kernel_name),
          kernel_args(kernel_args),
          worker_idx(worker_idx),
          group_count(group_count) {
    global_data::get().ze_data->cache->get(worker_idx, module, kernel_name, &kernel);
    CCL_THROW_IF_NOT(kernel);
    get_suggested_group_size(kernel, elem_count, &group_size);
}

ze_kernel::ze_kernel(ze_kernel &&other) noexcept
        : module(std::move(other.module)),
          kernel_name(std::move(other.kernel_name)),
          kernel_args(std::move(other.kernel_args)),
          worker_idx(std::move(other.worker_idx)),
          group_count(std::move(other.group_count)),
          group_size(std::move(other.group_size)),
          kernel(std::move(other.kernel)) {
    other.module = nullptr;
    other.kernel_name.clear();
    other.kernel_args = {};
    other.worker_idx = 0;
    other.group_count = { 0, 0, 0 };
    other.group_size = { 0, 0, 0 };
    other.kernel = nullptr;
};

ze_kernel::~ze_kernel() noexcept {
    try {
        if (kernel) {
            global_data::get().ze_data->cache->push(worker_idx, module, kernel_name, kernel);
        }
    }
    catch (...) {
        LOG_ERROR("error pushing to the kernel cache");
    }
}

void ze_kernel::actually_call_ze(ze_command_list_handle_t list,
                                 ze_event_handle_t out_event,
                                 const std::vector<ze_event_handle_t> &wait_events) {
    LOG_DEBUG("launch kernel set_group_size {",
              " x ",
              this->group_size.groupSizeX,
              " y ",
              this->group_size.groupSizeY,
              " z ",
              this->group_size.groupSizeZ,
              " }");
    ZE_CALL(zeKernelSetGroupSize,
            (this->kernel,
             this->group_size.groupSizeX,
             this->group_size.groupSizeY,
             this->group_size.groupSizeZ));

    set_kernel_args(kernel, kernel_args);
    ZE_CALL(zeCommandListAppendLaunchKernel,
            (list,
             this->kernel,
             &this->group_count,
             out_event,
             wait_events.size(),
             const_cast<std::vector<ze_event_handle_t> &>(wait_events).data()));
}

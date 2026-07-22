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

#include "sched/entry/ze/ze_primitives.hpp"

class ze_cmd_launch_kernel;

namespace ccl {
namespace ze {

class ze_kernel {
public:
    ze_kernel(ze_module_handle_t module,
              const std::string &kernel_name,
              const ze_kernel_args_t &kernel_args,
              size_t elem_count,
              size_t worker_idx = 0);

    ze_kernel(ze_module_handle_t module,
              const std::string &kernel_name,
              const ze_kernel_args_t &kernel_args,
              size_t elem_count,
              const ze_group_count_t &group_count,
              size_t worker_idx = 0);

    ze_kernel(const ze_kernel &) = delete;
    ze_kernel(ze_kernel &&other) noexcept;
    ze_kernel &operator=(const ze_kernel &) = delete;
    ze_kernel &operator=(ze_kernel &&other) = delete;
    ~ze_kernel() noexcept;

private:
    void actually_call_ze(ze_command_list_handle_t list,
                          ze_event_handle_t out_event,
                          const std::vector<ze_event_handle_t> &wait_events);
    friend class ::ze_cmd_launch_kernel;

    ze_module_handle_t module{};
    std::string kernel_name{};
    ze_kernel_args_t kernel_args{};
    size_t worker_idx{};

    ze_group_count_t group_count{};
    ze_group_size_t group_size{};
    ze_kernel_handle_t kernel{};
};

} // namespace ze
} // namespace ccl

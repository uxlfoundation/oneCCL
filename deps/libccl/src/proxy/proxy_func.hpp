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

/* proxy_func.hpp
*
*  A set of host-side algorithms executed by the proxy thread in response to host-initiated operations.
*/

#pragma once

#include "proxy/proxy.hpp"

#include "proxy/proxy_types.hpp"
#include "proxy/proxy_staging.hpp"
#include "proxy/host_initiated/cmd_ring_types.hpp"
#include "proxy/host_initiated/cmd_ring_host.hpp"

#include <array>
#include <functional>

/* ======== Proxy algorithm function pointer type ======== */

// Function pointer type for proxy algorithms
// All algorithms receive the operation descriptor and proxy context (infrastructure resources)
// The lifecycle wait/signal is handled by RAII guard in the dispatch layer
using proxy_algo_func_t = void (*)(const ccl_proxy_op& op,
                                    ccl_proxy_context& ctx);

/* ======== Proxy algorithm dispatch table ======== */

// Dispatch table indexed by ccl_proxy_op_type
// Initialize this table via init_proxy_algo_table() before use
extern std::array<proxy_algo_func_t, static_cast<size_t>(ccl_proxy_op_type::_count)> proxy_algo_upcall;

// Initialize the proxy algorithm dispatch table
// Must be called once during proxy service initialization
void init_proxy_algo_table();

/* ======== Main dispatch function ======== */

void process_host_initiated_op(const ccl_proxy_op& op,
                               ccl_cmd_ring_producer* producer,
                               ccl_kernel_lifecycle* lifecycle,
                               ccl_proxy_staging* staging,
                               sycl::queue* queue);

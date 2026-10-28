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

#include <map>
#include <memory>
#include <string>
#include <utility>

class ccl_comm;

enum class ccl_p2p_domain_level { p2p_auto = 0, p2p_node, p2p_numa, p2p_switch, p2p_none };

class ccl_p2p_domain {
public:
    static std::map<ccl_p2p_domain_level, std::string> p2p_names;

    static ccl_p2p_domain_level sycl_get_default_p2p_level(ccl_comm* global_comm,
                                                           size_t msg_size,
                                                           size_t threshold);
    static ccl_p2p_domain_level sycl_get_allreduce_p2p_level(ccl_comm* global_comm,
                                                             size_t msg_size,
                                                             size_t threshold);
    static ccl_p2p_domain_level sycl_get_allgatherv_p2p_level(ccl_comm* global_comm,
                                                              size_t msg_size,
                                                              size_t threshold);
    static ccl_p2p_domain_level sycl_get_reduce_scatter_p2p_level(ccl_comm* global_comm,
                                                                  size_t msg_size,
                                                                  size_t threshold);
    static ccl_p2p_domain_level sycl_get_alltoall_p2p_level(ccl_comm* global_comm,
                                                            size_t msg_size,
                                                            size_t small_threshold,
                                                            size_t threshold);
    static ccl_p2p_domain_level sycl_get_broadcast_p2p_level(ccl_comm* global_comm,
                                                             size_t msg_size,
                                                             size_t threshold);

    static std::pair<std::shared_ptr<ccl_comm>, std::shared_ptr<ccl_comm>> get_p2p_comms(
        ccl_comm* global_comm,
        ccl_p2p_domain_level p2p_level);
};

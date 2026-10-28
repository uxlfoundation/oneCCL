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

#include "comm/comm.hpp"
#include "coll/algorithms/utils/p2p_domain.hpp"

std::map<ccl_p2p_domain_level, std::string> ccl_p2p_domain::p2p_names = {
    std::make_pair(ccl_p2p_domain_level::p2p_auto, "auto"),
    std::make_pair(ccl_p2p_domain_level::p2p_node, "node"),
    std::make_pair(ccl_p2p_domain_level::p2p_numa, "numa"),
    std::make_pair(ccl_p2p_domain_level::p2p_switch, "switch"),
    std::make_pair(ccl_p2p_domain_level::p2p_none, "none"),
};

ccl_p2p_domain_level ccl_p2p_domain::sycl_get_default_p2p_level(ccl_comm* global_comm,
                                                                size_t msg_size,
                                                                size_t threshold) {
    ccl_p2p_domain_level p2p_level = ccl::global_data::env().sycl_p2p_level;
    if (p2p_level == ccl_p2p_domain_level::p2p_switch && !global_comm->has_pcie_switch()) {
        CCL_THROW("Invalid P2P domain level: switch");
    }

    if (p2p_level == ccl_p2p_domain_level::p2p_auto) {
        p2p_level = ccl_p2p_domain_level::p2p_node;
    }

    return p2p_level;
}

ccl_p2p_domain_level ccl_p2p_domain::sycl_get_allreduce_p2p_level(ccl_comm* global_comm,
                                                                  size_t msg_size,
                                                                  size_t threshold) {
    ccl_p2p_domain_level p2p_level = ccl::global_data::env().sycl_p2p_level;

    if (p2p_level == ccl_p2p_domain_level::p2p_switch && !global_comm->has_pcie_switch()) {
        CCL_THROW("Invalid P2P domain level: switch");
    }

    if (p2p_level == ccl_p2p_domain_level::p2p_auto) {
        std::shared_ptr<ccl_comm> node_comm = global_comm->get_node_comm();
        bool cross_upi = node_comm->is_cross_upi();
        if (!cross_upi) {
            if (global_comm->has_pcie_switch() && node_comm->is_cross_pcie_switch() && msg_size > threshold) {
                p2p_level = ccl_p2p_domain_level::p2p_switch;
            }
            else {
                p2p_level = ccl_p2p_domain_level::p2p_node;
            }
        }
        else if (threshold == 0 || msg_size <= threshold) {
            p2p_level = ccl_p2p_domain_level::p2p_node;
        }
        else {
            std::shared_ptr<ccl_comm> numa_comm = global_comm->get_numa_comm();
            if (numa_comm->has_pcie_switch() && numa_comm->is_cross_pcie_switch())
                p2p_level = ccl_p2p_domain_level::p2p_switch;
            else
                p2p_level = ccl_p2p_domain_level::p2p_numa;
        }
    }

    LOG_DEBUG("allreduce p2p level for size: ", msg_size, " is ", p2p_names[p2p_level]);
    return p2p_level;
}

ccl_p2p_domain_level ccl_p2p_domain::sycl_get_reduce_scatter_p2p_level(ccl_comm* global_comm,
                                                                       size_t msg_size,
                                                                       size_t threshold) {
    ccl_p2p_domain_level p2p_level = ccl::global_data::env().sycl_p2p_level;

    if (p2p_level == ccl_p2p_domain_level::p2p_switch && !global_comm->has_pcie_switch()) {
        CCL_THROW("Invalid P2P domain level: switch");
    }

    if (p2p_level == ccl_p2p_domain_level::p2p_auto) {
        if (threshold == 0 || msg_size <= threshold) {
            p2p_level = ccl_p2p_domain_level::p2p_node;
        }
        else {
            std::shared_ptr<ccl_comm> numa_comm = global_comm->get_numa_comm();
            if (numa_comm->has_pcie_switch() && numa_comm->is_cross_pcie_switch())
                p2p_level = ccl_p2p_domain_level::p2p_switch;
            else
                p2p_level = ccl_p2p_domain_level::p2p_numa;
        }
    }
    LOG_DEBUG("reduce_scatter p2p level for size: ", msg_size, " is ", p2p_names[p2p_level]);

    return p2p_level;
}

ccl_p2p_domain_level ccl_p2p_domain::sycl_get_allgatherv_p2p_level(ccl_comm* global_comm,
                                                                   size_t msg_size,
                                                                   size_t threshold) {
    ccl_p2p_domain_level p2p_level = ccl::global_data::env().sycl_p2p_level;

    if (p2p_level == ccl_p2p_domain_level::p2p_switch && !global_comm->has_pcie_switch()) {
        CCL_THROW("Invalid P2P domain level: switch");
    }

    if (p2p_level == ccl_p2p_domain_level::p2p_auto) {
        std::shared_ptr<ccl_comm> node_comm = global_comm->get_node_comm();
        bool cross_upi = node_comm->is_cross_upi();
        if (!cross_upi) {
            if (global_comm->has_pcie_switch() && node_comm->is_cross_pcie_switch() && msg_size > threshold) {
                p2p_level = ccl_p2p_domain_level::p2p_switch;
            }
            else {
                p2p_level = ccl_p2p_domain_level::p2p_node;
            }
        }
        else if (threshold == 0 || msg_size <= threshold) {
            p2p_level = ccl_p2p_domain_level::p2p_node;
        }
        else {
            std::shared_ptr<ccl_comm> numa_comm = global_comm->get_numa_comm();
            if (numa_comm->has_pcie_switch() && numa_comm->is_cross_pcie_switch())
                p2p_level = ccl_p2p_domain_level::p2p_switch;
            else
                p2p_level = ccl_p2p_domain_level::p2p_numa;
        }
    }
    LOG_DEBUG("allgatherv p2p level for size: ", msg_size, " is ", p2p_names[p2p_level]);

    return p2p_level;
}

ccl_p2p_domain_level ccl_p2p_domain::sycl_get_alltoall_p2p_level(ccl_comm* global_comm,
                                                                 size_t msg_size,
                                                                 size_t small_threshold,
                                                                 size_t threshold) {
    ccl_p2p_domain_level p2p_level = ccl::global_data::env().sycl_p2p_level;

    if (p2p_level == ccl_p2p_domain_level::p2p_switch && !global_comm->has_pcie_switch()) {
        CCL_THROW("Invalid P2P domain level: switch");
    }

    if (p2p_level == ccl_p2p_domain_level::p2p_auto) {
        std::shared_ptr<ccl_comm> node_comm = global_comm->get_node_comm();
        if (small_threshold == 0 || msg_size <= small_threshold) {
            p2p_level = ccl_p2p_domain_level::p2p_node;
        }
	else if (threshold == 0 || msg_size <= threshold) {
            p2p_level = ccl_p2p_domain_level::p2p_none;
        }
        else {
            std::shared_ptr<ccl_comm> numa_comm = global_comm->get_numa_comm();
            if (numa_comm->has_pcie_switch() && numa_comm->is_cross_pcie_switch())
                p2p_level = ccl_p2p_domain_level::p2p_switch;
            else
                p2p_level = ccl_p2p_domain_level::p2p_numa;
        }
    }
    LOG_DEBUG("alltoall p2p level for size: ", msg_size, " is ", p2p_names[p2p_level]);

    return p2p_level;
}

ccl_p2p_domain_level ccl_p2p_domain::sycl_get_broadcast_p2p_level(ccl_comm* global_comm,
                                                                  size_t msg_size,
                                                                  size_t threshold) {
    ccl_p2p_domain_level p2p_level = ccl::global_data::env().sycl_p2p_level;

    if (p2p_level == ccl_p2p_domain_level::p2p_switch && !global_comm->has_pcie_switch()) {
        CCL_THROW("Invalid P2P domain level: switch");
    }

    if (p2p_level == ccl_p2p_domain_level::p2p_auto) {
	if (threshold == 0 || msg_size <= threshold) {
            p2p_level = ccl_p2p_domain_level::p2p_node;
        }
        else {
            std::shared_ptr<ccl_comm> numa_comm = global_comm->get_numa_comm();
            if (numa_comm->has_pcie_switch() && numa_comm->is_cross_pcie_switch())
                p2p_level = ccl_p2p_domain_level::p2p_switch;
            else
                p2p_level = ccl_p2p_domain_level::p2p_numa;
        }
    }
    LOG_DEBUG("broadcast p2p level for size: ", msg_size, " is ", p2p_names[p2p_level]);

    return p2p_level;
}

std::pair<std::shared_ptr<ccl_comm>, std::shared_ptr<ccl_comm>> ccl_p2p_domain::get_p2p_comms(
    ccl_comm* global_comm,
    ccl_p2p_domain_level p2p_level) {
    std::shared_ptr<ccl_comm> p2p_comm;
    std::shared_ptr<ccl_comm> r2r_comm;

    switch (p2p_level) {
        case ccl_p2p_domain_level::p2p_node:
            p2p_comm = global_comm->get_node_comm();
            r2r_comm = global_comm->get_r2r_comm();
            break;
        case ccl_p2p_domain_level::p2p_numa:
            p2p_comm = global_comm->get_numa_comm();
            r2r_comm = global_comm->get_numa_r2r_comm();
            break;
        case ccl_p2p_domain_level::p2p_switch:
            p2p_comm = global_comm->get_pcie_switch_comm();
            r2r_comm = global_comm->get_pcie_switch_r2r_comm();
            break;
        case ccl_p2p_domain_level::p2p_none:
            p2p_comm = global_comm->get_self_comm();
            r2r_comm = global_comm->get_self_r2r_comm();
            break;
        default: CCL_THROW("Invalid P2P domain level");
    }

    // not uniformly distributed, fallback to node level
    if (p2p_level != ccl_p2p_domain_level::p2p_node && (!p2p_comm->is_uniform || !r2r_comm->is_uniform))
        return get_p2p_comms(global_comm, ccl_p2p_domain_level::p2p_node);

    return { p2p_comm, r2r_comm };
}

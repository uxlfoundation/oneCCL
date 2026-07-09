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

#include "coll/selection/selection.hpp"

template <>
std::map<ccl_coll_bcast_algo, std::string>
    ccl_algorithm_selector_helper<ccl_coll_bcast_algo>::algo_names = {
        std::make_pair(ccl_coll_bcast_direct, "direct"),
        std::make_pair(ccl_coll_bcast_ring, "ring"),
        std::make_pair(ccl_coll_bcast_double_tree, "double_tree"),
        std::make_pair(ccl_coll_bcast_naive, "naive"),
#ifdef CCL_ENABLE_SYCL
        std::make_pair(ccl_coll_bcast_topo, "topo")
#endif // CCL_ENABLE_SYCL
    };

ccl_algorithm_selector<ccl_coll_bcast>::ccl_algorithm_selector() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_bcast_topo);
#else // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_bcast_naive);
        insert(main_table, 0, CCL_BCAST_SHORT_MSG_SIZE, ccl_coll_bcast_double_tree);
    }
    else if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_bcast_direct);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_bcast_naive);

    // bcast currently does not support scale-out selection, but the table
    // has to be defined, therefore duplicating main table
    scaleout_table = main_table;
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_bcast_algo>::can_use(
    ccl_coll_bcast_algo algo,
    const ccl_selector_param& param,
    const ccl_selection_table_t<ccl_coll_bcast_algo>& table) {
    bool can_use = true;

    ccl_coll_algo algo_param;
    algo_param.bcast = algo;
    can_use = ccl_can_use_datatype(algo_param, param);

    if (ccl::global_data::env().enable_unordered_coll && algo == ccl_coll_bcast_double_tree) {
        /* TODO: stabilize double_tree bcast for unordered_coll case */
        can_use = false;
    }
    else if (algo == ccl_coll_bcast_direct &&
             (ccl::global_data::env().atl_transport == ccl_atl_ofi)) {
        can_use = false;
    }
    else if (algo == ccl_coll_bcast_topo && !ccl_can_use_topo_algo(param)) {
        can_use = false;
    }

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_bcast_algo,
                                    ccl_coll_bcast,
                                    ccl::global_data::env().bcast_algo_raw,
                                    param.count,
                                    ccl::global_data::env().bcast_scaleout_algo_raw);

// broadcast algorithm
template <>
std::map<ccl_coll_broadcast_algo, std::string>
    ccl_algorithm_selector_helper<ccl_coll_broadcast_algo>::algo_names = {
        std::make_pair(ccl_coll_broadcast_direct, "direct"),
        std::make_pair(ccl_coll_broadcast_ring, "ring"),
        std::make_pair(ccl_coll_broadcast_double_tree, "double_tree"),
        std::make_pair(ccl_coll_broadcast_naive, "naive"),
#ifdef CCL_ENABLE_SYCL
        std::make_pair(ccl_coll_broadcast_topo, "topo")
#endif // CCL_ENABLE_SYCL
    };

ccl_algorithm_selector<ccl_coll_broadcast>::ccl_algorithm_selector() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_broadcast_topo);
#else // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_broadcast_naive);
        insert(main_table, 0, CCL_BCAST_SHORT_MSG_SIZE, ccl_coll_broadcast_double_tree);
    }
    else if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_broadcast_direct);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_broadcast_naive);

    // broadcast currently does not support scale-out selection, but the table
    // has to be defined, therefore duplicating main table
    scaleout_table = main_table;
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_broadcast_algo>::can_use(
    ccl_coll_broadcast_algo algo,
    const ccl_selector_param& param,
    const ccl_selection_table_t<ccl_coll_broadcast_algo>& table) {
    bool can_use = true;

    ccl_coll_algo algo_param;
    algo_param.broadcast = algo;
    can_use = ccl_can_use_datatype(algo_param, param);

    if (ccl::global_data::env().enable_unordered_coll && algo == ccl_coll_broadcast_double_tree) {
        /* TODO: stabilize double_tree bcast for unordered_coll case */
        can_use = false;
    }
    else if (algo == ccl_coll_broadcast_direct &&
             (ccl::global_data::env().atl_transport == ccl_atl_ofi)) {
        can_use = false;
    }
    else if (algo == ccl_coll_broadcast_topo && !ccl_can_use_topo_algo(param)) {
        can_use = false;
    }

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_broadcast_algo,
                                    ccl_coll_broadcast,
                                    ccl::global_data::env().broadcast_algo_raw,
                                    param.count,
                                    ccl::global_data::env().broadcast_scaleout_algo_raw);

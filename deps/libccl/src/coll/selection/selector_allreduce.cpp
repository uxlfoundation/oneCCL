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
std::map<ccl_coll_allreduce_algo, std::string>
    ccl_algorithm_selector_helper<ccl_coll_allreduce_algo>::algo_names = {
        std::make_pair(ccl_coll_allreduce_direct, "direct"),
        std::make_pair(ccl_coll_allreduce_rabenseifner, "rabenseifner"),
        std::make_pair(ccl_coll_allreduce_nreduce, "nreduce"),
        std::make_pair(ccl_coll_allreduce_ring, "ring"),
        std::make_pair(ccl_coll_allreduce_ring_rma, "ring_rma"),
        std::make_pair(ccl_coll_allreduce_double_tree, "double_tree"),
        std::make_pair(ccl_coll_allreduce_recursive_doubling, "recursive_doubling"),
        std::make_pair(ccl_coll_allreduce_2d, "2d"),
#ifdef CCL_ENABLE_SYCL
        std::make_pair(ccl_coll_allreduce_topo, "topo"),
#endif // CCL_ENABLE_SYCL
    };

ccl_algorithm_selector<ccl_coll_allreduce>::ccl_algorithm_selector() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_topo);
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_ring);
        insert(
            fallback_table, 0, CCL_ALLREDUCE_SHORT_MSG_SIZE, ccl_coll_allreduce_recursive_doubling);
    }
    else {
        // TODO: restore direct algo in fallback table
        insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_ring);
    }
#else // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi) {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_ring);
        insert(main_table, 0, CCL_ALLREDUCE_SHORT_MSG_SIZE, ccl_coll_allreduce_recursive_doubling);
        insert(main_table,
               CCL_ALLREDUCE_SHORT_MSG_SIZE + 1,
               CCL_ALLREDUCE_MEDIUM_MSG_SIZE,
               ccl_coll_allreduce_nreduce);
    }
    else if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_direct);
    }
    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_ring);
    insert(fallback_table, 0, CCL_ALLREDUCE_SHORT_MSG_SIZE, ccl_coll_allreduce_recursive_doubling);
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    insert(scaleout_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_ring);
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_allreduce_algo>::can_use(
    ccl_coll_allreduce_algo algo,
    const ccl_selector_param& param,
    const ccl_selection_table_t<ccl_coll_allreduce_algo>& table) {
    bool can_use = true;

    ccl_coll_algo algo_param;
    algo_param.allreduce = algo;
    can_use = ccl_can_use_datatype(algo_param, param);

    if (algo == ccl_coll_allreduce_rabenseifner &&
        static_cast<int>(param.count) < param.comm->pof2())
        can_use = false;
    else if (algo == ccl_coll_allreduce_ring_rma && !atl_base_comm::attr.out.enable_rma)
        can_use = false;
    else if (algo == ccl_coll_allreduce_nreduce && !(param.count / param.comm->size()))
        can_use = false;
    else if (algo == ccl_coll_allreduce_direct &&
             (ccl::global_data::env().atl_transport == ccl_atl_ofi))
        can_use = false;
    else if (algo == ccl_coll_allreduce_topo && !ccl_can_use_topo_algo(param))
        can_use = false;
    else if (algo == ccl_coll_allreduce_2d && param.is_scaleout)
        // MLSL-1762: scale-up topo + scale-out 2d combination fails.
        // Algorithms are not compatible.
        can_use = false;
    else if (algo == ccl_coll_allreduce_direct && param.is_scaleout &&
             ccl::global_data::env().worker_count > 1
#ifdef CCL_ENABLE_SYCL
             && ccl::global_data::env().ze_multi_workers
#endif // CCL_ENABLE_SYCL
    )
        // MLSL-1757: scale-up topo + scale-out direct combination hangs
        // for CCL_ZE_MULTI_WORKERS=1 + CC_WORKER_COUNT > 1 cases
        can_use = false;

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_allreduce_algo,
                                    ccl_coll_allreduce,
                                    ccl::global_data::env().allreduce_algo_raw,
                                    param.count,
                                    ccl::global_data::env().allreduce_scaleout_algo_raw);

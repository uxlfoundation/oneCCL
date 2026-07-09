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
std::map<ccl_coll_alltoallv_algo, std::string>
    ccl_algorithm_selector_helper<ccl_coll_alltoallv_algo>::algo_names = {
        std::make_pair(ccl_coll_alltoallv_direct, "direct"),
        std::make_pair(ccl_coll_alltoallv_naive, "naive"),
        std::make_pair(ccl_coll_alltoallv_scatter, "scatter"),
#ifdef CCL_ENABLE_SYCL
        std::make_pair(ccl_coll_alltoallv_topo, "topo")
#endif // CCL_ENABLE_SYCL
    };

ccl_algorithm_selector<ccl_coll_alltoallv>::ccl_algorithm_selector() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoallv_topo);
#else // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoallv_scatter);
    if (ccl::global_data::env().atl_transport == ccl_atl_mpi) {
        insert(main_table, 0, CCL_ALLTOALL_MEDIUM_MSG_SIZE, ccl_coll_alltoallv_direct);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    insert(scaleout_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoallv_scatter);
    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_alltoallv_scatter);
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_alltoallv_algo>::can_use(
    ccl_coll_alltoallv_algo algo,
    const ccl_selector_param& param,
    const ccl_selection_table_t<ccl_coll_alltoallv_algo>& table) {
    bool can_use = true;

    if (algo == ccl_coll_alltoallv_topo && !ccl_can_use_topo_algo(param)) {
        can_use = false;
    }
    else if (param.is_vector_buf && algo != ccl_coll_alltoallv_scatter &&
             algo != ccl_coll_alltoallv_naive && algo != ccl_coll_alltoallv_topo) {
        can_use = false;
    }
    else if (algo == ccl_coll_alltoallv_direct &&
             (ccl::global_data::env().atl_transport == ccl_atl_ofi)) {
        can_use = false;
    }
    else if (algo == ccl_coll_alltoallv_direct && param.is_scaleout) {
        // TODO: MLSL-1877: scale-up topo + scale-out direct combination hangs
        // for all cases, with enabled and disabled CCL_ZE_MULTI_WORKERS
        can_use = false;
    }

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_alltoallv_algo,
                                    ccl_coll_alltoallv,
                                    ccl::global_data::env().alltoallv_algo_raw,
                                    param.count,
                                    ccl::global_data::env().alltoallv_scaleout_algo_raw);

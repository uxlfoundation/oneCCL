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
std::map<ccl_coll_barrier_algo, std::string>
    ccl_algorithm_selector_helper<ccl_coll_barrier_algo>::algo_names = {
        std::make_pair(ccl_coll_barrier_direct, "direct"),
        std::make_pair(ccl_coll_barrier_ring, "ring")
    };

ccl_algorithm_selector<ccl_coll_barrier>::ccl_algorithm_selector() {
    // TODO: make ring barrier default after MLSL-1915 is done
    if (ccl::global_data::env().atl_transport == ccl_atl_ofi)
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_barrier_ring);
    else if (ccl::global_data::env().atl_transport == ccl_atl_mpi)
        insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_barrier_direct);
    insert(fallback_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_barrier_ring);

    // barrier currently does not support scale-out selection, but the table
    // has to be defined, therefore duplicating main table
    scaleout_table = main_table;
}

template <>
bool ccl_algorithm_selector_helper<ccl_coll_barrier_algo>::can_use(
    ccl_coll_barrier_algo algo,
    const ccl_selector_param& param,
    const ccl_selection_table_t<ccl_coll_barrier_algo>& table) {
    bool can_use = true;

    if (algo == ccl_coll_barrier_direct && (ccl::global_data::env().atl_transport == ccl_atl_ofi))
        can_use = false;

    return can_use;
}

CCL_SELECTION_DEFINE_HELPER_METHODS(ccl_coll_barrier_algo,
                                    ccl_coll_barrier,
                                    ccl::global_data::env().barrier_algo_raw,
                                    0,
                                    ccl::global_data::env().barrier_scaleout_algo_raw);

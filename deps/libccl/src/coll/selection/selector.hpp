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

#include "coll/algorithms/algorithms.hpp"
#include "coll/coll.hpp"
#include "common/global/global.hpp"

#include <map>
#include <string>

#define CCL_ALLGATHER_SHORT_MSG_SIZE  32768
#define CCL_ALLGATHERV_SHORT_MSG_SIZE 32768
#define CCL_ALLREDUCE_SHORT_MSG_SIZE  8192
#define CCL_ALLREDUCE_MEDIUM_MSG_SIZE (1024 * 1024)
#define CCL_ALLTOALL_MEDIUM_MSG_SIZE  (1024 * 1024)
#define CCL_BCAST_SHORT_MSG_SIZE      8192
#define CCL_REDUCE_SHORT_MSG_SIZE     8192

struct ccl_selector_param {
    ccl_coll_type ctype = ccl_coll_last_value;
    size_t count = 0;
    ccl_datatype dtype = ccl_datatype_int8;
    ccl_comm* comm = nullptr;
    ccl_stream* stream = nullptr;
    void* buf = nullptr;

    ccl::reduction reduction = ccl::reduction::custom;

    const size_t* send_counts = nullptr;
    const size_t* recv_counts = nullptr;
    int is_vector_buf = 0;

#ifdef CCL_ENABLE_SYCL
    int is_sycl_buf = 0;
#endif // CCL_ENABLE_SYCL

    int peer_rank = CCL_INVALID_PEER_RANK_IDX;

    ccl_coll_algo hint_algo = {};

    bool is_scaleout = false;

    static ccl_selector_param create(ccl_coll_type ctype,
                                     size_t count,
                                     ccl::datatype dtype,
                                     ccl_comm* comm,
                                     ccl_stream* stream,
                                     void* buf,
                                     ccl::reduction reduction,
                                     bool is_vector_buf,
                                     bool is_sycl_buf,
                                     int peer_rank,
                                     ccl_coll_algo hint_algo,
                                     bool is_scaleout) {
        ccl_selector_param param;
        param.ctype = ctype;
        param.count = count;
        param.dtype = ccl::global_data::get().dtypes->get(dtype);
        param.reduction = reduction;
        param.comm = comm;
        param.stream = stream;
        param.buf = buf;
        param.is_vector_buf = is_vector_buf;

#ifdef CCL_ENABLE_SYCL
        param.is_sycl_buf = is_sycl_buf;
#endif

        param.peer_rank = peer_rank;
        param.hint_algo = hint_algo;
        param.is_scaleout = is_scaleout;

        return param;
    }
};

template <ccl_coll_type coll_id>
struct ccl_algorithm_selector;

template <typename algo_group_type>
using ccl_selection_table_iter_t = typename ccl_selection_table_t<algo_group_type>::const_iterator;

template <typename algo_group_type>
struct ccl_algorithm_selector_base {
    ccl_selection_table_t<algo_group_type> main_table;
    ccl_selection_table_t<algo_group_type> fallback_table;
    ccl_selection_table_t<algo_group_type> scaleout_table;
    void init();
    void print() const;
    algo_group_type get(const ccl_selector_param& param) const;
    static void insert(ccl_selection_table_t<algo_group_type>& table,
                       size_t left,
                       size_t right,
                       algo_group_type algo_id);
    static std::string table_to_str(const ccl_selection_table_t<algo_group_type>& table);
    static algo_group_type get_value_from_table(
        size_t size,
        const ccl_selection_table_t<algo_group_type>& table);
};

#define CCL_SELECTION_DECLARE_ALGO_SELECTOR(coll_id, algo_group_type) \
    template <> \
    struct ccl_algorithm_selector<coll_id> : public ccl_algorithm_selector_base<algo_group_type> { \
        using type = algo_group_type; \
        ccl_algorithm_selector(); \
    };

CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_allgather, ccl_coll_allgather_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_allgatherv, ccl_coll_allgatherv_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_allreduce, ccl_coll_allreduce_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_alltoall, ccl_coll_alltoall_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_alltoallv, ccl_coll_alltoallv_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_barrier, ccl_coll_barrier_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_bcast, ccl_coll_bcast_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_broadcast, ccl_coll_broadcast_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_recv, ccl_coll_recv_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_reduce, ccl_coll_reduce_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_reduce_scatter, ccl_coll_reduce_scatter_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_scatter, ccl_coll_scatter_algo);
CCL_SELECTION_DECLARE_ALGO_SELECTOR(ccl_coll_send, ccl_coll_send_algo);

#include "coll/selection/selector_impl.hpp"

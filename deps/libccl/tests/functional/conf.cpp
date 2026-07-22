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

#include "conf.hpp"
#include "lp.hpp"

std::vector<test_param> test_params;

ccl_data_type first_data_type = DATATYPE_INT8;
ccl_data_type last_data_type = DATATYPE_LAST;
std::map<int, std::string> data_type_names = {
    { DATATYPE_INT8, "DATATYPE_INT8" },       { DATATYPE_UINT8, "DATATYPE_UINT8" },
    { DATATYPE_INT16, "DATATYPE_INT16" },     { DATATYPE_UINT16, "DATATYPE_UINT16" },
    { DATATYPE_INT32, "DATATYPE_INT32" },     { DATATYPE_UINT32, "DATATYPE_UINT32" },
    { DATATYPE_INT64, "DATATYPE_INT64" },     { DATATYPE_UINT64, "DATATYPE_UINT64" },
    { DATATYPE_FLOAT16, "DATATYPE_FLOAT16" }, { DATATYPE_FLOAT32, "DATATYPE_FLOAT32" },
    { DATATYPE_FLOAT64, "DATATYPE_FLOAT64" }, { DATATYPE_BFLOAT16, "DATATYPE_BFLOAT16" },
};
std::map<int, ccl::datatype> data_type_values = {
    { DATATYPE_INT8, ccl::datatype::int8 },       { DATATYPE_UINT8, ccl::datatype::uint8 },
    { DATATYPE_INT16, ccl::datatype::int16 },     { DATATYPE_UINT16, ccl::datatype::uint16 },
    { DATATYPE_INT32, ccl::datatype::int32 },     { DATATYPE_UINT32, ccl::datatype::uint32 },
    { DATATYPE_INT64, ccl::datatype::int64 },     { DATATYPE_UINT64, ccl::datatype::uint64 },
    { DATATYPE_FLOAT16, ccl::datatype::float16 }, { DATATYPE_FLOAT32, ccl::datatype::float32 },
    { DATATYPE_FLOAT64, ccl::datatype::float64 }, { DATATYPE_BFLOAT16, ccl::datatype::bfloat16 },
};
std::map<int, size_t> data_type_size_values = {
    { DATATYPE_INT8, 1 },    { DATATYPE_UINT8, 1 },   { DATATYPE_INT16, 2 },
    { DATATYPE_UINT16, 2 },  { DATATYPE_INT32, 4 },   { DATATYPE_UINT32, 4 },
    { DATATYPE_INT64, 8 },   { DATATYPE_UINT64, 8 },  { DATATYPE_FLOAT16, 2 },
    { DATATYPE_FLOAT32, 4 }, { DATATYPE_FLOAT64, 8 }, { DATATYPE_BFLOAT16, 2 },
};

ccl_size_type first_size_type = SIZE_SMALL;
ccl_size_type last_size_type = SIZE_LAST;
std::map<int, std::string> size_type_names = { { SIZE_SMALL, "SIZE_SMALL" },
                                               { SIZE_MEDIUM, "SIZE_MEDIUM" },
                                               { SIZE_LARGE, "SIZE_LARGE" } };
std::map<int, size_t> size_type_values = { { SIZE_SMALL, 17 },
                                           { SIZE_MEDIUM, 32771 },
                                           { SIZE_LARGE, 262144 } };

ccl_buf_count_type first_buf_count_type = BUF_COUNT_SMALL;
ccl_buf_count_type last_buf_count_type = BUF_COUNT_LAST;
std::map<int, std::string> buf_count_type_names = { { BUF_COUNT_SMALL, "BUF_COUNT_SMALL" },
                                                    { BUF_COUNT_LARGE, "BUF_COUNT_LARGE" } };
std::map<int, size_t> buf_count_values = { { BUF_COUNT_SMALL, 1 }, { BUF_COUNT_LARGE, 4 } };

ccl_place_type first_place_type = PLACE_IN;
#ifdef TEST_CCL_BCAST
ccl_place_type last_place_type = static_cast<ccl_place_type>(first_place_type + 1);
#else
ccl_place_type last_place_type = PLACE_LAST;
#endif
std::map<int, std::string> place_type_names = { { PLACE_IN, "PLACE_IN" },
                                                { PLACE_OUT, "PLACE_OUT" } };

ccl_order_type first_start_order = ORDER_DIRECT;
ccl_order_type last_start_order = ORDER_LAST;
ccl_order_type first_complete_order = ORDER_DIRECT;
ccl_order_type last_complete_order = static_cast<ccl_order_type>(first_complete_order + 1);
std::map<int, std::string> order_type_names = { { ORDER_DIRECT, "ORDER_DIRECT" },
                                                { ORDER_INDIRECT, "ORDER_INDIRECT" },
                                                { ORDER_RANDOM, "ORDER_RANDOM" } };

ccl_complete_type first_complete_type = COMPLETE_TEST; //COMPLETE_WAIT;
ccl_complete_type last_complete_type = COMPLETE_LAST;
std::map<int, std::string> complete_type_names = { { COMPLETE_WAIT, "COMPLETE_WAIT" },
                                                   { COMPLETE_TEST, "COMPLETE_TEST" } };

ccl_cache_type first_cache_type = CACHE_FALSE;
ccl_cache_type last_cache_type = CACHE_LAST;
std::map<int, std::string> cache_type_names = { { CACHE_FALSE, "CACHE_FALSE" },
                                                { CACHE_TRUE, "CACHE_TRUE" } };

ccl_sync_type first_sync_type = SYNC_FALSE;
ccl_sync_type last_sync_type = SYNC_TRUE; //SYNC_LAST;
std::map<int, std::string> sync_type_names = { { SYNC_FALSE, "SYNC_FALSE" },
                                               { SYNC_TRUE, "SYNC_TRUE" } };

ccl_reduction_type first_reduction_type = REDUCTION_SUM;
#ifdef TEST_CCL_REDUCE
ccl_reduction_type last_reduction_type = REDUCTION_LAST;
#else
ccl_reduction_type last_reduction_type = static_cast<ccl_reduction_type>(first_reduction_type + 1);
#endif
std::map<int, std::string> reduction_type_names = {
    { REDUCTION_SUM, "REDUCTION_SUM" },       { REDUCTION_PROD, "REDUCTION_PROD" },
    { REDUCTION_MIN, "REDUCTION_MIN" },       { REDUCTION_MAX, "REDUCTION_MAX" },
#ifdef TEST_CCL_CUSTOM_REDUCE
    { REDUCTION_CUSTOM, "REDUCTION_CUSTOM" }, { REDUCTION_CUSTOM_NULL, "REDUCTION_CUSTOM_NULL" }
#endif
};
std::map<int, ccl::reduction> reduction_values = {
    { REDUCTION_SUM, ccl::reduction::sum },       { REDUCTION_PROD, ccl::reduction::prod },
    { REDUCTION_MIN, ccl::reduction::min },       { REDUCTION_MAX, ccl::reduction::max },
#ifdef TEST_CCL_CUSTOM_REDUCE
    { REDUCTION_CUSTOM, ccl::reduction::custom }, { REDUCTION_CUSTOM_NULL, ccl::reduction::custom }
#endif
};

POST_AND_PRE_INCREMENTS(ccl_data_type, DATATYPE_LAST);
POST_AND_PRE_INCREMENTS(ccl_size_type, SIZE_LAST);
POST_AND_PRE_INCREMENTS(ccl_buf_count_type, BUF_COUNT_LAST);
POST_AND_PRE_INCREMENTS(ccl_place_type, PLACE_LAST);
POST_AND_PRE_INCREMENTS(ccl_order_type, ORDER_LAST);
POST_AND_PRE_INCREMENTS(ccl_complete_type, COMPLETE_LAST);
POST_AND_PRE_INCREMENTS(ccl_cache_type, CACHE_LAST);
POST_AND_PRE_INCREMENTS(ccl_sync_type, SYNC_LAST);
POST_AND_PRE_INCREMENTS(ccl_reduction_type, REDUCTION_LAST);

size_t get_elem_count(const test_param& param) {
    return size_type_values[param.size_type];
}

size_t get_buffer_count(const test_param& param) {
    return buf_count_values[param.buf_count_type];
}

ccl::datatype get_ccl_datatype(const test_param& param) {
    return data_type_values[param.datatype];
}

size_t get_ccl_datatype_size(const test_param& param) {
    return data_type_size_values[param.datatype];
}

ccl::reduction get_ccl_reduction(const test_param& param) {
    return reduction_values[param.reduction];
}

bool should_skip_datatype(ccl_data_type dt) {
    if (dt == DATATYPE_BFLOAT16 && !is_bf16_enabled())
        return true;

    if (dt == DATATYPE_FLOAT16 && !is_fp16_enabled())
        return true;

    if (dt == DATATYPE_INT8 || dt == DATATYPE_UINT8 || dt == DATATYPE_INT16 ||
        dt == DATATYPE_UINT16 || dt == DATATYPE_UINT32 || dt == DATATYPE_INT64 ||
        dt == DATATYPE_UINT64 || dt == DATATYPE_FLOAT64)
        return true;

    return false;
}

void init_test_dims() {
    char* data_type_env = getenv("CCL_TEST_DATA_TYPE");
    char* size_type_env = getenv("CCL_TEST_SIZE_TYPE");
    char* buf_count_env = getenv("CCL_TEST_BUF_COUNT_TYPE");
    char* place_type_env = getenv("CCL_TEST_PLACE_TYPE");
    char* start_order_type_env = getenv("CCL_TEST_START_ORDER_TYPE");
    char* complete_order_type_env = getenv("CCL_TEST_COMPLETE_ORDER_TYPE");
    char* complete_type_env = getenv("CCL_TEST_COMPLETE_TYPE");
    char* cache_type_env = getenv("CCL_TEST_CACHE_TYPE");
    char* sync_env = getenv("CCL_TEST_SYNC_TYPE");
    char* reduction_type_env = getenv("CCL_TEST_REDUCTION_TYPE");

    /* limit test dimensions */

    if (data_type_env && atoi(data_type_env) == 0) {
        first_data_type = DATATYPE_FLOAT32;
        last_data_type = static_cast<ccl_data_type>(first_data_type + 1);
    }

    if (size_type_env && atoi(size_type_env) == 0) {
        first_size_type = SIZE_MEDIUM;
        last_size_type = static_cast<ccl_size_type>(first_size_type + 1);
    }

    if (buf_count_env && atoi(buf_count_env) == 0) {
        first_buf_count_type = BUF_COUNT_LARGE;
        last_buf_count_type = static_cast<ccl_buf_count_type>(first_buf_count_type + 1);
    }

    if (place_type_env && atoi(place_type_env) == 0) {
        first_place_type = PLACE_IN;
        last_place_type = static_cast<ccl_place_type>(first_place_type + 1);
    }

    if (start_order_type_env && atoi(start_order_type_env) == 0) {
        first_start_order = ORDER_DIRECT;
        last_start_order = static_cast<ccl_order_type>(first_start_order + 1);
    }

    if (complete_order_type_env && atoi(complete_order_type_env) == 0) {
        first_complete_order = ORDER_DIRECT;
        last_complete_order = static_cast<ccl_order_type>(first_complete_order + 1);
    }

    if (complete_type_env && atoi(complete_type_env) == 0) {
        first_complete_type = COMPLETE_TEST;
        last_complete_type = static_cast<ccl_complete_type>(first_complete_type + 1);
    }

    if (cache_type_env && atoi(cache_type_env) == 0) {
        first_cache_type = CACHE_TRUE;
        last_cache_type = static_cast<ccl_cache_type>(first_cache_type + 1);
    }

    if (sync_env && atoi(sync_env) == 0) {
        first_sync_type = SYNC_FALSE;
        last_sync_type = static_cast<ccl_sync_type>(first_sync_type + 1);
    }

    if (reduction_type_env && atoi(reduction_type_env) == 0) {
        first_reduction_type = REDUCTION_SUM;
        last_reduction_type = static_cast<ccl_reduction_type>(first_reduction_type + 1);
    }
}

void init_test_params() {
    init_test_dims();

#ifdef CCL_ENABLE_SYCL
    printf("FUNC_TESTS: CCL_ENABLE_SYCL ON\n");
#endif
    printf("FUNC_TESTS: BF16 enabled %d\n", is_bf16_enabled());
    printf("FUNC_TESTS: FP16 enabled %d\n", is_fp16_enabled());

    for (auto data_type = first_data_type; data_type < last_data_type; data_type++) {
        if (should_skip_datatype(data_type))
            continue;
        for (auto size_type = first_size_type; size_type < last_size_type; size_type++) {
            for (auto buf_count_type = first_buf_count_type; buf_count_type < last_buf_count_type;
                 buf_count_type++) {
                for (auto place_type = first_place_type; place_type < last_place_type;
                     place_type++) {
#ifdef TEST_CCL_BCAST
                    if (place_type == PLACE_OUT)
                        continue;
#endif
                    for (auto start_order = first_start_order; start_order < last_start_order;
                         start_order++) {
                        if (start_order == ORDER_RANDOM) {
                            char* unordered_coll_env = getenv("CCL_UNORDERED_COLL");
                            if (!unordered_coll_env || atoi(unordered_coll_env) == 0) {
                                continue;
                            }
                        }
                        for (auto complete_order = first_complete_order;
                             complete_order < last_complete_order;
                             complete_order++) {
                            for (auto complete_type = first_complete_type;
                                 complete_type < last_complete_type;
                                 complete_type++) {
                                for (auto sync_type = first_sync_type; sync_type < last_sync_type;
                                     sync_type++) {
                                    for (auto cache_type = first_cache_type;
                                         cache_type < last_cache_type;
                                         cache_type++) {
                                        for (auto reduction_type = first_reduction_type;
                                             reduction_type < last_reduction_type;
                                             reduction_type++) {
                                            test_param param;
                                            param.datatype = data_type;
                                            param.size_type = size_type;
                                            param.buf_count_type = buf_count_type;
                                            param.place_type = place_type;
                                            param.start_order = start_order;
                                            param.complete_order = complete_order;
                                            param.complete_type = complete_type;
                                            param.cache_type = cache_type;
                                            param.sync_type = sync_type;
                                            param.reduction = reduction_type;
                                            test_params.push_back(param);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

std::ostream& operator<<(std::ostream& stream, const test_param& param) {
    std::stringstream sstream;
    sstream << "["
            << " " << data_type_names[param.datatype] << " " << size_type_names[param.size_type]
            << " " << buf_count_type_names[param.buf_count_type] << " "
            << place_type_names[param.place_type] << " " << order_type_names[param.start_order]
            << " " << complete_type_names[param.complete_type] << " "
            << cache_type_names[param.cache_type] << " " << sync_type_names[param.sync_type] << " "
            << reduction_type_names[param.reduction] << " ]";
    return stream << sstream.str();
}

void print_err_message(char* message, std::ostream& output) {
    auto& comm = transport_data::instance().get_service_comm();
    int comm_size = comm.size();
    int comm_rank = comm.rank();
    size_t message_len = strlen(message);
    std::vector<size_t> message_lens(comm_size, 0);
    std::vector<size_t> recv_counts(comm_size, 1);
    ccl::allgatherv(&message_len, 1, message_lens.data(), recv_counts, comm).wait();

    auto total_message_len = std::accumulate(message_lens.begin(), message_lens.end(), 0);

    if (total_message_len == 0) {
        return;
    }

    std::vector<char> messages(total_message_len);
    ccl::allgatherv(message, message_len, messages.data(), message_lens, ccl::datatype::int8, comm)
        .wait();

    if (comm_rank == 0) {
        output << messages.data();
    }
}

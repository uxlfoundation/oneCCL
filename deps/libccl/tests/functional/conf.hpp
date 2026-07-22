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
#include "mpi.h"
#include <vector>

#include "oneapi/ccl.hpp"

#define POST_AND_PRE_INCREMENTS_DECLARE(EnumName) \
    EnumName& operator++(EnumName& orig); \
    EnumName operator++(EnumName& orig, int);

#define POST_AND_PRE_INCREMENTS(EnumName, LAST_ELEM) \
    EnumName& operator++(EnumName& orig) { \
        if (orig != LAST_ELEM) \
            orig = static_cast<EnumName>(orig + 1); \
        return orig; \
    } \
    EnumName operator++(EnumName& orig, int) { \
        EnumName rVal = orig; \
        ++orig; \
        return rVal; \
    }

#define SOME_VALUE (0xdeadbeef)
#define ROOT_RANK  (0)

typedef enum {
    DATATYPE_INT8 = 0,
    DATATYPE_UINT8,
    DATATYPE_INT16,
    DATATYPE_UINT16,
    DATATYPE_INT32,
    DATATYPE_UINT32,
    DATATYPE_INT64,
    DATATYPE_UINT64,
    DATATYPE_FLOAT16,
    DATATYPE_FLOAT32,
    DATATYPE_FLOAT64,
    DATATYPE_BFLOAT16,
    DATATYPE_LAST
} ccl_data_type;
extern ccl_data_type first_data_type;
extern ccl_data_type last_data_type;
extern std::map<int, std::string> data_type_names;
extern std::map<int, ccl::datatype> data_type_values;
extern std::map<int, size_t> data_type_sizes_values;

typedef enum { SIZE_SMALL = 0, SIZE_MEDIUM, SIZE_LARGE, SIZE_LAST } ccl_size_type;
extern ccl_size_type first_size_type;
extern ccl_size_type last_size_type;
extern std::map<int, std::string> size_type_names;
extern std::map<int, size_t> size_type_values;

typedef enum { BUF_COUNT_SMALL = 0, BUF_COUNT_LARGE, BUF_COUNT_LAST } ccl_buf_count_type;
extern ccl_buf_count_type first_buffer_count;
extern ccl_buf_count_type last_buffer_count;
extern std::map<int, std::string> buf_count_type_names;
extern std::map<int, size_t> buf_count_values;

typedef enum { PLACE_IN = 0, PLACE_OUT, PLACE_LAST } ccl_place_type;
extern ccl_place_type first_place_type;
extern ccl_place_type last_place_type;
extern std::map<int, std::string> place_type_names;

typedef enum { ORDER_DIRECT = 0, ORDER_INDIRECT, ORDER_RANDOM, ORDER_LAST } ccl_order_type;
extern ccl_order_type first_start_order;
extern ccl_order_type last_start_order;
extern ccl_order_type first_complete_order;
extern ccl_order_type last_complete_order;
extern std::map<int, std::string> order_type_names;

typedef enum { COMPLETE_WAIT = 0, COMPLETE_TEST, COMPLETE_LAST } ccl_complete_type;
extern ccl_complete_type first_complete_type;
extern ccl_complete_type last_complete_type;
extern std::map<int, std::string> complete_type_names;

typedef enum { CACHE_FALSE = 0, CACHE_TRUE, CACHE_LAST } ccl_cache_type;
extern ccl_cache_type first_cache_type;
extern ccl_cache_type last_cache_type;
extern std::map<int, std::string> cache_type_names;

typedef enum { SYNC_FALSE = 0, SYNC_TRUE, SYNC_LAST } ccl_sync_type;
extern ccl_sync_type first_sync_type;
extern ccl_sync_type last_sync_type;
extern std::map<int, std::string> sync_type_names;

typedef enum {
    REDUCTION_SUM = 0,
    REDUCTION_PROD,
    REDUCTION_MIN,
    REDUCTION_MAX,
#ifdef TEST_CCL_CUSTOM_REDUCE
    REDUCTION_CUSTOM,
    REDUCTION_CUSTOM_NULL,
#endif
    REDUCTION_LAST
} ccl_reduction_type;
extern ccl_reduction_type first_reduction_type;
extern ccl_reduction_type last_reduction_type;
extern std::map<int, std::string> reduction_type_names;
extern std::map<int, ccl::reduction> reduction_values;

POST_AND_PRE_INCREMENTS_DECLARE(ccl_data_type);
POST_AND_PRE_INCREMENTS_DECLARE(ccl_size_type);
POST_AND_PRE_INCREMENTS_DECLARE(ccl_buf_count_type);
POST_AND_PRE_INCREMENTS_DECLARE(ccl_place_type);
POST_AND_PRE_INCREMENTS_DECLARE(ccl_order_type);
POST_AND_PRE_INCREMENTS_DECLARE(ccl_complete_type);
POST_AND_PRE_INCREMENTS_DECLARE(ccl_cache_type);
POST_AND_PRE_INCREMENTS_DECLARE(ccl_sync_type);
POST_AND_PRE_INCREMENTS_DECLARE(ccl_reduction_type);

struct test_param {
    ccl_data_type datatype;
    ccl_size_type size_type;
    ccl_buf_count_type buf_count_type;
    ccl_place_type place_type;
    ccl_order_type start_order;
    ccl_order_type complete_order;
    ccl_complete_type complete_type;
    ccl_cache_type cache_type;
    ccl_sync_type sync_type;
    ccl_reduction_type reduction;
};

extern std::vector<test_param> test_params;

std::ostream& operator<<(std::ostream& stream, test_param const& conf);

size_t get_elem_count(const test_param& param);
size_t get_buffer_count(const test_param& param);
ccl::datatype get_ccl_datatype(const test_param& param);
size_t get_ccl_datatype_size(const test_param& param);
ccl::reduction get_ccl_reduction(const test_param& param);
void init_test_dims();
void init_test_params();
void print_err_message(char* err_message, std::ostream& output);

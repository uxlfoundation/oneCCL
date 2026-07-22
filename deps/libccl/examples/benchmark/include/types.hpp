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

#include "oneapi/ccl.hpp"
#include "lp.hpp"

#define PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__);

#ifndef PRINT_BY_ROOT
#define PRINT_BY_ROOT(comm, fmt, ...) \
    if (comm.rank() == 0) { \
        printf(fmt "\n", ##__VA_ARGS__); \
    }
#endif // PRINT_BY_ROOT

constexpr std::initializer_list<ccl::datatype> all_dtypes = {
    ccl::datatype::int8,    ccl::datatype::int32,   ccl::datatype::int64,   ccl::datatype::uint64,
    ccl::datatype::float16, ccl::datatype::float32, ccl::datatype::float64, ccl::datatype::bfloat16
};

typedef enum { BACKEND_HOST, BACKEND_SYCL } backend_type_t;
typedef enum { ITER_POLICY_OFF, ITER_POLICY_AUTO } iter_policy_t;
typedef enum { CHECK_OFF, CHECK_LAST_ITER, CHECK_ALL_ITERS } check_values_t;
typedef enum { EXT_OFF, EXT_AUTO, EXT_ON } ext_values_t;

typedef enum { SYCL_DEV_HOST, SYCL_DEV_CPU, SYCL_DEV_GPU } sycl_dev_type_t;
typedef enum { SYCL_MEM_USM, SYCL_MEM_BUF } sycl_mem_type_t;
typedef enum { SYCL_USM_SHARED, SYCL_USM_DEVICE } sycl_usm_type_t;
typedef enum { SYCL_QUEUE_OUT_ORDER, SYCL_QUEUE_IN_ORDER } sycl_queue_type_t;

std::map<backend_type_t, std::string> backend_names = { std::make_pair(BACKEND_HOST, "host"),
                                                        std::make_pair(BACKEND_SYCL, "sycl") };

std::map<iter_policy_t, std::string> iter_policy_names = { std::make_pair(ITER_POLICY_OFF, "off"),
                                                           std::make_pair(ITER_POLICY_AUTO,
                                                                          "auto") };

std::map<check_values_t, std::string> check_values_names = {
    std::make_pair(CHECK_OFF, "off"),
    std::make_pair(CHECK_LAST_ITER, "last"),
    std::make_pair(CHECK_ALL_ITERS, "all")
};

std::map<ext_values_t, std::string> ext_values_names = { std::make_pair(EXT_OFF, "off"),
                                                         std::make_pair(EXT_AUTO, "auto"),
                                                         std::make_pair(EXT_ON, "on") };

#ifdef CCL_ENABLE_SYCL
std::map<sycl_dev_type_t, std::string> sycl_dev_names = { std::make_pair(SYCL_DEV_HOST, "host"),
                                                          std::make_pair(SYCL_DEV_CPU, "cpu"),
                                                          std::make_pair(SYCL_DEV_GPU, "gpu") };

std::map<sycl_mem_type_t, std::string> sycl_mem_names = { std::make_pair(SYCL_MEM_USM, "usm"),
                                                          std::make_pair(SYCL_MEM_BUF, "buf") };

std::map<sycl_usm_type_t, std::string> sycl_usm_names = { std::make_pair(SYCL_USM_SHARED, "shared"),
                                                          std::make_pair(SYCL_USM_DEVICE,
                                                                         "device") };

std::map<sycl_queue_type_t, std::string> sycl_queue_names = {
    std::make_pair(SYCL_QUEUE_OUT_ORDER, "out_order"),
    std::make_pair(SYCL_QUEUE_IN_ORDER, "in_order")
};
#endif // CCL_ENABLE_SYCL

std::map<ccl::datatype, std::string> dtype_names = {
    std::make_pair(ccl::datatype::int8, "int8"),
    std::make_pair(ccl::datatype::int32, "int32"),
    std::make_pair(ccl::datatype::int64, "int64"),
    std::make_pair(ccl::datatype::uint64, "uint64"),
    std::make_pair(ccl::datatype::float16, "float16"),
    std::make_pair(ccl::datatype::float32, "float32"),
    std::make_pair(ccl::datatype::float64, "float64"),
    std::make_pair(ccl::datatype::bfloat16, "bfloat16")
};

std::map<ccl::reduction, std::string> reduction_names = {
    std::make_pair(ccl::reduction::sum, "sum"),
    std::make_pair(ccl::reduction::prod, "prod"),
    std::make_pair(ccl::reduction::min, "min"),
    std::make_pair(ccl::reduction::max, "max"),
};

template <typename T>
std::list<T> tokenize(const std::string& input, char delimeter) {
    std::istringstream ss(input);
    std::list<T> ret;
    std::string str;
    while (std::getline(ss, str, delimeter)) {
        std::stringstream converter;
        converter << str;
        T value;
        converter >> value;
        ret.push_back(value);
    }
    return ret;
}

void generate_counts(std::list<size_t>& counts, size_t min_count, size_t max_count) {
    counts.clear();
    size_t count = min_count;
    if (count == 0) {
        counts.push_back(0);
        count++;
    }
    for (; count <= max_count; count *= 2) {
        counts.push_back(count);
    }
    if (*counts.rbegin() != max_count)
        counts.push_back(max_count);
}

typedef struct user_options_t {
    backend_type_t backend;
    size_t iters;
    size_t warmup_iters;
    iter_policy_t iter_policy;
    size_t buf_count;
    size_t min_elem_count;
    size_t max_elem_count;
    std::list<size_t> elem_counts;
    size_t elem_offset;
    check_values_t check_values;
    int cache_ops;
    int inplace;
    size_t ranks_per_proc; // not exposed in bench options
    int numa_node;
    int verbosity;
#ifdef CCL_ENABLE_SYCL
    sycl_dev_type_t sycl_dev_type;
    int sycl_root_dev;
    sycl_mem_type_t sycl_mem_type;
    sycl_usm_type_t sycl_usm_type;
    sycl_queue_type_t sycl_queue_type;
#endif // CCL_ENABLE_SYCL
    std::list<std::string> coll_names;
    std::list<std::string> dtypes;
    std::list<std::string> reductions;
    std::string csv_filepath;

    bool min_elem_count_set;
    bool max_elem_count_set;
    bool elem_counts_set;
    ext_values_t show_additional_info;

    user_options_t() {
        backend = DEFAULT_BACKEND;
        iters = DEFAULT_ITERS;
        warmup_iters = DEFAULT_WARMUP_ITERS;
        iter_policy = DEFAULT_ITER_POLICY;
        buf_count = DEFAULT_BUF_COUNT;
        min_elem_count = DEFAULT_MIN_ELEM_COUNT;
        max_elem_count = DEFAULT_MAX_ELEM_COUNT;
        generate_counts(elem_counts, min_elem_count, max_elem_count);
        elem_offset = DEFAULT_ELEM_OFFSET;
        check_values = DEFAULT_CHECK_VALUES;
        cache_ops = DEFAULT_CACHE_OPS;
        inplace = DEFAULT_INPLACE;
        ranks_per_proc = DEFAULT_RANKS_PER_PROC;
        numa_node = DEFAULT_NUMA_NODE;
        verbosity = DEFAULT_VERBOSITY;
#ifdef CCL_ENABLE_SYCL
        sycl_dev_type = DEFAULT_SYCL_DEV_TYPE;
        sycl_root_dev = DEFAULT_SYCL_ROOT_DEV;
        sycl_mem_type = DEFAULT_SYCL_MEM_TYPE;
        sycl_usm_type = DEFAULT_SYCL_USM_TYPE;
        sycl_queue_type = DEFAULT_SYCL_QUEUE_TYPE;
#endif // CCL_ENABLE_SYCL
        coll_names = tokenize<std::string>(DEFAULT_COLL_LIST, ',');
        dtypes = tokenize<std::string>(DEFAULT_DTYPES_LIST, ',');
        reductions = tokenize<std::string>(DEFAULT_REDUCTIONS_LIST, ',');
        csv_filepath = std::string(DEFAULT_CSV_FILEPATH);

        min_elem_count_set = false;
        max_elem_count_set = false;
        elem_counts_set = false;
        show_additional_info = DEFAULT_EXT_VALUES;
    }
} user_options_t;

template <class Dtype>
ccl::datatype get_ccl_dtype() {
    return ccl::native_type_info<typename std::remove_pointer<Dtype>::type>::dtype;
}

template <class T>
size_t cast_to_size_t(T v) {
    return static_cast<size_t>(v);
}

template <>
size_t cast_to_size_t<ccl::bfloat16>(ccl::bfloat16 v) {
    return static_cast<size_t>(v.get_data());
}

template <>
size_t cast_to_size_t<ccl::float16>(ccl::float16 v) {
    return static_cast<size_t>(v.get_data());
}

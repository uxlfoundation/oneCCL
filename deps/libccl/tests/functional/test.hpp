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

#include <algorithm>
#include <chrono>
#include <functional>
#include <random>
#include <vector>

#include "oneapi/ccl.hpp"
#include "conf.hpp"
#include "transport.hpp"
#include "utils.hpp"

bool is_lp_datatype(ccl_data_type dtype);

template <typename T>
struct test_operation {
    test_param param;

    size_t elem_count;
    size_t buffer_count;
    ccl::datatype datatype;
    size_t datatype_size;
    ccl::reduction reduction;

    int comm_size;
    int comm_rank;

    std::vector<size_t> buf_indexes;

    float first_fp_coeff;
    float second_fp_coeff;

#ifdef CCL_ENABLE_SYCL
    std::vector<aligned_vector<T>> send_bufs;
    std::vector<aligned_vector<T>> recv_bufs;
    std::vector<void*> device_send_bufs;
    std::vector<void*> device_recv_bufs;
#else // CCL_ENABLE_SYCL
    std::vector<std::vector<T>> send_bufs;
    std::vector<std::vector<T>> recv_bufs;
#endif // CCL_ENABLE_SYCL

    std::vector<ccl::event> events;
    ccl::string_class match_id;

    test_operation(test_param param)
            : param(param),
              elem_count(get_elem_count(param)),
              buffer_count(get_buffer_count(param)),
              datatype(get_ccl_datatype(param)),
              datatype_size(get_ccl_datatype_size(param)),
              reduction(get_ccl_reduction(param)) {
        comm_size = transport_data::instance().get_comm().size();
        comm_rank = transport_data::instance().get_comm().rank();
        buf_indexes.resize(buffer_count);

        // fp coefficients are required to avoid overflow
        // or accuracy issue on large scale
        if (param.datatype == DATATYPE_BFLOAT16) {
            first_fp_coeff = 0.5;
            second_fp_coeff = 1;
        }
        else if (param.datatype == DATATYPE_FLOAT16) {
            first_fp_coeff = 0.01;
            second_fp_coeff = 0.01;
        }
        else {
            first_fp_coeff = 1;
            second_fp_coeff = 1;
        }
    }

    template <class coll_attr_type>
    void prepare_attr(coll_attr_type& coll_attr, size_t idx);

    std::string create_match_id(size_t buf_idx);
    size_t generate_priority_value(size_t buf_idx);
    void define_start_order(std::default_random_engine& rand_engine);

    bool complete_events();
    bool complete_event(ccl::event& e);

    const test_param& get_param() {
        return param;
    }

    void print(std::ostream& output);

    void* get_send_buf(size_t buf_idx) {
#ifdef CCL_ENABLE_SYCL
        return device_send_bufs[buf_idx];
#else // CCL_ENABLE_SYCL
        return send_bufs[buf_idx].data();
#endif // CCL_ENABLE_SYCL
    }

    void* get_recv_buf(size_t buf_idx) {
#ifdef CCL_ENABLE_SYCL
        return device_recv_bufs[buf_idx];
#else // CCL_ENABLE_SYCL
        return recv_bufs[buf_idx].data();
#endif // CCL_ENABLE_SYCL
    }

    size_t get_check_step(size_t elem_idx) {
        size_t step;
        if (param.size_type == SIZE_SMALL)
            step = 1;
        else if (param.size_type == SIZE_MEDIUM)
            step = 4;
        else
            step = 32;

        if ((step > 1) && (elem_idx + step >= elem_count)) {
            /* 
                to process tail elements
                when elem_count is not dividable by step
            */
            step = 1;
        }

        return step;
    }
};

template <typename T>
class base_test {
public:
    char err_message[ERR_MESSAGE_MAX_LEN]{};
    std::random_device rand_device;
    std::default_random_engine rand_engine;

    char* get_err_message() {
        return err_message;
    }

    base_test();
    virtual ~base_test() = default;

    void alloc_buffers_base(test_operation<T>& op);
    virtual void alloc_buffers(test_operation<T>& op);
    void free_buffers(test_operation<T>& op);

    virtual void fill_send_buffers(test_operation<T>& op);
    virtual void fill_recv_buffers(test_operation<T>& op);
    void change_buffers(test_operation<T>& op);

#ifdef CCL_ENABLE_SYCL
    void copy_to_device_send_buffers(test_operation<T>& op);
    void copy_from_device_recv_buffers(test_operation<T>& op);
#endif // CCL_ENABLE_SYCL

    virtual T calculate_reduce_value(test_operation<T>& op, size_t buf_idx, size_t elem_idx);

    int run(test_operation<T>& op);
    virtual void run_derived(test_operation<T>& op) = 0;

    virtual int check(test_operation<T>& op) = 0;
    virtual int check_error(test_operation<T>& op, T expected, size_t buf_idx, size_t elem_idx);
};

class MainTest : public ::testing ::TestWithParam<test_param> {
    template <typename T>
    int run(test_param param);

public:
    int test(test_param& param) {
        switch (param.datatype) {
            case DATATYPE_INT8: return run<int8_t>(param);
            case DATATYPE_UINT8: return run<uint8_t>(param);
            case DATATYPE_INT16: return run<int16_t>(param);
            case DATATYPE_UINT16: return run<uint16_t>(param);
            case DATATYPE_INT32: return run<int32_t>(param);
            case DATATYPE_UINT32: return run<uint32_t>(param);
            case DATATYPE_INT64: return run<int64_t>(param);
            case DATATYPE_UINT64: return run<uint64_t>(param);
            case DATATYPE_FLOAT16:
                // TODO: use C++23 std::float16_t. Then, remove convert_fp32_to_lp and all of lp_impl.hpp
                return run<float>(param);
            case DATATYPE_FLOAT32: return run<float>(param);
            case DATATYPE_FLOAT64: return run<double>(param);
            case DATATYPE_BFLOAT16:
                // TODO: use C++23 std::bfloat16_t. Then, remove convert_fp32_to_lp and all of lp_impl.hpp
                return run<float>(param);
            default:
                EXPECT_TRUE(false) << "Unexpected data type: " << param.datatype;
                return TEST_FAILURE;
        }
    }
};

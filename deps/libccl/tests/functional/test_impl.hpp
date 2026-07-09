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

#include <math.h>

#include "lp.hpp"
#include "test.hpp"
#include "transport.hpp"

#ifdef CCL_ENABLE_SYCL
void* alloc_buffer(size_t bytes) {
    auto& allocator = transport_data::instance().get_allocator();
    return allocator.allocate(bytes, sycl::usm::alloc::device);
}

void free_buffer(void* ptr) {
    auto& allocator = transport_data::instance().get_allocator();
    allocator.deallocate(static_cast<char*>(ptr));
}

void copy_buffer(void* dst, void* src, size_t bytes) {
    transport_data::instance().get_stream().get_native().memcpy(dst, src, bytes).wait();
}

void fill_buffer(void* ptr, int value, size_t bytes) {
    transport_data::instance().get_stream().get_native().memset(ptr, value, bytes).wait();
}
#endif // CCL_ENABLE_SYCL

template <typename T>
template <class coll_attr_type>
void test_operation<T>::prepare_attr(coll_attr_type& attr, size_t idx) {
    attr.template set<ccl::operation_attr_id::priority>(generate_priority_value(idx));
    attr.template set<ccl::operation_attr_id::to_cache>(param.cache_type == CACHE_TRUE ? true
                                                                                       : false);

    char* unordered_coll_env = getenv("CCL_UNORDERED_COLL");
    if (unordered_coll_env && atoi(unordered_coll_env) == 1) {
        attr.template set<ccl::operation_attr_id::synchronous>(false);
    }
    else {
        attr.template set<ccl::operation_attr_id::synchronous>(
            param.sync_type == SYNC_TRUE ? true : false);
    }

    match_id = create_match_id(idx);
    attr.template set<ccl::operation_attr_id::match_id>(match_id);
}

template <typename T>
std::string test_operation<T>::create_match_id(size_t buf_idx) {
    return (std::to_string(param.datatype) + std::to_string(param.size_type) +
            std::to_string(param.buf_count_type) + std::to_string(param.place_type) +
            std::to_string(param.start_order) + std::to_string(param.complete_order) +
            std::to_string(param.complete_type) + std::to_string(param.cache_type) +
            std::to_string(param.sync_type) + std::to_string(param.reduction) +
            std::to_string(buf_idx) + std::to_string(comm_size));
}

template <typename T>
void test_operation<T>::define_start_order(std::default_random_engine& rand_engine) {
    std::iota(buf_indexes.begin(), buf_indexes.end(), 0);
    if (param.start_order == ORDER_INDIRECT) {
        std::reverse(buf_indexes.begin(), buf_indexes.end());
    }
    else if (param.start_order == ORDER_RANDOM) {
        char* unordered_coll_env = getenv("CCL_UNORDERED_COLL");
        if (unordered_coll_env && atoi(unordered_coll_env) == 1) {
            std::shuffle(buf_indexes.begin(), buf_indexes.end(), rand_engine);
        }
    }
}

template <typename T>
bool test_operation<T>::complete_events() {
    size_t idx, msg_idx;
    size_t completions = 0;
    std::vector<bool> msg_completions(buffer_count, false);

    while (completions < buffer_count) {
        for (idx = 0; idx < buffer_count; idx++) {
            if (param.complete_order == ORDER_DIRECT) {
                msg_idx = idx;
            }
            else if (param.complete_order == ORDER_INDIRECT) {
                msg_idx = (buffer_count - idx - 1);
            }
            else if (param.complete_order == ORDER_RANDOM) {
                msg_idx = rand() % buffer_count;
            }
            else {
                msg_idx = idx;
            }

            if (msg_completions[msg_idx])
                continue;

            if (complete_event(events[msg_idx])) {
                completions++;
                msg_completions[msg_idx] = true;
            }
        }
    }

    events.clear();

    return TEST_SUCCESS;
}

template <typename T>
bool test_operation<T>::complete_event(ccl::event& e) {
    if (param.complete_type == COMPLETE_TEST) {
        return e.test();
    }
    else if (param.complete_type == COMPLETE_WAIT) {
        e.wait();
        return true;
    }
    else {
        ASSERT(0, "unexpected completion type %d", param.complete_type);
        return false;
    }
}

template <typename T>
size_t test_operation<T>::generate_priority_value(size_t buf_idx) {
    return buf_idx++;
}

template <typename T>
void test_operation<T>::print(std::ostream& output) {
    output << "test op:\n"
           << param << "\ncomm_size: " << comm_size << "\ncomm_rank: " << comm_rank
           << "\nelem_count: " << elem_count << "\nbuffer_count: " << buffer_count
           << "\nmatch_id: " << match_id << "\n-------------\n"
           << std::endl;
}

template <typename T>
base_test<T>::base_test() {
    memset(err_message, '\0', ERR_MESSAGE_MAX_LEN);
    rand_engine = std::default_random_engine{ rand_device() };
}

template <typename T>
int base_test<T>::check_error(test_operation<T>& op, T expected, size_t buf_idx, size_t elem_idx) {
    double max_error = 0;
    double precision = 0;

    if (op.param.datatype == DATATYPE_FLOAT16) {
        precision = 2 * FP16_PRECISION;
    }
    else if (op.param.datatype == DATATYPE_FLOAT32) {
        precision = FP32_PRECISION;
    }
    else if (op.param.datatype == DATATYPE_FLOAT64) {
        precision = FP64_PRECISION;
    }
    else if (op.param.datatype == DATATYPE_BFLOAT16) {
        precision = 2 * BF16_PRECISION;
    }

    if (precision) {
        if (op.comm_size == 1) {
            max_error = precision;
        }
        else {
            /* https://www.mcs.anl.gov/papers/P4093-0713_1.pdf */
            double log_base2 = log(op.comm_size) / log(2);
            double g = (log_base2 * precision) / (1 - (log_base2 * precision));
            max_error = g * expected;
        }
    }

    if (std::fabs(max_error) <
        std::fabs((double)expected - (double)op.recv_bufs[buf_idx][elem_idx])) {
        printf("[%d] got op.recvBuf[%zu][%zu] = %0.7f, but expected = %0.7f, "
               "max_error = %0.10f, precision = %0.7f\n",
               op.comm_rank,
               buf_idx,
               elem_idx,
               (double)op.recv_bufs[buf_idx][elem_idx],
               (double)expected,
               (double)max_error,
               precision);
        return TEST_FAILURE;
    }

    return TEST_SUCCESS;
}

template <typename T>
void base_test<T>::free_buffers(test_operation<T>& op) {
    op.send_bufs.clear();
    op.recv_bufs.clear();

#ifdef CCL_ENABLE_SYCL
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        free_buffer(op.device_send_bufs[buf_idx]);
        free_buffer(op.device_recv_bufs[buf_idx]);
    }
#endif // CCL_ENABLE_SYCL
}

template <typename T>
void base_test<T>::alloc_buffers_base(test_operation<T>& op) {
    op.send_bufs.resize(op.buffer_count);
    op.recv_bufs.resize(op.buffer_count);
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        op.send_bufs[buf_idx].resize(op.elem_count * op.comm_size);
        op.recv_bufs[buf_idx].resize(op.elem_count * op.comm_size);
    }

#ifdef CCL_ENABLE_SYCL
    op.device_send_bufs.resize(op.buffer_count);
    op.device_recv_bufs.resize(op.buffer_count);
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        op.device_send_bufs[buf_idx] = alloc_buffer(op.elem_count * sizeof(T) * op.comm_size);
        op.device_recv_bufs[buf_idx] = alloc_buffer(op.elem_count * sizeof(T) * op.comm_size);
    }
#endif // CCL_ENABLE_SYCL
}

template <typename T>
void base_test<T>::alloc_buffers(test_operation<T>& op) {}

template <typename T>
void base_test<T>::fill_send_buffers(test_operation<T>& op) {
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        for (size_t elem_idx = 0; elem_idx < op.send_bufs[buf_idx].size(); elem_idx++) {
            op.send_bufs[buf_idx][elem_idx] = op.comm_rank + buf_idx;

            if (op.param.reduction == REDUCTION_PROD) {
                op.send_bufs[buf_idx][elem_idx] += 1;
            }
        }
    }
}

template <typename T>
void base_test<T>::fill_recv_buffers(test_operation<T>& op) {
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        if (op.param.place_type == PLACE_IN) {
            std::copy(op.send_bufs[buf_idx].begin(),
                      op.send_bufs[buf_idx].end(),
                      op.recv_bufs[buf_idx].begin());
        }
        else {
            std::fill(op.recv_bufs[buf_idx].begin(), op.recv_bufs[buf_idx].end(), (T)SOME_VALUE);
        }
    }
}

template <typename T>
T base_test<T>::calculate_reduce_value(test_operation<T>& op, size_t buf_idx, size_t elem_idx) {
    T expected = 0;
    switch (op.param.reduction) {
        case REDUCTION_SUM:
            expected = (op.comm_size * (op.comm_size - 1)) / 2 + op.comm_size * buf_idx;
            break;
        case REDUCTION_PROD:
            expected = 1;
            for (int rank = 0; rank < op.comm_size; rank++) {
                expected *= rank + buf_idx + 1;
            }
            break;
        case REDUCTION_MIN: expected = (T)(buf_idx); break;
        case REDUCTION_MAX: expected = (T)(op.comm_size - 1 + buf_idx); break;
        default: ASSERT(0, "unexpected reduction %d", op.param.reduction); break;
    }
    return expected;
}

template <>
void base_test<float>::fill_send_buffers(test_operation<float>& op) {
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        for (size_t elem_idx = 0; elem_idx < op.send_bufs[buf_idx].size(); elem_idx++) {
            op.send_bufs[buf_idx][elem_idx] =
                op.first_fp_coeff * op.comm_rank + op.second_fp_coeff * buf_idx;
            if (op.param.reduction == REDUCTION_PROD) {
                op.send_bufs[buf_idx][elem_idx] += 1;
            }
        }
    }
}

template <>
float base_test<float>::calculate_reduce_value(test_operation<float>& op,
                                               size_t buf_idx,
                                               size_t elem_idx) {
    float expected = 0;
    switch (op.param.reduction) {
        case REDUCTION_SUM:
            expected = op.comm_size *
                       (op.first_fp_coeff * (op.comm_size - 1) / 2 + op.second_fp_coeff * buf_idx);
            break;
        case REDUCTION_PROD:
            expected = 1;
            for (int rank = 0; rank < op.comm_size; rank++) {
                expected *= op.first_fp_coeff * rank + op.second_fp_coeff * buf_idx + 1;
            }
            break;
        case REDUCTION_MIN: expected = op.second_fp_coeff * buf_idx; break;
        case REDUCTION_MAX:
            expected = op.first_fp_coeff * (op.comm_size - 1) + op.second_fp_coeff * buf_idx;
            break;
        default: ASSERT(0, "unexpected reduction %d", op.param.reduction); break;
    }
    return expected;
}

template <typename T>
void base_test<T>::change_buffers(test_operation<T>& op) {
    char* dynamic_pointer_env = getenv("CCL_TEST_DYNAMIC_POINTER");
    if (dynamic_pointer_env && atoi(dynamic_pointer_env) == 1) {
        void* send_buf = op.send_bufs[0].data();
        void* recv_buf = op.recv_bufs[0].data();
        /*
            create deep copy of vector with buffers and swap it with original one
            as result buffers in updated vector will have original content
            but in new memory locations
        */
#ifdef CCL_ENABLE_SYCL
        using vector_t = aligned_vector<T>;
#else // CCL_ENABLE_SYCL
        using vector_t = std::vector<T>;
#endif // CCL_ENABLE_SYCL

        std::vector<vector_t>(op.send_bufs.begin(), op.send_bufs.end()).swap(op.send_bufs);
        std::vector<vector_t>(op.recv_bufs.begin(), op.recv_bufs.end()).swap(op.recv_bufs);
        void* new_send_buf = op.send_bufs[0].data();
        void* new_recv_buf = op.recv_bufs[0].data();
        ASSERT(send_buf != new_send_buf, "send buffers should differ");
        ASSERT(recv_buf != new_recv_buf, "recv buffers should differ");

#ifdef CCL_ENABLE_SYCL
        /* do regular reallocation */
        void* device_send_buf = op.device_send_bufs[0];
        void* device_recv_buf = op.device_recv_bufs[0];
        std::vector<void*> new_device_send_bufs(op.buffer_count);
        std::vector<void*> new_device_recv_bufs(op.buffer_count);
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            new_device_send_bufs[buf_idx] = alloc_buffer(op.elem_count * sizeof(T) * op.comm_size);
            new_device_recv_bufs[buf_idx] = alloc_buffer(op.elem_count * sizeof(T) * op.comm_size);
        }
        for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
            free_buffer(op.device_send_bufs[buf_idx]);
            free_buffer(op.device_recv_bufs[buf_idx]);
            op.device_send_bufs[buf_idx] = new_device_send_bufs[buf_idx];
            op.device_recv_bufs[buf_idx] = new_device_recv_bufs[buf_idx];
        }
        void* new_device_send_buf = op.device_send_bufs[0];
        void* new_device_recv_buf = op.device_recv_bufs[0];
        ASSERT(device_send_buf != new_device_send_buf, "device send buffers should differ");
        ASSERT(device_recv_buf != new_device_recv_buf, "device recv buffers should differ");
#endif // CCL_ENABLE_SYCL
    }
}

#ifdef CCL_ENABLE_SYCL

template <typename T>
void base_test<T>::copy_to_device_send_buffers(test_operation<T>& op) {
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
#ifdef TEST_CCL_BCAST
        void* host_buf = op.recv_bufs[buf_idx].data();
        void* device_buf = op.device_recv_bufs[buf_idx];
#else // TEST_CCL_BCAST
        void* host_buf = (op.param.place_type == PLACE_IN) ? op.recv_bufs[buf_idx].data()
                                                           : op.send_bufs[buf_idx].data();
        void* device_buf = (op.param.place_type == PLACE_IN) ? op.device_recv_bufs[buf_idx]
                                                             : op.device_send_bufs[buf_idx];
#endif // TEST_CCL_BCAST
        size_t bytes = op.send_bufs[buf_idx].size() * sizeof(T);
        copy_buffer(device_buf, host_buf, bytes);
    }
}

template <typename T>
void base_test<T>::copy_from_device_recv_buffers(test_operation<T>& op) {
    for (size_t buf_idx = 0; buf_idx < op.buffer_count; buf_idx++) {
        copy_buffer(op.recv_bufs[buf_idx].data(),
                    op.device_recv_bufs[buf_idx],
                    op.recv_bufs[buf_idx].size() * sizeof(T));
    }
}
#endif // CCL_ENABLE_SYCL

template <typename T>
int base_test<T>::run(test_operation<T>& op) {
    static int run_counter = 0;
    size_t iter = 0, result = 0;

    char* algo = getenv(ALGO_SELECTION_ENV);
    if (algo)
        std::cout << ALGO_SELECTION_ENV << " = " << algo << "\n";
    std::cout << op.param << "\n";

    /*
        Buffer management logic for single operation
        SYCL-specific logic is marked with (*)
        LP-specific logic is marked with (**)

        1. alloc host send and recv buffers
        2. alloc device send and recv buffers (*)
        3. fill host send and recv buffers
        4. do in-place FP32->LP cast for host send buffer (**)
        5. copy from host send buffer into device send buffer (*)
        6. invoke comm operation on host or device (*) send and recv buffers
        7. copy device recv buffer into host recv buffer (*)
        8. do in-place LP->FP32 cast for host recv buffer (**)
        9. check result correctness on host recv buffer
        10. free host send and recv buffers
        11. free device send and recv buffers (*)
    */

    try {
        alloc_buffers_base(op);
        alloc_buffers(op);
        for (iter = 0; iter < ITER_COUNT; iter++) {
            if (iter > 0) {
                change_buffers(op);
            }

            fill_send_buffers(op);
            fill_recv_buffers(op);

            if (is_lp_datatype(op.param.datatype)) {
                make_lp_prologue(op, op.comm_size * op.elem_count);
            }

#ifdef CCL_ENABLE_SYCL
            copy_to_device_send_buffers(op);
#endif // CCL_ENABLE_SYCL

            op.define_start_order(rand_engine);
            run_derived(op);
            op.complete_events();

#ifdef CCL_ENABLE_SYCL
            copy_from_device_recv_buffers(op);
#endif // CCL_ENABLE_SYCL

            if (is_lp_datatype(op.param.datatype)) {
                make_lp_epilogue(op, op.comm_size * op.elem_count);
            }

            result += check(op);
            if ((run_counter % 10) == 0) {
                ccl::barrier(transport_data::instance().get_service_comm());
            }
        }
        run_counter++;
        free_buffers(op);
    }
    catch (const std::exception& ex) {
        result += TEST_FAILURE;
        printf("WARNING! %s iter number: %zu", ex.what(), iter);
    }

    return result;
}

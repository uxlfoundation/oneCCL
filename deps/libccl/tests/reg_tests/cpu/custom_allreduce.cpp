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

#include <string>

#include "base.hpp"

#define CUSTOM_BASE_DTYPE        int
#define CUSTOM_REPEAT_COUNT      8
#define CUSTOM_BASE_DTYPE_FORMAT "%d"
#define CUSTOM_DTYPE_SIZE        (CUSTOM_REPEAT_COUNT * sizeof(CUSTOM_BASE_DTYPE))

int size, rank;
ccl::datatype custom_dtype;
ccl::string_class global_match_id;

typedef void (*expected_fn_t)(void*, size_t);
typedef void (*fill_fn_t)(void*, size_t, size_t);
typedef int (*check_fn_t)(void*, size_t, expected_fn_t);

#define RUN_COLLECTIVE(start_cmd, fill_fn, check_fn, expected_fn, name) \
    do { \
        double t1 = 0, t2 = 0, t = 0; \
        for (int iter_idx = 0; iter_idx < ITERS; iter_idx++) { \
            global_match_id = match_id; \
            fill_fn(send_buf, MSG_SIZE_COUNT, rank + 1); \
            fill_fn(recv_buf, MSG_SIZE_COUNT, 1); \
            t1 = when(); \
            auto req = start_cmd; \
            req.wait(); \
            t2 = when(); \
            t += (t2 - t1); \
        } \
        check_values(recv_buf, MSG_SIZE_COUNT, check_fn, expected_fn); \
        printf("[%d] avg %s time: %8.2lf us\n", rank, name, t / ITERS); \
        fflush(stdout); \
    } while (0)

/* primitive operations for custom datatype */
void custom_sum(void* in_elem, void* inout_elem) {
    for (size_t idx = 0; idx < CUSTOM_REPEAT_COUNT; idx++) {
        ((CUSTOM_BASE_DTYPE*)inout_elem)[idx] += ((CUSTOM_BASE_DTYPE*)in_elem)[idx];
    }
}

void custom_set(void* elem, size_t base_value) {
    for (size_t idx = 0; idx < CUSTOM_REPEAT_COUNT; idx++) {
        ((CUSTOM_BASE_DTYPE*)elem)[idx] = (CUSTOM_BASE_DTYPE)(base_value);
    }
}

void custom_zeroize(void* elem) {
    memset(elem, 0, CUSTOM_DTYPE_SIZE);
}

int custom_compare(void* elem1, void* elem2) {
    return memcmp(elem1, elem2, CUSTOM_DTYPE_SIZE);
}

void custom_print(void* elem) {
    char print_buf[1024] = { 0 };
    size_t print_bytes = 0;

    for (size_t idx = 0; idx < CUSTOM_REPEAT_COUNT; idx++) {
        print_bytes += sprintf(
            print_buf + print_bytes, CUSTOM_BASE_DTYPE_FORMAT " ", ((CUSTOM_BASE_DTYPE*)elem)[idx]);
    }
    printf("%s\n", print_buf);
}

void fill_value_float(void* buf, size_t count, size_t base_value) {
    for (size_t idx = 0; idx < count; idx++) {
        ((float*)buf)[idx] = (float)base_value;
    }
}

void fill_value_custom(void* buf, size_t count, size_t base_value) {
    for (size_t idx = 0; idx < count; idx++) {
        custom_set(((char*)buf) + idx * CUSTOM_DTYPE_SIZE, base_value);
    }
}

void check_values(void* buf, size_t count, check_fn_t check_fn, expected_fn_t expected_fn) {
    for (size_t idx = 0; idx < MSG_SIZE_COUNT; idx++) {
        if (check_fn(buf, idx, expected_fn)) {
            ASSERT(0, "unexpected value on idx %zu", idx);
        }
    }
}

int check_value_float(void* buf, size_t idx, expected_fn_t expected_fn) {
    int res = 0;

    float expected = 0;
    expected_fn(&expected, idx);

    float got = ((float*)buf)[idx];

    if (got != expected) {
        printf("expected %f, got %f\n", expected, got);
        res = 1;
    }

    return res;
}

int check_value_custom(void* buf, size_t idx, expected_fn_t expected_fn) {
    int res = 0;

    char expected[CUSTOM_DTYPE_SIZE] = { 0 };
    expected_fn(expected, idx);

    char* got = (char*)buf + CUSTOM_DTYPE_SIZE * idx;

    if (custom_compare(expected, got)) {
        printf("expected ");
        custom_print(expected);
        printf(", got ");
        custom_print(got);
        printf("\n");
        res = 1;
    }

    return res;
}

void expected_float_1(void* elem, size_t idx) {
    *((float*)elem) = (size + 1) * ((float)size / 2);
}

void expected_float_2(void* elem, size_t idx) {
    *((float*)elem) = (size + 1) * size;
}

void expected_float_3(void* elem, size_t idx) {
    *((float*)elem) = 0;
}

void expected_float_4(void* elem, size_t idx) {
    *((float*)elem) = 2 * (size + 1) * size;
}

void expected_float_5(void* elem, size_t idx) {
    *((float*)elem) = (float)idx;
}

void expected_float_6(void* elem, size_t idx) {
    *((float*)elem) = (float)(2 * idx);
}

void expected_custom_1(void* elem, size_t idx) {
    custom_set(elem, (size + 1) * ((float)size / 2));
}

void expected_custom_2(void* elem, size_t idx) {
    custom_set(elem, (size + 1) * size);
}

void expected_custom_3(void* elem, size_t idx) {
    custom_zeroize(elem);
}

void expected_custom_4(void* elem, size_t idx) {
    custom_set(elem, 2 * (size + 1) * size);
}

void expected_custom_5(void* elem, size_t idx) {
    custom_set(elem, idx);
}

void expected_custom_6(void* elem, size_t idx) {
    custom_set(elem, 2 * idx);
}

void do_reduction_sum(const void* in_buf,
                      size_t in_count,
                      void* inout_buf,
                      size_t* out_count,
                      ccl::datatype dtype,
                      const ccl::fn_context* context) {
    size_t dtype_size;
    dtype_size = ccl::get_datatype_size(dtype);

    ASSERT((dtype == ccl::datatype::int8) || (dtype == ccl::datatype::float32) ||
               (dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(dtype));
    ASSERT(context->offset < MSG_SIZE_COUNT * dtype_size,
           "wrong offset for reduction_sum func, should be less than COUNT");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_count)
        *out_count = in_count;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (dtype == ccl::datatype::int8) {
            ((char*)inout_buf)[idx] += ((char*)in_buf)[idx];
        }
        else if (dtype == ccl::datatype::float32) {
            ((float*)inout_buf)[idx] += ((float*)in_buf)[idx];
        }
        else if (dtype == custom_dtype) {
            custom_sum((char*)in_buf + idx * CUSTOM_DTYPE_SIZE,
                       (char*)inout_buf + idx * CUSTOM_DTYPE_SIZE);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(dtype));
        }
    }
}

void do_reduction_null(const void* in_buf,
                       size_t in_count,
                       void* inout_buf,
                       size_t* out_count,
                       ccl::datatype dtype,
                       const ccl::fn_context* context) {
    size_t dtype_size;
    dtype_size = ccl::get_datatype_size(dtype);

    ASSERT((dtype == ccl::datatype::int8) || (dtype == ccl::datatype::float32) ||
               (dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(dtype));
    ASSERT(context->offset < MSG_SIZE_COUNT * dtype_size,
           "wrong offset for reduction_null func, should be less than COUNT");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_count)
        *out_count = in_count;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (dtype == ccl::datatype::int8) {
            ((char*)inout_buf)[idx] = (char)0;
        }
        else if (dtype == ccl::datatype::float32) {
            ((float*)inout_buf)[idx] = (float)0;
        }
        else if (dtype == custom_dtype) {
            custom_zeroize((char*)inout_buf + idx * CUSTOM_DTYPE_SIZE);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(dtype));
        }
    }
}

void do_reduction_custom(const void* in_buf,
                         size_t in_count,
                         void* inout_buf,
                         size_t* out_count,
                         ccl::datatype dtype,
                         const ccl::fn_context* context) {
    size_t dtype_size;
    dtype_size = ccl::get_datatype_size(dtype);

    ASSERT((dtype == ccl::datatype::float32) || (dtype == custom_dtype),
           "unexpected in_dtype %d",
           static_cast<int>(dtype));
    ASSERT(context->offset < MSG_SIZE_COUNT * dtype_size,
           "wrong offset for reduction_sum func, should be less than COUNT");
    ASSERT(!strcmp(context->match_id, global_match_id.c_str()), "wrong match_id");

    if (out_count)
        *out_count = in_count;

    size_t global_elem_offset = context->offset / dtype_size;

    for (size_t idx = 0; idx < in_count; idx++) {
        if (dtype == ccl::datatype::float32) {
            ((float*)inout_buf)[idx] = (float)(global_elem_offset + idx);
        }
        else if (dtype == custom_dtype) {
            custom_set((char*)inout_buf + idx * CUSTOM_DTYPE_SIZE, global_elem_offset + idx);
        }
        else {
            ASSERT(0, "unexpected dtype %d", static_cast<int>(dtype));
        }
    }
}

int main() {
    setenv("CCL_ATL_TRANSPORT", "ofi", 1);

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(size, rank, kvs);
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();

    float float_send_buf[MSG_SIZE_COUNT];
    float float_recv_buf[MSG_SIZE_COUNT];

    auto dt_attr =
        ccl::create_datatype_attr(ccl::attr_val<ccl::datatype_attr_id::size>(CUSTOM_DTYPE_SIZE));
    custom_dtype = ccl::register_datatype(dt_attr);

    char custom_send_buf[MSG_SIZE_COUNT * CUSTOM_DTYPE_SIZE];
    char custom_recv_buf[MSG_SIZE_COUNT * CUSTOM_DTYPE_SIZE];

    ccl::string_class base_match_id = attr.get<ccl::operation_attr_id::match_id>();
    attr.set<ccl::operation_attr_id::to_cache>(true);
    ccl::string_class match_id;

    for (size_t idx = 0; idx < 2; idx++) {
        if (rank == 0)
            printf("Running tests for %s datatype\n", (idx == 0) ? "FLOAT" : "CUSTOM");
        ccl::datatype dtype = (idx == 0) ? ccl::datatype::float32 : custom_dtype;
        void* send_buf = (idx == 0) ? (void*)float_send_buf : (void*)custom_send_buf;
        void* recv_buf = (idx == 0) ? (void*)float_recv_buf : (void*)custom_recv_buf;

        fill_fn_t fill_fn = (idx == 0) ? fill_value_float : fill_value_custom;
        check_fn_t check_fn = (idx == 0) ? check_value_float : check_value_custom;
        expected_fn_t expected_fn;

        if (idx == 0) {
            /* regular sum allreduce */
            expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
            match_id = base_match_id + "_regular_" + std::to_string(idx);
            attr.set<ccl::operation_attr_id::match_id>(match_id);
            RUN_COLLECTIVE(
                ccl::allreduce(
                    send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::sum, comm, attr),
                fill_fn,
                check_fn,
                expected_fn,
                "regular_allreduce");
        }

        /* reduction_sum */
        expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
        match_id = base_match_id + "_reduction_sum_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_sum);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_reduction_sum");

        /* reduction_null */
        if (size == 1)
            expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
        else
            expected_fn = (idx == 0) ? expected_float_3 : expected_custom_3;
        match_id = base_match_id + "_reduction_null_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_null);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_reduction_null");

        /* reduction_custom */
        if (size == 1)
            expected_fn = (idx == 0) ? expected_float_1 : expected_custom_1;
        else
            expected_fn = (idx == 0) ? expected_float_5 : expected_custom_5;
        match_id = base_match_id + "_reduction_custom_" + std::to_string(idx);
        attr.set<ccl::operation_attr_id::match_id>(match_id);
        attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)do_reduction_custom);
        RUN_COLLECTIVE(
            ccl::allreduce(
                send_buf, recv_buf, MSG_SIZE_COUNT, dtype, ccl::reduction::custom, comm, attr),
            fill_fn,
            check_fn,
            expected_fn,
            "allreduce_with_reduction_custom");
    }

    if (rank == 0)
        printf("PASSED\n");

    return 0;
}

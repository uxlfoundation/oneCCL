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

#include "base.hpp"

#define COUNT (1048576 / 256)

void custom_reduce(const void *in_buf,
                   size_t in_count,
                   void *inout_buf,
                   size_t *out_count,
                   ccl::datatype dtype,
                   const ccl::fn_context *context) {
    size_t dtype_size = ccl::get_datatype_size(dtype);

    ASSERT(dtype_size != 0, "unexpected datatype size");

    ASSERT((dtype != ccl::datatype::int8) && (dtype != ccl::datatype::uint8) &&
               (dtype != ccl::datatype::int16) && (dtype != ccl::datatype::uint16) &&
               (dtype != ccl::datatype::int32) && (dtype != ccl::datatype::uint32) &&
               (dtype != ccl::datatype::int64) && (dtype != ccl::datatype::uint64) &&
               (dtype != ccl::datatype::float16) && (dtype != ccl::datatype::float32) &&
               (dtype != ccl::datatype::float64) && (dtype != ccl::datatype::bfloat16),
           "unexpected datatype %d",
           static_cast<int>(dtype));

    for (size_t idx = 0; idx < in_count; idx++) {
        ((float *)inout_buf)[idx] += ((float *)in_buf)[idx];
    }
}

void check_allreduce(const ccl::communicator &comm) {
    const size_t max_dtype_count = 1024;

    std::vector<ccl::datatype> dtypes(max_dtype_count);
    std::vector<ccl::event> reqs(max_dtype_count);
    std::vector<std::vector<float>> send_bufs(max_dtype_count);
    std::vector<std::vector<float>> recv_bufs(max_dtype_count);

    auto dt_attr =
        ccl::create_datatype_attr(ccl::attr_val<ccl::datatype_attr_id::size>(sizeof(float)));

    for (size_t idx = 0; idx < max_dtype_count; idx++) {
        dtypes[idx] = ccl::register_datatype(dt_attr);
        send_bufs[idx].resize(COUNT, comm.rank() + 1);
        recv_bufs[idx].resize(COUNT, 0);
    }

    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
    attr.set<ccl::allreduce_attr_id::reduction_fn>((ccl::reduction_fn)custom_reduce);

    for (size_t idx = 0; idx < max_dtype_count; idx++) {
        reqs[idx] = ccl::allreduce(send_bufs[idx].data(),
                                   recv_bufs[idx].data(),
                                   COUNT,
                                   dtypes[idx],
                                   ccl::reduction::custom,
                                   comm,
                                   attr);
    }

    for (size_t idx = 0; idx < max_dtype_count; idx++) {
        reqs[idx].wait();
    }

    float expected = (comm.size() + 1) * ((float)comm.size() / 2);

    for (size_t idx = 0; idx < max_dtype_count; idx++) {
        for (size_t elem_idx = 0; elem_idx < recv_bufs[idx].size(); ++elem_idx) {
            if (recv_bufs[idx][elem_idx] != expected) {
                ASSERT(0,
                       "buf_idx %zu, elem_idx %zu: expected %f, got %f",
                       idx,
                       elem_idx,
                       expected,
                       recv_bufs[idx][elem_idx]);
            }
        }
    }

    for (size_t idx = 0; idx < max_dtype_count; idx++) {
        ccl::deregister_datatype(dtypes[idx]);
    }
}

void check_create_and_free() {
    auto attr = ccl::create_datatype_attr(ccl::attr_val<ccl::datatype_attr_id::size>(1));

    const size_t max_dtype_count = 16 * 1024;
    const size_t iter_count = 16;
    std::vector<ccl::datatype> dtypes(max_dtype_count);

    for (size_t iter = 0; iter < iter_count; iter++) {
        dtypes.clear();

        for (size_t idx = 0; idx < max_dtype_count; idx++) {
            attr.set<ccl::datatype_attr_id::size>(idx + 1);
            dtypes[idx] = ccl::register_datatype(attr);
            size_t dtype_size = ccl::get_datatype_size(dtypes[idx]);

            if (dtype_size != (idx + 1)) {
                printf("FAILED\n");
                throw std::runtime_error("unexpected datatype size: got " +
                                         std::to_string(dtype_size) + " expected " +
                                         std::to_string((idx + 1)));
            }
        }

        for (size_t idx = 0; idx < max_dtype_count; idx++) {
            ccl::deregister_datatype(dtypes[idx]);
        }
    }
}

int main() {
    /**
     * The example only works with CCL_ATL_TRANSPORT=ofi
     */
    setenv("CCL_ATL_TRANSPORT", "ofi", 0);

    ccl::init();

    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(size, rank, kvs);

    PRINT_BY_ROOT(comm, "\n- Check register and unregister");
    check_create_and_free();
    PRINT_BY_ROOT(comm, "PASSED");

    /* ofi atl is needed for this check */
    PRINT_BY_ROOT(comm, "\n- Check allreduce with custom datatype");
    check_allreduce(comm);
    PRINT_BY_ROOT(comm, "PASSED");

    return 0;
}

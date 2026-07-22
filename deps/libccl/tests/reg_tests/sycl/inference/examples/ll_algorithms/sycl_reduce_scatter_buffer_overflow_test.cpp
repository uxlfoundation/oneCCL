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

#include "oneapi/ccl.hpp"

#include <mpi.h>

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cassert>
#include <thread>
#include <vector>
#include <iomanip>

using namespace std;
using namespace sycl;

static void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized) {
        MPI_Finalize();
    }
}

static std::vector<sycl::device> select_devices() {
    std::vector<sycl::device> devices;
    auto platform_list = sycl::platform::get_platforms();
    for (const auto &platform : platform_list) {
        auto platform_name = platform.get_info<sycl::info::platform::name>();
        bool is_level_zero = platform_name.find("Level-Zero") != std::string::npos;
        if (is_level_zero) {
            std::cout << "Platform_name is:  " << platform_name << std::endl;
            auto device_list = platform.get_devices();
            for (const auto &device : device_list) {
                if (device.is_gpu()) {
                    devices.push_back(device);
                }
            }
        }
    }

    return devices;
}

template <typename T>
void write_by_byte(void *target, T value, size_t offset_bytes) {
    uint8_t *target_bytes = reinterpret_cast<uint8_t *>(target) + offset_bytes;
    uint8_t *src_bytes = reinterpret_cast<uint8_t *>(&value);
    for (size_t i = 0; i < sizeof(T); ++i) {
        target_bytes[i] = src_bytes[i];
    }
}

template <typename T>
T read_by_byte(void *src, size_t offset_bytes) {
    T ret{};
    uint8_t *target_bytes = reinterpret_cast<uint8_t *>(&ret);
    uint8_t *src_bytes = reinterpret_cast<uint8_t *>(src) + offset_bytes;
    for (size_t i = 0; i < sizeof(T); ++i) {
        target_bytes[i] = src_bytes[i];
    }
    return ret;
}

void tests_of_tests() {
    int32_t a = 0x7ABBCCDD;
    int32_t b = 0x6EFF0011;
    int32_t c = 0x22334455;

    std::vector<uint8_t> d(3 * 4);

    write_by_byte<int32_t>(d.data(), a, 0);
    write_by_byte<int32_t>(d.data(), b, 4);
    write_by_byte<int32_t>(d.data(), c, 8);

    // the test below works on little endian
    assert(d[3] == 0x7A);
    assert(d[2] == 0xBB);
    assert(d[1] == 0xCC);
    assert(d[0] == 0xDD);
    assert(d[7] == 0x6E);
    assert(d[6] == 0xFF);
    assert(d[5] == 0x00);
    assert(d[4] == 0x11);
    assert(d[11] == 0x22);
    assert(d[10] == 0x33);
    assert(d[9] == 0x44);
    assert(d[8] == 0x55);

    int32_t aa = read_by_byte<int32_t>(d.data(), 0);
    int32_t bb = read_by_byte<int32_t>(d.data(), 4);
    int32_t cc = read_by_byte<int32_t>(d.data(), 8);

    assert(aa == 0x7ABBCCDD);
    assert(bb == 0x6EFF0011);
    assert(cc == 0x22334455);
}

int sum_reduce_scatter(size_t element_count,
                       size_t offset_bytes,
                       queue &q,
                       ccl::communicator &comm,
                       ccl::stream &stream,
                       int comm_size,
                       int rank) {
    uint32_t send_canary[] = { 0xFFFFFFFF, 0xDDDDDDDD, 0xBBBBBBBB, 0x99999999,
                               0x77777777, 0x55555555, 0x44444444, 0x22222222,
                               0xFFFFFFFF, 0xDDDDDDDD, 0xBBBBBBBB, 0x99999999,
                               0x77777777, 0x55555555, 0x44444444, 0x22222222 };
    uint32_t recv_canary[] = { 0xEEEEEEEE, 0xCCCCCCCC, 0xAAAAAAAA, 0x88888888,
                               0x66666666, 0x44444444, 0x33333333, 0x11111111,
                               0xEEEEEEEE, 0xCCCCCCCC, 0xAAAAAAAA, 0x88888888,
                               0x66666666, 0x44444444, 0x33333333, 0x11111111 };

    size_t canary_count = sizeof(send_canary) / sizeof(send_canary[0]);
    size_t canary_size_bytes = canary_count * sizeof(int32_t);
    size_t send_element_count = element_count * comm_size;
    size_t recv_main_data_bytes = element_count * sizeof(int32_t);
    size_t send_main_data_bytes = send_element_count * sizeof(int32_t);
    size_t send_allocation_size = offset_bytes + send_main_data_bytes + canary_size_bytes;
    size_t recv_allocation_size = offset_bytes + recv_main_data_bytes + canary_size_bytes;

    std::vector<uint8_t> send_buf_cpu(send_allocation_size, 0);
    std::vector<uint8_t> recv_buf_cpu(recv_allocation_size, 0);
    std::vector<uint8_t> expected_recv_cpu(recv_allocation_size, 0);

    // Allocate device buffers
    uint8_t *send_buf =
        sycl::aligned_alloc_device<uint8_t>(4 * 1024, send_allocation_size, q); // +1 for safety
    uint8_t *recv_buf = sycl::aligned_alloc_device<uint8_t>(4 * 1024, recv_allocation_size, q);

    // Initialize front canary
    for (size_t i = 0; i < offset_bytes; ++i) {
        send_buf_cpu[i] = 0xAA;
        expected_recv_cpu[i] = 0xFF;
    }

    // Initialize send data: each rank sends different data to each destination
    // Data for src is: (rank * element_count + element_index)
    for (size_t i = 0; i < send_element_count; ++i) {
        size_t t_offset_bytes = i * sizeof(int32_t) + offset_bytes;
        write_by_byte<int32_t>(send_buf_cpu.data(), 1 + comm_size + rank + i, t_offset_bytes);
    }

    for (size_t i = 0; i < element_count; ++i) {
        size_t t_offset_bytes = i * sizeof(int32_t) + offset_bytes;
        size_t idx = i + rank * element_count;
        size_t expected_recv =
            comm_size + comm_size * comm_size + comm_size * (comm_size - 1) / 2 + idx * comm_size;

        write_by_byte<int32_t>(expected_recv_cpu.data(), expected_recv, t_offset_bytes);
    }

    // Initialize back canary
    size_t send_canary_offset = send_main_data_bytes + offset_bytes;
    size_t recv_canary_offset = recv_main_data_bytes + offset_bytes;
    for (size_t i = 0; i < canary_size_bytes; i += sizeof(int32_t)) {
        size_t canary_idx = i / sizeof(int32_t);
        size_t current_send_offset = send_canary_offset + i;
        size_t current_recv_offset = recv_canary_offset + i;

        if (canary_idx < sizeof(send_canary) / sizeof(send_canary[0])) {
            write_by_byte<int32_t>(
                send_buf_cpu.data(), send_canary[canary_idx], current_send_offset);
            write_by_byte<int32_t>(
                expected_recv_cpu.data(), recv_canary[canary_idx], current_recv_offset);
        }
    }

    // Copy data to device
    q.memcpy(send_buf, send_buf_cpu.data(), send_buf_cpu.size()).wait();
    q.memset(recv_buf, static_cast<uint8_t>(0x00), recv_allocation_size).wait();
    q.memset(recv_buf, static_cast<uint8_t>(0xFF), offset_bytes)
        .wait(); // Initialize front canary with 0xFF
    q.memcpy(recv_buf + offset_bytes + recv_main_data_bytes, recv_canary, canary_size_bytes).wait();

    // Execute reduce scatter
    auto attr = ccl::create_operation_attr<ccl::reduce_scatter_attr>();

    ccl::reduce_scatter((void *)(send_buf + offset_bytes),
                        (void *)(recv_buf + offset_bytes),
                        element_count,
                        ccl::datatype::int32,
                        ccl::reduction::sum,
                        comm,
                        stream,
                        attr)
        .wait();

    // Copy result back to host
    q.memcpy(recv_buf_cpu.data(), recv_buf, recv_buf_cpu.size()).wait();

    // Verify results
    int ret = 0;
    {
        size_t i;
        for (i = 0; i < recv_allocation_size; ++i) {
            int32_t received = recv_buf_cpu[i];
            int32_t expected = expected_recv_cpu[i];
            if (received != expected) {
                std::cout << "Size " << element_count << ", offset bytes " << offset_bytes
                          << " failure at idx [" << i << "], element [" << i / sizeof(int32_t)
                          << "]; found [" << std::hex << received << "] instead of [" << expected
                          << "]" << std::endl;
                std::cout << "FAILED\n";
                ret = 1;
                break;
            }
        }

        if (i == recv_allocation_size) {
            std::cout << "element_count " << element_count << ", offset bytes " << offset_bytes
                      << std::endl
                      << "PASSED\n";
        }
        else if (rank == 0) {
            // dump buffer
            std::cout << "rank " << rank << " out buffer contents: " << std::endl;
            for (size_t t = 0; t < recv_allocation_size; ++t) {
                std::cout << std::hex << (int)recv_buf_cpu[t] << " ";
            }
            std::cout << std::endl;
        }
    }

    sycl::free(send_buf, q);
    sycl::free(recv_buf, q);

    return ret;
}

int main(int argc, char *argv[]) {
    tests_of_tests();

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    auto devices = select_devices();
    assert(devices.size() >= size);

    sycl::property_list props{ sycl::property::queue::in_order{} };
    sycl::queue q(devices[rank], props);

    /* create kvs */
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

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* run examples */
    int ret = 0;
    std::vector<size_t> offsets{ 0, sizeof(uint32_t), 1, 2, 3, 4, 5, 6, 7, 8 };
    for (size_t offset_bytes : offsets) {
        // smallest size that is fully and correctly divided without a remainder
        if (ret == 0)
            ret = sum_reduce_scatter(16, offset_bytes, q, comm, stream, size, rank);
        // other cases
        if (ret == 0)
            ret = sum_reduce_scatter(4, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(1, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(7, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(8, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(15, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(16, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(31, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(32, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(56, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(57, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(58, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(59, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(63, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(64, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096 - 7, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096 - 6, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096 - 5, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096 - 4, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096 - 2, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096 - 1, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096 + 1, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(40960, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(40961, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(40962, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(40963, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(409600, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(409601, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(409602, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(409603, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096000, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096001, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096002, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduce_scatter(4096003, offset_bytes, q, comm, stream, size, rank);
    }

    return ret;
}

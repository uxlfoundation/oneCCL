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
    // Find and initialize Level-Zero devices and queues
    std::vector<sycl::device> devices;
    std::vector<sycl::queue> queues;
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
        // endianess - system-default, no conversion
        target_bytes[i] = src_bytes[i];
    }
}

template <typename T>
T read_by_byte(void *src, size_t offset_bytes) {
    T ret{};
    uint8_t *target_bytes = reinterpret_cast<uint8_t *>(&ret);
    uint8_t *src_bytes = reinterpret_cast<uint8_t *>(src) + offset_bytes;
    for (size_t i = 0; i < sizeof(T); ++i) {
        // endianess - system-default, no conversion
        target_bytes[i] = src_bytes[i];
    }
    return ret;
}

void tests_of_tests() {
    uint32_t a = 0xAABBCCDD;
    uint32_t b = 0xEEFF0011;
    uint32_t c = 0x22334455;

    std::vector<uint8_t> d(3 * 4);

    write_by_byte<uint32_t>(d.data(), a, 0);
    write_by_byte<uint32_t>(d.data(), b, 4);
    write_by_byte<uint32_t>(d.data(), c, 8);

    // the test below works on little endian
    assert(d[3] == 0xAA);
    assert(d[2] == 0xBB);
    assert(d[1] == 0xCC);
    assert(d[0] == 0xDD);
    assert(d[7] == 0xEE);
    assert(d[6] == 0xFF);
    assert(d[5] == 0x00);
    assert(d[4] == 0x11);
    assert(d[11] == 0x22);
    assert(d[10] == 0x33);
    assert(d[9] == 0x44);
    assert(d[8] == 0x55);

    uint32_t aa = read_by_byte<uint32_t>(d.data(), 0);
    uint32_t bb = read_by_byte<uint32_t>(d.data(), 4);
    uint32_t cc = read_by_byte<uint32_t>(d.data(), 8);

    assert(aa == 0xAABBCCDD);
    assert(bb == 0xEEFF0011);
    assert(cc == 0x22334455);
}

int sum_reduction(size_t base_size,
                  size_t offset_bytes, // used for testing unaligned data
                  queue &q,
                  ccl::communicator &comm,
                  ccl::stream &stream,
                  int size,
                  int rank) {
    uint32_t send_canary[] = { 0xFFFFFFFF, 0xDDDDDDDD, 0xBBBBBBBB, 0x99999999,
                               0x77777777, 0x55555555, 0x44444444, 0x22222222,
                               0xFFFFFFFF, 0xDDDDDDDD, 0xBBBBBBBB, 0x99999999,
                               0x77777777, 0x55555555, 0x44444444, 0x22222222 };
    uint32_t recv_canary[] = { 0xEEEEEEEE, 0xCCCCCCCC, 0xAAAAAAAA, 0x88888888,
                               0x66666666, 0x44444444, 0x33333333, 0x11111111,
                               0xEEEEEEEE, 0xCCCCCCCC, 0xAAAAAAAA, 0x88888888,
                               0x66666666, 0x44444444, 0x33333333, 0x11111111 };

    size_t allocation_size =
        (base_size + sizeof(send_canary) / sizeof(send_canary[0])) * sizeof(uint32_t) +
        offset_bytes;
    size_t base_size_bytes = base_size * sizeof(uint32_t);

    std::vector<uint8_t> send_buf_cpu(allocation_size);
    std::vector<uint8_t> recv_buf_cpu(allocation_size);
    std::vector<uint8_t> expected_sum_cpu(allocation_size);

    uint8_t *send_buf = sycl::aligned_alloc_device<uint8_t>(4 * 1024, allocation_size, q);
    uint8_t *recv_buf = sycl::aligned_alloc_device<uint8_t>(4 * 1024, allocation_size, q);

    // add front canary
    for (size_t i = 0; i < offset_bytes; ++i) {
        send_buf_cpu[i] = 0xAA;
        expected_sum_cpu[i] = 0xFF;
    }
    // add data
    for (size_t i = 0; i < base_size; ++i) {
        size_t t_offset_bytes = i * sizeof(uint32_t) + offset_bytes;
        write_by_byte<uint32_t>(send_buf_cpu.data(), 1 + size + rank + i, t_offset_bytes);
        // the expected sum:
        // (size + size*size + sum(0:size) + i*size
        // size + size*size +size*(size-1)/2
        write_by_byte<uint32_t>(expected_sum_cpu.data(),
                                size + size * size + size * (size - 1) / 2 + i * size,
                                t_offset_bytes);
    }

    // add canary
    for (size_t t_offset_bytes = base_size_bytes + offset_bytes, k = 0;
         t_offset_bytes < allocation_size;
         t_offset_bytes += sizeof(uint32_t), ++k) {
        assert(k < sizeof(send_canary) / sizeof(send_canary[0]));
        assert(k < sizeof(recv_canary) / sizeof(recv_canary[0]));

        write_by_byte<uint32_t>(send_buf_cpu.data(), send_canary[k], t_offset_bytes);
        write_by_byte<uint32_t>(expected_sum_cpu.data(), recv_canary[k], t_offset_bytes);
    }

    q.memcpy(send_buf, send_buf_cpu.data(), send_buf_cpu.size()).wait();
    q.memset(recv_buf, static_cast<uint8_t>(0x00), allocation_size).wait();
    // copy recv canary
    q.memset(recv_buf, static_cast<uint8_t>(0xFF), offset_bytes).wait();
    q.memcpy(recv_buf + base_size_bytes + offset_bytes, recv_canary, sizeof(recv_canary)).wait();

    /* invoke allreduce */
    auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
    ccl::allreduce(
        (void *)(send_buf + offset_bytes),
        (void *)(recv_buf + offset_bytes),
        base_size,
        ccl::datatype::uint32,
        ccl::reduction::sum, // TODO check other reductions, they are currently unsupported on BMG
        comm,
        stream,
        attr)
        .wait();

    /* open recv_buf and check its correctness on the device side */
    q.memcpy(recv_buf_cpu.data(), recv_buf, recv_buf_cpu.size()).wait();
    int ret = 0;
    /* print out the result of the test on the host side */
    {
        size_t i;
        for (i = 0; i < allocation_size; ++i) {
            uint32_t received = recv_buf_cpu[i];
            uint32_t expected = expected_sum_cpu[i];
            uint32_t sent = send_buf_cpu[i];

            if (received != expected) {
                std::cout << "size " << base_size << ", offset bytes " << offset_bytes
                          << " failure at idx [" << i << "], element [" << i / sizeof(uint32_t)
                          << "]; found [" << std::hex << received << "] instead of [" << expected
                          << "]" << std::endl;
                std::cout << "send_buf_cpu[i]: " << sent << std::endl;
                std::cout << "FAILED\n";
                ret = 1;
                break;
            }
        }
        if (i == allocation_size) {
            std::cout << "size " << base_size << std::endl;
            std::cout << "PASSED\n";
        }
        else if (rank == 0) {
            // dump buffer
            std::cout << "out buffer contents: " << std::endl;
            for (size_t t = 0; t < allocation_size; ++t) {
                std::cout << std::hex << (int)recv_buf_cpu[t] << " ";
            }
            std::cout << std::endl << std::endl;
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
            ret = sum_reduction(16, offset_bytes, q, comm, stream, size, rank);
        // other cases
        if (ret == 0)
            ret = sum_reduction(4, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(1, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(7, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(8, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(15, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(16, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(31, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(32, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(56, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(57, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(58, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(59, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(63, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(64, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096 - 7, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096 - 6, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096 - 5, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096 - 4, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096 - 2, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096 - 1, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096 + 1, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(40960, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(40961, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(40962, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(40963, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(409600, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(409601, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(409602, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(409603, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096000, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096001, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096002, offset_bytes, q, comm, stream, size, rank);
        if (ret == 0)
            ret = sum_reduction(4096003, offset_bytes, q, comm, stream, size, rank);
    }

    return ret;
}

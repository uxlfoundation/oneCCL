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

int broadcast_test(sycl::queue &q,
                   size_t element_count,
                   size_t offset_bytes,
                   ccl::datatype dtype,
                   ccl::communicator &comm,
                   ccl::stream &stream,
                   int comm_size,
                   int rank) {
    int32_t send_canary[] = { 0x7FFF, 0x6DDD, 0x5BBB, 0x4999, 0x3777, 0x2555, 0x1444, 0x0222,
                              0x7FFF, 0x6DDD, 0x5BBB, 0x4999, 0x3777, 0x2555, 0x1444, 0x0222 };
    int32_t recv_canary[] = { 0x6EEE, 0x5CCC, 0x4AAA, 0x3888, 0x2666, 0x1444, 0x0333, 0x0111,
                              0x6EEE, 0x5CCC, 0x4AAA, 0x3888, 0x2666, 0x1444, 0x0333, 0x0111 };

    size_t canary_count = sizeof(send_canary) / sizeof(send_canary[0]);
    size_t canary_size_bytes = canary_count * sizeof(int32_t);
    size_t main_data_bytes = element_count * sizeof(int32_t);
    size_t allocation_size = offset_bytes + main_data_bytes + canary_size_bytes;

    std::vector<uint8_t> send_buf_cpu(allocation_size, 0);
    std::vector<uint8_t> recv_buf_cpu(allocation_size, 0);
    std::vector<uint8_t> expected_recv_cpu(allocation_size, 0);

    int root_rank = 1;
    // Allocate device buffers
    uint8_t *send_buf = nullptr;
    if (rank == root_rank) {
        send_buf = sycl::aligned_alloc_device<uint8_t>(4 * 1024, allocation_size, q);
    }
    uint8_t *recv_buf = sycl::aligned_alloc_device<uint8_t>(4 * 1024, allocation_size, q);

    // Initialize front canary
    for (size_t i = 0; i < offset_bytes; ++i) {
        send_buf_cpu[i] = 0xAA;
        expected_recv_cpu[i] = 0xFF;
    }

    // Debug: Print initialization info
    std::cout << "Rank " << rank << ": element_count=" << element_count
              << ", offset_bytes=" << offset_bytes << ", comm_size=" << comm_size
              << ", allocation_size=" << allocation_size << std::endl;

    // Initialize send data: each rank sends different data to each destination
    // Data for src is: (rank * element_count + element_index)

    for (size_t i = 0; i < element_count; ++i) {
        size_t t_offset_bytes = i * sizeof(int32_t) + offset_bytes;
        if (rank == root_rank) {
            write_by_byte<int32_t>(send_buf_cpu.data(), element_count + i, t_offset_bytes);
        }
        write_by_byte<int32_t>(expected_recv_cpu.data(), element_count + i, t_offset_bytes);
    }

    std::cout << "\n=== Rank " << rank << " Initialize Complete ===" << std::endl;
    // Initialize back canary
    size_t canary_offset = main_data_bytes + offset_bytes;
    for (size_t i = 0; i < canary_size_bytes; i += sizeof(int32_t)) {
        size_t canary_idx = i / sizeof(int32_t);
        size_t current_offset = canary_offset + i;

        if (canary_idx < sizeof(send_canary) / sizeof(send_canary[0])) {
            write_by_byte<int32_t>(send_buf_cpu.data(), send_canary[canary_idx], current_offset);
            write_by_byte<int32_t>(
                expected_recv_cpu.data(), recv_canary[canary_idx], current_offset);
        }
    }

    // Copy data to device
    if (send_buf) {
        q.memcpy(send_buf, send_buf_cpu.data(), send_buf_cpu.size()).wait();
    }
    q.memset(recv_buf, static_cast<uint8_t>(0x00), allocation_size).wait();
    q.memset(recv_buf, static_cast<uint8_t>(0xFF), offset_bytes)
        .wait(); // Initialize front canary with 0xFF
    q.memcpy(recv_buf + offset_bytes + main_data_bytes, recv_canary, canary_size_bytes).wait();

    // Execute broadcast
    auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();

    std::cout << "Rank " << rank << ": broadcast start" << std::endl;

    // don't offset nullptr
    void *send_buf_ptr = (void *)(send_buf ? send_buf + offset_bytes : send_buf);
    ccl::broadcast(send_buf_ptr,
                   (void *)(recv_buf + offset_bytes),
                   element_count,
                   ccl::datatype::int32,
                   root_rank,
                   comm,
                   stream,
                   attr)
        .wait();

    std::cout << "Rank " << rank << ": broadcast completed" << std::endl;

    // Copy result back to host
    q.memcpy(recv_buf_cpu.data(), recv_buf, recv_buf_cpu.size()).wait();

    // Verify results
    int ret = 0;
    bool has_error = false;
    {
        size_t i;
        for (i = 0; i < allocation_size; ++i) {
            int32_t received = recv_buf_cpu[i];
            int32_t expected = expected_recv_cpu[i];
            if (received != expected) {
                std::cout << "Size " << element_count << ", offset bytes " << offset_bytes
                          << " failure at idx [" << i << "], element [" << i / sizeof(int32_t)
                          << "]; found [" << std::hex << received << "] instead of [" << expected
                          << "]" << std::endl;
                has_error = true;
                ret = 1;
                break;
            }
        }

        if (i == allocation_size) {
            ret = 0;
        }
        else if (rank == 0) {
            // dump buffer
            std::cout << "rank " << rank << " out buffer contents: " << std::endl;
            for (size_t t = 0; t < allocation_size; ++t) {
                std::cout << std::hex << (int)recv_buf_cpu[t] << " ";
            }
            std::cout << std::endl;
        }
    }

    if (!has_error) {
        std::cout << "rank" << rank << ": element_count " << element_count << ", offset bytes "
                  << offset_bytes << " PASSED\n"
                  << std::endl;
    }
    else {
        std::cout << "rank" << rank << ": element_count " << element_count << ", offset bytes "
                  << offset_bytes << " FAILED\n"
                  << std::endl;
    }

    sycl::free(send_buf, q);
    sycl::free(recv_buf, q);

    return ret;
}

int main(int argc, char *argv[]) {
    tests_of_tests();
    int comm_size = 0;
    int rank = 0;

    ccl::init();
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    auto devices = select_devices();
    if (devices.size() < static_cast<size_t>(comm_size)) {
        std::cerr << "Error: Not enough GPUs available. Need " << comm_size << ", found "
                  << devices.size() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    sycl::property_list props{ sycl::property::queue::in_order{} };
    sycl::queue q(devices[rank], props);

    // Create KVS
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

    // Create communicator
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(comm_size, rank, dev, ctx, kvs);

    // Create stream
    auto stream = ccl::create_stream(q);

    // Run broadcast tests
    int ret = 0;
    //Front canary offset, now unimplement rt front unaligment , so default set 0
    std::vector<size_t> offsets{ 0, sizeof(int32_t), 1, 2, 3, 4, 5, 6, 7, 8 };
    // broadcast elment count
    std::vector<size_t> element_counts{ 1,  3,  5,  7,  8,  15,   16,   31,   32,    56,
                                        57, 58, 59, 63, 64, 4095, 4096, 4097, 10240, 10241 };

    if (rank == 0) {
        std::cout << "\n=== Starting broadcast Tests ===" << std::endl;
        std::cout << "Total MPI ranks: " << comm_size << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t offset_bytes : offsets) {
        for (size_t element_count : element_counts) {
            if (rank == 0) {
                std::cout << "\n=== Testing: element_count=" << element_count
                          << ", comm_size=" << comm_size << ", offset_bytes=" << offset_bytes
                          << " ===" << std::endl;
            }

            MPI_Barrier(MPI_COMM_WORLD);

            int local_ret = broadcast_test(q,
                                           element_count,
                                           offset_bytes,
                                           ccl::datatype::int32,
                                           comm,
                                           stream,
                                           comm_size,
                                           rank);

            if (local_ret != 0) {
                ret = local_ret;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        if (ret == 0) {
            std::cout << "\n=== All tests PASSED ===" << std::endl;
        }
        else {
            std::cout << "\n=== Some tests FAILED ===" << std::endl;
        }
    }

    return ret;
}

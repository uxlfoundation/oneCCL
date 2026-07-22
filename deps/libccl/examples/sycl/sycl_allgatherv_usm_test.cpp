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

#include <thread>

#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

int main(int argc, char *argv[]) {
    size_t count = 10 * 1024 * 1024;
    size_t loops = 1;

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);
    if (!check_example_args(argc, argv))
        exit(1);

    if (args.count != args.DEFAULT_COUNT) {
        count = args.count;
    }

    if (args.loops != args.DEFAULT_LOOPS) {
        loops = args.loops;
    }

    string device_type = argv[1];
    string alloc_type = argv[2];

    sycl::queue q;
    if (!create_test_sycl_queue(device_type, rank, q, args))
        return -1;

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm_alloc_type_from_string(alloc_type);

    if (!check_sycl_usm(q, usm_alloc_type)) {
        return -1;
    }

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

    /* create buffers */
    auto send_buf = allocator.allocate(count, usm_alloc_type);
    auto recv_buf = allocator.allocate(count * size, usm_alloc_type);

    sycl::buffer<int> expected_buf(count * size);
    sycl::buffer<int> check_buf(count * size);
    std::vector<size_t> recv_counts(size, count);

    for (size_t loop = 0; loop < loops; loop++)
    {
        /* open buffers and modify them on the device side */
        auto e = q.submit([&](auto &h) {
            sycl::accessor expected_buf_acc(expected_buf, h, sycl::write_only);
            h.parallel_for(count, [=](auto id) {
                send_buf[id] = rank + 1;
                for (int i = 0; i < size; i++) {
                    expected_buf_acc[i * count + id] = i + 1;
                }
            });
        });

        /* do not wait completion of kernel and provide it as dependency for operation */
        vector<ccl::event> deps;
        deps.push_back(ccl::create_event(e));

        /* invoke allgatherv */
        auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();
        ccl::event ev = ccl::allgatherv(
            send_buf, count, recv_buf, recv_counts, ccl::datatype::int32, comm, stream, attr, deps);
        ev.wait();

        /* open recv_buf and check its correctness on the device side */
        q.submit([&](auto &h) {
            sycl::accessor expected_buf_acc(expected_buf, h, sycl::read_only);
            sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
            h.parallel_for(size * count, [=](auto id) {
                if (recv_buf[id] != expected_buf_acc[id]) {
                    check_buf_acc[id] = -1;
                }
                else {
                    check_buf_acc[id] = 0;
                }
            });
        });

        if (!handle_exception(q))
            return -1;

        if (args.is_verbose && rank == 0 && (loop == 0 || loop == loops - 1))
        {
            std::cout << "Verbose mode enabled, printing results\n";
            
            // allocate host buffer and copy data from device to host for printing
            size_t total_count = count * size;
            std::vector<int> host_recv_buf(total_count);
            q.memcpy(host_recv_buf.data(), recv_buf, total_count * sizeof(int)).wait();
            
            size_t print_count = std::min(total_count, 32UL); // limit print size
            std::cout << "Rank(" << rank << ") Allgatherv first 32 results: {";
            for (size_t i = 0; i < print_count; i++)
            {
                if (i % 32 == 0)
                {
                    std::cout << "\n";
                }
                std::cout << host_recv_buf[i] << ", ";
            }
            std::cout << "\n}\n";

            if (total_count > print_count * 2)
            {
                std::cout << "...\n";
                std::cout << "Rank(" << rank << ") Allgatherv last 32 results: {";
                for (size_t i = total_count - print_count; i < total_count; i++)
                {
                    if (i % 32 == 0)
                    {
                        std::cout << "\n";
                    }
                    std::cout << host_recv_buf[i] << ", ";
                }
                std::cout << "\n}\n";
            }
            
        }

        /* print out the result of the test on the host side */
        {
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            size_t i;
            for (i = 0; i < size * count; i++) 
            {
                if (check_buf_acc[i] == -1) 
                {
                    std::cout << "loop(" << loop << ") FAILED\n";
                    break;
                }
            }
            if (i == size * count) 
            {
                std::cout << "loop(" << loop << ") PASSED\n";
            }
        }
    }

    return 0;
}

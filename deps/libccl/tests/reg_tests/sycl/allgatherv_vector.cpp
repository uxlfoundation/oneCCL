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

#include <numeric>
#include <vector>
#include <iostream>

#include "sycl_base.hpp"

int main(int argc, char *argv[]) {
    std::list<int> elem_counts = { 0, 1, 2, 16, 17, 1024, 1025, 32768, 262145 };

    int size;
    int rank;
    int use_inplace = 0;

    int fail_counter = 0;

    if (argc > 1) {
        use_inplace = atoi(argv[1]);
    }

    std::cout << "use_inplace: " << use_inplace << "\n";

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    atexit(mpi_finalize);

    test_args args(argc, argv, rank);
    sycl::queue q;
    if (!create_test_sycl_queue("gpu", rank, q, args))
        return -1;

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
    for (auto elem_count : elem_counts) {
        std::cout << "check elem_count: " << elem_count << "\n";

        /* create buffers */
        std::vector<int *> recv_bufs;
        for (int i = 0; i < size; ++i) {
            recv_bufs.push_back(sycl::malloc_device<int>(elem_count, q));
        }
        auto send_buf = (use_inplace) ? recv_bufs[rank] : sycl::malloc_device<int>(elem_count, q);

        /* sycl buffers can't be empty so for the elem_count == 0 case 
           allocate single-element buffer */
        sycl::buffer<int> expected_buf((elem_count ? elem_count : 1) * size);
        sycl::buffer<int> check_buf((elem_count ? elem_count : 1) * size);
        std::vector<size_t> recv_counts(size, elem_count);

        std::vector<sycl::event> events;

        /* fill recv buffers */
        for (int i = 0; i < size; ++i) {
            if (use_inplace && (i == rank)) {
                /* buffer will be filled with send values in separate kernel below */
                continue;
            }
            events.push_back(q.submit([&](auto &h) {
                h.parallel_for(elem_count, [=, rb = recv_bufs[i]](auto id) {
                    rb[id] = -1;
                });
            }));
        }

        /* fill send buffer and expected buffer */
        events.push_back(q.submit([&](auto &h) {
            sycl::accessor expected_buf_acc(expected_buf, h, sycl::write_only);
            h.parallel_for(elem_count, [=, rnk = rank, sz = size](auto id) {
                send_buf[id] = rnk + id;
                for (int send_rank = 0; send_rank < sz; send_rank++) {
                    expected_buf_acc[send_rank * elem_count + id] = send_rank + id;
                }
            });
        }));

        /* do not wait completion of kernels and provide them as dependency for operation */
        std::vector<ccl::event> deps;
        for (auto e : events) {
            deps.push_back(ccl::create_event(e));
        }

        /* invoke allgatherv */
        auto attr = ccl::create_operation_attr<ccl::allgatherv_attr>();
        ccl::allgatherv(send_buf, elem_count, recv_bufs, recv_counts, comm, stream, attr, deps)
            .wait();

        /* open recv_buf and check its correctness on the device side */
        for (int recv_rank = 0; recv_rank < size; ++recv_rank) {
            q.submit([&](auto &h) {
                 sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
                 h.parallel_for(elem_count, [=, rb = recv_bufs[recv_rank]](auto id) {
                     check_buf_acc[recv_rank * elem_count + id] = rb[id];
                 });
             }).wait();
        }

        bool print_errors = true;
        /* print out the result of the test on the host side */
        {
            sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
            sycl::host_accessor expected_buf_acc(expected_buf, sycl::read_only);
            for (size_t recv_rank = 0; recv_rank < (size_t)size; ++recv_rank) {
                for (size_t id = 0; id < (size_t)elem_count; id++) {
                    size_t elem_id = recv_rank * elem_count + id;
                    if (check_buf_acc[elem_id] != expected_buf_acc[elem_id]) {
                        fail_counter++;
                        if (print_errors) {
                            // could probably be more efficient, but it's just test
                            std::cout << "rank " << rank << ", elem_count: " << elem_count
                                      << ", unexpected value at rank " << recv_rank << " element "
                                      << id << ": " << check_buf_acc[elem_id] << ", expected "
                                      << expected_buf_acc[elem_id] << "\n ";
                        }
                        break;
                    }
                }
            }
        }

        /* print out the contents of the buffers on the host side */
        bool print_buf_contents = false;
        if (elem_count < 20 && print_buf_contents) {
            sycl::buffer<int> tmp_send_buf(elem_count);
            sycl::buffer<int> tmp_recv_buf(elem_count * size);

            q.submit([&](auto &h) {
                 sycl::accessor tmp_send_buf_acc(tmp_send_buf, h, sycl::write_only);
                 h.parallel_for(elem_count, [=](auto id) {
                     tmp_send_buf_acc[id] = send_buf[id];
                 });
             }).wait();
            for (int i = 0; i < size; ++i) {
                q.submit([&](auto &h) {
                     sycl::accessor tmp_recv_buf_acc(tmp_recv_buf, h, sycl::write_only);
                     h.parallel_for(elem_count, [=, rb = recv_bufs[i]](auto id) {
                         tmp_recv_buf_acc[i * elem_count + id] = rb[id];
                     });
                 }).wait();
            }

            sycl::host_accessor send_buf_acc(tmp_send_buf, sycl::read_only);
            sycl::host_accessor recv_buf_acc(tmp_recv_buf, sycl::read_only);
            sycl::host_accessor expected_buf_acc(expected_buf, sycl::read_only);

            std::string s;

            for (int i = 0; i < elem_count; i++) {
                s += " elem " + std::to_string(i) + " send " + std::to_string(send_buf_acc[i]) +
                     " expected ";
                for (int j = 0; j < size; ++j) {
                    s += std::to_string(expected_buf_acc[i + elem_count * j]) + ",";
                }
                s += " recv[rank] ";
                for (int j = 0; j < size; ++j) {
                    // exchange printing order
                    s += std::to_string(recv_buf_acc[i + j * elem_count]) + ",";
                }
                s += "\n";
            }

            std::cout << "rank " << rank << " " << s;
        }
        for (size_t i = 0; i < recv_bufs.size(); ++i) {
            sycl::free(recv_bufs[i], q);
        }

        if (!use_inplace) {
            sycl::free(send_buf, q);
        }

        if (!handle_exception(q)) {
            return -1;
        }
    } // for elem_counts

    if (fail_counter) {
        std::cout << "FAILED\n";
        return -1;
    }
    else {
        std::cout << "PASSED\n";
        return 0;
    }
}
